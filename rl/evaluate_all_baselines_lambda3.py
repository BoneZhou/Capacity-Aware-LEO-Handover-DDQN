from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RL_DIR))

import Dueling_ddqn as proposed_module  # noqa: E402
import Dueling_ddqn_no_mshbo as dueling_module  # noqa: E402
import Standard_ddqn as standard_module  # noqa: E402
from config_loader import experiment_config, method_config  # noqa: E402

_SCENARIO = experiment_config()["ue_sensitivity"]
UE_COUNTS = _SCENARIO["ue_counts"]
CAPACITY = _SCENARIO["satellite_capacity"]
ORDER_SEED = 42
TOPK = method_config()["rl"]["top_k"]
INPUT_PATH = ROOT / "input.txt"
RATE_PATH = ROOT / "input_angle.txt"
NONRL_TP = ROOT / "evaluation_results" / "nonrl_throughput_mapping_fix_20261017_224524" / "nonrl_throughput_c5_corrected.csv"
NONRL_HB = ROOT / "improvement" / "results_handover_blocking.csv"
PROPOSED_METRICS = ROOT / "evaluation_results" / "proposed_vs_nonrl_preview_20261017_210355" / "proposed_rl_metrics_c5.csv"

METHODS = {
    "Proposed": {
        "module": proposed_module,
        "network": proposed_module.QNet,
        "checkpoint": RL_DIR / "runs" / "proposed_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_imi.pt",
        "reuse": True,
    },
    "Dueling DDQN": {
        "module": dueling_module,
        "network": dueling_module.QNet,
        "checkpoint": RL_DIR / "runs" / "no_mshbo_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_no_mshbo.pt",
        "reuse": False,
    },
    "Standard DDQN": {
        "module": standard_module,
        "network": standard_module.QNet,
        "checkpoint": RL_DIR / "runs" / "ddqn_persistent_topk_seed42_300ep" / "standard_ddqn_no_mshbo.pt",
        "reuse": False,
    },
}

METHOD_ORDER = ["GBW2", "MAC", "MVT", "MSH", "MSHBO", "Standard DDQN", "Dueling DDQN", "Proposed"]
STYLE = {
    "GBW2": dict(color="#2ca02c", marker="*", linestyle="-", markersize=11),
    "MAC": dict(color="#1f77b4", marker="P", linestyle="-", markersize=9),
    "MVT": dict(color="#9467bd", marker="X", linestyle="-", markersize=9),
    "MSH": dict(color="#ff7f0e", marker="s", linestyle="-", markersize=8),
    "MSHBO": dict(color="#d62728", marker="^", linestyle="-", markersize=8),
    "Standard DDQN": dict(color="#7f7f7f", marker="d", linestyle="--", markersize=8),
    "Dueling DDQN": dict(color="#17becf", marker="v", linestyle="--", markersize=8),
    "Proposed": dict(color="black", marker="o", linestyle="-", markersize=8),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_models() -> tuple[dict, dict]:
    models, provenance = {}, {}
    topk_sources = {inspect.getsource(spec["module"].topk_mask) for spec in METHODS.values()}
    if len(topk_sources) != 1:
        raise RuntimeError("The three formal topk_mask implementations differ")
    for method, spec in METHODS.items():
        checkpoint = spec["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = spec["network"](251, 51)
        loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"Strict checkpoint load failed: {method}")
        model.eval()
        models[method] = model
        provenance[method] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}

    no_mshbo_metadata = json.loads(
        (RL_DIR / "runs" / "no_mshbo_persistent_topk_seed42_300ep" / "run_metadata.json").read_text(encoding="utf-8")
    )
    if float(no_mshbo_metadata["reward_weights"]["handover"]) != 3.0:
        raise RuntimeError("No-MSHBO run is not lambda_handover=3.0")
    return models, provenance


def create_scenario(destination: Path, ue_count: int) -> None:
    lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    n, _, tb, te = map(int, lines[0].replace(",", " ").split()[:4])
    if len(lines) != n + 1:
        raise RuntimeError("Unexpected input.txt satellite row count")
    satellites = []
    for line in lines[1:]:
        begin, end, _ = map(int, line.split())
        satellites.append(f"{begin} {end} {CAPACITY}")
    destination.write_text("\n".join([f"{n} {ue_count} {tb} {te}"] + satellites) + "\n", encoding="utf-8")


def evaluate(method: str, model: torch.nn.Module, scenario: Path, policy: Path, provenance: dict) -> dict:
    module = METHODS[method]["module"]
    env = proposed_module.HandoverEnv(
        str(scenario), str(RATE_PATH), lambda_h=3.0,
        seed=ORDER_SEED, training_order_seed=ORDER_SEED,
        demonstration_order_seed=ORDER_SEED, evaluation_order_seed=ORDER_SEED,
    )
    env.reset_multi_user_episode()
    env.reset_evaluation_order_rng(ORDER_SEED)
    blocked = np.zeros(env.m, dtype=np.int64)
    handovers = np.zeros(env.m, dtype=np.int64)
    rate_sum = np.zeros(env.m, dtype=np.float64)
    mask_failures = 0

    with policy.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "user", "sat"])
        with torch.inference_mode():
            for t in range(env.T):
                for raw_user in env.next_evaluation_user_order():
                    user = int(raw_user)
                    previous = int(env.last_sat[user])
                    state = env.make_state(user, t)
                    full_legal = env.legal_mask(user, t)
                    selection = module.topk_mask(
                        full_legal, env.used, env.cap, t, previous, env.block_action, TOPK
                    )
                    if selection[env.block_action] <= 0 or np.any((selection > 0) & (full_legal <= 0)):
                        mask_failures += 1
                    if previous >= 0 and full_legal[previous] > 0 and selection[previous] <= 0:
                        mask_failures += 1
                    action = module.masked_argmax(model(torch.from_numpy(state)), selection)
                    if selection[action] <= 0 or full_legal[action] <= 0:
                        mask_failures += 1
                    _, was_blocked = env.step_user(user, t, action)
                    writer.writerow([t + env.Tb, user, -1 if was_blocked else int(action)])
                    if was_blocked:
                        blocked[user] += 1
                    else:
                        rate_sum[user] += float(env.rate[action, t])
                    if env.last_reward_components[user]["handover"] < 0:
                        handovers[user] += 1
    if mask_failures:
        raise RuntimeError(f"Top-K mask failures for {method}, UE={env.m}: {mask_failures}")
    return {
        "method": method,
        "ue_count": env.m,
        "capacity": CAPACITY,
        "average_throughput": float(rate_sum.sum()) / (env.m * env.T),
        "average_handover_count": int(handovers.sum()) / env.m,
        "average_blocking_rate": int(blocked.sum()) / env.T,
        "metric_source": "rl_evaluation_lambda3_topk3_persistent",
        "throughput_source": "RL evaluator",
        "handover_blocking_source": "RL evaluator",
        "checkpoint_path": provenance["path"],
        "checkpoint_sha256": provenance["sha256"],
        "policy_path": str(policy.resolve()),
        "policy_sha256": sha256(policy),
        "evaluation_order_seed": ORDER_SEED,
        "top_k": TOPK,
        "capacity_accounting": "connection_persistent",
    }


def load_proposed(provenance: dict) -> list[dict]:
    frame = pd.read_csv(PROPOSED_METRICS)
    expected_hash = provenance["sha256"]
    if len(frame) != 5 or set(frame["ue_count"]) != set(UE_COUNTS):
        raise RuntimeError("Incomplete reusable Proposed metrics")
    checks = (
        (frame["checkpoint_sha256"].str.upper() == expected_hash).all()
        and (frame["capacity"] == CAPACITY).all()
        and (frame["evaluation_order_seed"] == ORDER_SEED).all()
        and (frame["top_k"] == TOPK).all()
        and (frame["capacity_accounting"] == "connection_persistent").all()
    )
    if not checks:
        raise RuntimeError("Reusable Proposed evaluation provenance/settings mismatch")
    rows = []
    for _, row in frame.sort_values("ue_count").iterrows():
        rows.append(
            {
                "method": "Proposed",
                "ue_count": int(row["ue_count"]),
                "capacity": CAPACITY,
                "average_throughput": float(row["average_throughput"]),
                "average_handover_count": float(row["average_handover_count"]),
                "average_blocking_rate": float(row["average_blocking_rate"]),
                "metric_source": "reused_formal_rl_evaluation_lambda3_topk3_persistent",
                "throughput_source": str(PROPOSED_METRICS.resolve()),
                "handover_blocking_source": str(PROPOSED_METRICS.resolve()),
                "checkpoint_path": provenance["path"],
                "checkpoint_sha256": expected_hash,
                "policy_path": str(row["policy_path"]),
                "policy_sha256": str(row["policy_sha256"]),
                "evaluation_order_seed": ORDER_SEED,
                "top_k": TOPK,
                "capacity_accounting": "connection_persistent",
            }
        )
    return rows


def load_nonrl() -> list[dict]:
    tp = pd.read_csv(NONRL_TP)
    hb = pd.read_csv(NONRL_HB)
    rows = []
    for display in METHOD_ORDER[:5]:
        hb_name = "GBW" if display == "GBW2" else display
        for ue in UE_COUNTS:
            tp_row = tp[(tp["algorithm"] == display) & (tp["m"] == ue) & (tp["c"] == CAPACITY)]
            hb_row = hb[(hb["algorithm"] == hb_name) & (hb["m"] == ue) & (hb["c"] == CAPACITY)]
            if len(tp_row) != 1 or len(hb_row) != 1:
                raise RuntimeError(f"Missing/duplicate non-RL row: {display}, UE={ue}")
            rows.append(
                {
                    "method": display,
                    "ue_count": ue,
                    "capacity": CAPACITY,
                    "average_throughput": float(tp_row.iloc[0]["avg_throughput_all_users_mbps"]),
                    "average_handover_count": float(hb_row.iloc[0]["Average h"]),
                    "average_blocking_rate": float(hb_row.iloc[0]["avg_blocking_rate"]),
                    "metric_source": "nonrl_formal_sources",
                    "throughput_source": str(NONRL_TP.resolve()),
                    "handover_blocking_source": str(NONRL_HB.resolve()),
                    "checkpoint_path": "",
                    "checkpoint_sha256": "",
                    "policy_path": "",
                    "policy_sha256": "",
                    "evaluation_order_seed": "",
                    "top_k": "",
                    "capacity_accounting": "",
                }
            )
    return rows


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for extension in ("png", "eps"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_vs_ues(data: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, stem: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    for method in METHOD_ORDER:
        rows = data[data["method"] == method].sort_values("ue_count")
        ax.plot(rows["ue_count"], rows[metric], label=method, **STYLE[method])
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(UE_COUNTS)
    ax.grid(True, linestyle=":", color="lightgray")
    ax.legend(ncol=2)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_tradeoff(data: pd.DataFrame, output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for method in METHOD_ORDER:
        rows = data[data["method"] == method].sort_values("ue_count")
        ax.plot(rows["average_handover_count"], rows["average_blocking_rate"], label=method, **STYLE[method])
    ax.set_xlabel("Average Handover Count")
    ax.set_ylabel("Average Blocking Rate")
    ax.grid(True, linestyle=":", color="lightgray")
    ax.legend(ncol=2)
    fig.tight_layout()
    return save_figure(fig, output_dir, "tradeoff_blocking_vs_handover")


def main() -> None:
    models, provenance = load_models()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "ue_sensitivity" / f"all_baselines_lambda3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    scenario_dir = output_dir / "scenario_inputs"
    policy_dir = output_dir / "rl_policies"
    scenario_dir.mkdir()
    policy_dir.mkdir()

    rl_rows = load_proposed(provenance["Proposed"])
    for ue in UE_COUNTS:
        scenario = scenario_dir / f"input_m{ue}_c{CAPACITY}.txt"
        create_scenario(scenario, ue)
        for method in ("Dueling DDQN", "Standard DDQN"):
            print(f"Evaluating {method}: UE={ue}, capacity={CAPACITY}", flush=True)
            policy = policy_dir / f"{method.replace(' ', '_')}_m{ue}_c{CAPACITY}_topk3_policy.csv"
            row = evaluate(method, models[method], scenario, policy, provenance[method])
            rl_rows.append(row)
            print(
                f"  throughput={row['average_throughput']:.6f}, "
                f"handover={row['average_handover_count']:.6f}, "
                f"blocking={row['average_blocking_rate']:.6f}", flush=True,
            )

    rl_metrics = pd.DataFrame(rl_rows).sort_values(["method", "ue_count"]).reset_index(drop=True)
    rl_metrics.to_csv(output_dir / "rl_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    merged = pd.DataFrame(load_nonrl() + rl_rows)
    merged["method"] = pd.Categorical(merged["method"], METHOD_ORDER, ordered=True)
    merged = merged.sort_values(["method", "ue_count"]).reset_index(drop=True)
    merged["method"] = merged["method"].astype(str)
    if len(merged) != 40 or merged.duplicated(["method", "ue_count"]).any():
        raise RuntimeError("Merged plotting data is incomplete or duplicated")
    required = ["average_throughput", "average_handover_count", "average_blocking_rate"]
    if merged[required].isna().any().any():
        raise RuntimeError("Merged plotting data contains missing metrics")
    merged_path = output_dir / "merged_plot_data.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    configure_plotting()
    figures = {}
    figures["throughput"] = plot_vs_ues(
        merged, output_dir, "average_throughput", "Average Throughput (Mbps)", "throughput_vs_ues"
    )
    figures["handover"] = plot_vs_ues(
        merged, output_dir, "average_handover_count", "Average Handover Count", "handover_vs_ues"
    )
    figures["blocking"] = plot_vs_ues(
        merged, output_dir, "average_blocking_rate", "Average Blocking Rate", "blocking_vs_ues"
    )
    figures["tradeoff"] = plot_tradeoff(merged, output_dir)

    lines = [
        "lambda_handover=3.0 全 Baseline 比較",
        "",
        "RL Checkpoints",
    ]
    for method in ("Proposed", "Dueling DDQN", "Standard DDQN"):
        lines += [f"{method}: {provenance[method]['path']}", f"SHA-256: {provenance[method]['sha256']}"]
    lines += [
        "",
        f"Proposed metrics reused: {PROPOSED_METRICS.resolve()}",
        "Dueling DDQN and Standard DDQN: newly evaluated with Persistent Capacity, Top-K=3, capacity=5, evaluation order seed=42.",
        "Standard DDQN training provenance was not audited, as requested; only strict checkpoint loading and current evaluator execution were required.",
        f"Non-RL throughput: {NONRL_TP.resolve()}",
        f"Non-RL handover/blocking: {NONRL_HB.resolve()} (GBW source displayed as GBW2)",
        "Blocking definition: Average Blocked UE Count per Slot.",
        "Trade-off axes: X=Average Handover Count, Y=Average Blocking Rate.",
        "",
        f"Merged data: {merged_path.resolve()}",
        "Figures:",
    ]
    for paths in figures.values():
        lines.extend(paths)
    summary = output_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"OUTPUT_DIR={output_dir.resolve()}", flush=True)
    print(f"SUMMARY={summary.resolve()}", flush=True)


if __name__ == "__main__":
    main()
