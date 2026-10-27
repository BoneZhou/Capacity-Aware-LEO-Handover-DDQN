from __future__ import annotations

import hashlib
import inspect
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from config_loader import experiment_config, method_config, simulation_config  # noqa: E402

_SIM = simulation_config()
_EXPERIMENTS = experiment_config()
_SCENARIO = _EXPERIMENTS["ue_sensitivity"]
SEEDS = _SIM["evaluation_order_seeds"]
NEW_SEEDS = [seed for seed in SEEDS if seed != 42]
UE_COUNTS = _SCENARIO["ue_counts"]
CAPACITY = _SCENARIO["satellite_capacity"]
TOPK = method_config()["rl"]["top_k"]
RATE_PATH = ROOT / "input_angle.txt"
INPUT_PATH = ROOT / "input.txt"
METHOD_ORDER = ["Standard DDQN", "Dueling DDQN", "Proposed"]
METRICS = ["average_throughput", "average_handover_count", "average_blocking_rate"]

SPECS = {
    "Proposed": {
        "module": proposed_module,
        "network": proposed_module.QNet,
        "checkpoint": RL_DIR / "runs" / "proposed_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_imi.pt",
    },
    "Dueling DDQN": {
        "module": dueling_module,
        "network": dueling_module.QNet,
        "checkpoint": RL_DIR / "runs" / "no_mshbo_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_no_mshbo.pt",
    },
    "Standard DDQN": {
        "module": standard_module,
        "network": standard_module.QNet,
        "checkpoint": RL_DIR / "runs" / "ddqn_persistent_topk_seed42_300ep" / "standard_ddqn_no_mshbo.pt",
    },
}

STYLE = {
    "Standard DDQN": dict(color="#7f7f7f", marker="d", linestyle="--"),
    "Dueling DDQN": dict(color="#17becf", marker="v", linestyle="--"),
    "Proposed": dict(color="black", marker="o", linestyle="-"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_provenance() -> dict:
    provenance = {}
    sources = {inspect.getsource(spec["module"].topk_mask) for spec in SPECS.values()}
    if len(sources) != 1:
        raise RuntimeError("Formal topk_mask implementations differ")
    for method, spec in SPECS.items():
        path = spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        model = spec["network"](251, 51)
        loaded = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"Strict checkpoint load failed: {method}")
        provenance[method] = {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256(path)}

    no_mshbo_metadata = json.loads(
        (RL_DIR / "runs" / "no_mshbo_persistent_topk_seed42_300ep" / "run_metadata.json").read_text(encoding="utf-8")
    )
    if float(no_mshbo_metadata["reward_weights"]["handover"]) != 3.0:
        raise RuntimeError("No-MSHBO checkpoint is not from lambda_handover=3.0 run")
    return provenance


def create_scenarios(directory: Path) -> dict[int, str]:
    lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    n, _, tb, te = map(int, lines[0].replace(",", " ").split()[:4])
    if len(lines) != n + 1:
        raise RuntimeError("Unexpected input.txt row count")
    rows = []
    for line in lines[1:]:
        begin, end, _ = map(int, line.split())
        rows.append(f"{begin} {end} {CAPACITY}")
    paths = {}
    for ue in UE_COUNTS:
        path = directory / f"input_m{ue}_c{CAPACITY}.txt"
        path.write_text("\n".join([f"{n} {ue} {tb} {te}"] + rows) + "\n", encoding="utf-8")
        paths[ue] = str(path.resolve())
    return paths


def find_seed42_results(provenance: dict) -> tuple[pd.DataFrame, Path]:
    candidates = sorted((ROOT / "outputs" / "ue_sensitivity").glob("all_baselines_lambda3_*"), reverse=True)
    for directory in candidates:
        path = directory / "rl_evaluation_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if set(frame["method"]) != set(METHOD_ORDER) or set(frame["ue_count"]) != set(UE_COUNTS):
            continue
        if len(frame) != 15 or not (frame["evaluation_order_seed"] == 42).all():
            continue
        valid = all(
            (frame.loc[frame["method"] == method, "checkpoint_sha256"].str.upper() == record["checkpoint_sha256"]).all()
            for method, record in provenance.items()
        )
        if valid and (frame["top_k"] == TOPK).all() and (frame["capacity_accounting"] == "connection_persistent").all():
            result = frame[["method", "ue_count", "capacity", *METRICS]].copy()
            result["evaluation_order_seed"] = 42
            result["result_source"] = str(path.resolve())
            return result, path
    raise FileNotFoundError("No reusable Seed=42 formal evaluation matches all three checkpoints")


def load_worker_models() -> dict:
    torch.set_num_threads(1)
    models = {}
    for method, spec in SPECS.items():
        model = spec["network"](251, 51)
        model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=False), strict=True)
        model.eval()
        models[method] = model
    return models


def evaluate_one(method: str, model: torch.nn.Module, scenario: str, seed: int) -> dict:
    module = SPECS[method]["module"]
    env = proposed_module.HandoverEnv(
        scenario, str(RATE_PATH), lambda_h=3.0,
        seed=42, training_order_seed=42, demonstration_order_seed=42,
        evaluation_order_seed=seed,
    )
    env.reset_multi_user_episode()
    env.reset_evaluation_order_rng(seed)
    blocks = handovers = 0
    rate_sum = 0.0
    mask_failures = 0
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
                _, blocked = env.step_user(user, t, action)
                if blocked:
                    blocks += 1
                else:
                    rate_sum += float(env.rate[action, t])
                if env.last_reward_components[user]["handover"] < 0:
                    handovers += 1
    if mask_failures:
        raise RuntimeError(f"Mask failures: seed={seed}, method={method}, UE={env.m}, n={mask_failures}")
    return {
        "method": method,
        "ue_count": env.m,
        "capacity": CAPACITY,
        "average_throughput": rate_sum / (env.m * env.T),
        "average_handover_count": handovers / env.m,
        "average_blocking_rate": blocks / env.T,
        "evaluation_order_seed": seed,
        "result_source": "new_multiseed_evaluation",
    }


def evaluate_seed(seed: int, scenarios: dict[int, str]) -> list[dict]:
    models = load_worker_models()
    rows = []
    for method in METHOD_ORDER:
        for ue in UE_COUNTS:
            rows.append(evaluate_one(method, models[method], scenarios[ue], seed))
    return rows


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, ue), group in raw.groupby(["method", "ue_count"], sort=False):
        row = {"method": method, "ue_count": ue, "capacity": CAPACITY, "n_evaluation_seeds": len(group)}
        for metric in METRICS:
            mean = float(group[metric].mean())
            sd = float(group[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_mean_plus_minus_sd"] = f"{mean:.6f} ± {sd:.6f}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "ue_count"]).reset_index(drop=True)


def paired_differences(raw: pd.DataFrame) -> pd.DataFrame:
    proposed = raw[raw["method"] == "Proposed"].set_index(["evaluation_order_seed", "ue_count"])
    rows = []
    for baseline in ("Dueling DDQN", "Standard DDQN"):
        other = raw[raw["method"] == baseline].set_index(["evaluation_order_seed", "ue_count"])
        for key in proposed.index:
            row = {
                "baseline_method": baseline,
                "evaluation_order_seed": key[0],
                "ue_count": key[1],
                "capacity": CAPACITY,
            }
            for metric in METRICS:
                row[f"proposed_minus_baseline_{metric}"] = float(proposed.loc[key, metric] - other.loc[key, metric])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["baseline_method", "evaluation_order_seed", "ue_count"]).reset_index(drop=True)


def handover_trend(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proposed = raw[raw["method"] == "Proposed"]
    for seed, group in proposed.groupby("evaluation_order_seed"):
        group = group.sort_values("ue_count")
        ue = group["ue_count"].to_numpy(dtype=float)
        values = group["average_handover_count"].to_numpy(dtype=float)
        diffs = np.diff(values)
        rows.append(
            {
                "evaluation_order_seed": int(seed),
                "linear_slope_handover_per_added_ue": float(np.polyfit(ue, values, 1)[0]),
                "ue30_minus_ue10": float(values[-1] - values[0]),
                "decreasing_adjacent_steps_out_of_4": int(np.count_nonzero(diffs < 0)),
                "strictly_decreasing": bool(np.all(diffs < 0)),
                "negative_overall_slope": bool(np.polyfit(ue, values, 1)[0] < 0),
                "ue30_lower_than_ue10": bool(values[-1] < values[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("evaluation_order_seed").reset_index(drop=True)


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for method in METHOD_ORDER:
        rows = summary[summary["method"] == method].sort_values("ue_count")
        ax.errorbar(
            rows["ue_count"], rows[f"{metric}_mean"], yerr=rows[f"{metric}_sd"],
            label=method, capsize=4, linewidth=2, markersize=8, **STYLE[method],
        )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(UE_COUNTS)
    ax.grid(True, linestyle=":", color="lightgray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, raw: pd.DataFrame, summary: pd.DataFrame, trend: pd.DataFrame, provenance: dict, seed42_source: Path) -> None:
    overall = raw.groupby("method")[METRICS].mean()
    sensitivity = summary.groupby("method")[[f"{metric}_sd" for metric in METRICS]].mean()
    neg_slopes = int(trend["negative_overall_slope"].sum())
    lower_endpoints = int(trend["ue30_lower_than_ue10"].sum())
    strict = int(trend["strictly_decreasing"].sum())
    lines = [
        "lambda_handover=3.0 Multi-Seed Evaluation",
        "",
        "這是固定三個 Checkpoint 的五組 Evaluation Order Seeds，不是五次獨立訓練。",
        f"Seeds: {SEEDS}",
        f"UE: {UE_COUNTS}; capacity={CAPACITY}; Persistent Capacity; Top-K={TOPK}",
        "SD 使用 sample standard deviation (ddof=1)。",
        f"Seed=42 reused from: {seed42_source.resolve()}",
        "",
        "Checkpoints",
    ]
    for method in ("Proposed", "Dueling DDQN", "Standard DDQN"):
        lines += [f"{method}: {provenance[method]['checkpoint_path']}", f"SHA-256: {provenance[method]['checkpoint_sha256']}"]
    lines += [
        "",
        "Proposed Handover 趨勢",
        f"Negative overall slope: {neg_slopes}/5 seeds",
        f"UE=30 lower than UE=10: {lower_endpoints}/5 seeds",
        f"Strictly decreasing at all four adjacent UE steps: {strict}/5 seeds",
        "",
        "跨 UE 與 Seed 的整體平均（僅作簡潔排名摘要）",
    ]
    for method in METHOD_ORDER:
        row = overall.loc[method]
        lines.append(
            f"{method}: throughput={row.average_throughput:.6f}, "
            f"handover={row.average_handover_count:.6f}, blocking={row.average_blocking_rate:.6f}"
        )
    lines += ["", "平均 Seed SD（跨五個 UE 場景平均）"]
    for method in METHOD_ORDER:
        row = sensitivity.loc[method]
        lines.append(
            f"{method}: throughput SD={row.average_throughput_sd:.6f}, "
            f"handover SD={row.average_handover_count_sd:.6f}, blocking SD={row.average_blocking_rate_sd:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    provenance = checkpoint_provenance()
    output = ROOT / "outputs" / "ue_sensitivity" / f"rl_multiseed_lambda3_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    scenario_dir = output / "scenario_inputs"
    scenario_dir.mkdir()
    scenarios = create_scenarios(scenario_dir)

    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_seed, seed, scenarios): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            rows.extend(result)
            print(f"Completed evaluation order seed {seed}: {len(result)} scenarios", flush=True)

    raw = pd.DataFrame(rows)
    raw = raw.sort_values(["method", "evaluation_order_seed", "ue_count"]).reset_index(drop=True)
    if len(raw) != 75 or raw.duplicated(["method", "evaluation_order_seed", "ue_count"]).any():
        raise RuntimeError("Raw results are incomplete or duplicated")
    if set(raw["evaluation_order_seed"]) != set(SEEDS) or raw[METRICS].isna().any().any():
        raise RuntimeError("Raw results have missing seeds or metrics")
    for method, record in provenance.items():
        raw.loc[raw["method"] == method, "checkpoint_path"] = record["checkpoint_path"]
        raw.loc[raw["method"] == method, "checkpoint_sha256"] = record["checkpoint_sha256"]
    raw["top_k"] = TOPK
    raw["capacity_accounting"] = "connection_persistent"
    raw_path = output / "raw_seed_results.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = aggregate(raw)
    summary_path = output / "mean_sd_comparison.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired = paired_differences(raw)
    paired_path = output / "proposed_paired_differences.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    trend = handover_trend(raw)
    trend_path = output / "proposed_handover_trend_by_seed.csv"
    trend.to_csv(trend_path, index=False, encoding="utf-8-sig")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13})
    plot_metric(summary, "average_throughput", "Average Throughput (Mbps)", output / "throughput_multiseed.png")
    plot_metric(summary, "average_handover_count", "Average Handover Count", output / "handover_multiseed.png")
    plot_metric(summary, "average_blocking_rate", "Average Blocking Rate", output / "blocking_multiseed.png")
    write_summary(output / "summary.txt", raw, summary, trend, provenance, raw_path)
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
