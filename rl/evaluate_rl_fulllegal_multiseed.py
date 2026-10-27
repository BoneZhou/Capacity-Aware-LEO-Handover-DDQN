from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_all_baselines_lambda3 as plotbase
import evaluate_rl_multiseed_lambda3 as formal


ROOT = formal.ROOT
METRICS = formal.METRICS


def latest_topk_results() -> Path:
    candidates = sorted(
        ROOT.glob("evaluation_results/rl_multiseed_lambda3_*/raw_seed_results.csv"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No formal Top-K Multi-Seed result found")
    return candidates[0]


def evaluate_one(method: str, model: torch.nn.Module, scenario: str, seed: int) -> dict:
    module = formal.SPECS[method]["module"]
    env = formal.proposed_module.HandoverEnv(
        scenario,
        str(formal.RATE_PATH),
        lambda_h=3.0,
        seed=42,
        training_order_seed=42,
        demonstration_order_seed=42,
        evaluation_order_seed=seed,
    )
    env.reset_multi_user_episode()
    env.reset_evaluation_order_rng(seed)
    blocks = handovers = 0
    rate_sum = 0.0
    invalid_actions = 0

    with torch.inference_mode():
        for t in range(env.T):
            for raw_user in env.next_evaluation_user_order():
                user = int(raw_user)
                state = env.make_state(user, t)
                full_legal = env.legal_mask(user, t)
                action = module.masked_argmax(model(torch.from_numpy(state)), full_legal)
                if full_legal[env.block_action] <= 0 or full_legal[action] <= 0:
                    invalid_actions += 1
                _, blocked = env.step_user(user, t, action)
                if blocked:
                    blocks += 1
                else:
                    rate_sum += float(env.rate[action, t])
                if env.last_reward_components[user]["handover"] < 0:
                    handovers += 1

    if invalid_actions:
        raise RuntimeError(f"Full-Legal action validation failed: {method}, UE={env.m}, seed={seed}")
    return {
        "method": method,
        "ue_count": env.m,
        "capacity": formal.CAPACITY,
        "evaluation_order_seed": seed,
        "evaluation_action_mask": "full_legal",
        "average_throughput": rate_sum / (env.m * env.T),
        "average_handover_count": handovers / env.m,
        "average_blocking_rate": blocks / env.T,
        "result_source": "new_fulllegal_multiseed_evaluation",
    }


def evaluate_seed(seed: int, scenarios: dict[int, str]) -> list[dict]:
    models = formal.load_worker_models()
    return [
        evaluate_one(method, models[method], scenarios[ue], seed)
        for method in formal.METHOD_ORDER
        for ue in formal.UE_COUNTS
    ]


def aggregate(raw: pd.DataFrame, provenance: dict) -> pd.DataFrame:
    rows = []
    for (method, ue), group in raw.groupby(["method", "ue_count"], sort=False):
        row = {
            "method": method,
            "ue_count": int(ue),
            "capacity": formal.CAPACITY,
            "n_evaluation_seeds": len(group),
            "evaluation_action_mask": "full_legal",
            **provenance[method],
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "ue_count"]).reset_index(drop=True)


def compare_with_topk(full: pd.DataFrame, topk_source: Path) -> pd.DataFrame:
    topk = pd.read_csv(topk_source)
    expected = len(formal.METHOD_ORDER) * len(formal.UE_COUNTS) * len(formal.SEEDS)
    if (
        len(topk) != expected
        or set(topk["method"]) != set(formal.METHOD_ORDER)
        or set(topk["evaluation_order_seed"]) != set(formal.SEEDS)
    ):
        raise RuntimeError("Existing Top-K Multi-Seed result is incomplete")
    topk_mean = topk.groupby(["method", "ue_count"], as_index=False)[METRICS].mean()
    topk_mean = topk_mean.rename(columns={metric: f"topk_{metric}" for metric in METRICS})
    comparison = full.merge(topk_mean, on=["method", "ue_count"], validate="one_to_one")
    for metric in METRICS:
        full_column = f"{metric}_mean"
        topk_column = f"topk_{metric}"
        comparison[f"fulllegal_minus_topk_{metric}"] = comparison[full_column] - comparison[topk_column]
        comparison[f"fulllegal_minus_topk_{metric}_percent"] = np.where(
            comparison[topk_column] != 0,
            comparison[f"fulllegal_minus_topk_{metric}"] / comparison[topk_column] * 100.0,
            np.nan,
        )
    return comparison.sort_values(["method", "ue_count"]).reset_index(drop=True)


def build_plot_data(full: pd.DataFrame) -> pd.DataFrame:
    rows = plotbase.load_nonrl()
    for _, row in full.iterrows():
        rows.append(
            {
                "method": row["method"],
                "ue_count": int(row["ue_count"]),
                "capacity": formal.CAPACITY,
                "average_throughput": float(row["average_throughput_mean"]),
                "average_handover_count": float(row["average_handover_count_mean"]),
                "average_blocking_rate": float(row["average_blocking_rate_mean"]),
                "metric_source": "rl_fulllegal_multiseed_mean",
                "throughput_source": "fulllegal_mean_metrics.csv",
                "handover_blocking_source": "fulllegal_mean_metrics.csv",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 40 or frame.duplicated(["method", "ue_count"]).any():
        raise RuntimeError("Plot data must contain exactly 8 methods x 5 UE scenarios")
    return frame


def write_summary(path: Path, comparison: pd.DataFrame, topk_source: Path, provenance: dict) -> None:
    lines = [
        "RL Full-Legal Multi-Seed Evaluation",
        "",
        f"Seeds: {formal.SEEDS}",
        f"UE: {formal.UE_COUNTS}; capacity={formal.CAPACITY}; Persistent Capacity",
        "Evaluation action mask: Full-Legal (visibility + capacity + existing reservation + BLOCK).",
        f"Top-K comparison source: {topk_source.resolve()}",
        "No checkpoint was retrained or modified.",
        "",
        "Checkpoints",
    ]
    for method in formal.METHOD_ORDER:
        lines.extend(
            [
                f"{method}: {provenance[method]['checkpoint_path']}",
                f"SHA-256: {provenance[method]['checkpoint_sha256']}",
            ]
        )
    lines.extend(["", "Full-Legal mean and change from Top-K=3"])
    columns = ["method", "ue_count"]
    for metric in METRICS:
        columns.extend([f"{metric}_mean", f"topk_{metric}", f"fulllegal_minus_topk_{metric}"])
    lines.append(comparison[columns].to_string(index=False))
    lines.extend(["", "Across-UE mean change (Full-Legal minus Top-K=3)"])
    changes = [f"fulllegal_minus_topk_{metric}" for metric in METRICS]
    lines.append(comparison.groupby("method")[changes].mean().to_string())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    provenance = formal.checkpoint_provenance()
    topk_source = latest_topk_results()
    scenario_dir = topk_source.parent / "scenario_inputs"
    scenarios = {
        ue: str((scenario_dir / f"input_m{ue}_c{formal.CAPACITY}.txt").resolve())
        for ue in formal.UE_COUNTS
    }
    if not all(Path(path).is_file() for path in scenarios.values()):
        raise FileNotFoundError("Formal Multi-Seed scenario inputs are incomplete")

    rows = []
    with ProcessPoolExecutor(max_workers=min(len(formal.SEEDS), os.cpu_count() or 1)) as pool:
        futures = {pool.submit(evaluate_seed, seed, scenarios): seed for seed in formal.SEEDS}
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"Completed Full-Legal seed {futures[future]}", flush=True)

    raw = pd.DataFrame(rows).sort_values(["method", "ue_count", "evaluation_order_seed"]).reset_index(drop=True)
    if len(raw) != 75 or raw.groupby(["method", "ue_count"]).size().ne(5).any():
        raise RuntimeError("Full-Legal result is incomplete")
    for method, record in provenance.items():
        selected = raw["method"] == method
        raw.loc[selected, "checkpoint_path"] = record["checkpoint_path"]
        raw.loc[selected, "checkpoint_sha256"] = record["checkpoint_sha256"]

    mean = aggregate(raw, provenance)
    comparison = compare_with_topk(mean, topk_source)
    plot_data = build_plot_data(mean)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "evaluation_results" / f"rl_fulllegal_multiseed_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    raw.to_csv(output / "fulllegal_per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    mean.to_csv(output / "fulllegal_mean_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "topk_vs_fulllegal_comparison.csv", index=False, encoding="utf-8-sig")
    plot_data.to_csv(output / "merged_plot_data.csv", index=False, encoding="utf-8-sig")
    write_summary(output / "summary.txt", comparison, topk_source, provenance)

    plotbase.configure_plotting()
    plotbase.plot_vs_ues(plot_data, output, "average_throughput", "Average Throughput (Mbps)", "throughput_vs_ues")
    plotbase.plot_vs_ues(plot_data, output, "average_handover_count", "Average Handover Count", "handover_vs_ues")
    plotbase.plot_vs_ues(plot_data, output, "average_blocking_rate", "Average Blocking Rate", "blocking_vs_ues")
    plotbase.plot_tradeoff(plot_data, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
