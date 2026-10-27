import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

import Dueling_ddqn_no_capacity_state as ablation_module
from config_loader import experiment_config, method_config, simulation_config


RL_DIR = Path(__file__).resolve().parent
PROJECT = RL_DIR.parent
EVALUATION_ROOT = PROJECT / "outputs" / "ablation_vs_ue"
PAPER = EVALUATION_ROOT
INPUT = PROJECT / "input.txt"
RATE = PROJECT / "input_angle.txt"
FORMAL_RL_RAW = EVALUATION_ROOT / "rl_multiseed_lambda3_20261018_191043" / "raw_seed_results.csv"
ABLATION_RUN = RL_DIR / "runs" / "no_capacity_state_persistent_topk_seed42_300ep"
ABLATION_METADATA = ABLATION_RUN / "run_metadata.json"
ABLATION_CHECKPOINT = ABLATION_RUN / "checkpoints" / "dqn_imi_no_capacity_state.pt"

_SCENARIO = experiment_config()["capacity_aware_ablation_vs_ue"]
UES = _SCENARIO["ue_counts"]
SEEDS = simulation_config()["evaluation_order_seeds"]
CAPACITY = _SCENARIO["satellite_capacity"]
TOP_K = method_config()["rl"]["top_k"]
TIME_SLOTS = simulation_config()["time"]["evaluated_slots"]
METHODS = [
    "Proposed",
    "Dueling DDQN (No-MSHBO)",
    "DDQN",
    "Proposed w/o Remaining-Capacity State",
]
SOURCE_NAMES = {
    "Proposed": "Proposed",
    "Dueling DDQN": "Dueling DDQN (No-MSHBO)",
    "Standard DDQN": "DDQN",
}
STYLE = {
    "Proposed": dict(color="#000000", linestyle="-", marker="o"),
    "Dueling DDQN (No-MSHBO)": dict(color="#17BECF", linestyle="--", marker="v"),
    "DDQN": dict(color="#7F7F7F", linestyle="-.", marker="d"),
    "Proposed w/o Remaining-Capacity State": dict(color="#9467BD", linestyle=":", marker="s"),
}
PAPER_OUTPUTS = [
    PAPER / "average_handover_count_ablation.png",
    PAPER / "average_handover_count_ablation.pdf",
    PAPER / "average_blocking_rate_ablation.png",
    PAPER / "average_blocking_rate_ablation.pdf",
    PAPER / "average_throughput_ablation.png",
    PAPER / "average_throughput_ablation.pdf",
    PAPER / "tradeoff_ablation.png",
    PAPER / "tradeoff_ablation.pdf",
    PAPER / "ablation_vs_ue_raw.csv",
    PAPER / "ablation_vs_ue_mean_std.csv",
    PAPER / "ablation_plot_report.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def unique_evaluation_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = EVALUATION_ROOT / f"capacity_ablation_vs_ue_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def validate_inputs() -> tuple[dict, str]:
    existing = [path for path in PAPER_OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite requested paper outputs: {existing}")
    metadata = json.loads(ABLATION_METADATA.read_text(encoding="utf-8"))
    expected = {
        "episodes": 300,
        "training_seed": 42,
        "ue_count": 25,
        "state_dimension": 201,
        "action_dimension": 51,
        "top_k": TOP_K,
        "capacity_accounting": "connection_persistent",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Ablation metadata mismatch: {key}={metadata.get(key)!r}, expected={value!r}")
    if set(metadata["satellite_capacity"]) != {CAPACITY}:
        raise RuntimeError("Training metadata capacity is not uniformly 5")
    if not metadata["mshbo_demonstration_enabled"] or metadata["mshbo_expert_status"] != "loaded":
        raise RuntimeError("MSHBO Expert Replay was not retained")
    if not ABLATION_CHECKPOINT.is_file():
        raise FileNotFoundError(ABLATION_CHECKPOINT)
    checkpoint_hash = sha256(ABLATION_CHECKPOINT)
    model = ablation_module.QNet(201, 51)
    loaded = model.load_state_dict(
        torch.load(ABLATION_CHECKPOINT, map_location="cpu", weights_only=False), strict=True
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Ablation checkpoint strict load failed")
    with torch.inference_mode():
        output = model(torch.zeros((2, 201), dtype=torch.float32))
    if output.shape != (2, 51) or not torch.isfinite(output).all():
        raise RuntimeError("Ablation checkpoint forward pass failed")
    return metadata, checkpoint_hash


def create_scenarios(directory: Path) -> dict[int, str]:
    lines = INPUT.read_text(encoding="utf-8").splitlines()
    n, _, tb, te = map(int, lines[0].replace(",", " ").split()[:4])
    if len(lines) != n + 1 or te - tb != TIME_SLOTS:
        raise RuntimeError("Unexpected base input dimensions")
    satellite_rows = []
    for line in lines[1:]:
        begin, end, _ = map(int, line.split())
        satellite_rows.append(f"{begin} {end} {CAPACITY}")
    scenario_dir = directory / "scenarios"
    scenario_dir.mkdir(parents=True)
    scenarios = {}
    for ue in UES:
        path = scenario_dir / f"input_m{ue}_c{CAPACITY}.txt"
        path.write_text(
            "\n".join([f"{n} {ue} {tb} {te}"] + satellite_rows) + "\n",
            encoding="utf-8",
        )
        scenarios[ue] = str(path.resolve())
    return scenarios


def evaluate_seed(seed: int, scenarios: dict[int, str], checkpoint_hash: str) -> list[dict]:
    torch.set_num_threads(1)
    model = ablation_module.QNet(201, 51)
    model.load_state_dict(
        torch.load(ABLATION_CHECKPOINT, map_location="cpu", weights_only=False), strict=True
    )
    model.eval()
    rows = []
    for ue in UES:
        env = ablation_module.HandoverEnv(
            scenarios[ue], str(RATE), lambda_h=3.0,
            seed=42, training_order_seed=42, demonstration_order_seed=42,
            evaluation_order_seed=seed,
        )
        if (env.m, env.T, env.state_dim, env.n_actions) != (ue, TIME_SLOTS, 201, 51):
            raise RuntimeError(f"Unexpected environment dimensions: UE={ue}, seed={seed}")
        if set(int(value) for value in env.cap) != {CAPACITY}:
            raise RuntimeError(f"Capacity mismatch: UE={ue}, seed={seed}")
        env.reset_multi_user_episode()
        env.reset_evaluation_order_rng(seed)
        decisions = blocks = handovers = mask_failures = capacity_violations = 0
        rate_sum = 0.0
        with torch.inference_mode():
            for t in range(env.T):
                order = env.next_evaluation_user_order()
                if sorted(int(user) for user in order) != list(range(env.m)):
                    raise RuntimeError(f"Invalid UE permutation: UE={ue}, seed={seed}, t={t}")
                for raw_user in order:
                    user = int(raw_user)
                    previous = int(env.last_sat[user])
                    state = env.make_state(user, t)
                    if state.shape != (201,) or not np.isfinite(state).all():
                        raise RuntimeError(f"Invalid state: UE={ue}, seed={seed}, t={t}, user={user}")
                    legal = env.legal_mask(user, t)
                    selection = ablation_module.topk_mask(
                        legal, env.used, env.cap, t, previous, env.block_action, TOP_K
                    )
                    if selection[env.block_action] <= 0 or np.any((selection > 0) & (legal <= 0)):
                        mask_failures += 1
                    if previous >= 0 and legal[previous] > 0 and selection[previous] <= 0:
                        mask_failures += 1
                    action = ablation_module.masked_argmax(model(torch.from_numpy(state)), selection)
                    if selection[action] <= 0 or legal[action] <= 0:
                        mask_failures += 1
                    _, blocked = env.step_user(user, t, action)
                    decisions += 1
                    if np.any(env.occupancy > env.cap) or np.any(env.occupancy < 0):
                        capacity_violations += 1
                    if blocked:
                        blocks += 1
                    else:
                        rate_sum += float(env.rate[action, t])
                    if env.last_reward_components[user]["handover"] < 0:
                        handovers += 1
        if decisions != ue * TIME_SLOTS or mask_failures or capacity_violations:
            raise RuntimeError(
                f"Evaluation validation failed: UE={ue}, seed={seed}, decisions={decisions}, "
                f"mask_failures={mask_failures}, capacity_violations={capacity_violations}"
            )
        values = [rate_sum, blocks, handovers]
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Non-finite metric: UE={ue}, seed={seed}")
        rows.append({
            "method": "Proposed w/o Remaining-Capacity State",
            "ue_count": ue,
            "capacity": CAPACITY,
            "evaluation_order_seed": seed,
            "time_slots": TIME_SLOTS,
            "total_decisions": decisions,
            "average_throughput_mbps": rate_sum / decisions,
            "total_blocks": blocks,
            "blocking_rate_percent": blocks / decisions * 100.0,
            "total_handovers": handovers,
            "average_handovers_per_ue": handovers / ue,
            "mask_failures": mask_failures,
            "capacity_violations": capacity_violations,
            "checkpoint_path": str(ABLATION_CHECKPOINT.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "result_source": "new_201d_ablation_evaluation",
        })
    return rows


def load_existing_rl() -> list[dict]:
    with FORMAL_RL_RAW.open("r", encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    selected = [
        row for row in source
        if row["method"] in SOURCE_NAMES
        and int(row["ue_count"]) in UES
        and int(row["capacity"]) == CAPACITY
        and int(row["evaluation_order_seed"]) in SEEDS
    ]
    if len(selected) != 75:
        raise RuntimeError(f"Expected 75 existing formal RL rows, found {len(selected)}")
    rows = []
    keys = set()
    for row in selected:
        method = SOURCE_NAMES[row["method"]]
        ue = int(row["ue_count"])
        seed = int(row["evaluation_order_seed"])
        key = (method, ue, seed)
        if key in keys:
            raise RuntimeError(f"Duplicate formal RL row: {key}")
        keys.add(key)
        checkpoint = Path(row["checkpoint_path"])
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != row["checkpoint_sha256"].upper():
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        blocks_per_slot = float(row["average_blocking_rate"])
        total_blocks = int(round(blocks_per_slot * TIME_SLOTS))
        handovers_per_ue = float(row["average_handover_count"])
        total_handovers = int(round(handovers_per_ue * ue))
        rows.append({
            "method": method,
            "ue_count": ue,
            "capacity": CAPACITY,
            "evaluation_order_seed": seed,
            "time_slots": TIME_SLOTS,
            "total_decisions": ue * TIME_SLOTS,
            "average_throughput_mbps": float(row["average_throughput"]),
            "total_blocks": total_blocks,
            "blocking_rate_percent": total_blocks / (ue * TIME_SLOTS) * 100.0,
            "total_handovers": total_handovers,
            "average_handovers_per_ue": handovers_per_ue,
            "mask_failures": 0,
            "capacity_violations": 0,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "result_source": str(FORMAL_RL_RAW.resolve()),
        })
    expected = {(method, ue, seed) for method in SOURCE_NAMES.values() for ue in UES for seed in SEEDS}
    if keys != expected:
        raise RuntimeError("Existing formal RL data is incomplete")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for method in METHODS:
        for ue in UES:
            group = [row for row in rows if row["method"] == method and row["ue_count"] == ue]
            if len(group) != 5 or sorted(row["evaluation_order_seed"] for row in group) != SEEDS:
                raise RuntimeError(f"Incomplete five-seed group: {method}, UE={ue}")
            item = {
                "method": method,
                "ue_count": ue,
                "capacity": CAPACITY,
                "sample_count": 5,
                "evaluation_order_seeds": "42;123;2024;2025;2026",
                "uncertainty_type": "sample_sd_ddof_1_across_evaluation_order_seeds",
            }
            for metric in ["average_throughput_mbps", "blocking_rate_percent", "average_handovers_per_ue"]:
                values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_sd"] = float(values.std(ddof=1))
            summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def method_rows(summary: list[dict], method: str) -> list[dict]:
    return sorted((row for row in summary if row["method"] == method), key=lambda row: row["ue_count"])


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = PAPER / f"{stem}.png"
    pdf = PAPER / f"{stem}.pdf"
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {png}, {pdf}")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_vs_ue(summary: list[dict], metric: str, ylabel: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    maximum = 0.0
    for method in METHODS:
        group = method_rows(summary, method)
        x = np.asarray([row["ue_count"] for row in group])
        mean = np.asarray([row[f"{metric}_mean"] for row in group])
        sd = np.asarray([row[f"{metric}_sd"] for row in group])
        maximum = max(maximum, float(np.max(mean + sd)))
        ax.errorbar(
            x, mean, yerr=sd, label=method,
            capsize=3, elinewidth=1.0, capthick=1.0,
            markersize=9, linewidth=2.7 if method == "Proposed" else 2.0,
            zorder=5 if method == "Proposed" else 3,
            **STYLE[method],
        )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(UES)
    ax.set_ylim(0, maximum * 1.18 if maximum > 0 else 1.0)
    ax.grid(True, which="major", linestyle=":", color="0.82", linewidth=0.8)
    ax.legend(
        loc="upper right", ncol=1, frameon=True,
        facecolor="white", edgecolor="lightgray", framealpha=1.0,
    )
    fig.tight_layout()
    save_figure(fig, stem)


def plot_tradeoff(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    max_x = max_y = 0.0
    offsets = {
        "Proposed": (5, 5),
        "Dueling DDQN (No-MSHBO)": (-15, -11),
        "DDQN": (5, -12),
        "Proposed w/o Remaining-Capacity State": (5, 9),
    }
    for method in METHODS:
        group = method_rows(summary, method)
        x = np.asarray([row["average_handovers_per_ue_mean"] for row in group])
        y = np.asarray([row["blocking_rate_percent_mean"] for row in group])
        max_x = max(max_x, float(x.max()))
        max_y = max(max_y, float(y.max()))
        ax.plot(
            x, y, label=method, markersize=9,
            linewidth=2.7 if method == "Proposed" else 2.0,
            zorder=5 if method == "Proposed" else 3,
            **STYLE[method],
        )
        for ue, x_value, y_value in zip(UES, x, y):
            ax.annotate(
                str(ue), (x_value, y_value), xytext=offsets[method],
                textcoords="offset points", fontsize=7.5,
                color=STYLE[method]["color"],
            )
    ax.set_xlabel("Average Handover Count per UE")
    ax.set_ylabel("Average Blocking Rate (%)")
    ax.set_xlim(0, max_x * 1.15 if max_x > 0 else 1.0)
    ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1.0)
    ax.grid(True, which="major", linestyle=":", color="0.82", linewidth=0.8)
    ax.legend(
        loc="upper right", ncol=1, frameon=True,
        facecolor="white", edgecolor="lightgray", framealpha=1.0,
    )
    fig.tight_layout()
    save_figure(fig, "tradeoff_ablation")


def write_report(
    evaluation_dir: Path,
    checkpoint_hash: str,
    raw: list[dict],
    summary: list[dict],
) -> None:
    lines = [
        "# Capacity-aware Ablation vs. Number of UEs",
        "",
        "## 資料與評估",
        "",
        "- 原始三種 RL 直接使用既有正式五-seed CSV，未重複評估。",
        "- 消融模型原先只有 UE=25 正式結果，本次使用同一個 201 維 Checkpoint 實際評估 UE=10、15、20、25、30。",
        "- Capacity=5，Time Slots=3599，Top-K=3，Evaluation Order Seeds=42、123、2024、2025、2026。",
        "- Throughput = rate_sum / (UE × T)。",
        "- Blocking Rate (%) = total_blocks / (UE × T) × 100。",
        "- Handovers/UE = total_handovers / UE。",
        "- SD 為五組 Evaluation Order Seeds 的 Sample SD (ddof=1)，不是五次 Independent Training Runs。",
        f"- Ablation Checkpoint SHA-256: `{checkpoint_hash}`。",
        f"- 獨立評估目錄：`{evaluation_dir.resolve()}`。",
        "",
        "## 消融模型結果（Mean ± Sample SD）",
        "",
        "| UE | Throughput (Mbps) | Blocking Rate (%) | Handovers/UE |",
        "|---:|---:|---:|---:|",
    ]
    for ue in UES:
        row = next(
            item for item in summary
            if item["method"] == "Proposed w/o Remaining-Capacity State" and item["ue_count"] == ue
        )
        lines.append(
            f"| {ue} | {row['average_throughput_mbps_mean']:.6f} ± {row['average_throughput_mbps_sd']:.6f} "
            f"| {row['blocking_rate_percent_mean']:.6f} ± {row['blocking_rate_percent_sd']:.6f} "
            f"| {row['average_handovers_per_ue_mean']:.6f} ± {row['average_handovers_per_ue_sd']:.6f} |"
        )
    lines += [
        "",
        "## 檢查結果",
        "",
        f"- Raw rows: {len(raw)} = 4 methods × 5 UE settings × 5 seeds。",
        "- 消融模型每個 UE/seed 的決策數均為 UE × 3599。",
        "- Illegal/Mask Failure=0，Capacity Violation=0，所有指標皆為有限值。",
        "- 四種方法均使用一致的 Blocking Rate (%) 單位。",
        "- 四張圖均無 Inset、Zoomed View 或局部放大子圖。",
        "- 消融方法顏色為紫色 `#9467BD`。",
        "",
        "## 輸出",
        "",
    ]
    for path in PAPER_OUTPUTS:
        lines.append(f"- `{path.resolve()}`")
    (PAPER / "ablation_plot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metadata, checkpoint_hash = validate_inputs()
    evaluation_dir = unique_evaluation_dir()
    evaluation_dir.mkdir(parents=True)
    scenarios = create_scenarios(evaluation_dir)

    ablation_rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(evaluate_seed, seed, scenarios, checkpoint_hash): seed
            for seed in SEEDS
        }
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            ablation_rows.extend(result)
            print(f"completed evaluation seed={seed}")
    ablation_rows.sort(key=lambda row: (row["ue_count"], row["evaluation_order_seed"]))
    write_csv(evaluation_dir / "ablation_evaluation_raw.csv", ablation_rows)

    raw = load_existing_rl() + ablation_rows
    raw.sort(key=lambda row: (METHODS.index(row["method"]), row["ue_count"], row["evaluation_order_seed"]))
    if len(raw) != 100:
        raise RuntimeError(f"Expected 100 combined raw rows, found {len(raw)}")
    summary = summarize(raw)
    write_csv(PAPER / "ablation_vs_ue_raw.csv", raw)
    write_csv(PAPER / "ablation_vs_ue_mean_std.csv", summary)

    manifest = {
        "model": metadata["model_variant"],
        "checkpoint_path": str(ABLATION_CHECKPOINT.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "state_dimension": 201,
        "ue_counts": UES,
        "capacity": CAPACITY,
        "time_slots": TIME_SLOTS,
        "top_k": TOP_K,
        "evaluation_order_seeds": SEEDS,
        "blocking_definition": "total_blocks / (UE * T) * 100",
        "uncertainty": "sample SD across evaluation-order seeds (ddof=1)",
        "mask_failures": sum(row["mask_failures"] for row in ablation_rows),
        "capacity_violations": sum(row["capacity_violations"] for row in ablation_rows),
    }
    (evaluation_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    configure_plotting()
    plot_vs_ue(
        summary, "average_handovers_per_ue", "Average Handover Count per UE",
        "average_handover_count_ablation",
    )
    plot_vs_ue(
        summary, "blocking_rate_percent", "Average Blocking Rate (%)",
        "average_blocking_rate_ablation",
    )
    plot_vs_ue(
        summary, "average_throughput_mbps", "Average Throughput (Mbps)",
        "average_throughput_ablation",
    )
    plot_tradeoff(summary)
    write_report(evaluation_dir, checkpoint_hash, raw, summary)

    for path in PAPER_OUTPUTS:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    print(f"evaluation_dir={evaluation_dir.resolve()}")
    print("all_outputs_completed")


if __name__ == "__main__":
    main()
