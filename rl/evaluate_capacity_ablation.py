import csv
import hashlib
import json
import math
from pathlib import Path
from config_loader import experiment_config, method_config, simulation_config

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

import Dueling_ddqn as proposed_module
import Dueling_ddqn_no_capacity_state as ablation_module


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = PROJECT / "outputs" / "ablation"
INPUT = PROJECT / "input.txt"
RATE = PROJECT / "input_angle.txt"
FORMAL_RAW = PROJECT / "evaluation_results" / "rl_multiseed_lambda3_20261018_191043" / "raw_seed_results.csv"
PROPOSED_RUN = ROOT / "runs" / "proposed_persistent_topk_seed42_300ep"
ABLATION_RUN = ROOT / "runs" / "no_capacity_state_persistent_topk_seed42_300ep"
ABLATION_CHECKPOINT = ABLATION_RUN / "checkpoints" / "dqn_imi_no_capacity_state.pt"
_SCENARIO = experiment_config()["capacity_aware_ablation"]
SEEDS = simulation_config()["evaluation_order_seeds"]
UE_COUNT = _SCENARIO["number_of_ues"]
CAPACITY = _SCENARIO["satellite_capacity"]
TOP_K = method_config()["rl"]["top_k"]
T_SLOTS = 3599
ABLATION_COLOR = "#D62728"

VARIANT_ORDER = [
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
COLORS = {
    "Proposed": "#000000",
    "Dueling DDQN (No-MSHBO)": "#17BECF",
    "DDQN": "#7F7F7F",
    "Proposed w/o Remaining-Capacity State": ABLATION_COLOR,
}
TARGETS = [
    OUTPUT / "ablation_raw_results.csv",
    OUTPUT / "ablation_table.csv",
    OUTPUT / "capacity_ablation_comparison.png",
    OUTPUT / "capacity_ablation_comparison.pdf",
    OUTPUT / "capacity_ablation_report.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_training() -> tuple[dict, dict, str, str]:
    new = load_json(ABLATION_RUN / "run_metadata.json")
    old = load_json(PROPOSED_RUN / "run_metadata.json")
    expected = {
        "episodes": 300,
        "training_seed": 42,
        "training_order_seed": 42,
        "demonstration_order_seed": 42,
        "ue_count": UE_COUNT,
        "time_slot_count": T_SLOTS,
        "state_dimension": 201,
        "action_dimension": 51,
        "top_k": TOP_K,
        "learning_rate": 0.001,
        "batch_size": 256,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_slots": 30000,
        "target_network_update_slots": 2000,
        "replay_buffer_capacity": 200000,
        "pretraining_updates_actual": 2000,
    }
    for key, value in expected.items():
        if new.get(key) != value:
            raise RuntimeError(f"Ablation metadata mismatch: {key}={new.get(key)!r}, expected {value!r}")
    if not new.get("mshbo_demonstration_enabled") or new.get("mshbo_expert_status") != "loaded":
        raise RuntimeError("Ablation MSHBO Expert Replay was not loaded")
    if new["expert_replay_stats"]["expert_transitions_loaded_into_replay"] != 72756:
        raise RuntimeError("Unexpected ablation expert replay size")
    if set(new["satellite_capacity"]) != {CAPACITY}:
        raise RuntimeError("Ablation capacity is not uniformly 5")

    same_keys = [
        "training_seed", "training_order_seed", "demonstration_order_seed", "episodes",
        "learning_rate", "batch_size", "gamma", "epsilon_start", "epsilon_end",
        "epsilon_decay_slots", "target_network_update_slots", "replay_buffer_capacity",
        "pretraining_updates_actual", "top_k", "capacity_accounting", "sticky_probability",
        "lookahead_slots", "throughput_normalization_max_mbps", "ue_count", "satellite_count",
        "time_slot_count", "input_data_sha256", "rate_table_sha256", "mshbo_expert_csv_sha256",
    ]
    unequal = [key for key in same_keys if new.get(key) != old.get(key)]
    if unequal or new["reward_weights"] != old["reward_weights"]:
        raise RuntimeError(f"Training fairness mismatch: {unequal or ['reward_weights']}")
    if old["state_dimension"] != 251:
        raise RuntimeError("Original Proposed state dimension is not 251")

    with (ABLATION_RUN / "training_log.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        log = list(csv.DictReader(stream))
    if len(log) != 300 or [int(log[0]["episode_index"]), int(log[-1]["episode_index"])] != [1, 300]:
        raise RuntimeError("Ablation training log is incomplete")
    returns = np.asarray([float(row["episode_return"]) for row in log], dtype=np.float64)
    if not np.isfinite(returns).all():
        raise RuntimeError("Ablation training return contains NaN or Inf")

    checkpoint = Path(new["checkpoint_path"])
    if checkpoint.resolve() != ABLATION_CHECKPOINT.resolve() or not checkpoint.is_file():
        raise RuntimeError("Ablation checkpoint path is missing or unexpected")
    model = ablation_module.QNet(201, 51)
    loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Ablation checkpoint strict load failed")
    with torch.inference_mode():
        forward = model(torch.zeros((4, 201), dtype=torch.float32))
    if forward.shape != (4, 51) or not torch.isfinite(forward).all():
        raise RuntimeError("Ablation checkpoint forward pass failed")

    original_hash = sha256(Path(old["checkpoint_path"]))
    if original_hash != "B16016D037AE465ECDF4033AC681A2242BF9AAC43B8484D82EFA1A714DAE0720":
        raise RuntimeError("Original Proposed checkpoint hash changed")
    return old, new, original_hash, sha256(checkpoint)


def load_existing_formal_results() -> list[dict]:
    with FORMAL_RAW.open("r", encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    selected = [row for row in source if int(row["ue_count"]) == UE_COUNT and row["method"] in SOURCE_NAMES]
    if len(selected) != 15:
        raise RuntimeError(f"Expected 15 existing formal RL rows, found {len(selected)}")
    rows = []
    for row in selected:
        seed = int(row["evaluation_order_seed"])
        if seed not in SEEDS or int(row["capacity"]) != CAPACITY:
            raise RuntimeError("Existing formal result uses a different seed or capacity")
        checkpoint = Path(row["checkpoint_path"])
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != row["checkpoint_sha256"].upper():
            raise RuntimeError(f"Checkpoint hash mismatch in formal result: {checkpoint}")
        blocked_per_slot = float(row["average_blocking_rate"])
        total_blocks = int(round(blocked_per_slot * T_SLOTS))
        handovers_per_ue = float(row["average_handover_count"])
        total_handovers = int(round(handovers_per_ue * UE_COUNT))
        rows.append({
            "variant": SOURCE_NAMES[row["method"]],
            "evaluation_order_seed": seed,
            "ue_count": UE_COUNT,
            "capacity": CAPACITY,
            "time_slots": T_SLOTS,
            "total_decisions": UE_COUNT * T_SLOTS,
            "throughput_mbps": float(row["average_throughput"]),
            "total_blocks": total_blocks,
            "blocking_rate_percent": total_blocks / (UE_COUNT * T_SLOTS) * 100.0,
            "total_handovers": total_handovers,
            "handovers_per_ue": handovers_per_ue,
            "mask_failures": 0,
            "capacity_violations": 0,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "result_source": str(FORMAL_RAW.resolve()),
        })
    for variant in SOURCE_NAMES.values():
        if sorted(row["evaluation_order_seed"] for row in rows if row["variant"] == variant) != SEEDS:
            raise RuntimeError(f"Incomplete existing formal seeds for {variant}")
    return rows


def evaluate_ablation(checkpoint_hash: str) -> list[dict]:
    env = ablation_module.HandoverEnv(
        str(INPUT), str(RATE), lambda_h=3.0,
        seed=42, training_order_seed=42, demonstration_order_seed=42,
        evaluation_order_seed=42,
    )
    if (env.m, env.T, env.state_dim, env.n_actions) != (UE_COUNT, T_SLOTS, 201, 51):
        raise RuntimeError("Ablation evaluation environment dimensions are incorrect")
    if set(int(value) for value in env.cap) != {CAPACITY}:
        raise RuntimeError("Ablation evaluation capacity is not uniformly 5")
    model = ablation_module.QNet(env.state_dim, env.n_actions)
    model.load_state_dict(torch.load(ABLATION_CHECKPOINT, map_location="cpu", weights_only=False), strict=True)
    model.eval()

    rows = []
    for seed in SEEDS:
        env.reset_multi_user_episode()
        env.reset_evaluation_order_rng(seed)
        blocks = handovers = decisions = mask_failures = capacity_violations = 0
        rate_sum = 0.0
        with torch.inference_mode():
            for t in range(env.T):
                order = env.next_evaluation_user_order()
                if sorted(int(user) for user in order) != list(range(env.m)):
                    raise RuntimeError(f"Invalid UE order at seed={seed}, t={t}")
                for raw_user in order:
                    user = int(raw_user)
                    previous = int(env.last_sat[user])
                    state = env.make_state(user, t)
                    if state.shape != (201,) or not np.isfinite(state).all():
                        raise RuntimeError(f"Invalid state at seed={seed}, t={t}, user={user}")
                    full_legal = env.legal_mask(user, t)
                    selection = ablation_module.topk_mask(
                        full_legal, env.used, env.cap, t, previous, env.block_action, TOP_K
                    )
                    if selection[env.block_action] <= 0 or np.any((selection > 0) & (full_legal <= 0)):
                        mask_failures += 1
                    if previous >= 0 and full_legal[previous] > 0 and selection[previous] <= 0:
                        mask_failures += 1
                    action = ablation_module.masked_argmax(model(torch.from_numpy(state)), selection)
                    if selection[action] <= 0 or full_legal[action] <= 0:
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
        if decisions != UE_COUNT * T_SLOTS or mask_failures or capacity_violations:
            raise RuntimeError(
                f"Evaluation validation failed: seed={seed}, decisions={decisions}, "
                f"mask_failures={mask_failures}, capacity_violations={capacity_violations}"
            )
        values = [rate_sum, blocks, handovers]
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Non-finite evaluation metric at seed={seed}")
        rows.append({
            "variant": "Proposed w/o Remaining-Capacity State",
            "evaluation_order_seed": seed,
            "ue_count": UE_COUNT,
            "capacity": CAPACITY,
            "time_slots": T_SLOTS,
            "total_decisions": decisions,
            "throughput_mbps": rate_sum / decisions,
            "total_blocks": blocks,
            "blocking_rate_percent": blocks / decisions * 100.0,
            "total_handovers": handovers,
            "handovers_per_ue": handovers / UE_COUNT,
            "mask_failures": mask_failures,
            "capacity_violations": capacity_violations,
            "checkpoint_path": str(ABLATION_CHECKPOINT.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "result_source": "new_capacity_ablation_evaluation",
        })
        print(
            f"seed={seed}: throughput={rows[-1]['throughput_mbps']:.9f}, "
            f"blocking={rows[-1]['blocking_rate_percent']:.9f}%, "
            f"handovers={rows[-1]['handovers_per_ue']:.9f}"
        )
    return rows


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for variant in VARIANT_ORDER:
        group = [row for row in rows if row["variant"] == variant]
        if len(group) != len(SEEDS):
            raise RuntimeError(f"Expected five seeds for {variant}, found {len(group)}")
        result[variant] = {}
        for metric in ["throughput_mbps", "blocking_rate_percent", "handovers_per_ue"]:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[variant][f"{metric}_mean"] = float(values.mean())
            result[variant][f"{metric}_sd"] = float(values.std(ddof=1))
    return result


def write_csvs(rows: list[dict], summary: dict[str, dict[str, float]]) -> None:
    with TARGETS[0].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with TARGETS[1].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Variant", "Throughput (Mbps)", "Blocking Rate (%)", "Handovers/UE"])
        for variant in VARIANT_ORDER:
            item = summary[variant]
            writer.writerow([
                variant,
                f"{item['throughput_mbps_mean']:.6f} ± {item['throughput_mbps_sd']:.6f}",
                f"{item['blocking_rate_percent_mean']:.6f} ± {item['blocking_rate_percent_sd']:.6f}",
                f"{item['handovers_per_ue_mean']:.6f} ± {item['handovers_per_ue_sd']:.6f}",
            ])


def plot(summary: dict[str, dict[str, float]]) -> None:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    labels = ["Proposed", "Dueling DDQN\n(No-MSHBO)", "DDQN", "Proposed w/o\nCapacity State"]
    metrics = [
        ("throughput_mbps", "Throughput (Mbps)"),
        ("blocking_rate_percent", "Blocking Rate (%)"),
        ("handovers_per_ue", "Handovers per UE"),
    ]
    x = np.arange(len(VARIANT_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        means = [summary[name][f"{metric}_mean"] for name in VARIANT_ORDER]
        sds = [summary[name][f"{metric}_sd"] for name in VARIANT_ORDER]
        for index, variant in enumerate(VARIANT_ORDER):
            ax.errorbar(
                index, means[index], yerr=sds[index], fmt="o", markersize=7,
                color=COLORS[variant], ecolor=COLORS[variant], capsize=4,
                elinewidth=1.5, markeredgecolor="white", markeredgewidth=0.6,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=8)
        if metric == "throughput_mbps":
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Capacity-aware Ablation (UE=25, C=5)\n"
        "Mean ± sample SD across five evaluation-order seeds",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(TARGETS[2], dpi=300, bbox_inches="tight")
    fig.savefig(TARGETS[3], bbox_inches="tight")
    plt.close(fig)


def write_report(old: dict, new: dict, old_hash: str, new_hash: str, summary: dict[str, dict[str, float]]) -> None:
    proposed = summary["Proposed"]
    ablation = summary["Proposed w/o Remaining-Capacity State"]
    differences = {
        metric: proposed[f"{metric}_mean"] - ablation[f"{metric}_mean"]
        for metric in ["throughput_mbps", "blocking_rate_percent", "handovers_per_ue"]
    }
    supports = (
        differences["throughput_mbps"] > 0
        and differences["blocking_rate_percent"] < 0
        and differences["handovers_per_ue"] < 0
    )
    conclusion = (
        "三項指標方向一致：完整 Proposed 的 Throughput 較高、Blocking 與 Handover 較低，"
        "因此本次固定 Checkpoint、固定環境的結果支持 Remaining-Capacity State 的有效性。"
        if supports else
        "三項指標未呈現一致優勢，因此結果不支持『移除容量狀態後必然全面變差』；應依各指標分別解讀。"
    )
    lines = [
        "# Capacity-aware Ablation 正式驗證與評估報告",
        "",
        "## 1. 正式訓練與 Checkpoint 檢查",
        "",
        "- 新模型已完成 300 Episodes，Training Seed=42，UE=25，Capacity=5，Time Slots=3599。",
        "- State Dimension=201，Action Dimension=51。",
        "- MSHBO Expert Replay 已載入 72,756 transitions，完成 2,000 次預訓練 update。",
        "- Reward、Top-K=3、epsilon schedule、learning rate、batch size、replay capacity 與原始 Proposed 相同。",
        "- 300 筆 Training Return 皆為有限值，無 NaN/Inf。",
        f"- 消融 Checkpoint SHA-256: `{new_hash}`。Strict load 與 201→51 forward pass 通過。",
        f"- 原始 Proposed Checkpoint SHA-256: `{old_hash}`，與既有正式結果一致。",
        "",
        "## 2. 評估一致性",
        "",
        "四種方法使用 UE=25、C=5、T=3599、Top-K=3 與 Evaluation Order Seeds "
        "42、123、2024、2025、2026。同一 seed 使用同一組 per-slot UE permutation。",
        "",
        "消融版與 Proposed 共用相同 QNet 架構（僅 input dimension 251→201）、DDQN update、"
        "Top-K、Action Mask、Capacity Enforcement、Reward 與指標定義。五個 seeds 均完成 89,975 次決策，"
        "Mask Failure=0，Capacity Violation=0。每種方法的五個 seeds 均使用同一個固定 Training Checkpoint。",
        "",
        "## 3. 結果（Mean ± Sample SD）",
        "",
        "| Variant | Throughput (Mbps) | Blocking Rate (%) | Handovers/UE |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        item = summary[variant]
        lines.append(
            f"| {variant} | {item['throughput_mbps_mean']:.6f} ± {item['throughput_mbps_sd']:.6f} "
            f"| {item['blocking_rate_percent_mean']:.6f} ± {item['blocking_rate_percent_sd']:.6f} "
            f"| {item['handovers_per_ue_mean']:.6f} ± {item['handovers_per_ue_sd']:.6f} |"
        )
    lines += [
        "",
        "SD 來自五組 Evaluation Order Seeds，不是五次 Independent Training Runs。",
        "",
        "## 4. Proposed − Proposed w/o Remaining-Capacity State",
        "",
        f"- Throughput: {differences['throughput_mbps']:+.6f} Mbps。",
        f"- Blocking Rate: {differences['blocking_rate_percent']:+.6f} percentage points。",
        f"- Handovers/UE: {differences['handovers_per_ue']:+.6f}。",
        "",
        "## 5. 解讀與限制",
        "",
        conclusion,
        "",
        "本消融只從 Agent State 移除 Remaining-Capacity Ratio。環境、Action Mask 與 Top-K "
        "仍然使用容量資訊來排除滿載衛星與篩選候選動作。因此結果測量的是「容量資訊不作為"
        "網路輸入特徵」的影響，不是移除整個容量感知與容量約束。",
        "",
        "## 6. 輸出檔案",
        "",
    ]
    for target in TARGETS:
        lines.append(f"- `{target.resolve()}`")
    TARGETS[4].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing = [path for path in TARGETS if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing ablation outputs: {existing}")
    old, new, old_hash, new_hash = validate_training()
    rows = load_existing_formal_results()
    rows.extend(evaluate_ablation(new_hash))
    rows.sort(key=lambda row: (VARIANT_ORDER.index(row["variant"]), SEEDS.index(row["evaluation_order_seed"])))
    summary = summarize(rows)
    write_csvs(rows, summary)
    plot(summary)
    write_report(old, new, old_hash, new_hash, summary)
    for target in TARGETS:
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {target}")
    print("All capacity-aware ablation outputs completed:")
    for target in TARGETS:
        print(target.resolve())


if __name__ == "__main__":
    main()
