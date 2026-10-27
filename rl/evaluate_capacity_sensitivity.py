from __future__ import annotations

import csv
import hashlib
import inspect
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

import Dueling_ddqn as proposed_module
import Dueling_ddqn_no_mshbo as dueling_module
import Dueling_ddqn_no_capacity_state as ablation_module
import Standard_ddqn as standard_module
from config_loader import experiment_config, method_config, simulation_config


RL_DIR = Path(__file__).resolve().parent
PROJECT = RL_DIR.parent
EVALUATION_ROOT = PROJECT / "outputs" / "capacity_sensitivity"
PAPER = EVALUATION_ROOT
INPUT = PROJECT / "input.txt"
RATE = PROJECT / "input_angle.txt"
NONRL_HB = PROJECT / "improvement" / "results_handover_blocking.csv"
NONRL_POLICY_ROOT = PROJECT / "throughput"
NONRL_C5_CORRECTED = (
    EVALUATION_ROOT
    / "nonrl_throughput_mapping_fix_20261017_224524"
    / "nonrl_throughput_c5_corrected.csv"
)

_SCENARIO = experiment_config()["capacity_sensitivity"]
UE_COUNT = _SCENARIO["number_of_ues"]
CAPACITIES = _SCENARIO["capacities"]
SEEDS = simulation_config()["evaluation_order_seeds"]
TOP_K = method_config()["rl"]["top_k"]
TIME_SLOTS = simulation_config()["time"]["evaluated_slots"]
NONRL_SOURCE_SLOTS = 3600

RL_SPECS = {
    "Proposed": {
        "module": proposed_module,
        "network": proposed_module.QNet,
        "state_dim": 251,
        "checkpoint": RL_DIR / "runs" / "proposed_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_imi.pt",
    },
    "Dueling DDQN (No-MSHBO)": {
        "module": dueling_module,
        "network": dueling_module.QNet,
        "state_dim": 251,
        "checkpoint": RL_DIR / "runs" / "no_mshbo_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_no_mshbo.pt",
    },
    "DDQN": {
        "module": standard_module,
        "network": standard_module.QNet,
        "state_dim": 251,
        "checkpoint": RL_DIR / "runs" / "ddqn_persistent_topk_seed42_300ep" / "standard_ddqn_no_mshbo.pt",
    },
    "Proposed w/o Remaining-Capacity State": {
        "module": ablation_module,
        "network": ablation_module.QNet,
        "state_dim": 201,
        "checkpoint": RL_DIR / "runs" / "no_capacity_state_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_imi_no_capacity_state.pt",
    },
}
NONRL_METHODS = ["MSHBO", "MSH", "GBW2", "MAC", "MVT"]
METHODS_ABLATION = list(RL_SPECS) + NONRL_METHODS
METHODS_STANDARD = ["Proposed", "Dueling DDQN (No-MSHBO)", "DDQN"] + NONRL_METHODS

STYLE = {
    "Proposed": dict(color="#000000", linestyle="-", marker="o", zorder=2.3),
    "Dueling DDQN (No-MSHBO)": dict(color="#17BECF", linestyle="--", marker="v", zorder=4.1),
    "DDQN": dict(color="#7F7F7F", linestyle="-.", marker="d", zorder=5.1),
    "Proposed w/o Remaining-Capacity State": dict(color="#9467BD", linestyle=":", marker="s", zorder=6.2),
    "MSHBO": dict(color="#D62728", linestyle="-", marker="^", zorder=2.2),
    "MSH": dict(color="#FF7F0E", linestyle="--", marker="s", zorder=4.2),
    "GBW2": dict(color="#2CA02C", linestyle="-.", marker="*", zorder=5.2),
    "MAC": dict(color="#1F77B4", linestyle="-", marker="P", zorder=2.1),
    "MVT": dict(color="#9467BD", linestyle=":", marker="X", zorder=6.1),
}

ACTION_TO_RATE_ID = (
    36, 27, 18, 10, 2, 44, 35, 26, 17, 9,
    1, 43, 34, 25, 16, 8, 0, 42, 33, 24,
    15, 7, 49, 41, 32, 23, 14, 6, 48, 40,
    31, 22, 13, 5, 47, 39, 30, 21, 12, 4,
    46, 38, 29, 3, 11, 19, 20, 28, 37, 45,
)

FIGURE_STEMS = [
    "average_handover_count_capacity_sensitivity_ablation",
    "average_blocking_rate_capacity_sensitivity_ablation",
    "average_throughput_capacity_sensitivity_ablation",
    "average_handover_count_capacity_sensitivity",
    "average_blocking_rate_capacity_sensitivity",
    "average_throughput_capacity_sensitivity",
]
PAPER_OUTPUTS = [PAPER / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")]
PAPER_OUTPUTS += [
    PAPER / "capacity_sensitivity_raw.csv",
    PAPER / "capacity_sensitivity_mean_std.csv",
    PAPER / "capacity_sensitivity_plot_report.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_evaluation_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = EVALUATION_ROOT / f"capacity_sensitivity_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def validate_inputs() -> dict[str, dict[str, str]]:
    existing = [path for path in PAPER_OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite requested outputs: {existing}")
    sources = {inspect.getsource(spec["module"].topk_mask) for spec in RL_SPECS.values()}
    if len(sources) != 1:
        raise RuntimeError("The four RL topk_mask implementations differ")
    provenance = {}
    for method, spec in RL_SPECS.items():
        path = spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        model = spec["network"](spec["state_dim"], 51)
        loaded = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"Strict checkpoint load failed: {method}")
        with torch.inference_mode():
            output = model(torch.zeros((2, spec["state_dim"]), dtype=torch.float32))
        if output.shape != (2, 51) or not torch.isfinite(output).all():
            raise RuntimeError(f"Checkpoint forward pass failed: {method}")
        provenance[method] = {
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": sha256(path),
            "state_dimension": spec["state_dim"],
        }
    metadata = json.loads(
        (RL_DIR / "runs" / "no_capacity_state_persistent_topk_seed42_300ep" / "run_metadata.json")
        .read_text(encoding="utf-8")
    )
    if metadata.get("state_dimension") != 201 or metadata.get("top_k") != TOP_K:
        raise RuntimeError("Ablation metadata does not describe the formal 201-dimensional Top-K=3 run")
    if not metadata.get("mshbo_demonstration_enabled") or metadata.get("mshbo_expert_status") != "loaded":
        raise RuntimeError("Ablation metadata does not retain MSHBO Expert Replay")
    for required in (INPUT, RATE, NONRL_HB, NONRL_C5_CORRECTED):
        if not required.is_file():
            raise FileNotFoundError(required)
    return provenance


def create_scenarios(directory: Path) -> dict[int, str]:
    lines = INPUT.read_text(encoding="utf-8").splitlines()
    n, _, tb, te = map(int, lines[0].replace(",", " ").split()[:4])
    if len(lines) != n + 1 or te - tb != TIME_SLOTS:
        raise RuntimeError("Unexpected base input dimensions")
    scenario_dir = directory / "scenarios"
    scenario_dir.mkdir(parents=True)
    scenarios = {}
    for capacity in CAPACITIES:
        satellite_rows = []
        for line in lines[1:]:
            begin, end, _ = map(int, line.split())
            satellite_rows.append(f"{begin} {end} {capacity}")
        path = scenario_dir / f"input_m{UE_COUNT}_c{capacity}.txt"
        path.write_text(
            "\n".join([f"{n} {UE_COUNT} {tb} {te}"] + satellite_rows) + "\n",
            encoding="utf-8",
        )
        scenarios[capacity] = str(path.resolve())
    return scenarios


def load_worker_models() -> dict[str, torch.nn.Module]:
    torch.set_num_threads(1)
    models = {}
    for method, spec in RL_SPECS.items():
        model = spec["network"](spec["state_dim"], 51)
        model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=False), strict=True)
        model.eval()
        models[method] = model
    return models


def evaluate_rl_one(
    method: str,
    model: torch.nn.Module,
    scenario: str,
    capacity: int,
    seed: int,
    provenance: dict[str, dict[str, str]],
) -> dict:
    spec = RL_SPECS[method]
    env_module = ablation_module if spec["state_dim"] == 201 else proposed_module
    env = env_module.HandoverEnv(
        scenario,
        str(RATE),
        lambda_h=3.0,
        seed=42,
        training_order_seed=42,
        demonstration_order_seed=42,
        evaluation_order_seed=seed,
    )
    if (env.m, env.T, env.state_dim, env.n_actions) != (UE_COUNT, TIME_SLOTS, spec["state_dim"], 51):
        raise RuntimeError(f"Unexpected environment dimensions: {method}, C={capacity}, seed={seed}")
    if set(int(value) for value in env.cap) != {capacity}:
        raise RuntimeError(f"Capacity mismatch: {method}, C={capacity}, seed={seed}")
    env.reset_multi_user_episode()
    env.reset_evaluation_order_rng(seed)
    decisions = blocks = handovers = mask_failures = capacity_violations = 0
    rate_sum = 0.0
    with torch.inference_mode():
        for t in range(env.T):
            order = env.next_evaluation_user_order()
            if sorted(int(user) for user in order) != list(range(env.m)):
                raise RuntimeError(f"Invalid UE permutation: {method}, C={capacity}, seed={seed}, t={t}")
            for raw_user in order:
                user = int(raw_user)
                previous = int(env.last_sat[user])
                state = env.make_state(user, t)
                if state.shape != (spec["state_dim"],) or not np.isfinite(state).all():
                    raise RuntimeError(f"Invalid state: {method}, C={capacity}, seed={seed}, t={t}")
                legal = env.legal_mask(user, t)
                selection = spec["module"].topk_mask(
                    legal, env.used, env.cap, t, previous, env.block_action, TOP_K
                )
                if selection[env.block_action] <= 0 or np.any((selection > 0) & (legal <= 0)):
                    mask_failures += 1
                if previous >= 0 and legal[previous] > 0 and selection[previous] <= 0:
                    mask_failures += 1
                action = spec["module"].masked_argmax(model(torch.from_numpy(state)), selection)
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
    if decisions != UE_COUNT * TIME_SLOTS or mask_failures or capacity_violations:
        raise RuntimeError(
            f"RL validation failed: {method}, C={capacity}, seed={seed}, decisions={decisions}, "
            f"mask_failures={mask_failures}, capacity_violations={capacity_violations}"
        )
    record = provenance[method]
    return {
        "method": method,
        "method_group": "RL",
        "ue_count": UE_COUNT,
        "capacity": capacity,
        "evaluation_order_seed": seed,
        "time_slots": TIME_SLOTS,
        "total_decisions": decisions,
        "average_throughput_mbps": rate_sum / decisions,
        "total_blocks": blocks,
        "blocking_rate_percent": blocks / decisions * 100.0,
        "total_handovers": handovers,
        "average_handovers_per_ue": handovers / UE_COUNT,
        "mask_failures": mask_failures,
        "capacity_violations": capacity_violations,
        "state_dimension": spec["state_dim"],
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "result_source": "new_fixed_c5_checkpoint_cross_capacity_evaluation",
        "metric_source_notes": "RL formal definitions; 3599 slots",
    }


def evaluate_seed(seed: int, scenarios: dict[int, str], provenance: dict) -> list[dict]:
    models = load_worker_models()
    rows = []
    for capacity in CAPACITIES:
        for method in RL_SPECS:
            rows.append(evaluate_rl_one(method, models[method], scenarios[capacity], capacity, seed, provenance))
    return rows


def load_rate_table() -> dict[tuple[int, int], float]:
    rates = {}
    with RATE.open("r", encoding="utf-8") as stream:
        for line in stream:
            sat, time, rate = line.split()
            rates[(int(sat), int(time))] = float(rate)
    return rates


def calculate_nonrl_throughput(method: str, capacity: int, rates: dict[tuple[int, int], float]) -> float:
    policy = NONRL_POLICY_ROOT / method / f"{method}_m{UE_COUNT}_c{capacity}_policy.csv"
    rows = read_csv(policy)
    seen = set()
    rate_sum = 0.0
    for row in rows:
        time = int(row["t"])
        if not 0 <= time < TIME_SLOTS:
            continue
        user = int(row["user"])
        action = int(row["sat"])
        if not 0 <= user < UE_COUNT or (time, user) in seen:
            raise RuntimeError(f"Invalid or duplicate non-RL policy row: {policy}, t={time}, user={user}")
        seen.add((time, user))
        if action == -1:
            continue
        if not 0 <= action < len(ACTION_TO_RATE_ID):
            raise RuntimeError(f"Invalid action in non-RL policy: {policy}, action={action}")
        rate_sum += rates[(ACTION_TO_RATE_ID[action], time)]
    expected = UE_COUNT * TIME_SLOTS
    if len(seen) != expected:
        raise RuntimeError(f"Incomplete non-RL policy window: {policy}, {len(seen)}/{expected}")
    return rate_sum / expected


def load_nonrl_rows() -> list[dict]:
    hb_rows = read_csv(NONRL_HB)
    lookup = {}
    for row in hb_rows:
        method = "GBW2" if row["algorithm"] == "GBW" else row["algorithm"]
        if method in NONRL_METHODS and int(row["m"]) == UE_COUNT and int(row["c"]) in CAPACITIES:
            key = (method, int(row["c"]))
            if key in lookup:
                raise RuntimeError(f"Duplicate non-RL handover/blocking result: {key}")
            lookup[key] = row
    expected = {(method, capacity) for method in NONRL_METHODS for capacity in CAPACITIES}
    if set(lookup) != expected:
        raise RuntimeError(f"Missing non-RL handover/blocking results: {sorted(expected - set(lookup))}")

    rates = load_rate_table()
    c5_expected = {
        row["algorithm"]: float(row["avg_throughput_all_users_mbps"])
        for row in read_csv(NONRL_C5_CORRECTED)
        if int(row["m"]) == UE_COUNT and int(row["c"]) == 5
    }
    rows = []
    for method in NONRL_METHODS:
        for capacity in CAPACITIES:
            source = lookup[(method, capacity)]
            throughput = calculate_nonrl_throughput(method, capacity, rates)
            if capacity == 5 and abs(throughput - c5_expected[method]) > 1e-6:
                raise RuntimeError(f"Corrected C=5 throughput mismatch: {method}")
            blocked_ues_per_slot = float(source["avg_blocking_rate"])
            handovers = float(source["Average h"])
            values = [throughput, blocked_ues_per_slot, handovers]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite non-RL metric: {method}, C={capacity}")
            rows.append({
                "method": method,
                "method_group": "Non-RL",
                "ue_count": UE_COUNT,
                "capacity": capacity,
                "evaluation_order_seed": "",
                "time_slots": NONRL_SOURCE_SLOTS,
                "total_decisions": "",
                "average_throughput_mbps": throughput,
                "total_blocks": "",
                "blocking_rate_percent": blocked_ues_per_slot / UE_COUNT * 100.0,
                "total_handovers": "",
                "average_handovers_per_ue": handovers,
                "mask_failures": "",
                "capacity_violations": "",
                "state_dimension": "",
                "checkpoint_path": "",
                "checkpoint_sha256": "",
                "result_source": f"{NONRL_HB.resolve()} | {NONRL_POLICY_ROOT.resolve()}",
                "metric_source_notes": (
                    "Handover/blocking: original 3600-slot results; blocking converted from blocked UEs/slot "
                    "to percent. Throughput: existing policy reevaluated over t=0..3598 with corrected action-to-rate mapping."
                ),
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for method in METHODS_ABLATION:
        for capacity in CAPACITIES:
            group = [row for row in rows if row["method"] == method and row["capacity"] == capacity]
            expected_count = len(SEEDS) if method in RL_SPECS else 1
            if len(group) != expected_count:
                raise RuntimeError(f"Incomplete group: {method}, C={capacity}, n={len(group)}")
            item = {
                "method": method,
                "method_group": group[0]["method_group"],
                "ue_count": UE_COUNT,
                "capacity": capacity,
                "sample_count": expected_count,
                "evaluation_order_seeds": "42;123;2024;2025;2026" if method in RL_SPECS else "",
                "uncertainty_type": (
                    "sample_sd_ddof_1_across_evaluation_order_seeds" if method in RL_SPECS
                    else "not_available_single_original_result"
                ),
            }
            for metric in ("average_throughput_mbps", "blocking_rate_percent", "average_handovers_per_ue"):
                values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            summary.append(item)
    return summary


def configure_plotting() -> None:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 8.3,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = PAPER / f"{stem}.png"
    pdf = PAPER / f"{stem}.pdf"
    if png.exists() or pdf.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {png}, {pdf}")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric(summary: list[dict], methods: list[str], metric: str, ylabel: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    maximum = 0.0
    handles = {}
    for method in methods:
        group = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: row["capacity"],
        )
        x = np.asarray([row["capacity"] for row in group], dtype=int)
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in group], dtype=float)
        maximum = max(maximum, float(mean.max()))
        style = STYLE[method].copy()
        zorder = style.pop("zorder")
        common = dict(
            label=method,
            markersize=8.5 if method != "GBW2" else 10.5,
            linewidth=2.6 if method == "Proposed" else 2.0,
            markeredgewidth=1.0,
            alpha=0.92,
            zorder=zorder,
            **style,
        )
        if method in RL_SPECS:
            sd = np.asarray([float(row[f"{metric}_sd"]) for row in group], dtype=float)
            maximum = max(maximum, float(np.max(mean + sd)))
            handles[method] = ax.errorbar(
                x,
                mean,
                yerr=sd,
                capsize=3,
                elinewidth=1.0,
                capthick=1.0,
                ecolor=style["color"],
                **common,
            )
        else:
            handles[method] = ax.plot(x, mean, **common)[0]
    ax.set_xlabel("Satellite Capacity (UEs/satellite)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CAPACITIES)
    ax.set_ylim(0, maximum * 1.15 if maximum > 0 else 1.0)
    ax.grid(True, which="major", linestyle=":", color="0.82", linewidth=0.8, zorder=0)
    legend_columns = 3
    legend_rows = math.ceil(len(methods) / legend_columns)
    legend_order = [
        row * legend_columns + column
        for column in range(legend_columns)
        for row in range(legend_rows)
        if row * legend_columns + column < len(methods)
    ]
    ax.legend(
        handles=[handles[methods[index]] for index in legend_order],
        labels=[methods[index] for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=legend_columns,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, stem)


def write_report(evaluation_dir: Path, provenance: dict, raw: list[dict], summary: list[dict]) -> None:
    lines = [
        "# Capacity Sensitivity 實驗報告",
        "",
        "## 實驗設定",
        "",
        "- 固定 UE=25、Top-K=3；Satellite Capacity=1、3、5、7、9。",
        "- RL 使用 C=5 訓練完成的固定策略跨容量評估，未重新訓練或更新權重。",
        "- RL Time Slots=3599；Evaluation Order Seeds=42、123、2024、2025、2026。",
        "- RL Error Bar 為五組 Evaluation Order Seeds 的 Mean ± 1 Sample SD，不是 Independent Training Runs。",
        "- Non-RL 使用既有 Capacity Sensitivity 結果與既有 Policy；未重新設計或重跑演算法。",
        "- Non-RL 原始 Handover/Blocking 統計使用 3600-slot 流程，與 RL 的 3599 slots 有一個 slot 的來源差異。",
        "- Non-RL 不具真正重複評估，因此圖中不繪製 Error Bar，也不以 SD=0 代替。",
        "",
        "## 指標與單位",
        "",
        "- RL Throughput = rate_sum / (UE × 3599)。",
        "- RL Blocking Rate (%) = total_blocks / (UE × 3599) × 100。",
        "- RL Handovers/UE = total_handovers / UE。",
        "- Non-RL Blocking 原始值為 average blocked UEs per slot；本次以原值 / 25 × 100 轉為百分比。",
        "- Non-RL Throughput 使用既有 Policy、t=0..3598、正式 Action-to-Rate mapping 重新計算，並以既有 C=5 修正結果交叉驗證。",
        "",
        "## RL Checkpoints",
        "",
    ]
    for method in RL_SPECS:
        record = provenance[method]
        lines += [
            f"- {method}: `{record['checkpoint_path']}`",
            f"  - State dimension: {record['state_dimension']}；SHA-256: `{record['checkpoint_sha256']}`",
        ]
    lines += [
        "",
        "## 資料來源",
        "",
        f"- Non-RL Handover/Blocking: `{NONRL_HB.resolve()}`。",
        f"- Non-RL Policies: `{NONRL_POLICY_ROOT.resolve()}`。",
        f"- Non-RL C=5 Throughput validation: `{NONRL_C5_CORRECTED.resolve()}`。",
        f"- 本次獨立 RL 評估目錄: `{evaluation_dir.resolve()}`。",
        "- 所有 9 種方法均有 C=1、3、5、7、9 的真實結果，無缺值、推估、插值或複製值。",
        "",
        "## 圖層與繪圖",
        "",
        "- Solid zorder=2.x；Dashed zorder=4.x；Dash-dot zorder=5.x；Dotted zorder=6.x。",
        "- Error Bar capsize=3、elinewidth=1.0；六張圖均無 Inset 或 Zoom。",
        "- 實線位於虛線、點畫線與點線下層。",
        "",
        "## 趨勢檢查",
        "",
    ]
    for method in METHODS_ABLATION:
        rows = sorted((row for row in summary if row["method"] == method), key=lambda row: row["capacity"])
        blocking = [float(row["blocking_rate_percent_mean"]) for row in rows]
        throughput = [float(row["average_throughput_mbps_mean"]) for row in rows]
        lines.append(
            f"- {method}: Blocking C1→C9 {blocking[0]:.6f}%→{blocking[-1]:.6f}%；"
            f"Throughput {throughput[0]:.6f}→{throughput[-1]:.6f} Mbps。"
        )
    lines += ["", "## 輸出", ""]
    for path in PAPER_OUTPUTS:
        lines.append(f"- `{path.resolve()}`")
    lines += [
        "",
        "## 完整性檢查",
        "",
        f"- Raw rows={len(raw)}；Summary rows={len(summary)}。",
        "- RL Mask Failure=0、Capacity Violation=0、每個 UE/slot 恰好一次決策。",
        "- 所有指標為有限值；原始圖片、CSV 與 Checkpoint 未覆寫。",
    ]
    (PAPER / "capacity_sensitivity_plot_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    paper_before = {
        path.resolve(): sha256(path)
        for path in PAPER.iterdir()
        if path.is_file() and path not in PAPER_OUTPUTS
    }
    provenance = validate_inputs()
    evaluation_dir = unique_evaluation_dir()
    evaluation_dir.mkdir(parents=True)
    scenarios = create_scenarios(evaluation_dir)

    rl_rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(evaluate_seed, seed, scenarios, provenance): seed
            for seed in SEEDS
        }
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            rl_rows.extend(result)
            print(f"completed RL evaluation seed={seed}: {len(result)} scenarios", flush=True)
    rl_rows.sort(key=lambda row: (METHODS_ABLATION.index(row["method"]), row["capacity"], row["evaluation_order_seed"]))
    if len(rl_rows) != 100:
        raise RuntimeError(f"Expected 100 RL rows, found {len(rl_rows)}")
    write_csv(evaluation_dir / "rl_capacity_sensitivity_raw.csv", rl_rows)

    nonrl_rows = load_nonrl_rows()
    raw = rl_rows + nonrl_rows
    raw.sort(key=lambda row: (METHODS_ABLATION.index(row["method"]), row["capacity"], str(row["evaluation_order_seed"])))
    if len(raw) != 125:
        raise RuntimeError(f"Expected 125 raw rows, found {len(raw)}")
    summary = summarize(raw)
    if len(summary) != 45:
        raise RuntimeError(f"Expected 45 summary rows, found {len(summary)}")
    write_csv(PAPER / "capacity_sensitivity_raw.csv", raw)
    write_csv(PAPER / "capacity_sensitivity_mean_std.csv", summary)

    manifest = {
        "experiment": "fixed policies trained at C=5 evaluated across satellite capacity",
        "ue_count": UE_COUNT,
        "capacities": CAPACITIES,
        "rl_time_slots": TIME_SLOTS,
        "nonrl_handover_blocking_source_slots": NONRL_SOURCE_SLOTS,
        "top_k": TOP_K,
        "evaluation_order_seeds": SEEDS,
        "rl_provenance": provenance,
        "rl_mask_failures": sum(int(row["mask_failures"]) for row in rl_rows),
        "rl_capacity_violations": sum(int(row["capacity_violations"]) for row in rl_rows),
        "blocking_definition_rl": "total_blocks / (UE * T) * 100",
        "blocking_conversion_nonrl": "average_blocked_ues_per_slot / UE * 100",
    }
    (evaluation_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    configure_plotting()
    for methods, suffix in ((METHODS_ABLATION, "_ablation"), (METHODS_STANDARD, "")):
        plot_metric(
            summary,
            methods,
            "average_handovers_per_ue",
            "Average Handover Count per UE",
            f"average_handover_count_capacity_sensitivity{suffix}",
        )
        plot_metric(
            summary,
            methods,
            "blocking_rate_percent",
            "Average Blocking Rate (%)",
            f"average_blocking_rate_capacity_sensitivity{suffix}",
        )
        plot_metric(
            summary,
            methods,
            "average_throughput_mbps",
            "Average Throughput (Mbps)",
            f"average_throughput_capacity_sensitivity{suffix}",
        )

    write_report(evaluation_dir, provenance, raw, summary)
    changed = [str(path) for path, digest in paper_before.items() if sha256(path) != digest]
    if changed:
        raise RuntimeError(f"Existing paper files changed unexpectedly: {changed}")
    for path in PAPER_OUTPUTS:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    print(f"evaluation_dir={evaluation_dir.resolve()}", flush=True)
    print("all_outputs_completed", flush=True)


if __name__ == "__main__":
    main()
