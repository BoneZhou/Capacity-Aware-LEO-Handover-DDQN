from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_ps import RendererPS
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs"
OUT = SOURCE_ROOT / "figures"


def latest_match(pattern: str, fallback: str) -> Path:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else ROOT / fallback


UE_PLOT = latest_match(
    "outputs/ue_sensitivity/all_baselines_lambda3_*/merged_plot_data.csv",
    "outputs/ue_sensitivity/merged_plot_data.csv",
)
UE_FORMAL_RAW = latest_match(
    "outputs/ue_sensitivity/rl_multiseed_lambda3_*/raw_seed_results.csv",
    "outputs/ue_sensitivity/raw_seed_results.csv",
)
UE_RAW = SOURCE_ROOT / "ablation_vs_ue" / "ablation_vs_ue_raw.csv"
UE_SUMMARY = SOURCE_ROOT / "ablation_vs_ue" / "ablation_vs_ue_mean_std.csv"
CAP_RAW = SOURCE_ROOT / "capacity_sensitivity" / "capacity_sensitivity_raw.csv"
CAP_SUMMARY = SOURCE_ROOT / "capacity_sensitivity" / "capacity_sensitivity_mean_std.csv"

SEEDS = {42, 123, 2024, 2025, 2026}
UE_VALUES = [10, 15, 20, 25, 30]
CAPACITY_VALUES = [1, 3, 5, 7, 9]
RL3 = ["Proposed", "Dueling DDQN (No-MSHBO)", "DDQN"]
ABLATION = "Proposed w/o Remaining-Capacity State"
RL4 = RL3 + [ABLATION]
NON_RL = ["MSHBO", "MSH", "GBW2", "MAC", "MVT"]

STYLES = {
    "Proposed": ("#000000", "-", "o", 2.6, 7, 2),
    "Dueling DDQN (No-MSHBO)": ("#17BECF", "--", "v", 2.2, 7, 4),
    "DDQN": ("#7F7F7F", "-.", "d", 2.2, 7, 5),
    ABLATION: ("#9467BD", ":", "s", 2.2, 7, 6),
    "MSHBO": ("#D62728", "-", "^", 2.0, 7, 2),
    "MSH": ("#FF7F0E", "--", "s", 2.0, 7, 4),
    "GBW2": ("#2CA02C", "-.", "*", 2.0, 9, 5),
    "MAC": ("#1F77B4", "-", "P", 2.0, 7, 2),
    "MVT": ("#9467BD", ":", "X", 2.0, 7, 6),
}

METRICS = [
    ("system_throughput_mbps", "average_throughput", "Time-Average System Throughput (Mbps)"),
    ("average_handovers_per_ue", "average_handover", "Average Handovers per UE"),
    ("blocking_rate_percent", "average_blocking", "Average Blocking Rate (%)"),
]

GROUPS = [
    {"name": "Number of UEs", "prefixes": ["Fig2a_throughput", "Fig2c_handover", "Fig2b_blocking"],
     "suffix": "UE", "methods": RL3 + NON_RL, "rl": RL3, "x": "ue_count", "x_values": UE_VALUES,
     "xlabel": "Number of UEs", "data_key": "ue"},
    {"name": "Capacity sensitivity", "prefixes": ["G1_average_throughput", "G1_average_handover", "G1_average_blocking"],
     "suffix": "channels", "methods": RL3 + NON_RL, "rl": RL3, "x": "capacity", "x_values": CAPACITY_VALUES,
     "xlabel": "Number of Channels", "data_key": "capacity"},
    {"name": "Capacity-aware ablation", "prefixes": ["G2_average_throughput", "G2_average_handover", "G2_average_blocking"],
     "suffix": "UE", "methods": RL4 + NON_RL, "rl": RL4, "x": "ue_count", "x_values": UE_VALUES,
     "xlabel": "Number of UEs", "data_key": "ue"},
    {"name": "Capacity-sensitivity with capacity-aware ablation",
     "prefixes": ["G3_average_throughput", "G3_average_handover", "G3_average_blocking"],
     "suffix": "channels", "methods": RL4 + NON_RL, "rl": RL4, "x": "capacity", "x_values": CAPACITY_VALUES,
     "xlabel": "Number of Channels", "data_key": "capacity"},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_source() -> pd.DataFrame:
    rows = []
    sources = [UE_PLOT, UE_FORMAL_RAW, UE_RAW, UE_SUMMARY, CAP_RAW, CAP_SUMMARY]
    for path in sorted(item for item in sources if item.is_file()):
        rows.append({"relative_path": str(path.relative_to(SOURCE_ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    assert rows, "ICCpaper source directory is empty"
    return pd.DataFrame(rows)


def validate_raw(frame: pd.DataFrame, x: str, values: list[int], methods: list[str]) -> None:
    selected = frame[frame["method"].isin(methods)]
    assert set(selected["method"]) == set(methods)
    assert set(selected[x].astype(int)) == set(values)
    assert selected.groupby(["method", x]).size().eq(5).all()
    assert all(set(group["evaluation_order_seed"].astype(int)) == SEEDS for _, group in selected.groupby(["method", x]))
    assert np.isfinite(selected.select_dtypes(include=[np.number]).to_numpy()).all()
    assert (selected["total_decisions"] == selected["ue_count"] * selected["time_slots"]).all()
    assert (selected["mask_failures"] == 0).all()
    assert (selected["capacity_violations"] == 0).all()


def load_data() -> dict[str, pd.DataFrame]:
    sources = [UE_PLOT, UE_FORMAL_RAW, UE_RAW, UE_SUMMARY, CAP_RAW, CAP_SUMMARY]
    missing = [str(path) for path in sources if not path.is_file()]
    assert not missing, f"Missing source files: {missing}"

    ue_raw = pd.read_csv(UE_RAW)
    validate_raw(ue_raw, "ue_count", UE_VALUES, RL4)
    assert set(ue_raw["capacity"].astype(int)) == {5}
    ue_raw["system_throughput_mbps"] = ue_raw["average_throughput_mbps"] * ue_raw["ue_count"]

    cap_all = pd.read_csv(CAP_RAW)
    cap_rl = cap_all[cap_all["method"].isin(RL4)].copy()
    validate_raw(cap_rl, "capacity", CAPACITY_VALUES, RL4)
    assert set(cap_all["ue_count"].astype(int)) == {25}
    cap_all["system_throughput_mbps"] = cap_all["average_throughput_mbps"] * cap_all["ue_count"]

    ue_plot = pd.read_csv(UE_PLOT)
    if "system_throughput_mean" not in ue_plot:
        ue_plot["system_throughput_mean"] = ue_plot["average_throughput"] * ue_plot["ue_count"]
        ue_plot["blocking_percent_mean"] = ue_plot["average_blocking_rate"] / ue_plot["ue_count"] * 100
        ue_plot["handovers_mean"] = ue_plot["average_handover_count"]
        ue_plot["average_throughput_mean"] = ue_plot["average_throughput"]
    assert len(ue_plot) == 40 and not ue_plot.duplicated(["method", "ue_count"]).any()
    assert set(ue_plot["method"]) == set(RL3 + NON_RL)
    assert set(ue_plot["ue_count"].astype(int)) == set(UE_VALUES)
    assert set(ue_plot["capacity"].astype(int)) == {5}
    assert np.allclose(ue_plot["system_throughput_mean"], ue_plot["average_throughput_mean"] * ue_plot["ue_count"])
    assert np.allclose(ue_plot["blocking_percent_mean"], ue_plot["average_blocking_rate_mean"] / ue_plot["ue_count"] * 100)
    assert np.allclose(ue_plot["handovers_mean"], ue_plot["average_handover_count_mean"])

    ue_nonrl = ue_plot[ue_plot["method"].isin(NON_RL)].rename(columns={
        "system_throughput_mean": "system_throughput_mbps", "blocking_percent_mean": "blocking_rate_percent",
        "handovers_mean": "average_handovers_per_ue"})
    ue_main = pd.concat([ue_raw, ue_nonrl[["method", "ue_count", "capacity", "system_throughput_mbps",
                                                          "blocking_rate_percent", "average_handovers_per_ue"]]],
                        ignore_index=True, sort=False)

    formal_raw = pd.read_csv(UE_FORMAL_RAW).rename(columns={
        "average_throughput": "average_throughput_mbps", "average_handover_count": "average_handovers_per_ue",
        "blocking_percent": "blocking_rate_percent"})
    keys = ["method", "ue_count", "capacity", "evaluation_order_seed"]
    left = ue_raw[ue_raw["method"].isin(RL3)].sort_values(keys).reset_index(drop=True)
    right = formal_raw.sort_values(keys).reset_index(drop=True)
    assert left[keys].equals(right[keys])
    for metric in ("average_throughput_mbps", "blocking_rate_percent", "average_handovers_per_ue"):
        assert np.allclose(left[metric], right[metric], rtol=0, atol=1e-12)

    for raw, summary_path, x in ((ue_raw, UE_SUMMARY, "ue_count"), (cap_rl, CAP_SUMMARY, "capacity")):
        summary = pd.read_csv(summary_path)
        summary = summary[summary["method"].isin(RL4)]
        assert len(summary) == 20
        for row in summary.itertuples(index=False):
            group = raw[(raw["method"] == row.method) & (raw[x] == getattr(row, x))]
            assert len(group) == 5
            for metric in ("average_throughput_mbps", "blocking_rate_percent", "average_handovers_per_ue"):
                assert np.isclose(group[metric].mean(), getattr(row, f"{metric}_mean"), atol=1e-9)
                assert np.isclose(group[metric].std(ddof=1), getattr(row, f"{metric}_sd"), atol=1e-9)

    for method in RL3:
        for ue_count in UE_VALUES:
            group = ue_raw[(ue_raw["method"] == method) & (ue_raw["ue_count"] == ue_count)]
            row = ue_plot[(ue_plot["method"] == method) & (ue_plot["ue_count"] == ue_count)].iloc[0]
            assert np.isclose(group["system_throughput_mbps"].mean(), row["system_throughput_mean"], atol=1e-8)
            assert np.isclose(group["blocking_rate_percent"].mean(), row["blocking_percent_mean"], atol=1e-10)
            assert np.isclose(group["average_handovers_per_ue"].mean(), row["handovers_mean"], atol=1e-10)
    return {"ue": ue_main, "capacity": cap_all}


def eps_text_as_paths(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
    RendererBase._draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.canvas.draw()
    content = figure.get_tightbbox(figure.canvas.get_renderer())
    width, height = figure.get_size_inches()
    output_bounds = Bbox.from_extents(max(0, content.x0 - 0.08), 0, min(width, content.x1 + 0.125), height)
    png_path = stem.with_suffix(".png")
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches=output_bounds, pad_inches=0)
    figure.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white", bbox_inches=output_bounds, pad_inches=0)
    original = RendererPS.draw_text
    try:
        RendererPS.draw_text = eps_text_as_paths
        figure.savefig(stem.with_suffix(".eps"), dpi=300, facecolor="white", bbox_inches=output_bounds, pad_inches=0)
    finally:
        RendererPS.draw_text = original


def plot_one(data: pd.DataFrame, spec: dict, metric: str, ylabel: str, filename: str) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    figure.subplots_adjust(left=0.10, right=0.65, bottom=0.14, top=0.75)
    handles = {}
    for method in spec["methods"]:
        color, linestyle, marker, width, marker_size, zorder = STYLES[method]
        stats = data[data["method"] == method].groupby(spec["x"])[metric].agg(["mean", "std"]).reindex(spec["x_values"])
        assert stats["mean"].notna().all()
        common = dict(color=color, linestyle=linestyle, marker=marker, linewidth=width, markersize=marker_size,
                      markeredgecolor=color, markeredgewidth=1.2, label=method, zorder=zorder)
        if method in spec["rl"]:
            assert stats["std"].notna().all()
            container = axis.errorbar(spec["x_values"], stats["mean"], yerr=stats["std"], markerfacecolor="white",
                                      capsize=3, elinewidth=1.0, capthick=1.0, barsabove=True, **common)
            handles[method] = container.lines[0]
        else:
            handles[method] = axis.plot(spec["x_values"], stats["mean"], markerfacecolor=color, **common)[0]

    axis.set_xlabel(spec["xlabel"])
    axis.set_ylabel(ylabel)
    axis.set_xticks(spec["x_values"])
    axis.set_xlim(min(spec["x_values"]) - 1, max(spec["x_values"]) + 1)
    axis.set_ylim(bottom=0)
    axis.grid(True, which="major", color="#D3D3D3", linestyle=":", linewidth=0.7)
    axis.minorticks_off()
    axis.legend([handles[method] for method in spec["methods"]], spec["methods"], loc="lower center",
                bbox_to_anchor=(0.5, 1.025), ncol=2, frameon=False, handlelength=3.0,
                columnspacing=1.4, borderaxespad=0)
    save_figure(figure, OUT / spec["name"] / filename)
    plt.close(figure)


def make_preview(folder: Path, filenames: list[str]) -> None:
    images = [Image.open(folder / f"{name}.png").convert("RGB") for name in filenames]
    target_height = min(image.height for image in images)
    resized = [image.resize((round(image.width * target_height / image.height), target_height), Image.Resampling.LANCZOS)
               for image in images]
    gap = 24
    preview = Image.new("RGB", (sum(image.width for image in resized) + gap * 2, target_height), "white")
    x = 0
    for image in resized:
        preview.paste(image, (x, 0))
        x += image.width + gap
    preview.thumbnail((6000, 1500), Image.Resampling.LANCZOS)
    preview.save(folder / "preview.png", dpi=(300, 300))


def expected_plot_files() -> list[Path]:
    paths = []
    for spec in GROUPS:
        for prefix in spec["prefixes"]:
            stem = OUT / spec["name"] / f"{prefix}_vs_{spec['suffix']}"
            paths.extend(stem.with_suffix(f".{extension}") for extension in ("png", "pdf", "eps"))
        paths.append(OUT / spec["name"] / "preview.png")
    return paths


def verify_outputs(source_before: pd.DataFrame, source_after: pd.DataFrame) -> None:
    assert source_before.equals(source_after), "Original ICCpaper changed during rendering"
    paths = expected_plot_files()
    assert len(paths) == 40 and all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    assert len([path for path in paths if path.suffix == ".png" and path.name != "preview.png"]) == 12
    assert len([path for path in paths if path.suffix == ".pdf"]) == 12
    assert len([path for path in paths if path.suffix == ".eps"]) == 12
    assert len([path for path in paths if path.name == "preview.png"]) == 4
    for path in paths:
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.width > 1000 and image.height > 500
                assert image.info.get("dpi", (0, 0))[0] >= 299
                image.verify()
        elif path.suffix == ".pdf":
            payload = path.read_bytes()
            assert payload.startswith(b"%PDF-") and b"%%EOF" in payload[-2048:]
        else:
            assert path.read_bytes().startswith(b"%!PS-Adobe")


def write_audit(source_before: pd.DataFrame) -> None:
    source_before.to_csv(OUT / "original_iccpaper_sha256.csv", index=False, encoding="utf-8-sig")
    table_rows = []
    for spec in GROUPS:
        methods = ", ".join(spec["methods"])
        rl = ", ".join(spec["rl"])
        x_values = ", ".join(str(value) for value in spec["x_values"])
        for (_, _, ylabel), prefix in zip(METRICS, spec["prefixes"]):
            filename = f"{prefix}_vs_{spec['suffix']}"
            table_rows.append(f"| {spec['name']} | {filename} | {ylabel} | {x_values} | {methods} | {rl} |")
    lines = [
        "# ICC 2027 主圖 Error Bars 整合查核", "", "## 完成摘要", "",
        "- 已將四個分類、每類三張圖，共 12 張圖改為單一主座標區域。",
        "- RL Error Bars 已直接畫在每個 X 軸場景的對應資料點，沒有右側 Subplot、Inset 或 Zoom。",
        "- 每張圖輸出 PNG、PDF、EPS，共 36 個正式圖檔；另輸出四張 `preview.png`。",
        "- 沒有重新訓練或 Evaluation，也沒有修改原始 CSV、Checkpoint 或 `ICCpaper`。", "",
        "## 資料來源", "", f"- Group A 正式均值與 Non-RL：`{UE_PLOT}`。",
        f"- Group A 三種 RL 五組 seeds：`{UE_FORMAL_RAW}`。",
        f"- Group C 四種 RL 五組 seeds：`{UE_RAW}`；Non-RL 沿用 `{UE_PLOT}`。",
        f"- Group B/D Capacity 資料：`{CAP_RAW}`。", f"- UE Summary 交叉驗證：`{UE_SUMMARY}`。",
        f"- Capacity Summary 交叉驗證：`{CAP_SUMMARY}`。", "", "## 圖表與 Error Bar 範圍", "",
        "| 分類 | 圖檔基名 | Y 軸 | X 場景 | 全部 Methods | 加 Error Bar 的 RL |",
        "|---|---|---|---|---|---|", *table_rows, "", "## 統計與指標驗證", "",
        "- RL 的每個 method × X 場景皆有 seeds 42、123、2024、2025、2026，共五筆真實 Evaluation Order 結果。",
        "- Mean 由五筆 Raw Data 平均；Error Bar 為 Sample SD (`ddof=1`)。SD=0 時保持零長度，沒有人工放大。",
        "- Throughput 使用 `rate_sum/T`；Raw CSV 的 `rate_sum/(UE×T)` Mean 與 SD 均乘以對應 UE 數一次。",
        "- Blocking 已是 `total_blocks/(UE×T)×100`，沒有重複換算；Handover 使用 `total_handovers/UE`。",
        "- Group A/C 的 Proposed、No-MSHBO、DDQN raw observations 完全一致；Group B/D 共用同一份 Capacity dataframe。",
        "- Group A 三種 RL 的新版 Mean 已逐場景核對正式 `plot_data.csv`，結果一致。UE/Capacity Summary CSV 亦能以 `ddof=1` 重現 Raw Mean 與 SD。",
        "- Non-RL 每組只有一筆正式 deterministic 結果，沒有捏造 SD，故不繪 Error Bars。",
        "- Raw Data 的 `total_decisions = UE × time_slots`，且 Mask Failure 與 Capacity Violation 皆為 0。", "",
        "## 視覺格式與輸出驗證", "",
        "- 顏色、線型、Marker、Line Width、Marker Size、zorder 與 Legend 順序沿用 ICCpaper 規格。",
        "- Times New Roman、白色背景、淺灰點狀 Major Grid；Error Bar 使用 capsize=3、elinewidth=1.0、capthick=1.0。",
        "- 12 張 PNG 均為至少 300 DPI；12 張 PDF 與 12 張 EPS 通過向量檔頭/結尾檢查。",
        "- 四張 Preview 均按 Throughput、Handover、Blocking 由左至右排列。", "", "## 原始檔案保護", "",
        f"- 繪圖前後共核對 {len(source_before)} 個 `ICCpaper` 原始檔案；路徑、大小與 SHA-256 全部一致。",
        f"- 完整清單：`{OUT / 'original_iccpaper_sha256.csv'}`。", "", "## 缺失與不一致", "",
        "- 未發現缺少的 method、X 場景、evaluation-order seed、Mean 或 SD。",
        "- 未發現 Group A/C 或 Group B/D 共同 RL 方法的數值不一致。",
    ]
    (OUT / "errorbar_integration_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_plots_only() -> None:
    data = load_data()
    findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.labelsize": 12.5,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.3,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.axisbelow": True})
    for spec in GROUPS:
        for (metric, _, ylabel), prefix in zip(METRICS, spec["prefixes"]):
            filename = f"{prefix}_vs_{spec['suffix']}"
            plot_one(data[spec["data_key"]], spec, metric, ylabel, filename)


def regenerate_previews() -> None:
    for spec in GROUPS:
        filenames = [f"{prefix}_vs_{spec['suffix']}" for prefix in spec["prefixes"]]
        make_preview(OUT / spec["name"], filenames)


def main() -> None:
    for spec in GROUPS:
        (OUT / spec["name"]).mkdir(parents=True, exist_ok=True)
    source_before = snapshot_source()
    render_plots_only()
    regenerate_previews()
    source_after = snapshot_source()
    verify_outputs(source_before, source_after)
    write_audit(source_before)
    assert source_before.equals(snapshot_source())
    print(f"Created 36 figure files, 4 previews, one SHA-256 manifest, and one audit report under {OUT}")


if __name__ == "__main__":
    main()
