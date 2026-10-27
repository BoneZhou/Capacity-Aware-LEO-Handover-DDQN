import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 基本設定
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "ue_sensitivity" / "results_handover_blocking.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "tradeoff"

# 欄位名稱
HANDOVER_COL = "Average h"
BLOCKING_COL = "avg_blocking_rate"

# 固定顏色
COLOR_MAP = {
    "Proposed": "black",
    "DDQN": "gray",
    "MSHBO": "#d62728",
    "MSH": "#ff7f0e",
    "MAC": "#1f77b4",
    "GBW": "#2ca02c",
    "MVT": "#9467bd",
}

# 其他未定義演算法自動配色
AUTO_COLORS = [
    "#8c564b", "#e377c2", "#17becf",
    "#7f7f7f", "#bcbd22", "#aec7e8", "#ff9896"
]

# 其他未定義演算法 marker
OTHER_MARKERS = ['s', 'D', 'P', 'v', '<', '>', 'h', '8']

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12
})


# =========================
# 工具函式
# =========================
def sanitize_filename(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def savefig_both(base_path: str, dpi: int = 300):
    """
    同時輸出 PNG 與 EPS
    base_path 不要加副檔名
    """
    plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{base_path}.eps", format="eps", bbox_inches="tight")


def pretty_algo_label(name: str) -> str:
    """
    圖例顯示名稱
    """
    label_map = {
        "Proposed": "Proposed",
        "DDQN": "DDQN",
        "MSHBO": "MSHBO",
        "MSH": "MSH",
        "MAC": "MAC",
        "GBW": "GBW",
        "MVT": "MVT",
    }
    return label_map.get(str(name).strip(), str(name).strip())


def get_algorithm_order(df: pd.DataFrame):
    """
    固定演算法順序：依照你指定的 legend 順序
    """
    preferred_order = [
        "GBW",
        "MAC",
        "MVT",
        "MSH",
        "MSHBO",
        "DDQN",
        "Proposed"
    ]

    algos = list(df["algorithm"].dropna().unique())
    ordered = [a for a in preferred_order if a in algos]
    others = sorted([a for a in algos if a not in preferred_order], key=lambda x: str(x).upper())
    return ordered + others


def apply_ordered_legend():
    """
    強制套用固定 legend 順序
    """
    handles, labels = plt.gca().get_legend_handles_labels()

    order = [
        "GBW",
        "MAC",
        "MVT",
        "MSH",
        "MSHBO",
        "DDQN",
        "Proposed"
    ]

    ordered_handles = [handles[labels.index(l)] for l in order if l in labels]
    ordered_labels = [l for l in order if l in labels]

    plt.legend(ordered_handles, ordered_labels)


# =========================
# 樣式設定
# =========================
def get_style(algo_name: str, other_index: int = 0):
    name = str(algo_name).strip()

    if name == "Proposed":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.2,
            "marker": "o",
            "markersize": 12
        }

    elif name == "DDQN":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "--",
            "linewidth": 2.2,
            "marker": "d",   # 菱形
            "markersize": 12
        }

    elif name == "MSH":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "s",
            "markersize": 12
        }

    elif name == "MSHBO":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "^",
            "markersize": 12
        }

    elif name == "GBW":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "*",
            "markersize": 14
        }

    elif name == "MVT":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "X",
            "markersize": 13
        }

    elif name == "MAC":
        return {
            "color": COLOR_MAP[name],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": "P",
            "markersize": 13
        }

    else:
        return {
            "color": AUTO_COLORS[other_index % len(AUTO_COLORS)],
            "linestyle": "-",
            "linewidth": 2.0,
            "marker": OTHER_MARKERS[other_index % len(OTHER_MARKERS)],
            "markersize": 10
        }


# =========================
# 資料讀取與整理
# =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("=== 讀到的欄位 ===")
    print(df.columns.tolist())

    required_cols = ["algorithm", "m", "c", HANDOVER_COL, BLOCKING_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要欄位: {col}")

    df["algorithm"] = df["algorithm"].astype(str).str.strip()
    df["m"] = pd.to_numeric(df["m"], errors="coerce")
    df["c"] = pd.to_numeric(df["c"], errors="coerce")
    df[HANDOVER_COL] = pd.to_numeric(df[HANDOVER_COL], errors="coerce")
    df[BLOCKING_COL] = pd.to_numeric(df[BLOCKING_COL], errors="coerce")

    print("\n=== 演算法名稱 ===")
    print(df["algorithm"].dropna().unique())

    df = df.dropna(subset=["algorithm", "m", "c", HANDOVER_COL, BLOCKING_COL]).copy()
    df["m"] = df["m"].astype(int)
    df["c"] = df["c"].astype(int)

    return df


# =========================
# 畫 trade-off 圖：固定 c，看 m 變化
# =========================
def plot_tradeoff_same_c(df: pd.DataFrame):
    os.makedirs(f"{OUTPUT_DIR}/same_c", exist_ok=True)

    for c_val in sorted(df["c"].unique()):
        sub = df[df["c"] == c_val].copy()
        if sub.empty:
            continue

        algorithms = get_algorithm_order(sub)

        plt.figure(figsize=(7, 6))

        other_idx = 0
        for algo in algorithms:
            data = sub[sub["algorithm"] == algo].sort_values("m")
            if data.empty:
                continue

            style = get_style(algo, other_idx)
            if algo not in COLOR_MAP:
                other_idx += 1

            plt.plot(
                data[HANDOVER_COL],
                data[BLOCKING_COL],
                label=pretty_algo_label(algo),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"]
            )

        plt.xlabel("Average Handover Count", fontsize=18)
        plt.ylabel("Average Blocking Rate (events/second)", fontsize=18)
        plt.grid(True, linestyle=":", color="lightgray")
        apply_ordered_legend()
        plt.tight_layout()

        savefig_both(f"{OUTPUT_DIR}/same_c/tradeoff_c_{sanitize_filename(c_val)}")
        plt.close()


# =========================
# 畫 trade-off 圖：固定 m，看 c 變化
# =========================
def plot_tradeoff_same_m(df: pd.DataFrame):
    os.makedirs(f"{OUTPUT_DIR}/same_m", exist_ok=True)

    for m_val in sorted(df["m"].unique()):
        sub = df[df["m"] == m_val].copy()
        if sub.empty:
            continue

        algorithms = get_algorithm_order(sub)

        plt.figure(figsize=(7, 6))

        other_idx = 0
        for algo in algorithms:
            data = sub[sub["algorithm"] == algo].sort_values("c")
            if data.empty:
                continue

            style = get_style(algo, other_idx)
            if algo not in COLOR_MAP:
                other_idx += 1

            plt.plot(
                data[HANDOVER_COL],
                data[BLOCKING_COL],
                label=pretty_algo_label(algo),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"]
            )

        plt.xlabel("Average Handover Count", fontsize=18)
        plt.ylabel("Average Blocking Rate (events/second)", fontsize=18)
        plt.grid(True, linestyle=":", color="lightgray")
        apply_ordered_legend()
        plt.tight_layout()

        savefig_both(f"{OUTPUT_DIR}/same_m/tradeoff_m_{sanitize_filename(m_val)}")
        plt.close()


# =========================
# 論文常用圖：固定 c = 5，點標示 U
# =========================
def plot_tradeoff_fixed_c5(df: pd.DataFrame):
    sub = df[df["c"] == 5].copy()
    if sub.empty:
        print("警告：資料中沒有 c = 5，略過 tradeoff_c5 圖。")
        return

    sub = sub.sort_values(["algorithm", "m"])
    algorithms = get_algorithm_order(sub)

    os.makedirs(f"{OUTPUT_DIR}/paper", exist_ok=True)

    plt.figure(figsize=(7, 6))

    other_idx = 0
    for algo in algorithms:
        data = sub[sub["algorithm"] == algo].sort_values("m")
        if data.empty:
            continue

        style = get_style(algo, other_idx)
        if algo not in COLOR_MAP:
            other_idx += 1

        plt.plot(
            data[HANDOVER_COL],
            data[BLOCKING_COL],
            label=pretty_algo_label(algo),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"]
        )

        for _, row in data.iterrows():
            plt.annotate(
                f"U={int(row['m'])}",
                (row[HANDOVER_COL], row[BLOCKING_COL]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9
            )

    plt.xlabel("Average Handover Count", fontsize=18)
    plt.ylabel("Average Blocking Rate (events/second)", fontsize=18)
    plt.grid(True, linestyle=":", color="lightgray")
    apply_ordered_legend()
    plt.tight_layout()

    savefig_both(f"{OUTPUT_DIR}/paper/tradeoff_c5")
    plt.close()


# =========================
# 論文常用圖：固定 m = 20，點標示 C=1,3,5,7,9
# =========================
def plot_tradeoff_fixed_m20(df: pd.DataFrame):
    sub = df[(df["m"] == 20) & (df["c"].isin([1, 3, 5, 7, 9]))].copy()
    if sub.empty:
        print("警告：資料中沒有 m = 20 且 c = 1,3,5,7,9 的資料，略過 tradeoff_m20 圖。")
        return

    sub = sub.sort_values(["algorithm", "c"])
    algorithms = get_algorithm_order(sub)

    os.makedirs(f"{OUTPUT_DIR}/paper", exist_ok=True)

    plt.figure(figsize=(7, 6))

    other_idx = 0
    for algo in algorithms:
        data = sub[sub["algorithm"] == algo].sort_values("c")
        if data.empty:
            continue

        style = get_style(algo, other_idx)
        if algo not in COLOR_MAP:
            other_idx += 1

        plt.plot(
            data[HANDOVER_COL],
            data[BLOCKING_COL],
            label=pretty_algo_label(algo),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"]
        )

        for _, row in data.iterrows():
            plt.annotate(
                f"C={int(row['c'])}",
                (row[HANDOVER_COL], row[BLOCKING_COL]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9
            )

    plt.xlabel("Average Handover Count", fontsize=18)
    plt.ylabel("Average Blocking Rate (events/second)", fontsize=18)
    plt.grid(True, linestyle=":", color="lightgray")
    apply_ordered_legend()
    plt.tight_layout()

    savefig_both(f"{OUTPUT_DIR}/paper/tradeoff_m20")
    plt.close()


# =========================
# 主程式
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(CSV_PATH)

    plot_tradeoff_same_c(df)
    plot_tradeoff_same_m(df)
    plot_tradeoff_fixed_c5(df)
    plot_tradeoff_fixed_m20(df)

    print(f"\n完成：trade-off 圖已輸出到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
