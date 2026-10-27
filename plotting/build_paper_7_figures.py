from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs" / "paper"
RL_SOURCE = PROJECT / "evaluation_results" / "rl_multiseed_lambda3_20261018_191043" / "raw_seed_results.csv"
NONRL_THROUGHPUT_SOURCE = (
    PROJECT
    / "evaluation_results"
    / "nonrl_throughput_mapping_fix_20261017_224524"
    / "nonrl_throughput_c5_corrected.csv"
)
NONRL_HB_SOURCE = PROJECT / "improvement" / "results_handover_blocking.csv"

SEEDS = [42, 123, 2024, 2025, 2026]
UES = [10, 15, 20, 25, 30]
RL_METHODS = ["Proposed", "Dueling DDQN (No-MSHBO)", "DDQN"]
NONRL_METHODS = ["GBW2", "MAC", "MVT", "MSH", "MSHBO"]
METHODS = RL_METHODS + NONRL_METHODS
METRICS = {
    "average_throughput": "Average Throughput (Mbps)",
    "average_handover_count": "Average Handover Count per UE",
    "average_blocking_rate": "Average Blocking Rate",
}
STYLE = {
    "Proposed": dict(color="black", marker="o", linestyle="-", markersize=9),
    "Dueling DDQN (No-MSHBO)": dict(color="#17becf", marker="v", linestyle="--", markersize=9),
    "DDQN": dict(color="#7f7f7f", marker="d", linestyle="--", markersize=9),
    "GBW2": dict(color="#2ca02c", marker="*", linestyle="-", markersize=12),
    "MAC": dict(color="#1f77b4", marker="P", linestyle="-", markersize=10),
    "MVT": dict(color="#9467bd", marker="X", linestyle="-", markersize=10),
    "MSH": dict(color="#ff7f0e", marker="s", linestyle="-", markersize=9),
    "MSHBO": dict(color="#d62728", marker="^", linestyle="-", markersize=9),
}


def load_rl() -> pd.DataFrame:
    raw = pd.read_csv(RL_SOURCE)
    raw = raw[
        raw["method"].isin(["Proposed", "Dueling DDQN", "Standard DDQN"])
        & raw["ue_count"].isin(UES)
        & raw["evaluation_order_seed"].isin(SEEDS)
        & (raw["capacity"] == 5)
    ].copy()
    keys = ["method", "ue_count", "evaluation_order_seed"]
    if raw.duplicated(keys).any():
        raise RuntimeError("Duplicate RL method/UE/seed rows")
    expected = {
        (method, ue, seed)
        for method in ("Proposed", "Dueling DDQN", "Standard DDQN")
        for ue in UES
        for seed in SEEDS
    }
    actual = set(raw[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError(f"Incomplete RL data: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    raw["method"] = raw["method"].replace(
        {"Dueling DDQN": "Dueling DDQN (No-MSHBO)", "Standard DDQN": "DDQN"}
    )

    rows = []
    for method in RL_METHODS:
        for ue in UES:
            part = raw[(raw["method"] == method) & (raw["ue_count"] == ue)]
            row = {
                "method": method,
                "method_group": "RL",
                "ue_count": ue,
                "capacity": 5,
                "sample_count": len(part),
                "evaluation_seeds": "42;123;2024;2025;2026",
                "uncertainty_type": "sample_sd_ddof_1",
                "throughput_source": str(RL_SOURCE),
                "handover_blocking_source": str(RL_SOURCE),
            }
            for metric in METRICS:
                mean = part[metric].mean()
                sd = part[metric].std(ddof=1)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_mean_minus_sd"] = mean - sd
                row[f"{metric}_mean_plus_sd"] = mean + sd
            rows.append(row)
    result = pd.DataFrame(rows)
    if not (result["sample_count"] == 5).all():
        raise RuntimeError("Every RL plotting point must contain five evaluation-order seeds")
    return result


def load_nonrl() -> pd.DataFrame:
    throughput = pd.read_csv(NONRL_THROUGHPUT_SOURCE)
    throughput = throughput[(throughput["c"] == 5) & throughput["m"].isin(UES)].copy()
    throughput = throughput.rename(
        columns={
            "algorithm": "method",
            "m": "ue_count",
            "c": "capacity",
            "avg_throughput_all_users_mbps": "average_throughput_mean",
        }
    )[["method", "ue_count", "capacity", "average_throughput_mean"]]

    hb = pd.read_csv(NONRL_HB_SOURCE)
    hb = hb[(hb["c"] == 5) & hb["m"].isin(UES)].copy()
    hb["algorithm"] = hb["algorithm"].replace({"GBW": "GBW2"})
    hb = hb.rename(
        columns={
            "algorithm": "method",
            "m": "ue_count",
            "c": "capacity",
            "Average h": "average_handover_count_mean",
            "avg_blocking_rate": "average_blocking_rate_mean",
        }
    )[["method", "ue_count", "capacity", "average_handover_count_mean", "average_blocking_rate_mean"]]

    result = throughput.merge(hb, on=["method", "ue_count", "capacity"], validate="one_to_one")
    expected = {(method, ue) for method in NONRL_METHODS for ue in UES}
    actual = set(result[["method", "ue_count"]].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError(f"Incomplete non-RL data: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for metric in METRICS:
        result[f"{metric}_sd"] = np.nan
        result[f"{metric}_mean_minus_sd"] = np.nan
        result[f"{metric}_mean_plus_sd"] = np.nan
    result["method_group"] = "Non-RL"
    result["sample_count"] = pd.NA
    result["evaluation_seeds"] = ""
    result["uncertainty_type"] = "not_available"
    result["throughput_source"] = str(NONRL_THROUGHPUT_SOURCE)
    result["handover_blocking_source"] = str(NONRL_HB_SOURCE)
    return result


def combined_data() -> pd.DataFrame:
    data = pd.concat([load_rl(), load_nonrl()], ignore_index=True, sort=False)
    order = {method: index for index, method in enumerate(METHODS)}
    data["method_order"] = data["method"].map(order)
    data = data.sort_values(["method_order", "ue_count"]).drop(columns="method_order")
    if len(data) != 40 or set(data["method"]) != set(METHODS):
        raise RuntimeError("Expected exactly eight methods × five UE settings")
    columns = [
        "method",
        "method_group",
        "ue_count",
        "capacity",
        "sample_count",
        "evaluation_seeds",
    ]
    for metric in METRICS:
        columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_mean_minus_sd",
                f"{metric}_mean_plus_sd",
            ]
        )
    columns.extend(["uncertainty_type", "throughput_source", "handover_blocking_source"])
    return data[columns]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_upper(data: pd.DataFrame, metric: str) -> float:
    candidates = data[f"{metric}_mean"].copy()
    rl = data["method_group"] == "RL"
    candidates.loc[rl] = data.loc[rl, f"{metric}_mean_plus_sd"]
    maximum = float(candidates.max())
    # Reserve in-axes space for the two-column legend above the data.
    return maximum * 1.25 if maximum > 0 else 1.0


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_rl_mean_inset(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    rl = data[data["method"].isin(RL_METHODS)].pivot(
        index="ue_count", columns="method", values=f"{metric}_mean"
    ).reindex(UES)
    positions = {
        "average_blocking_rate": [0.62, 0.69, 0.32, 0.24],
        "average_throughput": [0.62, 0.79, 0.32, 0.16],
        "average_handover_count": [0.47, 0.76, 0.30, 0.18],
    }
    detail = ax.inset_axes(positions[metric])

    for method in RL_METHODS:
        detail.plot(
            UES,
            rl[method],
            linewidth=2.0 if method == "Proposed" else 1.5,
            zorder=5 if method == "Proposed" else 3,
            **STYLE[method],
        )

    if metric in {"average_blocking_rate", "average_throughput"}:
        values = rl.loc[25]
        lower = float(values.min())
        upper = float(values.max())
        padding = max((upper - lower) * 0.25, 1e-6)
        detail.set_xlim(24.6, 25.4)
        detail.set_ylim(lower - padding, upper + padding)
        detail.set_xticks([25])
        if metric == "average_throughput":
            detail.ticklabel_format(style="plain", axis="y", useOffset=False)
            detail.yaxis.set_major_locator(MaxNLocator(nbins=5))
            detail.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        lower = float(rl.min().min())
        upper = float(rl.max().max())
        padding = (upper - lower) * 0.08
        detail.set_xlim(9.5, 30.5)
        detail.set_ylim(lower - padding, upper + padding)
        detail.set_xticks(UES)

    detail.tick_params(labelsize=7)
    detail.grid(True, linestyle=":", color="0.82", linewidth=0.6)
    detail.set_facecolor("white")
    for spine in detail.spines.values():
        spine.set_linewidth(1.1)


def plot_vs_ues(data: pd.DataFrame, metric: str, stem: str, errorbars: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for method in METHODS:
        part = data[data["method"] == method].sort_values("ue_count")
        linewidth = 2.7 if method == "Proposed" else 2.0
        zorder = 5 if method == "Proposed" else 3
        if errorbars and method in RL_METHODS:
            ax.errorbar(
                part["ue_count"],
                part[f"{metric}_mean"],
                yerr=part[f"{metric}_sd"],
                label=method,
                capsize=4,
                elinewidth=1.35,
                capthick=1.15,
                linewidth=linewidth,
                zorder=zorder,
                **STYLE[method],
            )
        else:
            ax.plot(
                part["ue_count"],
                part[f"{metric}_mean"],
                label=method,
                linewidth=linewidth,
                zorder=zorder,
                **STYLE[method],
            )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(METRICS[metric])
    ax.set_xticks(UES)
    ax.set_ylim(0, metric_upper(data, metric))
    ax.grid(True, linestyle=":", color="0.82", linewidth=0.8)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
    )
    if not errorbars:
        add_rl_mean_inset(ax, data, metric)
    fig.tight_layout()
    save(fig, stem)


def annotate_nonrl(ax: plt.Axes, part: pd.DataFrame, method: str) -> None:
    offsets = {
        "GBW2": (5, 4),
        "MAC": (5, 0),
        "MVT": (-18, 0),
        "MSH": (5, 2),
        "MSHBO": (5, 3),
    }
    for _, row in part.iterrows():
        ax.annotate(
            str(int(row["ue_count"])),
            (row["average_handover_count_mean"], row["average_blocking_rate_mean"]),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=6.5,
            color=STYLE[method]["color"],
        )


def plot_tradeoff(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    handles = []
    labels = []
    for method in METHODS:
        part = data[data["method"] == method].sort_values("ue_count")
        linewidth = 2.7 if method == "Proposed" else 2.0
        line = ax.plot(
            part["average_handover_count_mean"],
            part["average_blocking_rate_mean"],
            label=method,
            linewidth=linewidth,
            zorder=5 if method == "Proposed" else 3,
            **STYLE[method],
        )[0]
        handles.append(line)
        labels.append(method)
        if method in NONRL_METHODS:
            annotate_nonrl(ax, part, method)

    ax.set_xlabel("Average Handover Count")
    ax.set_ylabel("Average Blocking Rate")
    ax.set_xlim(left=0)
    ax.set_ylim(0, data["average_blocking_rate_mean"].max() * 1.25)
    ax.grid(True, linestyle=":", color="0.82", linewidth=0.8)

    ax.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=1.0,
    )
    fig.tight_layout()
    save(fig, "tradeoff_mean")


def write_summary(data: pd.DataFrame) -> None:
    figure_stems = [
        "blocking_rate_with_errorbar",
        "handover_count_with_errorbar",
        "throughput_with_errorbar",
        "blocking_rate_mean",
        "handover_count_mean",
        "throughput_mean",
        "tradeoff_mean",
    ]
    lines = [
        "# Paper comparison figures",
        "",
        "## Data sources",
        "",
        f"- RL raw multi-seed results: `{RL_SOURCE}`",
        f"- Non-RL corrected throughput: `{NONRL_THROUGHPUT_SOURCE}`",
        f"- Non-RL handover/blocking: `{NONRL_HB_SOURCE}`",
        "",
        "## Methods and samples",
        "",
        "Methods: Proposed; Dueling DDQN (No-MSHBO); DDQN; GBW2; MAC; MVT; MSH; MSHBO.",
        "The source name `Standard DDQN` is displayed as `DDQN` in every figure.",
        "Proposed is the original lambda_handover=3 model; Proposed + Expert Decay is excluded.",
        "RL values use five evaluation-order seeds: 42, 123, 2024, 2025, 2026. These are not five independent training runs.",
        "RL error bars are ±1 sample SD (ddof=1). Non-RL SD is unavailable, remains blank in plot_data.csv, and is not plotted or assumed to be zero.",
        "The blocking, handover, and throughput mean figures contain in-axes zooms of the original RL metric values; the trade-off figure is a single clean plot without an inset.",
        "No required method/UE/seed data is missing.",
        "",
        "## Metrics",
        "",
        "- Throughput: Average Throughput (Mbps).",
        "- Handover: Average Handover Count per UE.",
        "- Blocking: Average Blocked UEs per Time Slot. The historical field name `average_blocking_rate` and axis label `Average Blocking Rate` are retained, but the value is not a 0–1 probability or percentage.",
        "- Trade-off axes: X = Average Handover Count; Y = Average Blocking Rate. Point labels are UE counts.",
        "",
        "## Outputs",
        "",
        *[f"- `{OUT / (stem + '.png')}` and `{OUT / (stem + '.pdf')}`" for stem in figure_stems],
        f"- `{OUT / 'plot_data.csv'}` ({len(data)} rows)",
    ]
    (OUT / "plot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = combined_data()
    data.to_csv(OUT / "plot_data.csv", index=False, float_format="%.12g")
    configure_plotting()

    plot_vs_ues(data, "average_blocking_rate", "blocking_rate_with_errorbar", True)
    plot_vs_ues(data, "average_handover_count", "handover_count_with_errorbar", True)
    plot_vs_ues(data, "average_throughput", "throughput_with_errorbar", True)
    plot_vs_ues(data, "average_blocking_rate", "blocking_rate_mean", False)
    plot_vs_ues(data, "average_handover_count", "handover_count_mean", False)
    plot_vs_ues(data, "average_throughput", "throughput_mean", False)
    plot_tradeoff(data)
    write_summary(data)

    kinds = {
        "blocking_rate_with_errorbar": "error bar",
        "handover_count_with_errorbar": "error bar",
        "throughput_with_errorbar": "error bar",
        "blocking_rate_mean": "mean + inset zoom",
        "handover_count_mean": "mean + inset zoom",
        "throughput_mean": "mean + inset zoom",
        "tradeoff_mean": "mean",
    }
    print("Generated paper figures:")
    for stem, kind in kinds.items():
        print(f"- {stem}.png / {stem}.pdf | {kind}")


if __name__ == "__main__":
    main()
