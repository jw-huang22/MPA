#!/usr/bin/env python3
"""Plot ADP vs model-stealing accuracy for four BERT datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/ram/tmp/matplotlib-cache")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR
DEFAULT_OUTPUT = SCRIPT_DIR / "adp_vs_model_stealing_accuracy_four_datasets"
DEFAULT_DATASETS = ("mnli", "qnli", "qqp", "sst2")
DATASET_LABELS = {
    "mnli": "MNLI",
    "qnli": "QNLI",
    "qqp": "QQP",
    "sst2": "SST-2",
}


def load_points(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    points = [
        (float(row["constructed_adp"]), float(row["mean_metric"]))
        for row in summary.get("adp_curve", {}).get("points", [])
    ]
    if not points:
        raise ValueError(f"No adp_curve.points data found in {path}")
    return sorted(points)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(
    series: dict[str, list[tuple[float, float]]],
    output_stem: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    configure_matplotlib()

    colors = {
        "mnli": "#0072B2",
        "qnli": "#D55E00",
        "qqp": "#009E73",
        "sst2": "#CC79A7",
    }
    markers = {
        "mnli": "o",
        "qnli": "s",
        "qqp": "^",
        "sst2": "D",
    }
    linestyles = {
        "mnli": "-",
        "qnli": "--",
        "qqp": "-.",
        "sst2": ":",
    }

    fig, ax = plt.subplots(figsize=(1.55, 1.18), constrained_layout=True)
    ax.set_facecolor("white")

    all_xs = []
    all_ys = []
    for dataset, points in series.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        all_xs.extend(xs)
        all_ys.extend(ys)
        ax.plot(
            xs,
            ys,
            color=colors.get(dataset, "#111827"),
            marker=markers.get(dataset, "o"),
            linestyle=linestyles.get(dataset, "-"),
            markersize=2.25,
            markerfacecolor="white",
            markeredgewidth=0.65,
            markevery=2,
            linewidth=1.25,
            label=DATASET_LABELS.get(dataset, dataset.upper()),
            zorder=3,
        )

    x_padding = 0.02 * (max(all_xs) - min(all_xs))
    ax.set_xlim(min(all_xs) - x_padding, max(all_xs) + x_padding)
    ax.set_ylim(max(0.0, min(all_ys) - 0.035), min(1.0, max(all_ys) + 0.025))
    ax.set_xlabel("ADP")
    ax.set_ylabel("Acc.")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.35, linestyle=(0, (1.0, 2.0)), alpha=0.6)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.3, linestyle=(0, (1.0, 2.2)), alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.tick_params(axis="both", which="major", length=2.4, color="#303030", pad=1.5)
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        handlelength=1.25,
        handletextpad=0.25,
        columnspacing=0.45,
        borderaxespad=0.12,
        labelspacing=0.15,
    )

    written = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        path = output_stem.with_suffix(suffix)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
        if suffix == ".png":
            save_kwargs["dpi"] = 400
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = {
        dataset: load_points(args.results_dir / dataset / "weight_distance_summary.json")
        for dataset in args.datasets
    }
    written = plot(series, args.output)
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
