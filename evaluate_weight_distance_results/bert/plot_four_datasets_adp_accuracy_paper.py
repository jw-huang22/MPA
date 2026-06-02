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
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
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
        "mnli": "#2563EB",
        "qnli": "#D97706",
        "qqp": "#059669",
        "sst2": "#DC2626",
    }
    markers = {
        "mnli": "o",
        "qnli": "s",
        "qqp": "^",
        "sst2": "D",
    }

    fig, ax = plt.subplots(figsize=(3.55, 2.35), constrained_layout=True)
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
            markersize=3.3,
            markerfacecolor="white",
            markeredgewidth=0.9,
            linewidth=1.35,
            label=DATASET_LABELS.get(dataset, dataset.upper()),
            zorder=3,
        )

    x_padding = 0.02 * (max(all_xs) - min(all_xs))
    ax.set_xlim(min(all_xs) - x_padding, max(all_xs) + x_padding)
    ax.set_ylim(max(0.0, min(all_ys) - 0.035), min(1.0, max(all_ys) + 0.025))
    ax.set_xlabel("ADP")
    ax.set_ylabel("model-stealing accuracy")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, linestyle=(0, (2.2, 2.2)))
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45, linestyle=(0, (2.2, 2.2)))
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="both", which="major", length=3.0, color="#222222")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.2)

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
