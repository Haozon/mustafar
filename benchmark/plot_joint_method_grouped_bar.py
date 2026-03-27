#!/usr/bin/env python3
"""
Plot a grouped bar chart for the joint AIDCS + JSQKV throughput comparison.

Bars show absolute throughput, while annotations above non-Dense bars show
relative throughput change vs Dense at the same batch size.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BATCH_SIZES = [1, 2, 3, 4]

SERIES: Dict[str, List[float]] = {
    "Dense": [18.31, 23.96, 29.56, 32.67],
    "AIDCS-only": [23.72, 25.41, 29.98, 32.81],
    "JSQKV-only": [14.95, 27.06, 37.77, 47.12],
    "AIDCS + JSQKV": [19.08, 27.92, 38.11, 47.34],
}

COLORS = {
    "Dense": "#9EA3A8",
    "AIDCS-only": "#F28E2B",
    "JSQKV-only": "#4E79A7",
    "AIDCS + JSQKV": "#E15759",
}


def percent_delta(value: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else (value - baseline) / baseline * 100.0


def format_delta(delta_pct: float) -> str:
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.1f}%"


def plot(out_dir: str, out_name: str) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(12.0, 6.6))

    width = 0.18
    x_positions = list(range(len(BATCH_SIZES)))
    offsets = {
        "Dense": -1.5 * width,
        "AIDCS-only": -0.5 * width,
        "JSQKV-only": 0.5 * width,
        "AIDCS + JSQKV": 1.5 * width,
    }

    max_y = max(max(values) for values in SERIES.values())
    baseline = SERIES["Dense"]

    for label, values in SERIES.items():
        xs = [x + offsets[label] for x in x_positions]
        is_joint = label == "AIDCS + JSQKV"

        bars = ax.bar(
            xs,
            values,
            width=width,
            label=label,
            color=COLORS[label],
            edgecolor="#2F2F2F" if is_joint else "white",
            linewidth=1.4 if is_joint else 0.8,
            hatch="//" if is_joint else None,
            zorder=3,
        )

        if label == "Dense":
            continue

        for idx, bar in enumerate(bars):
            delta_pct = percent_delta(values[idx], baseline[idx])
            text_color = "#2E8B57" if delta_pct >= 0 else "#C44E52"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_y * 0.015,
                format_delta(delta_pct),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=text_color,
                fontweight="bold" if is_joint else "normal",
                zorder=6,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(bs) for bs in BATCH_SIZES], fontsize=11)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_ylim(0, max_y * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=10,
    )
    ax.text(
        0.99,
        0.965,
        "Annotations: % change vs Dense",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4A4A4A",
        ha="right",
        va="top",
    )

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{out_name}.png")
    pdf_path = os.path.join(out_dir, f"{out_name}.pdf")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot grouped throughput bar chart for joint method")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("benchmark", "doc", "figures"),
        help="Directory to save the generated figure",
    )
    parser.add_argument(
        "--out-name",
        default="joint_method_throughput_grouped_bar",
        help="Base filename without extension",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    png_path, pdf_path = plot(args.out_dir, args.out_name)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
