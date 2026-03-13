#!/usr/bin/env python3
"""
Plot end-to-end benchmark metrics from a controlled benchmark summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(summary_csv: str):
    with open(summary_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return float(value)


def short_label(config: str) -> str:
    mapping = {
        "dense": "Dense",
        "sparse_50": "Sparse50",
        "sparse_70": "Sparse70",
        "sparse_50_quant_2bit": "Sparse50+2bit",
        "sparse_70_quant_2bit": "Sparse70+2bit",
    }
    return mapping.get(config, config)


def plot(summary_csv: str, out_dir: str):
    rows = load_rows(summary_csv)
    labels = [short_label(row["config"]) for row in rows]
    ttft = [as_float(row["ttft_ms"]) for row in rows]
    tpot = [as_float(row["tpot_ms"]) for row in rows]
    throughput = [as_float(row["throughput_tps"]) for row in rows]

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    panels = [
        ("TTFT (ms)", ttft),
        ("TPOT (ms)", tpot),
        ("Throughput (tok/s)", throughput),
    ]

    for ax, (title, values) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, width=0.72)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("End-to-End Throughput Benchmark", fontsize=14)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "end_to_end_metrics.png")
    pdf_path = os.path.join(out_dir, "end_to_end_metrics.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot end-to-end benchmark metrics")
    parser.add_argument("summary_csv", help="Path to benchmark summary.csv")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("benchmark", "doc", "figures"),
        help="Output directory for figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    png_path, pdf_path = plot(args.summary_csv, args.out_dir)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
