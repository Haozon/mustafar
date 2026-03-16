#!/usr/bin/env python3
"""
Plot a single-model five-line batch-size throughput chart.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "dense",
    "sparse_50",
    "sparse_70",
    "sparse_50_quant_2bit",
    "sparse_70_quant_2bit",
]

CONFIG_LABELS = {
    "dense": "Dense",
    "sparse_50": "Sparse50",
    "sparse_70": "Sparse70",
    "sparse_50_quant_2bit": "Sparse50+2bit",
    "sparse_70_quant_2bit": "Sparse70+2bit",
}

CONFIG_STYLES = {
    "dense": {"color": "#4C78A8", "marker": "^", "linestyle": "--"},
    "sparse_50": {"color": "#F58518", "marker": "s", "linestyle": "-"},
    "sparse_70": {"color": "#54A24B", "marker": "D", "linestyle": "-"},
    "sparse_50_quant_2bit": {"color": "#E45756", "marker": "o", "linestyle": "--"},
    "sparse_70_quant_2bit": {"color": "#72B7B2", "marker": "P", "linestyle": "--"},
}


def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot(data: dict, out_dir: str, label: str, out_name: str):
    plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()

    all_bs = set()
    for cfg in CONFIG_ORDER:
        if cfg not in data:
            continue
        valid_bs = sorted(int(k) for k, v in data[cfg].items() if isinstance(v, dict) and "throughput" in v)
        if not valid_bs:
            continue
        all_bs.update(valid_bs)
        ys = [data[cfg][str(bs)]["throughput"] for bs in valid_bs]
        style = CONFIG_STYLES[cfg]
        ax.plot(
            valid_bs,
            ys,
            label=CONFIG_LABELS[cfg],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            markersize=7,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(label)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    if all_bs:
        ax.set_xticks(sorted(all_bs))
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, out_name + ".png")
    pdf_path = os.path.join(out_dir, out_name + ".pdf")
    plt.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot five-line batch-size throughput chart")
    parser.add_argument("results_json")
    parser.add_argument("--label", default="Llama-3 8B Throughput vs Batch Size")
    parser.add_argument("--out-dir", default="benchmark/output")
    parser.add_argument("--out-name", default="five_line_llama3_bs")
    args = parser.parse_args()

    data = load_results(args.results_json)
    png_path, pdf_path = plot(data, args.out_dir, args.label, args.out_name)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
