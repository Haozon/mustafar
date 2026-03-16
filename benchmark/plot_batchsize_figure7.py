#!/usr/bin/env python3
"""
Plot a Figure-7-style batch-size throughput chart for one model.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot(model_results: dict, out_dir: str, model_label: str):
    configs = ["dense", "sparse_70", "sparse_70_quant_2bit"]
    labels = {
        "dense": "Dense",
        "sparse_70": "Sparse70",
        "sparse_70_quant_2bit": "Sparse70+2bit",
    }
    styles = {
        "dense": {"color": "#4C78A8", "marker": "^", "linestyle": "--"},
        "sparse_70": {"color": "#54A24B", "marker": "s", "linestyle": "-"},
        "sparse_70_quant_2bit": {"color": "#E45756", "marker": "o", "linestyle": "--"},
    }

    plt.figure(figsize=(8.2, 5.2))
    ax = plt.gca()

    for cfg in configs:
        cfg_data = model_results[cfg]
        bs_list = sorted(
            int(k) for k, v in cfg_data.items() if isinstance(v, dict) and "throughput" in v
        )
        throughput = [cfg_data[str(bs)]["throughput"] for bs in bs_list]
        style = styles[cfg]
        ax.plot(
            bs_list,
            throughput,
            label=labels[cfg],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            markersize=7,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"{model_label}: End-to-End Throughput vs Batch Size")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_xticks([1, 2, 4, 6, 8])
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "figure7_like_llama3_bs.png")
    pdf_path = os.path.join(out_dir, "figure7_like_llama3_bs.pdf")
    plt.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot Figure-7-style batch throughput chart")
    parser.add_argument("results_json")
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--label", default="Llama-3 8B")
    parser.add_argument("--out-dir", default="benchmark/output")
    args = parser.parse_args()

    data = load_results(args.results_json)
    png_path, pdf_path = plot(data, args.out_dir, args.label)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
