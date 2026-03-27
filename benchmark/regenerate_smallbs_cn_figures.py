#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_STYLES = {
    "dense": {"color": "#4C78A8", "marker": "^", "linestyle": "--"},
    "sparse_50": {"color": "#F58518", "marker": "s", "linestyle": "-"},
    "sparse_70": {"color": "#54A24B", "marker": "D", "linestyle": "-"},
    "sparse_50_quant_2bit": {"color": "#E45756", "marker": "o", "linestyle": "--"},
    "sparse_70_quant_2bit": {"color": "#72B7B2", "marker": "P", "linestyle": "--"},
}

CONFIG_LABELS = {
    "dense": "Dense",
    "sparse_50": "Sparse50",
    "sparse_70": "Sparse70",
    "sparse_50_quant_2bit": "Sparse50+2bit",
    "sparse_70_quant_2bit": "Sparse70+2bit",
}


def configure_chinese_font():
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = None
    for name in candidates:
        if name in available:
            selected = name
            break
    if selected:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected] + candidates
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot(data: dict, configs: list[str], title: str, stem: str, out_dir: str):
    configure_chinese_font()
    plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()
    xs = list(range(1, 9))
    for cfg in configs:
        ys = [data[cfg][str(bs)]["throughput"] for bs in xs]
        style = CONFIG_STYLES[cfg]
        ax.plot(
            xs,
            ys,
            label=CONFIG_LABELS[cfg],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("批大小")
    ax.set_ylabel("吞吐量（tokens/s）")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, stem + ".png"), dpi=240, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, stem + ".pdf"), dpi=240, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Regenerate Chinese paper-ready throughput figures")
    parser.add_argument("results_json")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_json))
    os.makedirs(out_dir, exist_ok=True)

    plot(
        data,
        ["dense", "sparse_50", "sparse_70", "sparse_50_quant_2bit", "sparse_70_quant_2bit"],
        "不同批大小下的端到端吞吐量",
        "throughput_vs_bs_smallbs_quant_opt",
        out_dir,
    )
    plot(
        data,
        ["dense", "sparse_50", "sparse_50_quant_2bit"],
        "Dense、Sparse50 与 Sparse50+2bit 吞吐量对比",
        "throughput_vs_bs_dense_sparse50_quant50",
        out_dir,
    )
    plot(
        data,
        ["dense", "sparse_70", "sparse_70_quant_2bit"],
        "Dense、Sparse70 与 Sparse70+2bit 吞吐量对比",
        "throughput_vs_bs_dense_sparse70_quant70",
        out_dir,
    )


if __name__ == "__main__":
    main()
