#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "sparse_50",
    "sparse_70",
    "sparse_50_quant_2bit",
    "sparse_70_quant_2bit",
]

CONFIG_LABELS = {
    "sparse_50": "Sparse50",
    "sparse_70": "Sparse70",
    "sparse_50_quant_2bit": "Sparse50+2bit",
    "sparse_70_quant_2bit": "Sparse70+2bit",
}

CONFIG_STYLES = {
    "sparse_50": {"color": "#F58518", "marker": "s", "linestyle": "-"},
    "sparse_70": {"color": "#54A24B", "marker": "D", "linestyle": "-"},
    "sparse_50_quant_2bit": {"color": "#E45756", "marker": "o", "linestyle": "--"},
    "sparse_70_quant_2bit": {"color": "#72B7B2", "marker": "P", "linestyle": "--"},
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


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(rows, out_dir: str):
    configure_chinese_font()
    by_config = {cfg: [] for cfg in CONFIG_ORDER}
    for row in rows:
        cfg = row["config"]
        if cfg not in by_config:
            continue
        by_config[cfg].append(
            {
                "batch_size": int(row["batch_size"]),
                "compression_ratio": float(row["compression_ratio"]),
                "memory_saving_ratio": float(
                    row.get("memory_saving_ratio", "") or (float(row["memory_saving_pct"]) / 100.0)
                ),
                "compression_time_ms": float(row["compression_time_ms"]),
            }
        )
    for cfg in by_config:
        by_config[cfg].sort(key=lambda r: r["batch_size"])

    os.makedirs(out_dir, exist_ok=True)

    # compression time figure
    plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()
    for cfg in CONFIG_ORDER:
        rows = by_config[cfg]
        if not rows:
            continue
        xs = [r["batch_size"] for r in rows]
        ys = [r["compression_time_ms"] for r in rows]
        style = CONFIG_STYLES[cfg]
        ax.plot(xs, ys, label=CONFIG_LABELS[cfg], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.4, markersize=7)
    ax.set_xlabel("批大小")
    ax.set_ylabel("压缩时间（ms）")
    ax.set_title("不同批大小下的压缩时间")
    ax.set_xticks(sorted({r['batch_size'] for rows in by_config.values() for r in rows}))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compression_time_vs_bs.png"), dpi=240, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "compression_time_vs_bs.pdf"), dpi=240, bbox_inches="tight")
    plt.close()

    # compression ratio figure
    plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()
    for cfg in CONFIG_ORDER:
        rows = by_config[cfg]
        if not rows:
            continue
        xs = [r["batch_size"] for r in rows]
        ys = [r["compression_ratio"] for r in rows]
        style = CONFIG_STYLES[cfg]
        ax.plot(xs, ys, label=CONFIG_LABELS[cfg], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.4, markersize=7)
    ax.set_xlabel("批大小")
    ax.set_ylabel("压缩率（原始KV / 压缩后KV）")
    ax.set_title("不同批大小下的 KV Cache 压缩率")
    ax.set_xticks(sorted({r['batch_size'] for rows in by_config.values() for r in rows}))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compression_ratio_vs_bs.png"), dpi=240, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "compression_ratio_vs_bs.pdf"), dpi=240, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot compression statistics")
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv_path))
    plot(rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
