#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "dense",
    "sparse50",
    "sparse70",
    "sparse50_quant2bit",
    "sparse70_quant2bit",
]

CONFIG_LABELS = {
    "dense": "Dense",
    "sparse50": "Sparse50",
    "sparse70": "Sparse70",
    "sparse50_quant2bit": "Sparse50+2bit",
    "sparse70_quant2bit": "Sparse70+2bit",
}

CONFIG_STYLES = {
    "dense": {"color": "#4C78A8", "marker": "^", "linestyle": "--"},
    "sparse50": {"color": "#F58518", "marker": "s", "linestyle": "-"},
    "sparse70": {"color": "#54A24B", "marker": "D", "linestyle": "-"},
    "sparse50_quant2bit": {"color": "#E45756", "marker": "o", "linestyle": "--"},
    "sparse70_quant2bit": {"color": "#72B7B2", "marker": "P", "linestyle": "--"},
}

SUMMARY_FILES = {
    "dense": "dense_summary.csv",
    "sparse50": "sparse50_summary.csv",
    "sparse70": "sparse70_summary.csv",
    "sparse50_quant2bit": "quant50_summary.csv",
    "sparse70_quant2bit": "quant70_summary.csv",
}

METRIC_META = {
    "throughput_tps": {
        "ylabel": "Throughput (tokens/s)",
        "title": "Meta-Llama-3-8B-Instruct | input=4096, output=256 | Throughput vs Batch Size",
        "note": "Hollow markers: throughput available, TTFT/TPOT incomplete",
    },
    "ttft_ms": {
        "ylabel": "TTFT (ms)",
        "title": "Meta-Llama-3-8B-Instruct | input=4096, output=256 | TTFT vs Batch Size",
        "note": "Missing markers indicate unavailable TTFT under single-card memory limits",
    },
    "tpot_ms": {
        "ylabel": "TPOT (ms/token)",
        "title": "Meta-Llama-3-8B-Instruct | input=4096, output=256 | TPOT vs Batch Size",
        "note": "Missing markers indicate unavailable TPOT under single-card memory limits",
    },
}


def load_summary(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_records(summary_dir: str, metric_key: str) -> Dict[str, List[dict]]:
    records: Dict[str, List[dict]] = {}
    for cfg, filename in SUMMARY_FILES.items():
        path = os.path.join(summary_dir, filename)
        if not os.path.exists(path):
            continue
        rows = []
        for row in load_summary(path):
            raw = row.get(metric_key, "")
            if not raw:
                continue
            rows.append(
                {
                    "batch_size": int(row["batch_size"]),
                    "value": float(raw),
                    "status": row.get("status", ""),
                }
            )
        rows.sort(key=lambda r: r["batch_size"])
        records[cfg] = rows
    return records


def plot(records: Dict[str, List[dict]], out_png: str, out_pdf: str, title: str, ylabel: str, note: str):
    plt.figure(figsize=(9.2, 5.8))
    ax = plt.gca()

    all_bs = sorted(
        {
            row["batch_size"]
            for rows in records.values()
            for row in rows
        }
    )

    for cfg in CONFIG_ORDER:
        rows = records.get(cfg, [])
        if not rows:
            continue
        style = CONFIG_STYLES[cfg]
        xs = [row["batch_size"] for row in rows]
        ys = [row["value"] for row in rows]

        ax.plot(
            xs,
            ys,
            label=CONFIG_LABELS[cfg],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=7,
        )

        failed_rows = [row for row in rows if row["status"] != "ok"]
        if failed_rows:
            ax.scatter(
                [row["batch_size"] for row in failed_rows],
                [row["value"] for row in failed_rows],
                s=85,
                facecolors="none",
                edgecolors=style["color"],
                linewidths=1.8,
                zorder=5,
            )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    if all_bs:
        ax.set_xticks(all_bs)
    ax.legend()
    ax.text(
        0.99,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=240, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot end-to-end metrics vs BS from summary CSV files.")
    parser.add_argument("summary_dir", help="Directory containing five *_summary.csv files")
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_META.keys()),
        default="throughput_tps",
        help="Metric to plot",
    )
    parser.add_argument("--title", default="", help="Override default title")
    parser.add_argument("--out-name", default="", help="Override output filename stem")
    args = parser.parse_args()

    meta = METRIC_META[args.metric]
    title = args.title or meta["title"]
    out_name = args.out_name or args.metric.replace("_", "") + "_vs_bs_five_configs"

    records = build_records(args.summary_dir, args.metric)
    out_png = os.path.join(args.summary_dir, out_name + ".png")
    out_pdf = os.path.join(args.summary_dir, out_name + ".pdf")
    plot(records, out_png, out_pdf, title, meta["ylabel"], meta["note"])
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
