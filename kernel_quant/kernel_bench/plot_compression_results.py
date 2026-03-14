#!/usr/bin/env python3
"""
从 benchmark 输出 JSON 生成 Compression 性能对比图表。
默认读取 output/ 下最新的 compression_detailed_*.json。
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot compression benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="compression_detailed_*.json 路径；不传则自动选最新文件",
    )
    parser.add_argument(
        "--report-stat",
        choices=["avg", "median", "trimmed_mean"],
        default=None,
        help="绘图时间口径；不传则优先使用 JSON 中的 report_stat",
    )
    return parser.parse_args()


def select_latest_json(output_dir):
    candidates = sorted(glob.glob(os.path.join(output_dir, "compression_detailed_*.json")))
    if not candidates:
        raise FileNotFoundError(f"未找到 benchmark JSON: {output_dir}/compression_detailed_*.json")
    return candidates[-1]


def time_value(time_dict, stat_key):
    if stat_key in time_dict:
        return float(time_dict[stat_key])
    if "median" in time_dict:
        return float(time_dict["median"])
    return float(time_dict["avg"])


def load_plot_data(json_path, report_stat_arg):
    with open(json_path, "r") as f:
        raw = json.load(f)

    report_stat = report_stat_arg or raw.get("report_stat", "median")

    seq_len_by_name = {}
    if "test_configs" in raw:
        for cfg in raw["test_configs"]:
            seq_len_by_name[cfg["name"]] = int(cfg["seq_len"])

    data = {}
    for item in raw.get("results", []):
        if not item.get("sparse") or not item.get("quant"):
            continue

        cfg = item["config"]
        name = cfg["name"]
        seq_len = int(cfg["seq_len"])
        seq_len_by_name[name] = seq_len
        sparsity = int(round(float(item["sparsity"]) * 100))

        sparse_t = time_value(item["sparse"]["time"], report_stat)
        quant_t = time_value(item["quant"]["time"], report_stat)
        speedup = sparse_t / quant_t if quant_t > 0 else 0.0

        if name not in data:
            data[name] = {
                "seq_len": seq_len,
                "original_mem": float(item["original_size_mb"]),
                "sparsity": {},
            }

        data[name]["sparsity"][sparsity] = {
            "fp16": sparse_t,
            "quant": quant_t,
            "speedup": speedup,
            "fp16_mem": float(item["sparse"]["memory_mb"]),
            "quant_mem": float(item["quant"]["memory_mb"]),
        }

    if not data:
        raise ValueError(f"JSON 中没有可绘图结果: {json_path}")

    scales = sorted(data.keys(), key=lambda x: data[x]["seq_len"])
    sparsities = sorted(
        {s for scale in data.values() for s in scale["sparsity"].keys()}
    )
    return raw, data, scales, sparsities, report_stat


def ensure_sparsity_exists(data, scales, preferred):
    if preferred in {s for scale in data.values() for s in scale["sparsity"]}:
        return preferred
    for s in [50, 30, 70]:
        if s in {sp for scale in data.values() for sp in scale["sparsity"]}:
            return s
    # fallback: 第一个可用值
    for scale in scales:
        if data[scale]["sparsity"]:
            return sorted(data[scale]["sparsity"].keys())[0]
    raise ValueError("未找到可用稀疏度数据")


def set_plot_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14


def plot_speedup_vs_scale(data, scales, sparsities, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    seq_lens = [data[s]["seq_len"] for s in scales]

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#6A4C93", "#1B9E77"]
    markers = ["o", "s", "^", "D", "P"]

    for i, sparsity in enumerate(sparsities):
        # 只画每个配置都具备的数据点
        valid_scales = [s for s in scales if sparsity in data[s]["sparsity"]]
        if not valid_scales:
            continue
        x_vals = [data[s]["seq_len"] for s in valid_scales]
        y_vals = [data[s]["sparsity"][sparsity]["speedup"] for s in valid_scales]
        ax.plot(
            x_vals,
            y_vals,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2.5,
            markersize=8,
            label=f"{sparsity}% Sparsity",
        )

    ax.set_xlabel("Sequence Length", fontweight="bold")
    ax.set_ylabel("Speedup (x)", fontweight="bold")
    ax.set_title("Compression Speedup vs. Sequence Length", fontweight="bold")
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([f"{sl}\n({name})" for name, sl in zip(scales, seq_lens)])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_scale.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "speedup_vs_scale.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_comparison(data, scales, output_dir, report_stat):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scales))
    width = 0.35

    sparsity = ensure_sparsity_exists(data, scales, 50)
    fp16_times = [data[s]["sparsity"][sparsity]["fp16"] for s in scales]
    quant_times = [data[s]["sparsity"][sparsity]["quant"] for s in scales]

    bars1 = ax.bar(
        x - width / 2,
        fp16_times,
        width,
        label="FP16 Compression",
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        quant_times,
        width,
        label="Quantized Compression",
        color="#F18F01",
        edgecolor="black",
        linewidth=0.5,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_ylabel("Compression Time (ms)", fontweight="bold")
    ax.set_title(
        f"Compression Time Comparison ({sparsity}% Sparsity, {report_stat})",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n({data[name]['seq_len']} seq)" for name in scales])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_memory_comparison(data, scales, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scales))
    width = 0.25

    sparsity = ensure_sparsity_exists(data, scales, 50)
    original_mem = [data[s]["original_mem"] for s in scales]
    fp16_mem = [data[s]["sparsity"][sparsity]["fp16_mem"] for s in scales]
    quant_mem = [data[s]["sparsity"][sparsity]["quant_mem"] for s in scales]

    bars1 = ax.bar(
        x - width,
        original_mem,
        width,
        label="Original (FP16)",
        color="#CCCCCC",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        fp16_mem,
        width,
        label="Sparse (FP16)",
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        quant_mem,
        width,
        label="Sparse + Quant (2-bit)",
        color="#F18F01",
        edgecolor="black",
        linewidth=0.5,
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.1f}" if height > 1 else f"{height:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_ylabel("Memory Footprint (MB)", fontweight="bold")
    ax.set_title(f"Memory Footprint Comparison ({sparsity}% Sparsity)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n({data[name]['seq_len']} seq)" for name in scales])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_comprehensive(data, scales, sparsities, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(scales))
    width = 0.35
    seq_lens = [data[s]["seq_len"] for s in scales]

    sparsity = ensure_sparsity_exists(data, scales, 50)
    fp16_times = [data[s]["sparsity"][sparsity]["fp16"] for s in scales]
    quant_times = [data[s]["sparsity"][sparsity]["quant"] for s in scales]

    ax1.bar(x - width / 2, fp16_times, width, label="FP16", color="#2E86AB")
    ax1.bar(x + width / 2, quant_times, width, label="Quant", color="#F18F01")
    ax1.set_xlabel("Configuration", fontweight="bold")
    ax1.set_ylabel("Time (ms)", fontweight="bold")
    ax1.set_title("(a) Compression Time", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#6A4C93", "#1B9E77"]
    for i, sp in enumerate(sparsities):
        valid_scales = [s for s in scales if sp in data[s]["sparsity"]]
        if not valid_scales:
            continue
        x_vals = [data[s]["seq_len"] for s in valid_scales]
        y_vals = [data[s]["sparsity"][sp]["speedup"] for s in valid_scales]
        ax2.plot(
            x_vals,
            y_vals,
            marker="o",
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{sp}%",
        )

    ax2.set_xlabel("Sequence Length", fontweight="bold")
    ax2.set_ylabel("Speedup (x)", fontweight="bold")
    ax2.set_title("(b) Speedup vs. Scale", fontweight="bold")
    ax2.set_xticks(seq_lens)
    ax2.legend(title="Sparsity")
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "comprehensive.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    bench_output_dir = os.path.join(script_dir, "output")

    json_path = args.input if args.input else select_latest_json(bench_output_dir)
    raw, data, scales, sparsities, report_stat = load_plot_data(json_path, args.report_stat)

    set_plot_style()
    plot_speedup_vs_scale(data, scales, sparsities, output_dir)
    plot_time_comparison(data, scales, output_dir, report_stat)
    plot_memory_comparison(data, scales, output_dir)
    plot_comprehensive(data, scales, sparsities, output_dir)

    print("=" * 70)
    print("Compression 图表生成完成")
    print("=" * 70)
    print(f"输入数据: {json_path}")
    print(f"统计口径: {report_stat}")
    if "timestamp" in raw:
        print(f"数据时间: {raw['timestamp']}")
    print(f"输出目录: {output_dir}")
    print("生成文件:")
    print("  - speedup_vs_scale.pdf/png")
    print("  - time_comparison.pdf/png")
    print("  - memory_comparison.pdf/png")
    print("  - comprehensive.pdf/png")


if __name__ == "__main__":
    main()
