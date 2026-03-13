#!/usr/bin/env python3
"""
Draw 5-line BS curves from JSQKV benchmark raw JSON files.

Lines:
  - dense
  - sparse_50
  - sparse_70
  - sparse_50_quant_2bit
  - sparse_70_quant_2bit

Each workload scenario (model + input_length + output_length) outputs one plot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple

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
    "dense": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "sparse_50": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "sparse_70": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "sparse_50_quant_2bit": {"color": "#d62728", "marker": "D", "linestyle": "--"},
    "sparse_70_quant_2bit": {"color": "#9467bd", "marker": "P", "linestyle": "--"},
}


def parse_model_and_timestamp(filename: str) -> Tuple[str, int]:
    base = os.path.basename(filename)
    m = re.match(r"(.+)_results(?:_(\d{8})_(\d{6}))?\.json$", base)
    if not m:
        return os.path.splitext(base)[0], 0

    model = m.group(1)
    if m.group(2) and m.group(3):
        return model, int(m.group(2) + m.group(3))
    return model, 0


def _last_float(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _detect_model_from_text(text: str) -> str:
    if "Llama-2-7b" in text or "llama2_7b" in text:
        return "llama2_7b"
    if "Meta-Llama-3-8B" in text or "llama3_8b" in text:
        return "llama3_8b"
    return "llama3_8b"


def _upsert_record(
    store,
    model: str,
    input_len: int,
    output_len: int,
    cfg: str,
    bs: int,
    value: float,
    ts: int,
):
    scenario = (model, input_len, output_len)
    old = store[scenario][cfg].get(bs)
    if old is None or ts >= old[0]:
        store[scenario][cfg][bs] = (ts, value)


def _extract_from_json_results(store, results_dir: str, metric: str):
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    for path in files:
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        model, ts = parse_model_and_timestamp(path)
        for cfg in CONFIG_ORDER:
            cfg_dict = data.get(cfg, {})
            if not isinstance(cfg_dict, dict):
                continue
            for bs_str, rec in cfg_dict.items():
                if not isinstance(rec, dict) or metric not in rec:
                    continue
                try:
                    bs = int(bs_str)
                    input_len = int(rec["input_length"])
                    output_len = int(rec["output_length"])
                    value = float(rec[metric])
                except Exception:
                    continue
                _upsert_record(store, model, input_len, output_len, cfg, bs, value, ts)


def _extract_from_sweep_summaries(store, benchmark_root: str, metric: str):
    summary_files = sorted(
        glob.glob(os.path.join(benchmark_root, "benchmark_results_bs_output_sweep_*", "summary.csv"))
    )
    for path in summary_files:
        m = re.search(r"(\d{8}_\d{6})", path)
        ts = int(m.group(1).replace("_", "")) if m else 0
        try:
            rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
        except Exception:
            continue

        for row in rows:
            status = (row.get("status") or "").strip()
            if status != "ok":
                continue
            mode = (row.get("mode") or "").strip()
            if mode == "mustafar":
                cfg = "sparse_70"
            elif mode == "quant":
                cfg = "sparse_70_quant_2bit"
            else:
                continue

            try:
                bs = int(row["batch_size"])
                input_len = int(row["prompt_length"])
                output_len = int(row["output_length"])
            except Exception:
                continue

            if metric == "throughput":
                raw = row.get("throughput_tps", "")
                if raw in ("", None):
                    # Some mustafar rows do not export throughput directly in summary.csv.
                    # Derive it from batch_ms when available.
                    try:
                        batch_ms = float(row.get("batch_ms", ""))
                        raw = str(bs * output_len / (batch_ms / 1000.0))
                    except Exception:
                        raw = ""
            elif metric == "ttft":
                raw = row.get("ttft_ms", "")
            elif metric == "tpot":
                raw = row.get("tpot_ms", "")
            elif metric == "peak_memory":
                raw = row.get("peak_gb", "")
            else:  # avg_batch_time
                # summary.csv stores batch_ms
                raw = row.get("batch_ms", "")

            if raw in ("", None):
                continue
            try:
                value = float(raw)
                if metric == "avg_batch_time":
                    value = value / 1000.0
            except Exception:
                continue

            _upsert_record(store, "llama3_8b", input_len, output_len, cfg, bs, value, ts)


def _extract_metrics_from_output_text(text: str, metric: str):
    bs_io = re.search(
        r"Batch size:\s*(\d+).*?Input length:\s*(\d+).*?Output length:\s*(\d+)",
        text,
        flags=re.S,
    )
    if not bs_io:
        return None
    bs = int(bs_io.group(1))
    input_len = int(bs_io.group(2))
    output_len = int(bs_io.group(3))

    throughput = _last_float(r"Throughput:\s*([\d.]+)\s*tokens/sec", text)
    batch_ms = _last_float(r"(?:Batch generation time|Average time):\s*([\d.]+)\s*ms", text)
    ttft = _last_float(r"(?:^TTFT:\s*|Average TTFT:\s*)([\d.]+)\s*ms", text)
    tpot = _last_float(r"(?:^TPOT:\s*|Average TPOT:\s*)([\d.]+)\s*ms", text)
    peak = _last_float(r"(?:Peak memory|Peak mem):\s*([\d.]+)\s*GB", text)

    if metric == "throughput":
        if throughput is None and batch_ms is not None:
            throughput = bs * output_len / (batch_ms / 1000.0)
        value = throughput
    elif metric == "ttft":
        value = ttft
    elif metric == "tpot":
        value = tpot
    elif metric == "peak_memory":
        value = peak
    else:
        value = batch_ms / 1000.0 if batch_ms is not None else None

    if value is None:
        return None
    return bs, input_len, output_len, float(value)


def _extract_from_benchmark_outputs(store, benchmark_root: str, metric: str):
    # We only parse consolidated benchmark outputs (bs=8 style files).
    dir_paths = sorted(glob.glob(os.path.join(benchmark_root, "benchmark_results_*")))
    name_to_cfg = {
        "baseline_dense_output.txt": "dense",
        "mustafar_50_output.txt": "sparse_50",
        "mustafar_70_output.txt": "sparse_70",
        "mustafar_quant_50_output.txt": "sparse_50_quant_2bit",
        "mustafar_quant_70_output.txt": "sparse_70_quant_2bit",
    }
    for d in dir_paths:
        if "benchmark_results_bs_output_sweep_" in d:
            continue
        m = re.search(r"(\d{8}_\d{6})", d)
        ts = int(m.group(1).replace("_", "")) if m else 0

        for fname, cfg in name_to_cfg.items():
            path = os.path.join(d, fname)
            if not os.path.exists(path):
                continue
            try:
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            parsed = _extract_metrics_from_output_text(text, metric)
            if parsed is None:
                continue
            bs, input_len, output_len, value = parsed
            model = _detect_model_from_text(text)
            _upsert_record(store, model, input_len, output_len, cfg, bs, value, ts)


def load_records(results_dir: str, benchmark_root: str, metric: str):
    # scenario -> config -> bs -> (timestamp, metric_value)
    store: Dict[Tuple[str, int, int], Dict[str, Dict[int, Tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    _extract_from_json_results(store, results_dir, metric)
    _extract_from_sweep_summaries(store, benchmark_root, metric)
    _extract_from_benchmark_outputs(store, benchmark_root, metric)
    return store


def plot_scenarios(store, out_dir: str, metric: str) -> int:
    os.makedirs(out_dir, exist_ok=True)

    ylabel_map = {
        "throughput": "Throughput (tokens/sec)",
        "ttft": "TTFT (ms)",
        "tpot": "TPOT (ms/token)",
        "peak_memory": "Peak Memory (GB)",
        "avg_batch_time": "Batch Time (s)",
    }
    ylabel = ylabel_map.get(metric, metric)

    count = 0
    for (model, input_len, output_len), cfg_map in sorted(store.items()):
        all_bs = sorted(
            {
                bs
                for cfg in CONFIG_ORDER
                for bs in cfg_map.get(cfg, {}).keys()
            }
        )
        if not all_bs:
            continue

        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        for cfg in CONFIG_ORDER:
            style = CONFIG_STYLES[cfg]
            pts = cfg_map.get(cfg, {})
            if pts:
                xs = sorted(pts.keys())
                ys = [pts[x][1] for x in xs]
                ax.plot(
                    xs,
                    ys,
                    label=CONFIG_LABELS[cfg],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    markersize=6,
                )
            else:
                # Keep all 5 lines in legend even when missing.
                ax.plot(
                    [],
                    [],
                    label=f"{CONFIG_LABELS[cfg]} (no data)",
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                )

        ax.set_xlabel("Batch Size (BS)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_bs)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{model} | input={input_len}, output={output_len} | metric={metric}")
        ax.legend(fontsize=9)

        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        out_name = f"{safe_model}_in{input_len}_out{output_len}_{metric}.png"
        out_path = os.path.join(out_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Draw 5-line BS charts by scenario.")
    parser.add_argument(
        "--results-dir",
        default="/mnt/home/zh/mustafar/JSQKV_benchmark/results/raw_data",
        help="Directory containing *_results*.json",
    )
    parser.add_argument(
        "--benchmark-root",
        default="/mnt/home/zh/mustafar/benchmark",
        help="Benchmark root (for benchmark_results_* and sweep summary.csv files)",
    )
    parser.add_argument(
        "--metric",
        default="throughput",
        choices=["throughput", "ttft", "tpot", "peak_memory", "avg_batch_time"],
        help="Y-axis metric",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for plots (default: timestamped under current dir)",
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), f"five_line_plots_{args.metric}_{ts}")

    store = load_records(args.results_dir, args.benchmark_root, args.metric)
    num = plot_scenarios(store, out_dir, args.metric)

    print(f"Generated {num} plot(s).")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
