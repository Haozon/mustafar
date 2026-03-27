#!/usr/bin/env python3
"""
Collect prompt-side compression statistics under the real workload.

For each config and batch size, run the prompt once (plus warmup) and aggregate:
- original KV bytes
- compressed KV bytes
- compression ratio
- memory saving ratio
- compression time inside the prefill compression path
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, validate_config
from utils.metrics import build_fixed_length_inputs
from utils.model_loader import apply_quant_runtime_tuning, load_model


def reset_compression_stats(model):
    for module in model.modules():
        if hasattr(module, "latest_compression_stats"):
            module.latest_compression_stats = None


def aggregate_layer_stats(model):
    total = {
        "layer_count": 0,
        "original_kv_bytes": 0,
        "compressed_kv_bytes": 0,
        "compression_time_ms": 0.0,
        "compressed_length_tokens": 0,
    }
    for module in model.modules():
        stats = getattr(module, "latest_compression_stats", None)
        if not stats:
            continue
        total["layer_count"] += 1
        total["original_kv_bytes"] += stats["original_kv_bytes"]
        total["compressed_kv_bytes"] += stats["compressed_kv_bytes"]
        total["compression_time_ms"] += stats["compression_time_ms"]
        total["compressed_length_tokens"] = max(total["compressed_length_tokens"], stats["compressed_length"])
    if total["compressed_kv_bytes"] > 0:
        total["compression_ratio"] = total["original_kv_bytes"] / total["compressed_kv_bytes"]
    else:
        total["compression_ratio"] = 0.0
    if total["original_kv_bytes"] > 0:
        total["memory_saving_ratio"] = 1.0 - (total["compressed_kv_bytes"] / total["original_kv_bytes"])
    else:
        total["memory_saving_ratio"] = 0.0
    return total


def summarize_numeric(rows, key):
    values = [row[key] for row in rows]
    return sum(values) / len(values) if values else 0.0


def run_single_case(model_name, model_config, config_name, test_config, global_config, batch_size):
    print(f"\n{'=' * 70}")
    print(f"Compression stats: {model_name} | {config_name} | BS={batch_size}")
    print(f"{'=' * 70}")

    test_cfg = test_config.copy()
    test_cfg["group_size"] = global_config.get("group_size", 32)
    test_cfg["residual_length"] = global_config.get("residual_length", 256)

    model, tokenizer = load_model(model_config, test_cfg)
    try:
        tuning = apply_quant_runtime_tuning(model, model_config, test_cfg, batch_size)
        inputs = build_fixed_length_inputs(tokenizer, batch_size, model_config["input_length"], device="cuda")

        repeats = []
        with torch.no_grad():
            reset_compression_stats(model)
            _ = model(**inputs, use_cache=True, return_dict=True)
            torch.cuda.synchronize()

            for _ in range(global_config.get("num_repeats", 3)):
                reset_compression_stats(model)
                _ = model(**inputs, use_cache=True, return_dict=True)
                torch.cuda.synchronize()
                repeats.append(aggregate_layer_stats(model))

        result = {
            "batch_size": batch_size,
            "input_length": model_config["input_length"],
            "output_length": model_config["output_length"],
            "quant_runtime_tuning": tuning,
            "layer_count": int(summarize_numeric(repeats, "layer_count")),
            "original_kv_bytes": int(round(summarize_numeric(repeats, "original_kv_bytes"))),
            "compressed_kv_bytes": int(round(summarize_numeric(repeats, "compressed_kv_bytes"))),
            "compression_ratio": summarize_numeric(repeats, "compression_ratio"),
            "memory_saving_ratio": summarize_numeric(repeats, "memory_saving_ratio"),
            "compression_time_ms": summarize_numeric(repeats, "compression_time_ms"),
            "compressed_length_tokens": int(round(summarize_numeric(repeats, "compressed_length_tokens"))),
        }
        print(
            f"✅ ratio={result['compression_ratio']:.2f}x, "
            f"saved={result['memory_saving_ratio'] * 100:.1f}%, "
            f"compression_time={result['compression_time_ms']:.2f} ms"
        )
        return result
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Prompt-side compression stats benchmark")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/compression_stats")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    models_to_test = args.models or list(config["models"].keys())
    configs_to_test = args.configs or list(config["test_configs"].keys())
    batch_sizes = args.batch_sizes or config["batch_sizes"]

    all_results = {}
    for model_name in models_to_test:
        model_config = config["models"][model_name]
        all_results[model_name] = {}
        for config_name in configs_to_test:
            test_config = config["test_configs"][config_name]
            all_results[model_name][config_name] = {}
            for batch_size in batch_sizes:
                result = run_single_case(model_name, model_config, config_name, test_config, config, batch_size)
                all_results[model_name][config_name][batch_size] = result

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"compression_stats_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(args.output_dir, f"compression_stats_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "config",
            "batch_size",
            "input_length",
            "output_length",
            "layer_count",
            "compressed_length_tokens",
            "original_kv_bytes",
            "compressed_kv_bytes",
            "compression_ratio",
            "memory_saving_ratio",
            "compression_time_ms",
        ])
        for model_name, model_results in all_results.items():
            for config_name, config_results in model_results.items():
                for bs, row in sorted(config_results.items()):
                    writer.writerow([
                        model_name,
                        config_name,
                        bs,
                        row["input_length"],
                        row["output_length"],
                        row["layer_count"],
                        row["compressed_length_tokens"],
                        row["original_kv_bytes"],
                        row["compressed_kv_bytes"],
                        f"{row['compression_ratio']:.6f}",
                        f"{row['memory_saving_ratio']:.6f}",
                        f"{row['compression_time_ms']:.6f}",
                    ])
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

