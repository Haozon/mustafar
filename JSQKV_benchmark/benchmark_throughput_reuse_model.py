#!/usr/bin/env python3
"""
Batch-size throughput benchmark that reuses one loaded model for multiple batch sizes.
This is more suitable for long sequence settings such as 4096 -> 4096.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, validate_config
from utils.model_loader import load_model
from utils.metrics import measure_throughput


def run_model_config(model_name, model_config, config_name, test_config, global_config, measure_token_metrics):
    print(f"\n{'=' * 70}")
    print(f"Loading once: {model_name} | {config_name}")
    print(f"{'=' * 70}")

    test_cfg = test_config.copy()
    test_cfg["group_size"] = global_config.get("group_size", 32)
    test_cfg["residual_length"] = global_config.get("residual_length", 256)

    model, tokenizer = load_model(model_config, test_cfg)
    results = {}

    try:
        for batch_size in global_config["batch_sizes"]:
            print(f"\n{'-' * 70}")
            print(f"Benchmarking BS={batch_size}")
            print(f"{'-' * 70}")
            try:
                metrics = measure_throughput(
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    input_length=model_config["input_length"],
                    output_length=model_config["output_length"],
                    num_repeats=global_config.get("num_repeats", 1),
                    warmup_tokens=global_config.get("warmup_tokens", 10),
                    measure_token_metrics=measure_token_metrics,
                )
                results[batch_size] = metrics
                print(
                    f"✅ {model_name} | {config_name} | BS={batch_size}: "
                    f"{metrics['throughput']:.2f} tok/s"
                )
            except RuntimeError as e:
                msg = str(e)
                if "out of memory" in msg.lower() or "cuda" in msg.lower():
                    print(f"⚠️  BS={batch_size} failed: {msg}")
                    torch.cuda.empty_cache()
                    results[batch_size] = {"error": msg}
                else:
                    raise
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch-size throughput benchmark with model reuse")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results/raw_data")
    parser.add_argument(
        "--skip-token-metrics",
        action="store_true",
        help="Only measure throughput / batch time / memory, skip TTFT and TPOT",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    models_to_test = args.models or list(config["models"].keys())
    configs_to_test = args.configs or list(config["test_configs"].keys())

    all_results = {}
    for model_name in models_to_test:
        model_config = config["models"][model_name]
        all_results[model_name] = {}
        for config_name in configs_to_test:
            test_config = config["test_configs"][config_name]
            all_results[model_name][config_name] = run_model_config(
                model_name, model_config, config_name, test_config, config, not args.skip_token_metrics
            )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name, model_results in all_results.items():
        output_file = os.path.join(args.output_dir, f"{model_name}_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)
        backup_file = os.path.join(args.output_dir, f"{model_name}_results_{timestamp}.json")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)
        print(f"Saved: {output_file}")
        print(f"Saved: {backup_file}")


if __name__ == "__main__":
    main()
