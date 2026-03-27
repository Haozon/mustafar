#!/usr/bin/env python3
"""
Batch-size throughput benchmark with isolated runs.

Each batch size is measured with a freshly loaded model so the points are directly comparable.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, validate_config
from utils.metrics import measure_throughput
from utils.model_loader import apply_quant_runtime_tuning, load_model


def run_single_case(model_name, model_config, config_name, test_config, global_config, batch_size, measure_token_metrics):
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name} | {config_name} | BS={batch_size}")
    print(f"{'=' * 70}")

    test_cfg = test_config.copy()
    test_cfg["group_size"] = global_config.get("group_size", 32)
    test_cfg["residual_length"] = global_config.get("residual_length", 256)

    model, tokenizer = load_model(model_config, test_cfg)

    try:
        tuning = apply_quant_runtime_tuning(model, model_config, test_cfg, batch_size)
        if test_cfg.get("use_quant", False):
            print(
                "Runtime quant tuning: "
                f"split_k={tuning['quant_v_split_k']}, "
                f"tile={tuning['quant_v_tile_config']}, "
                f"decode_n1={tuning['quant_v_decode_n1']}, "
                f"auto_tuned={tuning['auto_tuned']}, "
                f"reason={tuning['reason']}"
            )

        metrics = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            input_length=model_config["input_length"],
            output_length=model_config["output_length"],
            num_repeats=global_config.get("num_repeats", 3),
            warmup_tokens=global_config.get("warmup_tokens", 10),
            measure_token_metrics=measure_token_metrics,
        )
        metrics["quant_runtime_tuning"] = tuning
        print(
            f"✅ {model_name} | {config_name} | BS={batch_size}: "
            f"{metrics['throughput']:.2f} tok/s"
        )
        return metrics
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Batch-size throughput benchmark with isolated runs")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
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
    batch_sizes = args.batch_sizes or config["batch_sizes"]

    all_results = {}
    for model_name in models_to_test:
        model_config = config["models"][model_name]
        all_results[model_name] = {}
        for config_name in configs_to_test:
            test_config = config["test_configs"][config_name]
            all_results[model_name][config_name] = {}
            for batch_size in batch_sizes:
                try:
                    metrics = run_single_case(
                        model_name,
                        model_config,
                        config_name,
                        test_config,
                        config,
                        batch_size,
                        not args.skip_token_metrics,
                    )
                    all_results[model_name][config_name][batch_size] = metrics
                except RuntimeError as e:
                    msg = str(e)
                    if "out of memory" in msg.lower() or "cuda" in msg.lower():
                        print(f"⚠️  {model_name} | {config_name} | BS={batch_size} failed: {msg}")
                        torch.cuda.empty_cache()
                        all_results[model_name][config_name][batch_size] = {"error": msg}
                    else:
                        raise

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
