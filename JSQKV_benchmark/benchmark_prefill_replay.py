#!/usr/bin/env python3
"""
Measure prompt-side cache build and cached decode under the real workload.

The goal is attribution within end-to-end inference:
- online_total_ms: full prompt-to-output generation
- prefill_ms: prompt processing with cache construction
- cached_decode_ms: decode after prompt cache is already available
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, validate_config
from utils.metrics import build_fixed_length_inputs
from utils.model_loader import apply_quant_runtime_tuning, load_model


def run_online_total(model, inputs, output_length, num_repeats, warmup_tokens):
    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model.generate(**inputs, max_new_tokens=warmup_tokens, eos_token_id=None)
        torch.cuda.synchronize()

    times = []
    peaks = []
    with torch.no_grad():
        for _ in range(num_repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.time()
            _ = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
    return times, peaks


def run_prefill(model, inputs, num_repeats):
    times = []
    peaks = []
    last_outputs = None
    with torch.no_grad():
        for _ in range(num_repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.time()
            last_outputs = model(**inputs, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
    return times, peaks, last_outputs


def run_cached_decode(model, prefill_outputs, output_length, num_repeats):
    if output_length <= 1:
        return [0.0] * num_repeats

    times = []
    with torch.no_grad():
        for _ in range(num_repeats):
            past_key_values = prefill_outputs.past_key_values
            next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)

            torch.cuda.synchronize()
            start = time.time()
            for _step in range(output_length - 1):
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000.0)
    return times


def summarize(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def run_single_case(model_name, model_config, config_name, test_config, global_config, batch_size):
    print(f"\n{'=' * 70}")
    print(f"Attribution: {model_name} | {config_name} | BS={batch_size}")
    print(f"{'=' * 70}")

    test_cfg = test_config.copy()
    test_cfg["group_size"] = global_config.get("group_size", 32)
    test_cfg["residual_length"] = global_config.get("residual_length", 256)

    model, tokenizer = load_model(model_config, test_cfg)

    try:
        tuning = apply_quant_runtime_tuning(model, model_config, test_cfg, batch_size)
        inputs = build_fixed_length_inputs(tokenizer, batch_size, model_config["input_length"], device="cuda")

        online_times, online_peaks = run_online_total(
            model,
            inputs,
            model_config["output_length"],
            global_config.get("num_repeats", 3),
            global_config.get("warmup_tokens", 10),
        )
        prefill_times, prefill_peaks, prefill_outputs = run_prefill(
            model, inputs, global_config.get("num_repeats", 3)
        )
        cached_decode_times = run_cached_decode(
            model, prefill_outputs, model_config["output_length"], global_config.get("num_repeats", 3)
        )

        result = {
            "batch_size": batch_size,
            "input_length": model_config["input_length"],
            "output_length": model_config["output_length"],
            "quant_runtime_tuning": tuning,
            "online_total_ms": summarize(online_times),
            "prefill_ms": summarize(prefill_times),
            "cached_decode_ms": summarize(cached_decode_times),
            "online_peak_gb": summarize(online_peaks),
            "prefill_peak_gb": summarize(prefill_peaks),
            "cached_decode_tokens": max(model_config["output_length"] - 1, 0),
        }
        if result["cached_decode_tokens"] > 0:
            result["cached_decode_tpot_ms"] = result["cached_decode_ms"]["mean"] / result["cached_decode_tokens"]
        else:
            result["cached_decode_tpot_ms"] = 0.0
        result["prefill_share_of_online"] = (
            result["prefill_ms"]["mean"] / result["online_total_ms"]["mean"]
            if result["online_total_ms"]["mean"] > 0
            else 0.0
        )
        return result
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Prompt-stage and cached-decode attribution benchmark")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/attribution")
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
                    result = run_single_case(
                        model_name, model_config, config_name, test_config, config, batch_size
                    )
                    all_results[model_name][config_name][batch_size] = result
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
    output_file = os.path.join(args.output_dir, f"prefill_replay_results_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
