#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSPARSE_ROOT = PROJECT_ROOT / "DiffSparseKV"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFSPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSPARSE_ROOT))

import torch
from transformers import AutoTokenizer

import eval_diff_sparse_kv_longbench as diff_eval  # noqa: E402
from JSQKV.integration import create_jsqkv_quant_config, load_jsqkv_llama  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JSQKV-lite on LongBench.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "trec", "lcc"],
    )
    parser.add_argument("--output_dir", default="jsqkv_runs")
    parser.add_argument("--output_tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=-1)
    parser.add_argument("--sample_indices_file", default="")
    parser.add_argument("--target_distribution", type=str, default="0.0,0.75,0.25")
    parser.add_argument("--sparsity_levels", type=str, default="0.0,0.6,1.0")
    parser.add_argument("--importance_mode", default="value_aware")
    parser.add_argument("--head_aggregation_mode", default="max")
    parser.add_argument("--value_sink_keep", type=int, default=2)
    parser.add_argument("--level_2_mode", default="evict", choices=["evict", "zero"])
    parser.add_argument("--protected_heavy_ratio", type=float, default=0.0)
    parser.add_argument("--protected_recent_ratio", type=float, default=1.0)
    parser.add_argument("--obs_window_size", type=int, default=128)
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--v_bits", type=int, default=2)
    parser.add_argument("--quant_impl", default="default", choices=["default", "kivi"])
    parser.add_argument("--k_quant_scheme", default="per-token-tile")
    parser.add_argument("--v_quant_scheme", default="per-token-tile")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--quant_granularity", default="per-token-tile")
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--enable_hadamard", action="store_true")
    parser.add_argument("--hadamard_mode", default="tile", choices=["none", "full", "tile"])
    parser.add_argument("--hadamard_group_size", type=int, default=64)
    parser.add_argument("--run_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    target_distribution = [float(x) for x in args.target_distribution.split(",")]
    sparsity_levels = [float(x) for x in args.sparsity_levels.split(",")]
    sample_indices_spec = None
    if args.sample_indices_file:
        with open(args.sample_indices_file, "r", encoding="utf-8") as f:
            sample_indices_spec = json.load(f)

    quant_cfg = create_jsqkv_quant_config(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        quant_impl=args.quant_impl,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        group_size=args.group_size,
        quant_granularity=args.quant_granularity,
        tile_size=args.tile_size,
        residual_length=args.residual_length,
        enable_hadamard=args.enable_hadamard,
        hadamard_mode=args.hadamard_mode,
        hadamard_group_size=args.hadamard_group_size,
    )

    model, config = load_jsqkv_llama(
        model_path=args.model_path,
        diff_target_distribution=target_distribution,
        diff_sparsity_levels=sparsity_levels,
        quant_cfg=quant_cfg,
        max_length=args.max_length,
        obs_window_size=args.obs_window_size,
        importance_mode=args.importance_mode,
        head_aggregation_mode=args.head_aggregation_mode,
        value_sink_keep=args.value_sink_keep,
        level_2_mode=args.level_2_mode,
        protected_heavy_ratio=args.protected_heavy_ratio,
        protected_recent_ratio=args.protected_recent_ratio,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_name = Path(args.model_path).name
    tag = args.output_tag or (
        f"{model_name}_{args.max_length}_jsqkv_budget_{sum(p*s for p,s in zip(target_distribution, sparsity_levels)):.2f}"
        f"_k{args.k_bits}_v{args.v_bits}"
    )
    output_path = Path(args.output_dir) / tag
    output_path.mkdir(parents=True, exist_ok=True)

    dataset2prompt = {
        "narrativeqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "qasper": "{context}\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_en": "{context}\n\nQuestion: {input}\nAnswer:",
        "hotpotqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "trec": "{input}",
        "lcc": "{input}",
    }
    dataset2maxlen = {
        "narrativeqa": 100,
        "qasper": 100,
        "multifieldqa_en": 100,
        "hotpotqa": 100,
        "trec": 50,
        "lcc": 100,
    }

    config_dir = PROJECT_ROOT / "config"
    try:
        with open(config_dir / "dataset2prompt.json", "r", encoding="utf-8") as f:
            dataset2prompt.update(json.load(f))
        with open(config_dir / "dataset2maxlen.json", "r", encoding="utf-8") as f:
            dataset2maxlen.update(json.load(f))
        print("Loaded dataset prompt/maxlen configuration from config/")
    except FileNotFoundError:
        print("Using built-in dataset prompt/maxlen defaults (config files not found)")

    for dataset in args.datasets:
        data = diff_eval.load_longbench_dataset(dataset)
        if sample_indices_spec is not None:
            if isinstance(sample_indices_spec, dict):
                selected_indices = sample_indices_spec.get(dataset, [])
            else:
                selected_indices = sample_indices_spec
            data = data.select([int(i) for i in selected_indices])
        elif args.limit > 0:
            sample_count = min(args.limit, len(data))
            if args.sample_seed >= 0:
                import numpy as np

                rng = np.random.default_rng(args.sample_seed)
                sampled_indices = sorted(
                    rng.choice(len(data), size=sample_count, replace=False).tolist()
                )
                data = data.select(sampled_indices)
            else:
                data = data.select(range(sample_count))

        preds = diff_eval.get_pred(
            model=model,
            tokenizer=tokenizer,
            data=data,
            max_length=args.max_length,
            max_gen=dataset2maxlen[dataset],
            prompt_format=dataset2prompt[dataset],
            dataset=dataset,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_name=model_name,
            kv_sparsity_wrapper=None,
        )

        with open(output_path / f"{dataset}.jsonl", "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    run_config = {
        "model_path": args.model_path,
        "max_length": args.max_length,
        "datasets": args.datasets,
        "target_distribution": target_distribution,
        "sparsity_levels": sparsity_levels,
        "importance_mode": args.importance_mode,
        "head_aggregation_mode": args.head_aggregation_mode,
        "value_sink_keep": args.value_sink_keep,
        "quant_cfg": quant_cfg.to_dict(),
        "jsqkv_type": "diffsparse_plus_rotatetile_fake_quant",
    }
    with open(output_path / "jsqkv_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"JSQKV predictions saved to: {output_path}")

    if args.run_eval:
        import subprocess

        subprocess.run(
            [sys.executable, str(DIFFSPARSE_ROOT / "eval_results.py"), "--result_dir", str(output_path)],
            check=False,
        )


if __name__ == "__main__":
    main()
