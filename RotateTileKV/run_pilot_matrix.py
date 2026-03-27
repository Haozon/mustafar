#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from RotateTileKV.modeling_llama_rotatetilekv import (
    RotateTileKVConfig,
    load_rotatetilekv_llama,
    set_rotatetilekv_config,
)
from RotateTileKV.run_longbench import (
    load_prompt_config,
    resolve_max_length,
    score_dataset,
    seed_everything,
    generate_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small RotateTileKV LongBench pilot matrix.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--datasets", default="trec,triviaqa,passage_count,qasper")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    bootstrap_cfg = RotateTileKVConfig(enable_hadamard=False, k_bits=16, v_bits=16, quant_granularity="per-token-tile")
    tokenizer, model = load_rotatetilekv_llama(
        args.model_name_or_path,
        quant_cfg=bootstrap_cfg,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=not args.allow_download,
        low_cpu_mem_usage=True,
    )

    prompt_cfg, gen_cfg = load_prompt_config()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    max_length = resolve_max_length(args.model_name_or_path, model.config, None)
    output_dir = Path(args.output_dir) if args.output_dir else Path("RotateTileKV/pilot_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cache = {}
    for dataset_name in datasets:
        ds = load_dataset("THUDM/LongBench", dataset_name, split="test", trust_remote_code=True)
        data_cache[dataset_name] = ds.select(range(min(args.limit, len(ds))))

    configs = [
        ("fp16", RotateTileKVConfig(enable_hadamard=False, k_bits=16, v_bits=16, quant_granularity="per-token-tile")),
        ("per_token_4bit", RotateTileKVConfig(enable_hadamard=False, k_bits=4, v_bits=4, quant_granularity="per-token")),
        ("per_token_head_4bit", RotateTileKVConfig(enable_hadamard=False, k_bits=4, v_bits=4, quant_granularity="per-token-head")),
        ("per_token_tile_4bit", RotateTileKVConfig(enable_hadamard=False, k_bits=4, v_bits=4, quant_granularity="per-token-tile")),
        ("per_token_tile_4bit_hadamard", RotateTileKVConfig(enable_hadamard=True, k_bits=4, v_bits=4, quant_granularity="per-token-tile")),
        ("per_token_tile_2bit", RotateTileKVConfig(enable_hadamard=False, k_bits=2, v_bits=2, quant_granularity="per-token-tile")),
        ("per_token_tile_2bit_hadamard", RotateTileKVConfig(enable_hadamard=True, k_bits=2, v_bits=2, quant_granularity="per-token-tile")),
    ]

    summary = {}
    for label, quant_cfg in configs:
        print("=" * 80)
        print(f"Running config: {label}")
        set_rotatetilekv_config(model, quant_cfg)
        summary[label] = {}
        for dataset_name in datasets:
            rows = generate_predictions(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                rows=data_cache[dataset_name],
                prompt_format=prompt_cfg[dataset_name],
                max_length=max_length,
                max_gen=gen_cfg[dataset_name],
            )
            score = score_dataset(dataset_name, rows)
            summary[label][dataset_name] = score
            print(f"{dataset_name:20s}: {score:6.2f}")
        summary[label]["average"] = round(
            sum(summary[label][dataset_name] for dataset_name in datasets) / max(len(datasets), 1),
            2,
        )
        print(f"{'average':20s}: {summary[label]['average']:6.2f}")
        with open(output_dir / "pilot_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        torch.cuda.empty_cache()

    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved to: {output_dir / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()
