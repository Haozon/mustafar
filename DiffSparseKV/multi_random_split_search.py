#!/usr/bin/env python3
"""
Run multi-random-split configuration search for DiffSparseKV.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


PYTHON = "/home/zh/miniconda3/envs/mustar/bin/python"


def candidate_configs():
    return [
        {
            "name": "default70_mean",
            "target_distribution": "0.05,0.75,0.20",
            "sparsity_levels": "0.0,0.6666666667,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "mean",
        },
        {
            "name": "default70_max",
            "target_distribution": "0.05,0.75,0.20",
            "sparsity_levels": "0.0,0.6666666667,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "max",
        },
        {
            "name": "conservative70_max",
            "target_distribution": "0.10,0.70,0.20",
            "sparsity_levels": "0.0,0.7142857143,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "max",
        },
        {
            "name": "lowevict70_max",
            "target_distribution": "0.10,0.80,0.10",
            "sparsity_levels": "0.0,0.75,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "max",
        },
        {
            "name": "nodense70_max",
            "target_distribution": "0.00,0.80,0.20",
            "sparsity_levels": "0.0,0.625,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "max",
        },
        {
            "name": "507010_a20_max",
            "target_distribution": "0.20,0.6666666667,0.1333333333",
            "sparsity_levels": "0.5,0.7,1.0",
            "importance_mode": "value_aware",
            "head_aggregation_mode": "max",
        },
    ]


def run_eval(task: str, limit: int, seed: int, cfg: dict, output_root: Path) -> float:
    tag = f"{task}_seed{seed}_{cfg['name']}"
    cmd = [
        PYTHON,
        "eval_diff_sparse_kv_longbench.py",
        "--model_path",
        "/home/zh/model/Meta-Llama-3-8B-Instruct",
        "--datasets",
        task,
        "--limit",
        str(limit),
        "--sample_seed",
        str(seed),
        "--output_dir",
        str(output_root),
        "--output_tag",
        tag,
        "--sparsity_type",
        "diff_sparse_kv",
        "--target_distribution",
        cfg["target_distribution"],
        "--sparsity_levels",
        cfg["sparsity_levels"],
        "--level_2_mode",
        "evict",
        "--importance_mode",
        cfg["importance_mode"],
        "--value_sink_keep",
        "2",
        "--head_aggregation_mode",
        cfg["head_aggregation_mode"],
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent)

    result_dir = next(output_root.glob(f"Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.00_{tag}"))
    score_cmd = [
        PYTHON,
        "eval_results.py",
        "--result_dir",
        str(result_dir),
    ]
    subprocess.run(score_cmd, check=True, cwd=Path(__file__).resolve().parent, stdout=subprocess.DEVNULL)
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    return float(result[task])


def summarize(rows):
    rows.sort(key=lambda x: x["mean_score"], reverse=True)
    header = "| rank | config | mean | std | splits |\n| --- | --- | --- | --- | --- |"
    body = []
    for idx, row in enumerate(rows, start=1):
        body.append(
            f"| {idx} | {row['name']} | {row['mean_score']:.2f} | {row['std_score']:.2f} | {row['scores']} |"
        )
    return "\n".join([header] + body)


def main():
    parser = argparse.ArgumentParser(description="Multi random split family search")
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output-root", type=str, default="tmp_eval")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    rows = []
    for cfg in candidate_configs():
        scores = []
        for seed in seeds:
            score = run_eval(args.task, args.limit, seed, cfg, output_root)
            scores.append(round(score, 2))
        rows.append(
            {
                "name": cfg["name"],
                "scores": scores,
                "mean_score": mean(scores),
                "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
            }
        )
    markdown = summarize(rows)
    print(markdown)
    if args.out:
        Path(args.out).write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
