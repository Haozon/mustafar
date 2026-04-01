#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/zh/miniconda3/envs/mustar/bin/python"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    jobs = [
        # Meta-Llama-3-8B 50%
        {
            "task": "narrativeqa",
            "cfg": "solver_runs_budget50/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.50_rep6_budget50_try1_narrativeqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Meta-Llama-3-8B-Instruct",
            "max_length": 8192,
            "budget": 0.50,
            "tag_prefix": "meta50_full",
        },
        {
            "task": "qasper",
            "cfg": "solver_runs_budget50/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.50_rep6_budget50_try1_qasper_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Meta-Llama-3-8B-Instruct",
            "max_length": 8192,
            "budget": 0.50,
            "tag_prefix": "meta50_full",
        },
        {
            "task": "multifieldqa_en",
            "cfg": "solver_runs_budget50/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.50_rep6_budget50_try1_multifieldqa_en_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Meta-Llama-3-8B-Instruct",
            "max_length": 8192,
            "budget": 0.50,
            "tag_prefix": "meta50_full",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_budget50/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.50_rep6_budget50_try1_trec_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Meta-Llama-3-8B-Instruct",
            "max_length": 8192,
            "budget": 0.50,
            "tag_prefix": "meta50_full",
        },

        # Llama-2-7B 70%
        {
            "task": "narrativeqa",
            "cfg": "solver_runs_llama2_7b_budget70_fix/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_narrativeqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_7b_70_full",
        },
        {
            "task": "multifieldqa_en",
            "cfg": "solver_runs_llama2_7b_budget70_fix/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_multifieldqa_en_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_7b_70_full",
        },
        {
            "task": "hotpotqa",
            "cfg": "solver_runs_llama2_7b_budget70_fix/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_hotpotqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_7b_70_full",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_llama2_7b_budget70_fix/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_trec_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_7b_70_full",
        },
        {
            "task": "lcc",
            "cfg": "solver_runs_llama2_7b_budget70_fix/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_lcc_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_7b_70_full",
        },

        # Llama-2-7B 50%
        {
            "task": "narrativeqa",
            "cfg": "solver_runs_llama2_7b_budget50/Llama-2-7b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_narrativeqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_7b_50_full",
        },
        {
            "task": "multifieldqa_en",
            "cfg": "solver_runs_llama2_7b_budget50/Llama-2-7b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_multifieldqa_en_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_7b_50_full",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_llama2_7b_budget50/Llama-2-7b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_trec_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_7b_50_full",
        },
        {
            "task": "lcc",
            "cfg": "solver_runs_llama2_7b_budget50/Llama-2-7b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_lcc_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_7b_50_full",
        },

        # Llama-2-13B 70%
        {
            "task": "narrativeqa",
            "cfg": "solver_runs_llama2_13b_budget70/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_narrativeqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_13b_70_full",
        },
        {
            "task": "qasper",
            "cfg": "solver_runs_llama2_13b_budget70/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_qasper_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_13b_70_full",
        },
        {
            "task": "hotpotqa",
            "cfg": "solver_runs_llama2_13b_budget70/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_hotpotqa_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_13b_70_full",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_llama2_13b_budget70/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_trec_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_13b_70_full",
        },
        {
            "task": "lcc",
            "cfg": "solver_runs_llama2_13b_budget70/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_rep6_budget70_try1_lcc_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.70,
            "tag_prefix": "llama2_13b_70_full",
        },

        # Llama-2-13B 50%
        {
            "task": "qasper",
            "cfg": "solver_runs_llama2_13b_budget50/Llama-2-13b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_qasper_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_13b_50_full",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_llama2_13b_budget50/Llama-2-13b-hf_4096_diff_sparse_kv_0.50_rep6_budget50_try1_trec_bestdiff_val/sparsity_config.json",
            "model": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "budget": 0.50,
            "tag_prefix": "llama2_13b_50_full",
        },
    ]

    for job in jobs:
        run([
            PY, "solver_runs/run_full_task_from_config.py",
            "--task", job["task"],
            "--config_json", job["cfg"],
            "--model_path", job["model"],
            "--max_length", str(job["max_length"]),
            "--target_budget", f"{job['budget']:.2f}",
            "--tag_prefix", job["tag_prefix"],
        ])


if __name__ == "__main__":
    main()
