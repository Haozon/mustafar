#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODEL_PATH = "/home/zh/model/Mistral-7B-v0.1"
MAX_LENGTH = "8192"
TARGET_BUDGET = "0.70"
TAG_PREFIX = "mistral70_repaired_full"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    jobs = [
        {
            "task": "multifieldqa_en",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_multifieldqa_en_bestdiff_val/sparsity_config.json",
        },
        {
            "task": "lcc",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_lcc_bestdiff_val/sparsity_config.json",
        },
        {
            "task": "hotpotqa",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_hotpotqa_bestdiff_val/sparsity_config.json",
        },
        {
            "task": "narrativeqa",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_narrativeqa_bestdiff_val/sparsity_config.json",
        },
        {
            "task": "trec",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_trec_bestdiff_val/sparsity_config.json",
        },
    ]

    for job in jobs:
        run([
            PY,
            "solver_runs/run_full_task_from_config.py",
            "--task", job["task"],
            "--config_json", job["cfg"],
            "--model_path", MODEL_PATH,
            "--max_length", MAX_LENGTH,
            "--target_budget", TARGET_BUDGET,
            "--tag_prefix", TAG_PREFIX,
        ])


if __name__ == "__main__":
    main()
