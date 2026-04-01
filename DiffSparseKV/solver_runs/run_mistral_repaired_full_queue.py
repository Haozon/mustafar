#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/zh/miniconda3/envs/mustar/bin/python"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    jobs = [
        {
            "task": "multifieldqa_en",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_multifieldqa_en_bestdiff_val/sparsity_config.json",
            "tag_prefix": "mistral70_repaired_full",
        },
        {
            "task": "lcc",
            "cfg": "solver_runs_mistral_budget70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_rep6_budget70_try1_lcc_bestdiff_val/sparsity_config.json",
            "tag_prefix": "mistral70_repaired_full",
        },
    ]

    for job in jobs:
        run([
            PY,
            "solver_runs/run_full_task_from_config.py",
            "--task", job["task"],
            "--config_json", job["cfg"],
            "--model_path", "/home/zh/model/Mistral-7B-Instruct-v0.1",
            "--max_length", "8192",
            "--target_budget", "0.70",
            "--tag_prefix", job["tag_prefix"],
        ])


if __name__ == "__main__":
    main()
