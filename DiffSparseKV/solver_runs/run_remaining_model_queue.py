#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    jobs = [
        {
            "name": "llama2_7b_70",
            "output_root": "solver_runs_llama2_7b_budget70_fix",
            "model_path": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "target_budget": 0.70,
            "p0_grid": "0.0,0.05,0.10,0.15",
            "rho1_grid": "0.50,0.55,0.60,0.65,0.70,0.75",
        },
        {
            "name": "mistral_70",
            "output_root": "solver_runs_mistral_budget70",
            "model_path": "/home/zh/model/Mistral-7B-v0.1",
            "max_length": 8192,
            "target_budget": 0.70,
            "p0_grid": "0.0,0.05,0.10,0.15",
            "rho1_grid": "0.50,0.55,0.60,0.65,0.70,0.75",
        },
        {
            "name": "llama2_13b_70",
            "output_root": "solver_runs_llama2_13b_budget70",
            "model_path": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "target_budget": 0.70,
            "p0_grid": "0.0,0.05,0.10,0.15",
            "rho1_grid": "0.50,0.55,0.60,0.65,0.70,0.75",
        },
        {
            "name": "llama2_7b_50",
            "output_root": "solver_runs_llama2_7b_budget50",
            "model_path": "/home/zh/model/Llama-2-7b-hf",
            "max_length": 4096,
            "target_budget": 0.50,
            "p0_grid": "0.0,0.05,0.10",
            "rho1_grid": "0.40,0.45,0.50,0.55,0.60,0.65",
        },
        {
            "name": "mistral_50",
            "output_root": "solver_runs_mistral_budget50",
            "model_path": "/home/zh/model/Mistral-7B-v0.1",
            "max_length": 8192,
            "target_budget": 0.50,
            "p0_grid": "0.0,0.05,0.10",
            "rho1_grid": "0.40,0.45,0.50,0.55,0.60,0.65",
        },
        {
            "name": "llama2_13b_50",
            "output_root": "solver_runs_llama2_13b_budget50",
            "model_path": "/home/zh/model/Llama-2-13b-hf",
            "max_length": 4096,
            "target_budget": 0.50,
            "p0_grid": "0.0,0.05,0.10",
            "rho1_grid": "0.40,0.45,0.50,0.55,0.60,0.65",
        },
    ]

    common = [
        "--solver_mode", "per_task",
        "--calib_limit", "8",
        "--calib_seed", "17",
        "--val_datasets", "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "trec", "lcc",
        "--val_limit", "20",
        "--val_seed", "29",
    ]

    for job in jobs:
        ensure_dir(job["output_root"])
        budget = int(round(job["target_budget"] * 100))
        tag = f"rep6_budget{budget}_try1"
        cmd = [
            PY, "search_diff_budget_solver.py",
            "--model_path", job["model_path"],
            "--max_length", str(job["max_length"]),
            "--target_budget", f"{job['target_budget']:.2f}",
            "--output_root", job["output_root"],
            "--search_tag", tag,
            "--p0_grid", job["p0_grid"],
            "--rho1_grid", job["rho1_grid"],
        ] + common
        print(f"\n===== START JOB {job['name']} =====", flush=True)
        run(cmd)
        print(f"===== END JOB {job['name']} =====\n", flush=True)


if __name__ == "__main__":
    main()
