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
    cmd = [
        PY,
        "eval_qwen_longbench_baseline.py",
        "--model_path", "/home/zh/model/Qwen2.5-7B",
        "--max_length", "8192",
        "--datasets",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "trec",
        "lcc",
        "--output_dir", "solver_runs_qwen_native_focus",
        "--output_subdir", "Qwen2.5-7B_8192_native_focus_baseline",
        "--run_eval",
    ]
    run(cmd)


if __name__ == "__main__":
    main()
