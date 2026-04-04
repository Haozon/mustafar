#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    output_root = "solver_runs_mistral_qasper_wide70"
    Path(ROOT / output_root).mkdir(parents=True, exist_ok=True)

    cmd = [
        PY,
        "search_diff_budget_solver.py",
        "--solver_mode", "per_task",
        "--model_path", "/home/zh/model/Mistral-7B-v0.1",
        "--max_length", "8192",
        "--target_budget", "0.70",
        "--output_root", output_root,
        "--search_tag", "mistral_qasper_wide70_r1",
        "--val_datasets", "qasper",
        "--calib_limit", "12",
        "--calib_seed", "17",
        "--val_limit", "30",
        "--val_seed", "29",
        # Wider 3-level family search than the original 22-candidate pass.
        "--p0_grid", "0.0,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20",
        "--rho1_grid", "0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        # Keep the sweep broad but still tractable for a single-task search.
        "--importance_mode_grid", "value_aware",
        "--head_aggregation_mode_grid", "max,mean,top2_mean",
        "--value_sink_keep_grid", "2,4,8",
        "--level_2_mode_grid", "evict",
        "--selector_mode_grid", "diffsparse",
        "--protected_heavy_ratio_grid", "0.0,0.1",
        "--protected_recent_ratio_grid", "0.5,0.75,1.0",
    ]
    run(cmd)


if __name__ == "__main__":
    main()
