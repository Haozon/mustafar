#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/zh/miniconda3/envs/mustar/bin/python"
OUTPUT_ROOT = ROOT / "solver_runs_meta50_negative_repair"
SEARCH_TAG = "meta50_negfix_r1"
MODEL_PATH = "/home/zh/model/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = "8192"
TARGET_BUDGET = "0.50"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    run([
        PY,
        "search_diff_budget_solver.py",
        "--solver_mode", "per_task",
        "--model_path", MODEL_PATH,
        "--max_length", MAX_LENGTH,
        "--target_budget", TARGET_BUDGET,
        "--output_root", str(OUTPUT_ROOT.relative_to(ROOT)),
        "--search_tag", SEARCH_TAG,
        "--val_datasets", "hotpotqa", "lcc",
        "--calib_limit", "16",
        "--calib_seed", "17",
        "--val_limit", "30",
        "--val_seed", "29",
        "--p0_grid", "0.0,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20",
        "--rho1_grid", "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        "--importance_mode_grid", "value_aware",
        "--head_aggregation_mode_grid", "max,mean,top2_mean",
        "--value_sink_keep_grid", "2,4",
        "--level_2_mode_grid", "evict",
        "--selector_mode_grid", "diffsparse",
        "--protected_heavy_ratio_grid", "0.0",
        "--protected_recent_ratio_grid", "0.75,1.0",
    ])

    summary_path = OUTPUT_ROOT / f"{SEARCH_TAG}_per_task_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for task in ("hotpotqa", "lcc"):
        item = summary["tasks"][task]
        if float(item["val_delta"]) <= 0.0:
            print(f"[skip-full] task={task} val_delta={item['val_delta']}", flush=True)
            continue
        run([
            PY,
            "solver_runs/run_full_task_from_config.py",
            "--task", task,
            "--config_json", str(Path(item["best_val_dir"]) / "sparsity_config.json"),
            "--model_path", MODEL_PATH,
            "--max_length", MAX_LENGTH,
            "--target_budget", TARGET_BUDGET,
            "--tag_prefix", "meta50_negfix_r1_full",
        ])


if __name__ == "__main__":
    main()
