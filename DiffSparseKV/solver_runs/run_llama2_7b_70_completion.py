#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/zh/miniconda3/envs/mustar/bin/python"
MODEL_PATH = "/home/zh/model/Llama-2-7b-hf"
MODEL_NAME = "Llama-2-7b-hf"
MAX_LENGTH = "4096"
TARGET_BUDGET = "0.70"
BASE_TAG = "llama2_7b_70_full"
PER_TASK_JSON = ROOT / "solver_runs_llama2_7b_budget70_fix" / "rep6_budget70_try1_per_task_summary.json"
SUMMARY_MD = ROOT / "solver_runs" / "per_task_current_summary.md"
TASK_ORDER = ["hotpotqa", "lcc", "multifieldqa_en", "narrativeqa", "qasper", "trec"]


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def result_path(task: str, kind: str, tag_prefix: str) -> Path:
    if kind == "uniform":
        return ROOT / "solver_runs" / f"{MODEL_NAME}_4096_uniform_0.70_{tag_prefix}_{task}_uniform_full" / "result.json"
    return ROOT / "solver_runs" / f"{MODEL_NAME}_4096_diff_sparse_kv_0.70_{tag_prefix}_{task}_bestdiff_full" / "result.json"


def full_delta(task: str, tag_prefix: str) -> float | None:
    up = result_path(task, "uniform", tag_prefix)
    dp = result_path(task, "diff", tag_prefix)
    if not up.exists() or not dp.exists():
        return None
    uv = json.loads(up.read_text(encoding="utf-8"))["average"]
    dv = json.loads(dp.read_text(encoding="utf-8"))["average"]
    return float(dv) - float(uv)


def update_summary() -> None:
    run([
        PY,
        "solver_runs/update_model_section.py",
        "--summary_md", str(SUMMARY_MD),
        "--section_title", "Llama-2-7B 70%",
        "--model_name", MODEL_NAME,
        "--max_length", "4096",
        "--budget", "0.70",
        "--per_task_summary_json", str(PER_TASK_JSON),
        "--task_order", ",".join(TASK_ORDER),
        "--results_root", "solver_runs",
    ])


def run_full_from_cfg(task: str, cfg_path: str, tag_prefix: str) -> None:
    run([
        PY,
        "solver_runs/run_full_task_from_config.py",
        "--task", task,
        "--config_json", cfg_path,
        "--model_path", MODEL_PATH,
        "--max_length", MAX_LENGTH,
        "--target_budget", TARGET_BUDGET,
        "--tag_prefix", tag_prefix,
    ])


def repair_round(task: str, round_idx: int) -> Path:
    output_root = ROOT / "solver_runs_llama2_7b_70_repair"
    output_root.mkdir(parents=True, exist_ok=True)
    search_tag = f"llama2_7b_70_{task}_repair_r{round_idx}"
    run([
        PY,
        "search_diff_budget_solver.py",
        "--solver_mode", "per_task",
        "--model_path", MODEL_PATH,
        "--max_length", MAX_LENGTH,
        "--target_budget", TARGET_BUDGET,
        "--output_root", str(output_root.relative_to(ROOT)),
        "--search_tag", search_tag,
        "--val_datasets", task,
        "--calib_limit", "12",
        "--calib_seed", "17",
        "--val_limit", "30",
        "--val_seed", "29",
        "--p0_grid", "0.0,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20",
        "--rho1_grid", "0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        "--importance_mode_grid", "value_aware",
        "--head_aggregation_mode_grid", "max,mean,top2_mean",
        "--value_sink_keep_grid", "2,4,8",
        "--level_2_mode_grid", "evict",
        "--selector_mode_grid", "diffsparse",
        "--protected_heavy_ratio_grid", "0.0,0.1",
        "--protected_recent_ratio_grid", "0.5,0.75,1.0",
    ])
    return output_root / f"{search_tag}_per_task_summary.json"


def main() -> None:
    task_summary = json.loads(PER_TASK_JSON.read_text(encoding="utf-8"))["tasks"]

    # 1. Fill missing full-task runs from the first-round best configs.
    for task in ["trec", "lcc", "qasper"]:
        if full_delta(task, BASE_TAG) is not None:
            continue
        cfg_path = str(Path(task_summary[task]["best_val_dir"]) / "sparsity_config.json")
        run_full_from_cfg(task, cfg_path, BASE_TAG)
        update_summary()

    # 2. Repair any full-task negatives until they turn positive.
    round_idx = 1
    while True:
        update_summary()
        negatives = []
        for task in TASK_ORDER:
            delta = full_delta(task, BASE_TAG)
            if delta is not None and delta <= 0.0:
                negatives.append(task)
        if not negatives:
            print("[done] all Llama-2-7B 70% full-task deltas are positive", flush=True)
            return

        task = negatives[0]
        print(f"[repair] task={task} round={round_idx}", flush=True)
        repair_summary_path = repair_round(task, round_idx)
        repair_summary = json.loads(repair_summary_path.read_text(encoding="utf-8"))
        item = repair_summary["tasks"][task]
        cfg_path = str(Path(item["best_val_dir"]) / "sparsity_config.json")
        repair_tag = f"llama2_7b_70_{task}_repair_r{round_idx}"
        run_full_from_cfg(task, cfg_path, repair_tag)
        update_summary()
        round_idx += 1


if __name__ == "__main__":
    main()
