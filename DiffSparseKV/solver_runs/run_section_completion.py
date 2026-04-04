#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/mnt/home/zh/mustafar/DiffSparseKV")
PY = sys.executable


def normalize_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)
    try:
        relative = path.relative_to(LEGACY_ROOT)
    except ValueError:
        return str(path)
    return str(ROOT / relative)


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def result_path(model_name: str, max_length: int, budget: float, task: str, kind: str, tag_prefix: str) -> Path:
    if kind == "uniform":
        return ROOT / "solver_runs" / f"{model_name}_{max_length}_uniform_{budget:.2f}_{tag_prefix}_{task}_uniform_full" / "result.json"
    return ROOT / "solver_runs" / f"{model_name}_{max_length}_diff_sparse_kv_{budget:.2f}_{tag_prefix}_{task}_bestdiff_full" / "result.json"


def full_delta(model_name: str, max_length: int, budget: float, task: str, tag_prefix: str) -> float | None:
    up = result_path(model_name, max_length, budget, task, "uniform", tag_prefix)
    dp = result_path(model_name, max_length, budget, task, "diff", tag_prefix)
    if not up.exists() or not dp.exists():
        return None
    uv = json.loads(up.read_text(encoding="utf-8"))["average"]
    dv = json.loads(dp.read_text(encoding="utf-8"))["average"]
    return float(dv) - float(uv)


def update_summary(section_title: str, model_name: str, max_length: int, budget: float, per_task_json: Path, task_order: list[str]) -> None:
    run([
        PY,
        "solver_runs/update_model_section.py",
        "--summary_md", "solver_runs/per_task_current_summary.md",
        "--section_title", section_title,
        "--model_name", model_name,
        "--max_length", str(max_length),
        "--budget", f"{budget:.2f}",
        "--per_task_summary_json", str(per_task_json),
        "--task_order", ",".join(task_order),
        "--results_root", "solver_runs",
    ])


def run_full_from_cfg(task: str, cfg_path: str, model_path: str, max_length: int, budget: float, tag_prefix: str) -> None:
    cfg_path = normalize_repo_path(cfg_path)
    run([
        PY,
        "solver_runs/run_full_task_from_config.py",
        "--task", task,
        "--config_json", cfg_path,
        "--model_path", model_path,
        "--max_length", str(max_length),
        "--target_budget", f"{budget:.2f}",
        "--tag_prefix", tag_prefix,
    ])


def repair_round(
    *,
    task: str,
    round_idx: int,
    repair_output_root: Path,
    model_path: str,
    max_length: int,
    budget: float,
    calib_limit: int,
    val_limit: int,
) -> Path:
    repair_output_root = repair_output_root.resolve()
    repair_output_root.mkdir(parents=True, exist_ok=True)
    search_tag = f"{Path(model_path).name}_{int(round(budget*100))}_{task}_repair_r{round_idx}".replace("-", "_")
    run([
        PY,
        "search_diff_budget_solver.py",
        "--solver_mode", "per_task",
        "--model_path", model_path,
        "--max_length", str(max_length),
        "--target_budget", f"{budget:.2f}",
        "--output_root", str(repair_output_root.relative_to(ROOT)),
        "--search_tag", search_tag,
        "--val_datasets", task,
        "--calib_limit", str(calib_limit),
        "--calib_seed", "17",
        "--val_limit", str(val_limit),
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
    return repair_output_root / f"{search_tag}_per_task_summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section_title", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--max_length", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--per_task_summary_json", required=True)
    ap.add_argument("--task_order", required=True)
    ap.add_argument("--base_tag", required=True)
    ap.add_argument("--repair_output_root", required=True)
    ap.add_argument("--repair_calib_limit", type=int, default=12)
    ap.add_argument("--repair_val_limit", type=int, default=30)
    args = ap.parse_args()

    model_name = Path(args.model_path).name
    per_task_json = Path(args.per_task_summary_json)
    task_order = [x.strip() for x in args.task_order.split(",") if x.strip()]
    task_summary = json.loads(per_task_json.read_text(encoding="utf-8"))["tasks"]

    # Fill missing full-task runs from first-round per-task best configs.
    for task in task_order:
        if full_delta(model_name, args.max_length, args.budget, task, args.base_tag) is not None:
            continue
        cfg_path = str(Path(normalize_repo_path(task_summary[task]["best_val_dir"])) / "sparsity_config.json")
        run_full_from_cfg(task, cfg_path, args.model_path, args.max_length, args.budget, args.base_tag)
        update_summary(args.section_title, model_name, args.max_length, args.budget, per_task_json, task_order)

    round_idx = 1
    while True:
        update_summary(args.section_title, model_name, args.max_length, args.budget, per_task_json, task_order)
        negatives = []
        for task in task_order:
            delta = full_delta(model_name, args.max_length, args.budget, task, args.base_tag)
            if delta is not None and delta <= 0.0:
                negatives.append((delta, task))
        if not negatives:
            print(f"[done] section complete: {args.section_title}", flush=True)
            return

        negatives.sort()
        _, task = negatives[0]
        print(f"[repair] section={args.section_title} task={task} round={round_idx}", flush=True)
        repair_summary_path = repair_round(
            task=task,
            round_idx=round_idx,
            repair_output_root=Path(args.repair_output_root),
            model_path=args.model_path,
            max_length=args.max_length,
            budget=args.budget,
            calib_limit=args.repair_calib_limit,
            val_limit=args.repair_val_limit,
        )
        repair_summary = json.loads(repair_summary_path.read_text(encoding="utf-8"))
        item = repair_summary["tasks"][task]
        cfg_path = str(Path(normalize_repo_path(item["best_val_dir"])) / "sparsity_config.json")
        repair_tag = f"{args.base_tag}_{task}_repair_r{round_idx}"
        run_full_from_cfg(task, cfg_path, args.model_path, args.max_length, args.budget, repair_tag)
        update_summary(args.section_title, model_name, args.max_length, args.budget, per_task_json, task_order)
        round_idx += 1


if __name__ == "__main__":
    main()
