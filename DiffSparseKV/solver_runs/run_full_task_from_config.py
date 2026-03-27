#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_JSON = ROOT / "solver_runs" / "per_task_results_manifest.json"
MANIFEST_MD = ROOT / "solver_runs" / "per_task_results_manifest.md"
CURRENT_MD = ROOT / "solver_runs" / "per_task_current_summary.md"


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def gpu_free_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


def wait_for_gpu(min_free_mb: int = 30000, poll_sec: int = 60) -> None:
    while True:
        free_mb = gpu_free_mb()
        print(f"[gpu] free={free_mb:.0f}MB required={min_free_mb}MB", flush=True)
        if free_mb >= min_free_mb:
            return
        time.sleep(poll_sec)


def eval_result_path(output_dir: Path) -> Path:
    return output_dir / "result.json"


def run_with_retries(cmd: list[str], out_dir: Path, retries: int = 3) -> Path:
    if eval_result_path(out_dir).exists():
        print(f"[reuse] {out_dir}", flush=True)
        return out_dir
    for attempt in range(retries):
        wait_for_gpu()
        try:
            run(cmd)
            run([sys.executable, "eval_results.py", "--result_dir", str(out_dir)])
            return out_dir
        except subprocess.CalledProcessError as e:
            print(f"[warn] attempt={attempt+1} failed for {out_dir}: {e}", flush=True)
            time.sleep(60)
    raise RuntimeError(f"failed after retries: {out_dir}")


def write_markdowns(manifest: dict) -> None:
    lines = []
    lines.append("# Per-Task Results Manifest")
    lines.append("")
    lines.append(f"- Model: `{manifest.get('model', '')}`")
    lines.append(f"- Target budget: `{manifest.get('target_budget', '')}`")
    lines.append("")
    lines.append("| Task | Status | Val Uniform | Val Diff | Val Delta | Full Uniform | Full Diff | Full Delta | Best target distribution | Best sparsity levels |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    tasks = manifest.get("tasks", {})
    for task in sorted(tasks):
        item = tasks[task]
        full_u = item.get("full_uniform_score")
        full_d = item.get("full_diff_score")
        full_delta = item.get("full_delta")
        lines.append(
            f"| `{task}` | {item.get('status','')} | "
            f"{item.get('uniform_validation','')} | {item.get('best_diff_validation','')} | {item.get('validation_delta','')} | "
            f"{'' if full_u is None else full_u} | {'' if full_d is None else full_d} | {'' if full_delta is None else full_delta} | "
            f"`{item.get('best_config',{}).get('target_distribution','')}` | "
            f"`{item.get('best_config',{}).get('sparsity_levels','')}` |"
        )
    MANIFEST_MD.write_text("\n".join(lines), encoding="utf-8")

    short = []
    short.append("# Current Per-Task Solver Summary")
    short.append("")
    short.append("This file tracks the latest finished `per-task` runs and full-dataset follow-ups.")
    short.append("")
    short.append("## Finished tasks")
    short.append("")
    short.append("| Task | Val Delta | Full Delta | Status |")
    short.append("|---|---:|---:|---|")
    for task in sorted(tasks):
        item = tasks[task]
        short.append(
            f"| `{task}` | {item.get('validation_delta','')} | "
            f"{'' if item.get('full_delta') is None else item.get('full_delta')} | {item.get('status','')} |"
        )
    CURRENT_MD.write_text("\n".join(short), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config_json", required=True)
    ap.add_argument("--model_path", default="/home/zh/model/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--target_budget", type=float, default=0.70)
    ap.add_argument("--tag_prefix", required=True)
    ap.add_argument("--manifest_task", default="")
    ap.add_argument("--update_manifest", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config_json, "r", encoding="utf-8"))
    task = args.task
    model_name = Path(args.model_path).name

    uniform_tag = f"{args.tag_prefix}_{task}_uniform_full"
    uniform_dir = ROOT / "solver_runs" / f"{model_name}_{args.max_length}_uniform_{args.target_budget:.2f}_{uniform_tag}"
    uniform_cmd = [
        sys.executable, "eval_diff_sparse_kv_longbench.py",
        "--model_path", args.model_path,
        "--max_length", str(args.max_length),
        "--datasets", task,
        "--output_dir", str(ROOT / "solver_runs"),
        "--kv_sparsity", f"{args.target_budget:.4f}",
        "--sparsity_type", "uniform",
        "--output_tag", uniform_tag,
    ]
    uniform_dir = run_with_retries(uniform_cmd, uniform_dir)

    diff_tag = f"{args.tag_prefix}_{task}_bestdiff_full"
    diff_dir = ROOT / "solver_runs" / f"{model_name}_{args.max_length}_diff_sparse_kv_{args.target_budget:.2f}_{diff_tag}"
    diff_cmd = [
        sys.executable, "eval_diff_sparse_kv_longbench.py",
        "--model_path", args.model_path,
        "--max_length", str(args.max_length),
        "--datasets", task,
        "--output_dir", str(ROOT / "solver_runs"),
        "--kv_sparsity", f"{args.target_budget:.4f}",
        "--sparsity_type", "diff_sparse_kv",
        "--target_distribution", ",".join(str(x) for x in cfg["target_distribution"]),
        "--sparsity_levels", ",".join(str(x) for x in cfg["sparsity_levels"]),
        "--importance_mode", str(cfg.get("importance_mode", "value_aware")),
        "--head_aggregation_mode", str(cfg.get("head_aggregation_mode", "max")),
        "--value_sink_keep", str(int(cfg.get("value_sink_keep", 2))),
        "--head_aggregation_alpha", str(float(cfg.get("head_aggregation_alpha", 0.5))),
        "--level_2_mode", str(cfg.get("level_2_mode", "evict")),
        "--selector_mode", str(cfg.get("selector_mode", "diffsparse")),
        "--protected_heavy_ratio", str(float(cfg.get("protected_heavy_ratio", 0.0))),
        "--output_tag", diff_tag,
    ]
    diff_dir = run_with_retries(diff_cmd, diff_dir)

    uniform_res = json.load(eval_result_path(uniform_dir).open("r", encoding="utf-8"))
    diff_res = json.load(eval_result_path(diff_dir).open("r", encoding="utf-8"))
    uniform_score = float(uniform_res["average"])
    diff_score = float(diff_res["average"])
    delta = diff_score - uniform_score
    print(f"[done] {task}: full uniform={uniform_score:.2f}, full diff={diff_score:.2f}, delta={delta:+.2f}", flush=True)

    if args.update_manifest and MANIFEST_JSON.exists():
        manifest = json.load(MANIFEST_JSON.open("r", encoding="utf-8"))
        key = args.manifest_task or task
        task_entry = manifest.setdefault("tasks", {}).setdefault(key, {})
        task_entry["full_uniform_dir"] = str(uniform_dir)
        task_entry["full_diff_dir"] = str(diff_dir)
        task_entry["full_uniform_score"] = uniform_score
        task_entry["full_diff_score"] = diff_score
        task_entry["full_delta"] = delta
        if "best_config" not in task_entry:
            task_entry["best_config"] = {
                "target_distribution": cfg["target_distribution"],
                "sparsity_levels": cfg["sparsity_levels"],
            }
        MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdowns(manifest)


if __name__ == "__main__":
    main()
