#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "/home/zh/model/Meta-Llama-3-8B-Instruct"
TASK = "narrativeqa"
TARGET_BUDGET = 0.70
SPLIT_FILE = ROOT / "solver_runs" / "narrative_joint_search1_per_task_indices" / "narrativeqa_calibration_indices.json"


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


def run_eval_if_needed(tag: str, extra_args: list[str]) -> Path:
    if "selector_mode" in " ".join(extra_args):
        out_dir = ROOT / "solver_runs" / f"{Path(MODEL_PATH).name}_8192_diff_sparse_kv_{TARGET_BUDGET:.2f}_{tag}"
    else:
        out_dir = ROOT / "solver_runs" / f"{Path(MODEL_PATH).name}_8192_uniform_{TARGET_BUDGET:.2f}_{tag}"
    if eval_result_path(out_dir).exists():
        print(f"[reuse] {out_dir}", flush=True)
        return out_dir

    wait_for_gpu()
    cmd = [
        sys.executable, "eval_diff_sparse_kv_longbench.py",
        "--model_path", MODEL_PATH,
        "--max_length", "8192",
        "--datasets", TASK,
        "--output_dir", str(ROOT / "solver_runs"),
        "--kv_sparsity", f"{TARGET_BUDGET:.4f}",
        "--sample_indices_file", str(SPLIT_FILE),
        "--output_tag", tag,
    ] + extra_args
    run(cmd)
    run([sys.executable, "eval_results.py", "--result_dir", str(out_dir)])
    return out_dir


def main() -> None:
    # 1) true uniform baseline
    uniform_tag = "snapkv_compare_uniform_calib"
    uniform_dir = run_eval_if_needed(uniform_tag, ["--sparsity_type", "uniform"])
    uniform_res = json.load(eval_result_path(uniform_dir).open("r", encoding="utf-8"))

    # 2) uniform-like diff_sparse fallback for reference
    diff_uniform_like_tag = "snapkv_compare_diff_uniformlike_calib"
    diff_uniform_like_dir = run_eval_if_needed(
        diff_uniform_like_tag,
        [
            "--sparsity_type", "diff_sparse_kv",
            "--target_distribution", "0.0,1.0,0.0",
            "--sparsity_levels", "0.0,0.7,1.0",
            "--importance_mode", "attention_only",
            "--head_aggregation_mode", "mean",
            "--value_sink_keep", "0",
            "--selector_mode", "diffsparse",
            "--level_2_mode", "evict",
        ],
    )
    diff_uniform_like_res = json.load(eval_result_path(diff_uniform_like_dir).open("r", encoding="utf-8"))

    # 3) minimal SnapKV-style runs
    snapkv_variants = [
        ("snapkv_mean", "mean"),
        ("snapkv_max", "max"),
        ("snapkv_top2_mean", "top2_mean"),
        ("snapkv_hybrid", "hybrid"),
    ]
    results = {}
    for name, head_mode in snapkv_variants:
        tag = f"snapkv_compare_{name}_calib"
        out_dir = run_eval_if_needed(
            tag,
            [
                "--sparsity_type", "diff_sparse_kv",
                "--target_distribution", "0.0,1.0,0.0",
                "--sparsity_levels", "0.0,0.7,1.0",
                "--importance_mode", "attention_only",
                "--head_aggregation_mode", head_mode,
                "--value_sink_keep", "0",
                "--selector_mode", "snapkv",
                "--level_2_mode", "zero",
            ],
        )
        results[name] = {
            "dir": str(out_dir),
            "result": json.load(eval_result_path(out_dir).open("r", encoding="utf-8")),
        }

    summary = {
        "task": TASK,
        "split_file": str(SPLIT_FILE),
        "target_budget": TARGET_BUDGET,
        "uniform": {"dir": str(uniform_dir), "result": uniform_res},
        "diff_uniform_like": {"dir": str(diff_uniform_like_dir), "result": diff_uniform_like_res},
        "snapkv_variants": results,
    }
    out_json = ROOT / "solver_runs" / "snapkv_narrative_compare_summary.json"
    out_md = ROOT / "solver_runs" / "snapkv_narrative_compare_summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# SnapKV-style NarrativeQA Compare")
    lines.append("")
    lines.append(f"- task: `{TASK}`")
    lines.append(f"- split: `{SPLIT_FILE}`")
    lines.append(f"- target budget: `{TARGET_BUDGET}`")
    lines.append("")
    lines.append("| Method | Score | Notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| `uniform` | {uniform_res['average']:.2f} | MUSTAFAR uniform baseline |")
    lines.append(f"| `diff_uniform_like` | {diff_uniform_like_res['average']:.2f} | diff path with `[0,1,0] + rho1=0.7` |")
    for name, item in results.items():
        lines.append(f"| `{name}` | {item['result']['average']:.2f} | `selector_mode=snapkv` |")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out_json}")
    print(f"[done] {out_md}")


if __name__ == "__main__":
    main()
