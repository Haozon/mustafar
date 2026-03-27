#!/usr/bin/env python3
"""
Lightweight budget solver for DiffSparseKV.

Workflow:
1. Run a uniform baseline on a small calibration split.
2. Search feasible differential-sparsity configurations under a fixed target budget.
3. Pick the best configuration on calibration.
4. Validate the best configuration on a larger validation split.

This script intentionally keeps the solver simple:
it searches over two degrees of freedom, `p0` and `rho1`, and solves `p1/p2`
analytically to satisfy the target average sparsity budget.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset


ROOT = Path(__file__).resolve().parent
CONFIG_TOL = 1e-4


@dataclass
class CandidateConfig:
    p0: float
    p1: float
    p2: float
    rho1: float
    target_budget: float

    @property
    def target_distribution(self) -> List[float]:
        return [self.p0, self.p1, self.p2]

    @property
    def sparsity_levels(self) -> List[float]:
        return [0.0, self.rho1, 1.0]

    @property
    def expected_budget(self) -> float:
        return self.p0 * 0.0 + self.p1 * self.rho1 + self.p2 * 1.0

    @property
    def short_name(self) -> str:
        return (
            f"p0_{self.p0:.2f}_p1_{self.p1:.2f}_p2_{self.p2:.2f}_rho1_{self.rho1:.4f}"
            .replace(".", "p")
        )


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def generate_candidates(
    target_budget: float,
    p0_grid: Iterable[float],
    rho1_grid: Iterable[float],
) -> List[CandidateConfig]:
    candidates: List[CandidateConfig] = []
    for p0 in p0_grid:
        for rho1 in rho1_grid:
            if rho1 >= 1.0:
                continue
            numerator = 1.0 - p0 - target_budget
            denominator = 1.0 - rho1
            if denominator <= 0:
                continue
            p1 = numerator / denominator
            p2 = 1.0 - p0 - p1
            if min(p0, p1, p2) < -1e-9:
                continue
            if max(p0, p1, p2) > 1.0 + 1e-9:
                continue
            cfg = CandidateConfig(
                p0=round(max(0.0, p0), 6),
                p1=round(max(0.0, p1), 6),
                p2=round(max(0.0, p2), 6),
                rho1=round(rho1, 6),
                target_budget=target_budget,
            )
            if abs(cfg.expected_budget - target_budget) > 1e-5:
                continue
            candidates.append(cfg)
    unique: Dict[str, CandidateConfig] = {}
    for cfg in candidates:
        unique[cfg.short_name] = cfg
    return list(unique.values())


def run_command(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def eval_result_path(output_dir: Path) -> Path:
    return output_dir / "result.json"


def load_result(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def same_list(a: List[float], b: List[float], tol: float = CONFIG_TOL) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def find_matching_candidate_dir(
    *,
    output_root: Path,
    model_path: str,
    max_length: int,
    search_tag: str,
    task: str,
    candidate: CandidateConfig,
    importance_mode: str,
    head_aggregation_mode: str,
    value_sink_keep: int,
    head_aggregation_alpha: float,
    level_2_mode: str,
    selector_mode: str,
    protected_heavy_ratio: float,
) -> Optional[Path]:
    model_name = Path(model_path).name
    pattern = f"{model_name}_{max_length}_diff_sparse_kv_0.70_{search_tag}_{task}_*"
    for path_str in glob.glob(str(output_root / pattern)):
        path = Path(path_str)
        cfg_file = path / "sparsity_config.json"
        result_file = path / "result.json"
        if not cfg_file.exists() or not result_file.exists():
            continue
        try:
            cfg = json.load(cfg_file.open("r", encoding="utf-8"))
        except Exception:
            continue
        if same_list(cfg.get("target_distribution", []), candidate.target_distribution) and same_list(
            cfg.get("sparsity_levels", []), candidate.sparsity_levels
        ) and cfg.get("importance_mode") == importance_mode and cfg.get(
            "head_aggregation_mode"
        ) == head_aggregation_mode and int(cfg.get("value_sink_keep", -999)) == int(value_sink_keep) and abs(
            float(cfg.get("head_aggregation_alpha", 0.5)) - float(head_aggregation_alpha)
        ) <= CONFIG_TOL and cfg.get("level_2_mode", "evict") == level_2_mode and cfg.get(
            "selector_mode", "diffsparse"
        ) == selector_mode and abs(
            float(cfg.get("protected_heavy_ratio", 0.0)) - float(protected_heavy_ratio)
        ) <= CONFIG_TOL:
            return path
    return None


def load_local_longbench_dataset(dataset_name: str) -> Dataset:
    cache_pattern = (
        f"/home/zh/.cache/huggingface/datasets/THUDM___long_bench/"
        f"{dataset_name}/1.0.0/*/long_bench-test.arrow"
    )
    matches = sorted(glob.glob(cache_pattern))
    if not matches:
        raise FileNotFoundError(f"No cached LongBench dataset found for {dataset_name}")
    return Dataset.from_file(matches[-1])


def build_disjoint_split_indices(
    dataset_name: str,
    calib_limit: int,
    calib_seed: int,
    val_limit: int,
    val_seed: int,
) -> Dict[str, List[int]]:
    dataset = load_local_longbench_dataset(dataset_name)
    n = len(dataset)
    all_indices = list(range(n))

    rng_calib = random.Random(calib_seed)
    rng_val = random.Random(val_seed)

    calib_count = min(calib_limit, n)
    calib_indices = sorted(rng_calib.sample(all_indices, calib_count))

    remaining = [idx for idx in all_indices if idx not in set(calib_indices)]
    val_count = min(val_limit, len(remaining))
    val_indices = sorted(rng_val.sample(remaining, val_count))
    return {
        "calibration": calib_indices,
        "validation": val_indices,
    }


def run_eval(
    *,
    model_path: str,
    max_length: int,
    datasets: List[str],
    limit: int,
    sample_seed: int,
    sample_indices_file: Optional[Path],
    output_root: Path,
    output_tag: str,
    sparsity_type: str,
    kv_sparsity: float,
    candidate: Optional[CandidateConfig] = None,
    importance_mode: str = "value_aware",
    head_aggregation_mode: str = "max",
    value_sink_keep: int = 2,
    head_aggregation_alpha: float = 0.5,
    level_2_mode: str = "evict",
    selector_mode: str = "diffsparse",
    protected_heavy_ratio: float = 0.0,
) -> Optional[Path]:
    args = [
        sys.executable,
        "eval_diff_sparse_kv_longbench.py",
        "--model_path", model_path,
        "--max_length", str(max_length),
        "--datasets", *datasets,
        "--output_dir", str(output_root),
        "--kv_sparsity", f"{kv_sparsity:.4f}",
        "--sparsity_type", sparsity_type,
        "--output_tag", output_tag,
    ]

    if sample_indices_file is not None:
        args += ["--sample_indices_file", str(sample_indices_file)]
    else:
        args += ["--limit", str(limit), "--sample_seed", str(sample_seed)]

    if sparsity_type == "diff_sparse_kv" and candidate is not None:
        args += [
            "--target_distribution",
            ",".join(f"{x:.6f}" for x in candidate.target_distribution),
            "--sparsity_levels",
            ",".join(f"{x:.6f}" for x in candidate.sparsity_levels),
            "--importance_mode", importance_mode,
            "--head_aggregation_mode", head_aggregation_mode,
            "--value_sink_keep", str(value_sink_keep),
            "--head_aggregation_alpha", f"{head_aggregation_alpha:.4f}",
            "--level_2_mode", level_2_mode,
            "--selector_mode", selector_mode,
            "--protected_heavy_ratio", f"{protected_heavy_ratio:.4f}",
        ]

    if sparsity_type == "uniform":
        subdir = f"{Path(model_path).name}_{max_length}_uniform_{kv_sparsity:.2f}_{output_tag}"
    elif sparsity_type == "none":
        subdir = f"{Path(model_path).name}_{max_length}_baseline_{output_tag}"
    else:
        subdir = f"{Path(model_path).name}_{max_length}_diff_sparse_kv_{kv_sparsity:.2f}_{output_tag}"

    # When diff_sparse uses custom levels, the script still names by kv_sparsity.
    if sparsity_type == "diff_sparse_kv":
        subdir = f"{Path(model_path).name}_{max_length}_diff_sparse_kv_{kv_sparsity:.2f}_{output_tag}"

    output_path = output_root / subdir
    if (output_path / "result.json").exists():
        print(f"[reuse] {output_path}")
        return output_path

    try:
        run_command(args)
        run_command([sys.executable, "eval_results.py", "--result_dir", str(output_path)])
    except subprocess.CalledProcessError as e:
        print(f"[warn] evaluation failed for {output_path}: {e}")
        return None
    if not (output_path / "result.json").exists():
        print(f"[warn] missing result.json for {output_path}, skipping")
        return None
    return output_path


def run_diff_eval_with_custom_dir(
    *,
    model_path: str,
    max_length: int,
    datasets: List[str],
    limit: int,
    sample_seed: int,
    sample_indices_file: Optional[Path],
    output_root: Path,
    output_tag: str,
    kv_sparsity: float,
    candidate: CandidateConfig,
    importance_mode: str,
    head_aggregation_mode: str,
    value_sink_keep: int,
    head_aggregation_alpha: float,
    level_2_mode: str,
    selector_mode: str,
    protected_heavy_ratio: float,
) -> Optional[Path]:
    existing = None
    if "cand" in output_tag:
        # Reuse same-task same-config candidate result if already available.
        task = datasets[0] if len(datasets) == 1 else ""
        if task:
            existing = find_matching_candidate_dir(
                output_root=output_root,
                model_path=model_path,
                max_length=max_length,
                search_tag=output_tag.split(f"_{task}_")[0],
                task=task,
                candidate=candidate,
                importance_mode=importance_mode,
                head_aggregation_mode=head_aggregation_mode,
                value_sink_keep=value_sink_keep,
                head_aggregation_alpha=head_aggregation_alpha,
                level_2_mode=level_2_mode,
                selector_mode=selector_mode,
                protected_heavy_ratio=protected_heavy_ratio,
            )
    if existing is not None:
        print(f"[reuse-candidate] {existing}")
        return existing

    args = [
        sys.executable,
        "eval_diff_sparse_kv_longbench.py",
        "--model_path", model_path,
        "--max_length", str(max_length),
        "--datasets", *datasets,
        "--output_dir", str(output_root),
        "--kv_sparsity", f"{kv_sparsity:.4f}",
        "--sparsity_type", "diff_sparse_kv",
        "--target_distribution", ",".join(f"{x:.6f}" for x in candidate.target_distribution),
        "--sparsity_levels", ",".join(f"{x:.6f}" for x in candidate.sparsity_levels),
        "--importance_mode", importance_mode,
        "--head_aggregation_mode", head_aggregation_mode,
        "--value_sink_keep", str(value_sink_keep),
        "--head_aggregation_alpha", f"{head_aggregation_alpha:.4f}",
        "--level_2_mode", level_2_mode,
        "--selector_mode", selector_mode,
        "--protected_heavy_ratio", f"{protected_heavy_ratio:.4f}",
        "--output_tag", output_tag,
    ]
    if sample_indices_file is not None:
        args += ["--sample_indices_file", str(sample_indices_file)]
    else:
        args += ["--limit", str(limit), "--sample_seed", str(sample_seed)]
    output_path = output_root / f"{Path(model_path).name}_{max_length}_diff_sparse_kv_{kv_sparsity:.2f}_{output_tag}"
    if (output_path / "result.json").exists():
        print(f"[reuse] {output_path}")
        return output_path
    try:
        run_command(args)
        run_command([sys.executable, "eval_results.py", "--result_dir", str(output_path)])
    except subprocess.CalledProcessError as e:
        print(f"[warn] diff evaluation failed for {output_path}: {e}")
        return None
    if not (output_path / "result.json").exists():
        print(f"[warn] missing result.json for {output_path}, skipping")
        return None
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Search a lightweight DiffSparseKV budget solver.")
    parser.add_argument("--solver_mode", type=str, default="shared", choices=["shared", "per_task"])
    parser.add_argument("--model_path", type=str, default="/home/zh/model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--target_budget", type=float, default=0.70)
    parser.add_argument("--calib_datasets", nargs="+", default=["narrativeqa", "qasper", "multifieldqa_en"])
    parser.add_argument("--calib_limit", type=int, default=10)
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--val_datasets", nargs="+", default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "trec", "lcc"])
    parser.add_argument("--val_limit", type=int, default=30)
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--p0_grid", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--rho1_grid", type=str, default="0.50,0.60,0.6667,0.7143,0.75")
    parser.add_argument("--importance_mode", type=str, default="value_aware")
    parser.add_argument("--head_aggregation_mode", type=str, default="max")
    parser.add_argument("--value_sink_keep", type=int, default=2)
    parser.add_argument("--importance_mode_grid", type=str, default="")
    parser.add_argument("--head_aggregation_mode_grid", type=str, default="")
    parser.add_argument("--value_sink_keep_grid", type=str, default="")
    parser.add_argument("--head_aggregation_alpha_grid", type=str, default="")
    parser.add_argument("--level_2_mode_grid", type=str, default="")
    parser.add_argument("--selector_mode_grid", type=str, default="")
    parser.add_argument("--protected_heavy_ratio_grid", type=str, default="")
    parser.add_argument("--output_root", type=str, default="solver_runs")
    parser.add_argument("--search_tag", type=str, default="solver_search")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    p0_grid = parse_float_list(args.p0_grid)
    rho1_grid = parse_float_list(args.rho1_grid)
    candidates = generate_candidates(args.target_budget, p0_grid, rho1_grid)
    if not candidates:
        raise RuntimeError("No feasible candidates generated.")
    importance_modes = parse_str_list(args.importance_mode_grid) if args.importance_mode_grid else [args.importance_mode]
    head_modes = parse_str_list(args.head_aggregation_mode_grid) if args.head_aggregation_mode_grid else [args.head_aggregation_mode]
    sink_values = parse_int_list(args.value_sink_keep_grid) if args.value_sink_keep_grid else [args.value_sink_keep]
    head_alphas = parse_float_list(args.head_aggregation_alpha_grid) if args.head_aggregation_alpha_grid else [0.5]
    level_2_modes = parse_str_list(args.level_2_mode_grid) if args.level_2_mode_grid else ["evict"]
    selector_modes = parse_str_list(args.selector_mode_grid) if args.selector_mode_grid else ["diffsparse"]
    protected_heavy_ratios = parse_float_list(args.protected_heavy_ratio_grid) if args.protected_heavy_ratio_grid else [0.0]
    search_settings = list(product(
        importance_modes,
        head_modes,
        sink_values,
        head_alphas,
        level_2_modes,
        selector_modes,
        protected_heavy_ratios,
    ))

    print(f"[info] generated {len(candidates)} feasible budget candidates and {len(search_settings)} importance settings")

    if args.solver_mode == "per_task":
        per_task_root = output_root / f"{args.search_tag}_per_task_indices"
        per_task_root.mkdir(parents=True, exist_ok=True)
        task_rows = []
        task_summaries = {}
        task_list = list(dict.fromkeys(args.val_datasets))
        for task in task_list:
            print(f"[per-task] task={task}")
            split = build_disjoint_split_indices(
                dataset_name=task,
                calib_limit=args.calib_limit,
                calib_seed=args.calib_seed,
                val_limit=args.val_limit,
                val_seed=args.val_seed,
            )
            calib_idx_file = per_task_root / f"{task}_calibration_indices.json"
            val_idx_file = per_task_root / f"{task}_validation_indices.json"
            if calib_idx_file.exists():
                split["calibration"] = json.load(calib_idx_file.open("r", encoding="utf-8"))
            else:
                calib_idx_file.write_text(json.dumps(split["calibration"]), encoding="utf-8")
            if val_idx_file.exists():
                split["validation"] = json.load(val_idx_file.open("r", encoding="utf-8"))
            else:
                val_idx_file.write_text(json.dumps(split["validation"]), encoding="utf-8")

            uniform_tag = f"{args.search_tag}_{task}_uniform_calib"
            uniform_output = run_eval(
                model_path=args.model_path,
                max_length=args.max_length,
                datasets=[task],
                limit=args.calib_limit,
                sample_seed=args.calib_seed,
                sample_indices_file=calib_idx_file,
                output_root=output_root,
                output_tag=uniform_tag,
                sparsity_type="uniform",
                kv_sparsity=args.target_budget,
            )
            if uniform_output is None:
                raise RuntimeError(f"Uniform calibration failed for {task}")
            uniform_result = load_result(eval_result_path(uniform_output))
            uniform_avg = uniform_result["average"]

            best_cfg: Optional[CandidateConfig] = None
            best_result: Optional[Dict[str, float]] = None
            best_dir: Optional[Path] = None
            best_setting = None
            total_combo = len(candidates) * len(search_settings)
            combo_idx = 0
            for idx, candidate in enumerate(candidates, start=1):
                for imp_mode, head_mode, sink_keep, head_alpha, level_2_mode, selector_mode, protected_heavy_ratio in search_settings:
                    combo_idx += 1
                    candidate_tag = (
                        f"{args.search_tag}_{task}_cand{idx}_{candidate.short_name}_"
                        f"imp_{imp_mode}_head_{head_mode}_sink_{sink_keep}_"
                        f"alpha_{head_alpha:.2f}_l2_{level_2_mode}_sel_{selector_mode}_phr_{protected_heavy_ratio:.2f}"
                    )
                    print(
                        f"[per-task-search] task={task} {combo_idx}/{total_combo} "
                        f"p0={candidate.p0:.2f} p1={candidate.p1:.2f} p2={candidate.p2:.2f} "
                        f"rho1={candidate.rho1:.4f} imp={imp_mode} head={head_mode} sink={sink_keep} "
                        f"alpha={head_alpha:.2f} l2={level_2_mode} sel={selector_mode} phr={protected_heavy_ratio:.2f}"
                    )
                    diff_output = run_diff_eval_with_custom_dir(
                        model_path=args.model_path,
                        max_length=args.max_length,
                        datasets=[task],
                        limit=args.calib_limit,
                        sample_seed=args.calib_seed,
                        sample_indices_file=calib_idx_file,
                        output_root=output_root,
                        output_tag=candidate_tag,
                        kv_sparsity=args.target_budget,
                        candidate=candidate,
                        importance_mode=imp_mode,
                        head_aggregation_mode=head_mode,
                        value_sink_keep=sink_keep,
                        head_aggregation_alpha=head_alpha,
                        level_2_mode=level_2_mode,
                        selector_mode=selector_mode,
                        protected_heavy_ratio=protected_heavy_ratio,
                    )
                    if diff_output is None:
                        continue
                    result = load_result(eval_result_path(diff_output))
                    if best_result is None or result["average"] > best_result["average"]:
                        best_cfg = candidate
                        best_result = result
                        best_dir = diff_output
                        best_setting = (
                            imp_mode, head_mode, sink_keep,
                            head_alpha, level_2_mode, selector_mode, protected_heavy_ratio,
                        )

            if best_cfg is None or best_result is None or best_dir is None or best_setting is None:
                raise RuntimeError(f"Per-task search failed for {task}")
            best_imp_mode, best_head_mode, best_sink_keep, best_head_alpha, best_level2_mode, best_selector_mode, best_protected_heavy_ratio = best_setting

            uniform_val_tag = f"{args.search_tag}_{task}_uniform_val"
            uniform_val_output = run_eval(
                model_path=args.model_path,
                max_length=args.max_length,
                datasets=[task],
                limit=args.val_limit,
                sample_seed=args.val_seed,
                sample_indices_file=val_idx_file,
                output_root=output_root,
                output_tag=uniform_val_tag,
                sparsity_type="uniform",
                kv_sparsity=args.target_budget,
            )
            if uniform_val_output is None:
                raise RuntimeError(f"Uniform validation failed for {task}")
            uniform_val_result = load_result(eval_result_path(uniform_val_output))

            diff_val_tag = f"{args.search_tag}_{task}_bestdiff_val"
            diff_val_output = run_diff_eval_with_custom_dir(
                model_path=args.model_path,
                max_length=args.max_length,
                datasets=[task],
                limit=args.val_limit,
                sample_seed=args.val_seed,
                sample_indices_file=val_idx_file,
                output_root=output_root,
                output_tag=diff_val_tag,
                kv_sparsity=args.target_budget,
                candidate=best_cfg,
                importance_mode=best_imp_mode,
                head_aggregation_mode=best_head_mode,
                value_sink_keep=best_sink_keep,
                head_aggregation_alpha=best_head_alpha,
                level_2_mode=best_level2_mode,
                selector_mode=best_selector_mode,
                protected_heavy_ratio=best_protected_heavy_ratio,
            )
            if diff_val_output is None:
                raise RuntimeError(f"Best diff validation failed for {task}")
            diff_val_result = load_result(eval_result_path(diff_val_output))

            task_row = {
                "task": task,
                "uniform_calib": uniform_result["average"],
                "diff_calib": best_result["average"],
                "calib_delta": best_result["average"] - uniform_result["average"],
                "uniform_val": uniform_val_result["average"],
                "diff_val": diff_val_result["average"],
                "val_delta": diff_val_result["average"] - uniform_val_result["average"],
                "p0": best_cfg.p0,
                "p1": best_cfg.p1,
                "p2": best_cfg.p2,
                "rho1": best_cfg.rho1,
                "importance_mode": best_imp_mode,
                "head_aggregation_mode": best_head_mode,
                "value_sink_keep": best_sink_keep,
                "head_aggregation_alpha": best_head_alpha,
                "level_2_mode": best_level2_mode,
                "selector_mode": best_selector_mode,
                "protected_heavy_ratio": best_protected_heavy_ratio,
                "calib_indices_file": str(calib_idx_file),
                "val_indices_file": str(val_idx_file),
                "best_calib_dir": str(best_dir),
                "best_val_dir": str(diff_val_output),
            }
            task_rows.append(task_row)
            task_summaries[task] = task_row

        task_rows.sort(key=lambda x: x["val_delta"], reverse=True)
        per_task_csv = output_root / f"{args.search_tag}_per_task_results.csv"
        with per_task_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task", "uniform_calib", "diff_calib", "calib_delta",
                    "uniform_val", "diff_val", "val_delta",
                    "p0", "p1", "p2", "rho1",
                    "importance_mode", "head_aggregation_mode", "value_sink_keep",
                    "head_aggregation_alpha", "level_2_mode", "selector_mode", "protected_heavy_ratio",
                    "calib_indices_file", "val_indices_file",
                    "best_calib_dir", "best_val_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(task_rows)

        summary_path = output_root / f"{args.search_tag}_per_task_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "solver_mode": "per_task",
                    "model_path": args.model_path,
                    "target_budget": args.target_budget,
                    "tasks": task_summaries,
                    "per_task_csv": str(per_task_csv),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[done] per-task csv: {per_task_csv}")
        print(f"[done] per-task summary: {summary_path}")
        return

    uniform_tag = f"{args.search_tag}_uniform_calib"
    uniform_output = run_eval(
        model_path=args.model_path,
        max_length=args.max_length,
        datasets=args.calib_datasets,
        limit=args.calib_limit,
        sample_seed=args.calib_seed,
        sample_indices_file=None,
        output_root=output_root,
        output_tag=uniform_tag,
        sparsity_type="uniform",
        kv_sparsity=args.target_budget,
    )
    if uniform_output is None:
        raise RuntimeError("Uniform calibration failed")
    uniform_result = load_result(eval_result_path(uniform_output))
    uniform_avg = uniform_result["average"]
    print(f"[info] calibration uniform average = {uniform_avg:.2f}")

    rows = []
    best_cfg: Optional[CandidateConfig] = None
    best_result: Optional[Dict[str, float]] = None
    best_setting = None

    total_combo = len(candidates) * len(search_settings)
    combo_idx = 0
    for idx, candidate in enumerate(candidates, start=1):
        for imp_mode, head_mode, sink_keep, head_alpha, level_2_mode, selector_mode, protected_heavy_ratio in search_settings:
            combo_idx += 1
            candidate_tag = (
                f"{args.search_tag}_cand{idx}_{candidate.short_name}_"
                f"imp_{imp_mode}_head_{head_mode}_sink_{sink_keep}_"
                f"alpha_{head_alpha:.2f}_l2_{level_2_mode}_sel_{selector_mode}_phr_{protected_heavy_ratio:.2f}"
            )
            print(
                f"[search] {combo_idx}/{total_combo} "
                f"p0={candidate.p0:.2f} p1={candidate.p1:.2f} p2={candidate.p2:.2f} "
                f"rho1={candidate.rho1:.4f} imp={imp_mode} head={head_mode} sink={sink_keep} "
                f"alpha={head_alpha:.2f} l2={level_2_mode} sel={selector_mode} phr={protected_heavy_ratio:.2f}"
            )
            diff_output = run_diff_eval_with_custom_dir(
                model_path=args.model_path,
                max_length=args.max_length,
                datasets=args.calib_datasets,
                limit=args.calib_limit,
                sample_seed=args.calib_seed,
                sample_indices_file=None,
                output_root=output_root,
                output_tag=candidate_tag,
                kv_sparsity=args.target_budget,
                candidate=candidate,
                importance_mode=imp_mode,
                head_aggregation_mode=head_mode,
                value_sink_keep=sink_keep,
                head_aggregation_alpha=head_alpha,
                level_2_mode=level_2_mode,
                selector_mode=selector_mode,
                protected_heavy_ratio=protected_heavy_ratio,
            )
            if diff_output is None:
                continue
            result = load_result(eval_result_path(diff_output))
            avg = result["average"]
            delta = avg - uniform_avg
            row = {
                **asdict(candidate),
                "importance_mode": imp_mode,
                "head_aggregation_mode": head_mode,
                "value_sink_keep": sink_keep,
                "head_aggregation_alpha": head_alpha,
                "level_2_mode": level_2_mode,
                "selector_mode": selector_mode,
                "protected_heavy_ratio": protected_heavy_ratio,
                "expected_budget": candidate.expected_budget,
                "calib_average": avg,
                "delta_vs_uniform": delta,
                "output_dir": str(diff_output),
            }
            rows.append(row)
            if best_result is None or avg > best_result["average"]:
                best_cfg = candidate
                best_result = result
                best_setting = (imp_mode, head_mode, sink_keep, head_alpha, level_2_mode, selector_mode, protected_heavy_ratio)

    rows.sort(key=lambda x: x["calib_average"], reverse=True)
    search_csv = output_root / f"{args.search_tag}_calibration_results.csv"
    with search_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
                fieldnames=[
                    "p0", "p1", "p2", "rho1",
                    "importance_mode", "head_aggregation_mode", "value_sink_keep",
                    "head_aggregation_alpha", "level_2_mode", "selector_mode", "protected_heavy_ratio",
                    "target_budget", "expected_budget",
                    "calib_average", "delta_vs_uniform", "output_dir"
                ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if best_cfg is None or best_result is None or best_setting is None:
        raise RuntimeError("Search finished without a valid best candidate.")
    best_imp_mode, best_head_mode, best_sink_keep, best_head_alpha, best_level2_mode, best_selector_mode, best_protected_heavy_ratio = best_setting

    print(
        "[best] "
        f"p0={best_cfg.p0:.2f} p1={best_cfg.p1:.2f} p2={best_cfg.p2:.2f} rho1={best_cfg.rho1:.4f} "
        f"imp={best_imp_mode} head={best_head_mode} sink={best_sink_keep} "
        f"alpha={best_head_alpha:.2f} l2={best_level2_mode} sel={best_selector_mode} phr={best_protected_heavy_ratio:.2f} "
        f"calib_avg={best_result['average']:.2f} delta={best_result['average'] - uniform_avg:+.2f}"
    )

    val_uniform_tag = f"{args.search_tag}_uniform_val"
    val_uniform_output = run_eval(
        model_path=args.model_path,
        max_length=args.max_length,
        datasets=args.val_datasets,
        limit=args.val_limit,
        sample_seed=args.val_seed,
        sample_indices_file=None,
        output_root=output_root,
        output_tag=val_uniform_tag,
        sparsity_type="uniform",
        kv_sparsity=args.target_budget,
    )
    if val_uniform_output is None:
        raise RuntimeError("Uniform validation failed")
    val_uniform_result = load_result(eval_result_path(val_uniform_output))

    val_diff_tag = f"{args.search_tag}_bestdiff_val"
    val_diff_output = run_diff_eval_with_custom_dir(
        model_path=args.model_path,
        max_length=args.max_length,
        datasets=args.val_datasets,
        limit=args.val_limit,
        sample_seed=args.val_seed,
        sample_indices_file=None,
        output_root=output_root,
        output_tag=val_diff_tag,
        kv_sparsity=args.target_budget,
        candidate=best_cfg,
        importance_mode=best_imp_mode,
        head_aggregation_mode=best_head_mode,
        value_sink_keep=best_sink_keep,
        head_aggregation_alpha=best_head_alpha,
        level_2_mode=best_level2_mode,
        selector_mode=best_selector_mode,
        protected_heavy_ratio=best_protected_heavy_ratio,
    )
    if val_diff_output is None:
        raise RuntimeError("Best diff validation failed")
    val_diff_result = load_result(eval_result_path(val_diff_output))

    summary = {
        "model_path": args.model_path,
        "max_length": args.max_length,
        "target_budget": args.target_budget,
        "calib_datasets": args.calib_datasets,
        "calib_limit": args.calib_limit,
        "calib_seed": args.calib_seed,
        "val_datasets": args.val_datasets,
        "val_limit": args.val_limit,
        "val_seed": args.val_seed,
        "importance_mode": args.importance_mode,
        "head_aggregation_mode": args.head_aggregation_mode,
        "value_sink_keep": args.value_sink_keep,
        "best_head_aggregation_alpha": best_head_alpha,
        "best_level_2_mode": best_level2_mode,
        "best_selector_mode": best_selector_mode,
        "best_protected_heavy_ratio": best_protected_heavy_ratio,
        "best_importance_mode": best_imp_mode,
        "best_head_aggregation_mode": best_head_mode,
        "best_value_sink_keep": best_sink_keep,
        "uniform_calibration": uniform_result,
        "best_candidate": asdict(best_cfg),
        "best_candidate_calibration": best_result,
        "uniform_validation": val_uniform_result,
        "best_candidate_validation": val_diff_result,
        "calibration_delta_vs_uniform": best_result["average"] - uniform_avg,
        "validation_delta_vs_uniform": val_diff_result["average"] - val_uniform_result["average"],
        "search_csv": str(search_csv),
        "uniform_calibration_dir": str(uniform_output),
        "uniform_validation_dir": str(val_uniform_output),
        "diff_validation_dir": str(val_diff_output),
    }
    summary_path = output_root / f"{args.search_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] search csv: {search_csv}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
