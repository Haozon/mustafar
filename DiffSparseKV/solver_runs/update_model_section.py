#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def format_delta(delta: float) -> str:
    return f"{delta:+.2f}"


def latest_result(model_name: str, max_length: int, budget: float, task: str, kind: str, root: Path) -> Path | None:
    if kind == "uniform":
        pattern = root / f"{model_name}_{max_length}_uniform_{budget:.2f}_*_{task}_uniform_full" / "result.json"
    else:
        pattern = root / f"{model_name}_{max_length}_diff_sparse_kv_{budget:.2f}_*_{task}_bestdiff_full" / "result.json"
    matches = [Path(p) for p in glob.glob(str(pattern))]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status_from_delta(delta: float | None, fallback_delta: float) -> str:
    if delta is None:
        delta = fallback_delta
    if delta > 0.10:
        return "positive"
    if delta > 0.0:
        return "weak_positive"
    if delta == 0.0:
        return "neutral"
    if delta > -0.10:
        return "weak_negative"
    return "negative"


def config_text(task_info: dict, diff_result_path: Path | None) -> str:
    if diff_result_path is not None:
        cfg_path = diff_result_path.parent / "sparsity_config.json"
        if cfg_path.exists():
            cfg = load_json(cfg_path)
            td = cfg.get("target_distribution")
            sl = cfg.get("sparsity_levels")
            extras = []
            if "importance_mode" in cfg:
                extras.append(str(cfg["importance_mode"]))
            if "head_aggregation_mode" in cfg:
                extras.append(str(cfg["head_aggregation_mode"]))
            if "value_sink_keep" in cfg:
                extras.append(f"sink={cfg['value_sink_keep']}")
            if "level_2_mode" in cfg:
                extras.append(str(cfg["level_2_mode"]))
            base = f"{td} / {sl}"
            if extras:
                return f"{base}; {', '.join(extras)}"
            return base

    return (
        f"[{task_info['p0']}, {task_info['p1']}, {task_info['p2']}] / "
        f"[0.0, {task_info['rho1']}, 1.0]; "
        f"{task_info['importance_mode']}, {task_info['head_aggregation_mode']}, "
        f"sink={task_info['value_sink_keep']}, {task_info['level_2_mode']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_md", required=True)
    ap.add_argument("--section_title", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_length", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--per_task_summary_json", required=True)
    ap.add_argument("--task_order", required=True)
    ap.add_argument("--results_root", default="solver_runs")
    args = ap.parse_args()

    summary_path = Path(args.summary_md)
    results_root = Path(args.results_root)
    task_order = [x.strip() for x in args.task_order.split(",") if x.strip()]
    task_summary = load_json(Path(args.per_task_summary_json))["tasks"]

    rows: list[str] = []
    full_uniform_scores = []
    full_diff_scores = []

    for task in task_order:
        info = task_summary[task]
        val_delta = float(info["val_delta"])
        val_text = f"{info['uniform_val']:.2f} -> {info['diff_val']:.2f} ({format_delta(val_delta)})"

        uniform_result = latest_result(args.model_name, args.max_length, args.budget, task, "uniform", results_root)
        diff_result = latest_result(args.model_name, args.max_length, args.budget, task, "diff", results_root)

        if uniform_result and diff_result:
            uv = float(load_json(uniform_result)["average"])
            dv = float(load_json(diff_result)["average"])
            full_uniform_scores.append(uv)
            full_diff_scores.append(dv)
            full_delta = dv - uv
            full_text = f"{uv:.2f} -> {dv:.2f} ({format_delta(full_delta)})"
            status = status_from_delta(full_delta, val_delta)
        else:
            full_text = "--"
            status = status_from_delta(None, val_delta)

        rows.append(
            f"| `{task}` | {val_text} | {full_text} | {status} | {config_text(info, diff_result)} |"
        )

    if len(full_uniform_scores) == len(task_order):
        uv = sum(full_uniform_scores) / len(full_uniform_scores)
        dv = sum(full_diff_scores) / len(full_diff_scores)
        avg_row = f"| **Average** | -- | {uv:.2f} -> {dv:.2f} ({format_delta(dv - uv)}) | -- | -- |"
    else:
        avg_row = "| **Average** | -- | -- | -- | -- |"

    section = []
    section.append(f"## {args.section_title}")
    section.append("")
    section.append("| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |")
    section.append("|---|---|---|---|---|")
    section.extend(rows)
    section.append(avg_row)

    content = summary_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^## {re.escape(args.section_title)}\n.*?(?=^## |\Z)"
    )
    new_section = "\n".join(section) + "\n\n"
    if not pattern.search(content):
        raise RuntimeError(f"Section not found: {args.section_title}")
    updated = pattern.sub(new_section, content)
    summary_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
