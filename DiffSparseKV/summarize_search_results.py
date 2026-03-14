#!/usr/bin/env python3
"""
Summarize DiffSparseKV search runs into a ranked markdown table.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(root: Path, task: str, tag_substr: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_path in sorted(root.glob("*/result.json")):
        run_dir = result_path.parent
        tag = run_dir.name
        if tag_substr and tag_substr not in tag:
            continue

        config_path = run_dir / "sparsity_config.json"
        if not config_path.exists():
            continue

        result = load_json(result_path)
        if task not in result:
            continue

        cfg = load_json(config_path)
        rows.append(
            {
                "run_dir": str(run_dir),
                "tag": cfg.get("output_tag", tag),
                "score": float(result[task]),
                "sparsity_type": cfg.get("sparsity_type"),
                "target_budget": cfg.get("target_budget"),
                "target_distribution": cfg.get("target_distribution"),
                "sparsity_levels": cfg.get("sparsity_levels"),
                "importance_mode": cfg.get("importance_mode"),
                "head_aggregation_mode": cfg.get("head_aggregation_mode"),
                "value_sink_keep": cfg.get("value_sink_keep"),
                "level_2_mode": cfg.get("level_2_mode"),
                "limit": cfg.get("limit"),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in value) + "]"
    return str(value)


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "rank",
        "score",
        "tag",
        "type",
        "budget",
        "distribution",
        "levels",
        "importance",
        "head_agg",
        "sink",
        "exit",
        "limit",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    fmt(row["score"]),
                    row["tag"],
                    fmt(row["sparsity_type"]),
                    fmt(row["target_budget"]),
                    fmt(row["target_distribution"]),
                    fmt(row["sparsity_levels"]),
                    fmt(row["importance_mode"]),
                    fmt(row["head_aggregation_mode"]),
                    fmt(row["value_sink_keep"]),
                    fmt(row["level_2_mode"]),
                    fmt(row["limit"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize search runs")
    parser.add_argument("--root", type=str, default="tmp_eval", help="Directory containing run outputs")
    parser.add_argument("--task", type=str, required=True, help="Task metric to summarize, e.g. qasper")
    parser.add_argument("--tag-substr", type=str, default="", help="Only include runs whose directory/tag contains this substring")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum number of rows to print")
    parser.add_argument("--out", type=str, default="", help="Optional markdown output path")
    args = parser.parse_args()

    rows = collect_runs(Path(args.root), args.task, args.tag_substr)
    rows = rows[: args.top_k]
    markdown = to_markdown(rows)
    print(markdown)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
