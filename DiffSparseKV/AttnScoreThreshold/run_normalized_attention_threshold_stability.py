#!/usr/bin/env python3
"""Collect cross-dataset threshold stability results on normalized attention importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "aidcs_repro"))
sys.path.append(str(REPO_ROOT))

from common import build_tokenized_samples  # noqa: E402
from importance_calculator import DiffKVImportanceCalculator  # noqa: E402
from threshold_manager import GlobalThresholdManager  # noqa: E402


DATASET_ORDER = [
    "wikitext2",
    "gsm8k",
    "qasper",
    "multifieldqa_en",
    "narrativeqa",
    "hotpotqa",
    "musique",
]
DATASET_LABELS = {
    "wikitext2": "WikiText-2",
    "gsm8k": "GSM8K",
    "qasper": "Qasper",
    "multifieldqa_en": "MultiFieldQA-EN",
    "narrativeqa": "NarrativeQA",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
}
LAYER_ORDER = [10, 20]
THRESHOLD_TYPES = ["threshold_high", "threshold_low"]
THRESHOLD_COLORS = {"threshold_high": "#D55E00", "threshold_low": "#0072B2"}


def parse_args():
    parser = argparse.ArgumentParser(description="Normalized attention threshold stability for DiffSparseKV.")
    parser.add_argument("--model-path", type=str, default="/home/zh/model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    parser.add_argument("--layers", nargs="+", type=int, default=LAYER_ORDER)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "AttnScoreThreshold" / "normalized_attention_threshold_llama3_8b"),
    )
    parser.add_argument("--seed", type=int, default=20260327)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_records(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "layer", "threshold_type"])["threshold_value"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    quantiles = (
        df.groupby(["dataset", "layer", "threshold_type"])["threshold_value"]
        .quantile([0.25, 0.75])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={0.25: "q25", 0.75: "q75"})
    )
    return summary.merge(quantiles, on=["dataset", "layer", "threshold_type"], how="left")


def summarize_cross_dataset(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, threshold_type), group in summary_df.groupby(["layer", "threshold_type"]):
        medians = group["median"].to_numpy(dtype=float)
        means = group["mean"].to_numpy(dtype=float)
        rows.append(
            {
                "layer": int(layer),
                "threshold_type": threshold_type,
                "dataset_count": int(len(group)),
                "median_min": float(medians.min()),
                "median_max": float(medians.max()),
                "median_range": f"{medians.min():.5f}--{medians.max():.5f}",
                "median_cv_percent": float(medians.std(ddof=0) / medians.mean() * 100.0),
                "mean_min": float(means.min()),
                "mean_max": float(means.max()),
                "mean_range": f"{means.min():.5f}--{means.max():.5f}",
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "threshold_type"]).reset_index(drop=True)


def create_layer_figure(df: pd.DataFrame, cross_df: pd.DataFrame, layer: int, paper_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=False)

    for ax, threshold_type in zip(axes, THRESHOLD_TYPES):
        subset = df[(df["layer"] == layer) & (df["threshold_type"] == threshold_type)]
        box_data = [subset[subset["dataset"] == dataset]["threshold_value"].values for dataset in DATASET_ORDER]
        bp = ax.boxplot(
            box_data,
            tick_labels=[DATASET_LABELS[d] for d in DATASET_ORDER],
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(THRESHOLD_COLORS[threshold_type])
            patch.set_alpha(0.24)
            patch.set_edgecolor(THRESHOLD_COLORS[threshold_type])

        medians = (
            subset.groupby("dataset")["threshold_value"]
            .median()
            .reindex(DATASET_ORDER)
            .to_numpy(dtype=float)
        )
        ax.plot(
            np.arange(1, len(DATASET_ORDER) + 1),
            medians,
            color=THRESHOLD_COLORS[threshold_type],
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
        cv = cross_df[
            (cross_df["layer"] == layer) & (cross_df["threshold_type"] == threshold_type)
        ]["median_cv_percent"].iloc[0]
        title = r"$\tau_h$" if threshold_type == "threshold_high" else r"$\tau_l$"
        ax.set_title(f"{title}  |  Median CV = {cv:.2f}%")
        ax.set_ylabel("Threshold")
        ax.tick_params(axis="x", rotation=32, labelsize=9)
        ax.grid(True, axis="y", alpha=0.22)

    fig.suptitle(f"Layer {layer}: Normalized Attention Thresholds Across Datasets", fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(paper_dir / f"layer{layer}_normalized_attention_thresholds.png", bbox_inches="tight")
    fig.savefig(paper_dir / f"layer{layer}_normalized_attention_thresholds.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(cross_df: pd.DataFrame, output_dir: Path, args) -> None:
    lines = []
    for _, row in cross_df.iterrows():
        symbol = r"$\tau_h$" if row["threshold_type"] == "threshold_high" else r"$\tau_l$"
        lines.append(
            f"- Layer {int(row['layer'])} {symbol}: median range {row['median_range']}, "
            f"cross-dataset median CV {row['median_cv_percent']:.2f}%."
        )

    md = f"""# Normalized Attention Threshold Stability

## Experimental Scope

- Model: `{args.model_path}`
- Datasets: {", ".join(args.datasets)}
- Layers: {", ".join(str(x) for x in args.layers)}
- Samples per dataset: {args.num_samples}
- Max length: {args.max_length}
- Threshold definition: `threshold_high` / `threshold_low` computed from normalized attention-derived importance scores using `DiffKVImportanceCalculator + GlobalThresholdManager`.

## Main Findings

{chr(10).join(lines)}

## Interpretation

These results evaluate the exact threshold type used by the DiffSparseKV prefill-to-decode reuse path: normalized attention-derived importance thresholds. If the cross-dataset CV remains low for a fixed layer, then a threshold estimated during prefill is likely to remain comparable to normalized decode-time scores under the same scaling rule.
"""
    (output_dir / "summary.md").write_text(md, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    paper_dir = output_dir / "paper_main"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    calculator = DiffKVImportanceCalculator()
    manager = GlobalThresholdManager()

    records: List[Dict] = []

    for dataset_name in args.datasets:
        print(f"[run] dataset={dataset_name}")
        samples = build_tokenized_samples(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            num_samples=args.num_samples,
            max_length=args.max_length,
        )
        for sample_idx, sample in enumerate(samples):
            input_ids = sample.unsqueeze(0).to("cuda")
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, use_cache=False, output_attentions=True)
            seq_len = int(input_ids.shape[1])
            for layer in args.layers:
                attn = outputs.attentions[layer]
                importance = calculator.compute_diffkv_importance(attn)
                threshold_high, threshold_low = manager.compute_per_layer_thresholds(importance)
                records.append(
                    {
                        "dataset": dataset_name,
                        "sample_idx": sample_idx,
                        "seq_len": seq_len,
                        "layer": layer,
                        "threshold_type": "threshold_high",
                        "threshold_value": float(threshold_high),
                    }
                )
                records.append(
                    {
                        "dataset": dataset_name,
                        "sample_idx": sample_idx,
                        "seq_len": seq_len,
                        "layer": layer,
                        "threshold_type": "threshold_low",
                        "threshold_value": float(threshold_low),
                    }
                )

            del outputs
            torch.cuda.empty_cache()

    records_df = pd.DataFrame(records)
    records_df.to_csv(output_dir / "raw_threshold_records.csv", index=False)

    summary_df = summarize_records(records_df)
    summary_df.to_csv(output_dir / "dataset_threshold_summary.csv", index=False)

    cross_df = summarize_cross_dataset(summary_df)
    cross_df.to_csv(output_dir / "cross_dataset_threshold_summary.csv", index=False)

    for layer in args.layers:
        create_layer_figure(records_df, cross_df, layer, paper_dir)

    metrics = {
        f"layer{int(row['layer'])}_{row['threshold_type']}_median_cv_percent": float(row["median_cv_percent"])
        for _, row in cross_df.iterrows()
    }
    save_json(metrics, output_dir / "metrics.json")
    write_summary_markdown(cross_df, output_dir, args)

    manifest = {
        "model_path": args.model_path,
        "datasets": args.datasets,
        "layers": args.layers,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "output_dir": str(output_dir),
    }
    save_json(manifest, output_dir / "manifest.json")
    print(f"[done] wrote normalized-attention threshold results to {output_dir}")


if __name__ == "__main__":
    main()
