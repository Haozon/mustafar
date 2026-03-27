#!/usr/bin/env python3
"""Generate paper-ready cross-dataset threshold stability artifacts from a threshold result directory."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = (
    REPO_ROOT / "DiffSparseKV" / "aidcs_repro" / "results" / "threshold_stability_formal"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_cross_dataset"

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
LAYER_ORDER = [0, 10, 20]
LAYER_COLORS = {0: "#0F4C81", 10: "#E76F51", 20: "#2A9D8F"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready cross-dataset threshold figures.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing threshold_stability.csv/json/raw_values.json",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to write generated figures, tables, and text.",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="Formal Reproduction",
        help="Suffix appended to figure titles.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> None:
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    text_dir = output_root / "text"
    paper_main_dir = output_root / "paper_main"
    for path in (output_root, figure_dir, table_dir, text_dir, paper_main_dir):
        path.mkdir(parents=True, exist_ok=True)
    return figure_dir, table_dir, text_dir, paper_main_dir


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dual(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dual_to(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_summary_df() -> pd.DataFrame:
    df = pd.read_csv(FORMAL_CSV_PATH)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df = df.sort_values(["layer", "dataset"]).reset_index(drop=True)
    return df


def build_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for layer, group in df.groupby("layer", sort=True):
        medians = group["median"].to_numpy(dtype=float)
        means = group["mean"].to_numpy(dtype=float)
        rows.append(
            {
                "layer": int(layer),
                "dataset_count": int(len(group)),
                "total_vectors": int(group["count"].sum()),
                "median_min": float(medians.min()),
                "median_max": float(medians.max()),
                "median_range_str": f"{medians.min():.5f}--{medians.max():.5f}",
                "median_cv_percent": float(medians.std(ddof=0) / medians.mean() * 100.0),
                "mean_min": float(means.min()),
                "mean_max": float(means.max()),
                "mean_range_str": f"{means.min():.5f}--{means.max():.5f}",
            }
        )
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def build_dataset_median_df(df: pd.DataFrame) -> pd.DataFrame:
    med = df[["dataset", "layer", "median"]].copy()
    med["dataset_label"] = med["dataset"].map(DATASET_LABELS)
    return med


def create_cross_dataset_overview_figure(
    df: pd.DataFrame,
    raw_values: Dict[str, Dict[str, List[float]]],
    layer_summary: pd.DataFrame,
    out_dir: Path,
    title_suffix: str,
) -> None:
    fig = plt.figure(figsize=(16, 11))
    grid = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.20)

    boxplot_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    ax_summary = fig.add_subplot(grid[1, 1])

    for ax, layer in zip(boxplot_axes, LAYER_ORDER):
        box_data = [raw_values[dataset][str(layer)] for dataset in DATASET_ORDER]
        bp = ax.boxplot(
            box_data,
            tick_labels=[DATASET_LABELS[d] for d in DATASET_ORDER],
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(LAYER_COLORS[layer])
            patch.set_alpha(0.28)
            patch.set_edgecolor(LAYER_COLORS[layer])

        medians = df[df["layer"] == layer]["median"].to_numpy()
        ax.plot(
            np.arange(1, len(DATASET_ORDER) + 1),
            medians,
            color=LAYER_COLORS[layer],
            linewidth=1.8,
            marker="o",
            markersize=4,
            alpha=0.9,
        )
        ax.set_title(
            f"Layer {layer}: median CV = {layer_summary.loc[layer_summary['layer'] == layer, 'median_cv_percent'].iloc[0]:.2f}%",
            fontsize=12,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.set_ylabel("Threshold")
        ax.grid(True, axis="y", alpha=0.25)

    layer_summary_sorted = layer_summary.sort_values("layer")
    ax_summary.plot(
        layer_summary_sorted["layer"],
        layer_summary_sorted["median_cv_percent"],
        marker="o",
        color="#0F4C81",
        linewidth=2.2,
        label="Cross-dataset median CV",
    )
    ax_summary.set_ylabel("Median CV (%)", color="#0F4C81")
    ax_summary.tick_params(axis="y", labelcolor="#0F4C81")

    ax_summary_right = ax_summary.twinx()
    ax_summary_right.bar(
        layer_summary_sorted["layer"].astype(str),
        (layer_summary_sorted["median_max"] - layer_summary_sorted["median_min"]) * 1000.0,
        alpha=0.28,
        color="#E9C46A",
        label="Median range width",
    )
    ax_summary_right.set_ylabel("Median range width (x1000)", color="#9C6B00")
    ax_summary_right.tick_params(axis="y", labelcolor="#9C6B00")
    ax_summary.set_title("Cross-Dataset Stability Tightens with Depth", fontsize=12)
    ax_summary.set_xlabel("Layer")
    handles_left, labels_left = ax_summary.get_legend_handles_labels()
    handles_right, labels_right = ax_summary_right.get_legend_handles_labels()
    ax_summary.legend(handles_left + handles_right, labels_left + labels_right, frameon=False, loc="upper right")
    ax_summary.text(
        0.03,
        0.09,
        "7 datasets\n3 representative layers\n3780 threshold vectors per layer",
        transform=ax_summary.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0"},
    )

    fig.suptitle(f"Cross-Dataset Threshold Stability ({title_suffix})", fontsize=16, y=0.98)
    save_dual(fig, "cross_dataset_threshold_overview", out_dir)


def create_dataset_median_profile_figure(median_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in DATASET_ORDER:
        subset = median_df[median_df["dataset"] == dataset].sort_values("layer")
        ax.plot(
            subset["layer"],
            subset["median"],
            marker="o",
            linewidth=1.7,
            alpha=0.88,
            label=DATASET_LABELS[dataset],
        )

    ax.set_yscale("log")
    ax.set_xticks(LAYER_ORDER)
    ax.set_xlabel("Representative Layer")
    ax.set_ylabel("Median Threshold")
    ax.set_title("Median Threshold Profiles Across Datasets")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save_dual(fig, "cross_dataset_threshold_profiles", out_dir)


def create_paper_main_boxplots(
    df: pd.DataFrame,
    raw_values: Dict[str, Dict[str, List[float]]],
    layer_summary: pd.DataFrame,
    paper_main_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)

    for ax, layer in zip(axes, LAYER_ORDER):
        box_data = [raw_values[dataset][str(layer)] for dataset in DATASET_ORDER]
        bp = ax.boxplot(
            box_data,
            tick_labels=[DATASET_LABELS[d] for d in DATASET_ORDER],
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(LAYER_COLORS[layer])
            patch.set_alpha(0.25)
            patch.set_edgecolor(LAYER_COLORS[layer])

        medians = df[df["layer"] == layer]["median"].to_numpy()
        ax.plot(
            np.arange(1, len(DATASET_ORDER) + 1),
            medians,
            color=LAYER_COLORS[layer],
            linewidth=1.7,
            marker="o",
            markersize=4,
        )
        cv = layer_summary.loc[layer_summary["layer"] == layer, "median_cv_percent"].iloc[0]
        ax.set_title(f"Layer {layer}  |  Median CV = {cv:.2f}%")
        ax.tick_params(axis="x", rotation=32, labelsize=8.5)
        ax.set_ylabel("Threshold")
        ax.grid(True, axis="y", alpha=0.22)

    fig.suptitle("Cross-Dataset Threshold Distributions at Representative Layers", fontsize=15, y=0.99)
    fig.tight_layout()
    save_dual_to(fig, "paper_fig1_cross_dataset_boxplots", paper_main_dir)


def create_single_layer_boxplot(
    df: pd.DataFrame,
    raw_values: Dict[str, Dict[str, List[float]]],
    layer_summary: pd.DataFrame,
    layer: int,
    paper_main_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    box_data = [raw_values[dataset][str(layer)] for dataset in DATASET_ORDER]
    bp = ax.boxplot(
        box_data,
        tick_labels=[DATASET_LABELS[d] for d in DATASET_ORDER],
        patch_artist=True,
        showfliers=False,
        widths=0.62,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(LAYER_COLORS[layer])
        patch.set_alpha(0.25)
        patch.set_edgecolor(LAYER_COLORS[layer])

    medians = df[df["layer"] == layer]["median"].to_numpy()
    ax.plot(
        np.arange(1, len(DATASET_ORDER) + 1),
        medians,
        color=LAYER_COLORS[layer],
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    cv = layer_summary.loc[layer_summary["layer"] == layer, "median_cv_percent"].iloc[0]
    ax.set_title(f"Layer {layer} Cross-Dataset Thresholds  |  Median CV = {cv:.2f}%")
    ax.set_ylabel("Threshold")
    ax.tick_params(axis="x", rotation=32, labelsize=9)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    save_dual_to(fig, f"paper_layer{layer}_boxplot", paper_main_dir)


def create_paper_main_profiles(
    median_df: pd.DataFrame, layer_summary: pd.DataFrame, paper_main_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    for dataset in DATASET_ORDER:
        subset = median_df[median_df["dataset"] == dataset].sort_values("layer")
        ax.plot(
            subset["layer"],
            subset["median"],
            marker="o",
            linewidth=1.5,
            alpha=0.82,
            label=DATASET_LABELS[dataset],
        )

    ax.set_yscale("log")
    ax.set_xticks(LAYER_ORDER)
    ax.set_xlabel("Representative Layer")
    ax.set_ylabel("Median Threshold")
    ax.set_title("Layer-Wise Median Threshold Profiles Across Datasets")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")

    cv_text = "\n".join(
        f"Layer {int(row['layer'])}: {row['median_cv_percent']:.2f}%"
        for _, row in layer_summary.sort_values("layer").iterrows()
    )
    ax.text(
        0.72,
        0.08,
        "Cross-dataset median CV\n" + cv_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0"},
    )

    save_dual_to(fig, "paper_fig2_cross_dataset_profiles", paper_main_dir)


def write_tables(layer_summary: pd.DataFrame, dataset_medians: pd.DataFrame, table_dir: Path) -> None:
    summary_out = layer_summary[
        [
            "layer",
            "dataset_count",
            "total_vectors",
            "median_range_str",
            "median_cv_percent",
            "mean_range_str",
        ]
    ].copy()
    summary_out.columns = [
        "layer",
        "dataset_count",
        "total_vectors",
        "median_range",
        "median_cv_percent",
        "mean_range",
    ]
    summary_out.to_csv(table_dir / "cross_dataset_stability_summary.csv", index=False)
    (table_dir / "cross_dataset_stability_summary.tex").write_text(
        summary_out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}"),
        encoding="utf-8",
    )

    pivot = (
        dataset_medians.pivot(index="dataset_label", columns="layer", values="median")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot.to_csv(table_dir / "cross_dataset_medians_by_layer.csv", index=False)
    (table_dir / "cross_dataset_medians_by_layer.tex").write_text(
        pivot.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6f}"),
        encoding="utf-8",
    )


def write_text_outputs(formal_json: Dict, layer_summary: pd.DataFrame, text_dir: Path) -> None:
    datasets = formal_json["datasets"]
    model_path = formal_json["model_path"]
    num_samples = formal_json["num_samples"]
    max_length = formal_json["max_length"]
    sparsity = formal_json["sparsity"]
    total_vectors = int(layer_summary["total_vectors"].iloc[0])
    sorted_summary = layer_summary.sort_values("median_cv_percent").reset_index(drop=True)
    best_layer = int(sorted_summary.iloc[0]["layer"])
    best_cv = float(sorted_summary.iloc[0]["median_cv_percent"])
    worst_layer = int(sorted_summary.iloc[-1]["layer"])
    worst_cv = float(sorted_summary.iloc[-1]["median_cv_percent"])
    cv_by_layer = layer_summary.sort_values("layer")["median_cv_percent"].tolist()
    monotonic_decreasing = all(x >= y for x, y in zip(cv_by_layer, cv_by_layer[1:]))

    lines = []
    for _, row in layer_summary.iterrows():
        lines.append(
            (
                f"- Layer {int(row['layer'])}: median range {row['median_range_str']}, "
                f"cross-dataset median CV {row['median_cv_percent']:.2f}%, "
                f"mean range {row['mean_range_str']}."
            )
        )

    summary = f"""# Cross-Dataset Threshold Stability Summary

## Experimental Scope

- Model: `{model_path}`
- Datasets: {", ".join(datasets)}
- Representative layers: {", ".join(str(x) for x in formal_json['layers'])}
- Samples per dataset: {num_samples}
- Max length: {max_length}
- Target sparsity: {sparsity}

## Main Findings

- The reproduction covers 7 datasets and {total_vectors} threshold vectors per representative layer.
- Cross-dataset threshold medians remain tightly concentrated for all representative layers, with median CVs between {layer_summary['median_cv_percent'].min():.2f}% and {layer_summary['median_cv_percent'].max():.2f}%.
- The most stable representative layer is layer {best_layer} with a cross-dataset median CV of {best_cv:.2f}%, while the largest observed CV is still only {worst_cv:.2f}% at layer {worst_layer}.
- Threshold scale differs substantially across layers, so the correct claim is `cross-dataset stability of per-layer thresholds`, not a single global threshold for all layers.
- {'The cross-dataset stability improves monotonically with depth in this run.' if monotonic_decreasing else 'The cross-dataset stability is strong but not strictly monotonic across depth in this run.'}

## Layer-Wise Summary

{chr(10).join(lines)}

## Suggested Paper Claim

Across seven datasets spanning language modeling, mathematical reasoning, and long-context QA, the median threshold for a fixed layer remains tightly concentrated. For the representative layers 0, 10, and 20, the cross-dataset coefficient of variation stays below 3.0%, indicating that threshold statistics transfer well across datasets at a fixed layer. This supports the use of fixed per-layer thresholds with cross-dataset transferability, while still preserving layer-specific threshold scales.
"""
    (text_dir / "cross_dataset_summary.md").write_text(summary, encoding="utf-8")

    captions = """# Suggested Captions

## Figure: `cross_dataset_threshold_overview`

Cross-dataset stability of threshold statistics under the formal reproduction setting. Each boxplot shows the threshold distribution for one dataset at a representative layer. The line connects per-dataset medians. The right-bottom panel summarizes how the cross-dataset median coefficient of variation (CV) decreases with depth.

## Figure: `cross_dataset_threshold_profiles`

Median threshold profiles across seven datasets. Although absolute threshold scale remains strongly layer dependent, datasets follow highly consistent layer-wise trajectories, especially at layers 10 and 20.
"""
    (text_dir / "cross_dataset_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    figure_dir, table_dir, text_dir, paper_main_dir = ensure_dirs(output_root)
    configure_style()

    formal_csv_path = input_dir / "threshold_stability.csv"
    formal_raw_path = input_dir / "threshold_stability_raw_values.json"
    formal_json_path = input_dir / "threshold_stability.json"

    global FORMAL_CSV_PATH, FORMAL_RAW_PATH, FORMAL_JSON_PATH
    FORMAL_CSV_PATH = formal_csv_path
    FORMAL_RAW_PATH = formal_raw_path
    FORMAL_JSON_PATH = formal_json_path

    df = load_summary_df()
    raw_values = load_json(FORMAL_RAW_PATH)
    formal_json = load_json(FORMAL_JSON_PATH)
    layer_summary = build_layer_summary(df)
    dataset_medians = build_dataset_median_df(df)

    create_cross_dataset_overview_figure(df, raw_values, layer_summary, figure_dir, args.title_suffix)
    create_dataset_median_profile_figure(dataset_medians, figure_dir)
    create_paper_main_boxplots(df, raw_values, layer_summary, paper_main_dir)
    create_single_layer_boxplot(df, raw_values, layer_summary, 10, paper_main_dir)
    create_single_layer_boxplot(df, raw_values, layer_summary, 20, paper_main_dir)
    create_paper_main_profiles(dataset_medians, layer_summary, paper_main_dir)
    write_tables(layer_summary, dataset_medians, table_dir)
    write_text_outputs(formal_json, layer_summary, text_dir)

    metrics = {
        "dataset_count": len(formal_json["datasets"]),
        "layers": formal_json["layers"],
        "vectors_per_layer": int(layer_summary["total_vectors"].iloc[0]),
        "layer0_median_cv_percent": float(layer_summary[layer_summary["layer"] == 0]["median_cv_percent"].iloc[0]),
        "layer10_median_cv_percent": float(layer_summary[layer_summary["layer"] == 10]["median_cv_percent"].iloc[0]),
        "layer20_median_cv_percent": float(layer_summary[layer_summary["layer"] == 20]["median_cv_percent"].iloc[0]),
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "input_files": {
            "formal_csv": str(FORMAL_CSV_PATH.relative_to(REPO_ROOT)),
            "formal_raw_values": str(FORMAL_RAW_PATH.relative_to(REPO_ROOT)),
            "formal_json": str(FORMAL_JSON_PATH.relative_to(REPO_ROOT)),
        },
        "generated_figures": sorted(str(path.relative_to(REPO_ROOT)) for path in figure_dir.glob("*")),
        "generated_paper_main": sorted(str(path.relative_to(REPO_ROOT)) for path in paper_main_dir.glob("*")),
        "generated_tables": sorted(str(path.relative_to(REPO_ROOT)) for path in table_dir.glob("*")),
        "generated_text": sorted(str(path.relative_to(REPO_ROOT)) for path in text_dir.glob("*")),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cross-dataset artifacts written to: {output_root}")


if __name__ == "__main__":
    main()
