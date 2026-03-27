#!/usr/bin/env python3
"""Generate paper-ready attention-threshold artifacts from existing experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

STABILITY_SESSION_PATH = (
    REPO_ROOT
    / "LeanSparseKV"
    / "threshold_stability_analysis"
    / "threshold_data"
    / "session_20251231_154207.json"
)
STABILITY_REPORT_PATH = (
    REPO_ROOT
    / "LeanSparseKV"
    / "threshold_stability_analysis"
    / "visualization_results"
    / "validation_report.json"
)

OUTPUT_ROOT = SCRIPT_DIR / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
TEXT_DIR = OUTPUT_ROOT / "text"

QUANTILE_ORDER = ["alpha_h", "alpha_mh", "alpha_m", "alpha_ml"]
QUANTILE_LABELS = {
    "alpha_h": r"$\alpha_h$",
    "alpha_mh": r"$\alpha_{mh}$",
    "alpha_m": r"$\alpha_m$",
    "alpha_ml": r"$\alpha_{ml}$",
}
QUANTILE_COLORS = {
    "alpha_h": "#0F4C81",
    "alpha_mh": "#E76F51",
    "alpha_m": "#2A9D8F",
    "alpha_ml": "#A68A64",
}
REPRESENTATIVE_LAYERS = [0, 10, 20, 31]


def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, FIGURE_DIR, TABLE_DIR, TEXT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_layer_id(layer_value) -> int:
    if isinstance(layer_value, int):
        return layer_value
    text = str(layer_value)
    if text.startswith("layer_"):
        return int(text.split("_")[1])
    return int(text)


def load_stability_records() -> pd.DataFrame:
    session = load_json(STABILITY_SESSION_PATH)
    df = pd.DataFrame(session["records"])
    if df.empty:
        raise ValueError(f"No records found in {STABILITY_SESSION_PATH}")
    df["layer_num"] = df["layer_id"].map(parse_layer_id)
    df["quantile_name"] = pd.Categorical(
        df["quantile_name"], categories=QUANTILE_ORDER, ordered=True
    )
    return df.sort_values(["layer_num", "quantile_name", "sample_size", "bootstrap_iteration"])


def compute_layer_quantile_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(["layer_num", "quantile_name"], observed=True)["threshold_value"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    stats["cv"] = stats["std"] / stats["mean"]
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95"] = 1.96 * stats["se"]
    stats["ci95_low"] = stats["mean"] - stats["ci95"]
    stats["ci95_high"] = stats["mean"] + stats["ci95"]
    return stats


def compute_sample_size_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pair_df = (
        df.groupby(["sample_size", "layer_num", "quantile_name"], observed=True)["threshold_value"]
        .mean()
        .reset_index()
        .pivot(index=["layer_num", "quantile_name"], columns="sample_size", values="threshold_value")
        .dropna()
        .reset_index()
        .rename(columns={20: "threshold_n20", 50: "threshold_n50"})
    )
    r, p = pearsonr(pair_df["threshold_n20"], pair_df["threshold_n50"])
    ratio = pair_df["threshold_n50"] / pair_df["threshold_n20"]
    metrics = {
        "pearson_r": float(r),
        "pearson_pvalue": float(p),
        "mean_ratio_50_to_20": float(ratio.mean()),
        "std_ratio_50_to_20": float(ratio.std(ddof=1)),
        "min_ratio_50_to_20": float(ratio.min()),
        "max_ratio_50_to_20": float(ratio.max()),
        "pair_count": int(len(pair_df)),
    }
    return pair_df, metrics


def build_quantile_summary(report_json: Dict) -> pd.DataFrame:
    rows = []
    for quantile in QUANTILE_ORDER:
        entry = report_json["quantile_analysis"][quantile]
        rows.append(
            {
                "quantile": quantile,
                "stable_layers": int(entry["stable_layers"]),
                "total_layers": int(entry["total_layers"]),
                "stability_rate": float(entry["stability_rate"]),
                "avg_cv": float(entry["avg_cv"]),
                "min_cv": float(entry["min_cv"]),
                "max_cv": float(entry["max_cv"]),
            }
        )
    return pd.DataFrame(rows)


def save_dual_format_figure(fig: plt.Figure, filename: str) -> None:
    png_path = FIGURE_DIR / f"{filename}.png"
    pdf_path = FIGURE_DIR / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, stem: str, latex_float_fmt: str = "%.4f") -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    tex_path = TABLE_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(index=False, escape=False, float_format=lambda x: latex_float_fmt % x)
    tex_path.write_text(tex, encoding="utf-8")


def create_overview_figure(
    stats_df: pd.DataFrame, pair_df: pd.DataFrame, pair_metrics: Dict[str, float]
) -> None:
    fig = plt.figure(figsize=(15, 10))
    grid = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.24)

    ax_heatmap = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_profile = fig.add_subplot(grid[1, :])

    heatmap_data = (
        stats_df.pivot(index="quantile_name", columns="layer_num", values="cv")
        .loc[QUANTILE_ORDER]
        .values
        * 100.0
    )
    im = ax_heatmap.imshow(heatmap_data, aspect="auto", cmap="YlGnBu", vmin=4.0, vmax=10.5)
    ax_heatmap.set_title("Per-Layer Threshold CV Heatmap", fontsize=13)
    ax_heatmap.set_xlabel("Layer")
    ax_heatmap.set_ylabel("Quantile")
    ax_heatmap.set_xticks(range(0, heatmap_data.shape[1], 4))
    ax_heatmap.set_xticklabels(range(0, heatmap_data.shape[1], 4))
    ax_heatmap.set_yticks(range(len(QUANTILE_ORDER)))
    ax_heatmap.set_yticklabels([QUANTILE_LABELS[q] for q in QUANTILE_ORDER])
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("CV (%)")

    for quantile in QUANTILE_ORDER:
        subset = pair_df[pair_df["quantile_name"] == quantile]
        ax_scatter.scatter(
            subset["threshold_n20"],
            subset["threshold_n50"],
            s=30,
            alpha=0.85,
            label=QUANTILE_LABELS[quantile],
            color=QUANTILE_COLORS[quantile],
        )

    bounds = np.array(
        [
            min(pair_df["threshold_n20"].min(), pair_df["threshold_n50"].min()),
            max(pair_df["threshold_n20"].max(), pair_df["threshold_n50"].max()),
        ]
    )
    ax_scatter.plot(bounds, bounds, linestyle="--", color="black", linewidth=1.0, label="y = x")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Mean Threshold (20 calibration samples)")
    ax_scatter.set_ylabel("Mean Threshold (50 calibration samples)")
    ax_scatter.set_title("Sample-Size Consistency", fontsize=13)
    ax_scatter.legend(frameon=False, fontsize=9, loc="upper left")
    ax_scatter.text(
        0.03,
        0.05,
        (
            f"Pearson r = {pair_metrics['pearson_r']:.4f}\n"
            f"Mean ratio (50/20) = {pair_metrics['mean_ratio_50_to_20']:.3f}"
        ),
        transform=ax_scatter.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0"},
    )

    for quantile in QUANTILE_ORDER:
        subset = (
            stats_df[stats_df["quantile_name"] == quantile]
            .sort_values("layer_num")
            .reset_index(drop=True)
        )
        ax_profile.plot(
            subset["layer_num"],
            subset["mean"],
            color=QUANTILE_COLORS[quantile],
            linewidth=2.2,
            label=QUANTILE_LABELS[quantile],
        )
        ax_profile.fill_between(
            subset["layer_num"],
            subset["ci95_low"],
            subset["ci95_high"],
            color=QUANTILE_COLORS[quantile],
            alpha=0.12,
        )

    ax_profile.set_title("Layer-Wise Threshold Profiles", fontsize=13)
    ax_profile.set_xlabel("Layer")
    ax_profile.set_ylabel("Mean Threshold")
    ax_profile.set_yscale("log")
    ax_profile.set_xlim(stats_df["layer_num"].min(), stats_df["layer_num"].max())
    ax_profile.legend(ncol=4, frameon=False, loc="upper right")

    fig.suptitle("Attention-Score Threshold Stability Overview", fontsize=16, y=0.98)
    save_dual_format_figure(fig, "attn_threshold_stability_overview")


def create_representative_layer_figure(stats_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    x_positions = np.arange(len(QUANTILE_ORDER))

    for ax, layer in zip(axes, REPRESENTATIVE_LAYERS):
        subset = (
            stats_df[stats_df["layer_num"] == layer]
            .set_index("quantile_name")
            .loc[QUANTILE_ORDER]
            .reset_index()
        )
        for idx, row in subset.iterrows():
            quantile = row["quantile_name"]
            ax.errorbar(
                x_positions[idx],
                row["mean"],
                yerr=row["ci95"],
                fmt="o",
                color=QUANTILE_COLORS[quantile],
                markersize=6,
                capsize=4,
                linewidth=1.5,
            )
        ax.plot(x_positions, subset["mean"], color="#666666", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Layer {layer} (avg CV = {subset['cv'].mean() * 100:.2f}%)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([QUANTILE_LABELS[q] for q in QUANTILE_ORDER])
        ax.set_yscale("log")
        ax.set_ylabel("Threshold")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Representative Layer Threshold Intervals", fontsize=15, y=0.98)
    fig.tight_layout()
    save_dual_format_figure(fig, "attn_threshold_representative_layers")


def create_normalized_variation_figure(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    merged = df.merge(
        stats_df[["layer_num", "quantile_name", "mean"]],
        on=["layer_num", "quantile_name"],
        how="left",
    )
    merged["normalized_threshold"] = merged["threshold_value"] / merged["mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [
        merged.loc[merged["quantile_name"] == quantile, "normalized_threshold"].values
        for quantile in QUANTILE_ORDER
    ]

    boxplot = ax.boxplot(
        box_data,
        tick_labels=[QUANTILE_LABELS[q] for q in QUANTILE_ORDER],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, quantile in zip(boxplot["boxes"], QUANTILE_ORDER):
        patch.set_facecolor(QUANTILE_COLORS[quantile])
        patch.set_alpha(0.35)
        patch.set_edgecolor(QUANTILE_COLORS[quantile])

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_ylabel("Threshold / Layer-Quantile Mean")
    ax.set_title("Normalized Bootstrap Variation Around Each Fixed Threshold")
    ax.set_ylim(0.75, 1.30)
    save_dual_format_figure(fig, "attn_threshold_normalized_variation")


def build_representative_tables(stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep_long = (
        stats_df[stats_df["layer_num"].isin(REPRESENTATIVE_LAYERS)]
        .copy()
        .sort_values(["layer_num", "quantile_name"])
        .reset_index(drop=True)
    )
    rep_long["cv_percent"] = rep_long["cv"] * 100.0

    wide_rows = []
    for layer in REPRESENTATIVE_LAYERS:
        layer_df = rep_long[rep_long["layer_num"] == layer].set_index("quantile_name")
        row = {"layer": layer}
        for quantile in QUANTILE_ORDER:
            row[quantile] = f"{layer_df.loc[quantile, 'mean']:.6f} ({layer_df.loc[quantile, 'cv_percent']:.2f}\\%)"
        wide_rows.append(row)
    rep_wide = pd.DataFrame(wide_rows)
    return rep_long, rep_wide


def write_compact_representative_table(rep_wide: pd.DataFrame) -> None:
    tex_path = TABLE_DIR / "representative_layer_compact.tex"
    latex = rep_wide.to_latex(index=False, escape=False)
    tex_path.write_text(latex, encoding="utf-8")


def write_text_outputs(
    report_json: Dict,
    quantile_summary_df: pd.DataFrame,
    pair_metrics: Dict[str, float],
    rep_long_df: pd.DataFrame,
) -> None:
    overall = report_json["overall_statistics"]
    layer0_alpha_h = rep_long_df[
        (rep_long_df["layer_num"] == 0) & (rep_long_df["quantile_name"] == "alpha_h")
    ]["mean"].iloc[0]
    layer20_alpha_h = rep_long_df[
        (rep_long_df["layer_num"] == 20) & (rep_long_df["quantile_name"] == "alpha_h")
    ]["mean"].iloc[0]
    alpha_h_ratio = layer0_alpha_h / layer20_alpha_h

    metrics = {
        "overall_stability_rate": float(overall["overall_stability_rate"]),
        "stable_combinations": int(overall["stable_combinations"]),
        "total_combinations": int(overall["total_layer_quantile_combinations"]),
        "sample_size_pearson_r": pair_metrics["pearson_r"],
        "sample_size_mean_ratio_50_to_20": pair_metrics["mean_ratio_50_to_20"],
        "alpha_h_layer0_to_layer20_ratio": float(alpha_h_ratio),
    }
    (OUTPUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    quantile_lines = []
    for _, row in quantile_summary_df.iterrows():
        quantile_lines.append(
            (
                f"- {row['quantile']}: stability rate {row['stability_rate'] * 100:.1f}%, "
                f"average CV {row['avg_cv'] * 100:.2f}% "
                f"(range {row['min_cv'] * 100:.2f}% to {row['max_cv'] * 100:.2f}%)."
            )
        )

    summary = f"""# Attention-Score Threshold Paper Summary

## Main Findings

- Fixed per-layer thresholds are statistically stable under the available bootstrap/sample-size analysis: {overall['stable_combinations']}/{overall['total_layer_quantile_combinations']} layer-quantile combinations satisfy the stability criterion, for an overall stability rate of {overall['overall_stability_rate'] * 100:.1f}%.
- Threshold estimates obtained from 20 and 50 calibration samples are nearly perfectly rank-preserving: Pearson r = {pair_metrics['pearson_r']:.4f}.
- The 50-sample estimates are systematically but mildly larger than the 20-sample estimates, with a mean ratio of {pair_metrics['mean_ratio_50_to_20']:.3f}.
- Threshold scale remains layer dependent. For example, $\\alpha_h$ at layer 0 is {alpha_h_ratio:.2f}x the value at layer 20, which supports per-layer thresholding instead of a single global threshold.

## Quantile-Wise Stability

{chr(10).join(quantile_lines)}

## Suggested Paper Wording

Normalization places attention-derived threshold statistics on a layer-wise comparable scale. On this normalized scale, threshold estimates remain stable across bootstrap resampling and calibration sample sizes, which supports fixed per-layer thresholds as a practical replacement for per-run dynamic threshold estimation.

## Scope Note

These artifacts are generated from the existing `session_20251231_154207` stability records in `LeanSparseKV/threshold_stability_analysis/threshold_data` and therefore support bootstrap/sample-size stability claims on the available WikiText calibration session. They should be described as evidence for `fixed per-layer threshold stability`, not as proof that a single global threshold works for all layers.
"""
    (TEXT_DIR / "paper_ready_summary.md").write_text(summary, encoding="utf-8")

    captions = """# Suggested Captions

## Figure: `attn_threshold_stability_overview`

Stability analysis for attention-score-based fixed thresholds. Left: coefficient of variation (CV) for each layer-quantile pair over 20 bootstrap/sample-size replicates. Top-right: threshold means estimated with 20 and 50 calibration samples are nearly perfectly correlated. Bottom: learned thresholds remain layer dependent, motivating per-layer thresholding.

## Figure: `attn_threshold_representative_layers`

Representative threshold intervals for shallow, middle, and late layers. Points show mean thresholds and error bars denote 95% confidence intervals over the available bootstrap/sample-size replicates.

## Figure: `attn_threshold_normalized_variation`

Normalized threshold variation after dividing each observation by its layer-quantile mean. The narrow boxes around 1.0 indicate that repeated estimates stay close to the fixed threshold assigned to each layer and quantile.
"""
    (TEXT_DIR / "suggested_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()

    df = load_stability_records()
    stats_df = compute_layer_quantile_stats(df)
    pair_df, pair_metrics = compute_sample_size_consistency(df)
    report_json = load_json(STABILITY_REPORT_PATH)
    quantile_summary_df = build_quantile_summary(report_json)
    rep_long_df, rep_wide_df = build_representative_tables(stats_df)

    create_overview_figure(stats_df, pair_df, pair_metrics)
    create_representative_layer_figure(stats_df)
    create_normalized_variation_figure(df, stats_df)

    write_table(
        quantile_summary_df.assign(
            stability_rate=quantile_summary_df["stability_rate"] * 100.0,
            avg_cv=quantile_summary_df["avg_cv"] * 100.0,
            min_cv=quantile_summary_df["min_cv"] * 100.0,
            max_cv=quantile_summary_df["max_cv"] * 100.0,
        ),
        "quantile_stability_summary",
        latex_float_fmt="%.2f",
    )

    write_table(
        rep_long_df.assign(
            cv=rep_long_df["cv"] * 100.0,
            ci95=rep_long_df["ci95"],
            ci95_low=rep_long_df["ci95_low"],
            ci95_high=rep_long_df["ci95_high"],
        )[
            ["layer_num", "quantile_name", "mean", "std", "cv", "ci95_low", "ci95_high", "count"]
        ],
        "representative_layer_stats",
        latex_float_fmt="%.6f",
    )

    sample_size_table = pd.DataFrame(
        [
            {
                "pearson_r": pair_metrics["pearson_r"],
                "mean_ratio_50_to_20": pair_metrics["mean_ratio_50_to_20"],
                "std_ratio_50_to_20": pair_metrics["std_ratio_50_to_20"],
                "min_ratio_50_to_20": pair_metrics["min_ratio_50_to_20"],
                "max_ratio_50_to_20": pair_metrics["max_ratio_50_to_20"],
                "pair_count": pair_metrics["pair_count"],
            }
        ]
    )
    write_table(sample_size_table, "sample_size_consistency", latex_float_fmt="%.4f")
    write_table(rep_wide_df, "representative_layer_compact", latex_float_fmt="%.4f")
    write_compact_representative_table(rep_wide_df)
    write_text_outputs(report_json, quantile_summary_df, pair_metrics, rep_long_df)

    manifest = {
        "input_files": {
            "stability_session": str(STABILITY_SESSION_PATH.relative_to(REPO_ROOT)),
            "stability_report": str(STABILITY_REPORT_PATH.relative_to(REPO_ROOT)),
        },
        "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
        "generated_figures": sorted(str(path.relative_to(REPO_ROOT)) for path in FIGURE_DIR.glob("*")),
        "generated_tables": sorted(str(path.relative_to(REPO_ROOT)) for path in TABLE_DIR.glob("*")),
        "generated_text": sorted(str(path.relative_to(REPO_ROOT)) for path in TEXT_DIR.glob("*")),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Artifacts written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
