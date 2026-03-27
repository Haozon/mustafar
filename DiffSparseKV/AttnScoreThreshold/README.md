# AttnScoreThreshold

This directory contains paper-ready artifacts for the attention-score threshold stability analysis.

## What It Produces

- `outputs/figures/attn_threshold_stability_overview.{png,pdf}`
- `outputs/figures/attn_threshold_representative_layers.{png,pdf}`
- `outputs/figures/attn_threshold_normalized_variation.{png,pdf}`
- `outputs/tables/*.csv`
- `outputs/tables/*.tex`
- `outputs/text/paper_ready_summary.md`
- `outputs/text/suggested_captions.md`
- `outputs/metrics.json`
- `outputs_cross_dataset/figures/cross_dataset_threshold_overview.{png,pdf}`
- `outputs_cross_dataset/figures/cross_dataset_threshold_profiles.{png,pdf}`
- `outputs_cross_dataset/tables/*.csv`
- `outputs_cross_dataset/tables/*.tex`
- `outputs_cross_dataset/text/cross_dataset_summary.md`

## Data Sources

The generator reuses the existing stability session and summary report:

- `LeanSparseKV/threshold_stability_analysis/threshold_data/session_20251231_154207.json`
- `LeanSparseKV/threshold_stability_analysis/visualization_results/validation_report.json`

The cross-dataset generator reuses the formal reproduction outputs:

- `DiffSparseKV/aidcs_repro/results/threshold_stability_formal/threshold_stability.csv`
- `DiffSparseKV/aidcs_repro/results/threshold_stability_formal/threshold_stability_raw_values.json`
- `DiffSparseKV/aidcs_repro/results/threshold_stability_formal/threshold_stability.json`

## Run

```bash
python DiffSparseKV/AttnScoreThreshold/generate_paper_artifacts.py
python DiffSparseKV/AttnScoreThreshold/generate_cross_dataset_artifacts.py
```

## Notes

- The generated figures support `fixed per-layer threshold stability` claims.
- They should not be described as evidence for a single global threshold shared by all layers.
- The summary text in `outputs/text/paper_ready_summary.md` is meant to be dropped into a paper draft and edited as needed.
- The `outputs_cross_dataset` artifacts provide the stronger cross-dataset evidence and are better suited for the main paper claim.
