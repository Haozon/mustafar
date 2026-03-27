# AIDCS Repro Results Summary

## Generated artifacts

- Layer sensitivity:
  - `aidcs_repro/results/layer_type_sensitivity_formal/layer_type_sensitivity.csv`
  - `aidcs_repro/results/layer_type_sensitivity_formal/safe_sparsity_summary.csv`
  - `aidcs_repro/results/layer_type_sensitivity_formal/layer_type_sensitivity_plot.pdf`
- Threshold stability:
  - `aidcs_repro/results/threshold_stability_formal/threshold_stability.csv`
  - `aidcs_repro/results/threshold_stability_formal/layer_0_threshold_boxplot.pdf`
  - `aidcs_repro/results/threshold_stability_formal/layer_10_threshold_boxplot.pdf`
  - `aidcs_repro/results/threshold_stability_formal/layer_20_threshold_boxplot.pdf`
- Massive token:
  - `aidcs_repro/results/massive_token_formal/massive_token_analysis.csv`
- Revision snippets:
  - `aidcs_repro/tex/aidcs_revised_experiment_snippets.tex`

## Key takeaways

- `qkv` and `up/gate` are the most sensitive layer types.
- `out` and `down` are consistently more robust.
- The chapter should not claim that 50% retained features are broadly "safe" for all layer types.
- Cross-dataset threshold medians become more stable with depth, but the claim should be phrased as a statistical tendency rather than an absolute fact.
- Massive-activation token ratio is around 1%, not 0.4%--0.6% in the reproduced setup.
- The current CATS/TEAL/AIDCS comparison is not methodologically fair without matched layer scope.
- The current performance section is not reproducible from the available workspace.

