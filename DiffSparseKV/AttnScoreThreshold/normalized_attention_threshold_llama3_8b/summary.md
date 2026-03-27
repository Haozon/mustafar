# Normalized Attention Threshold Stability

## Experimental Scope

- Model: `/home/zh/model/Meta-Llama-3-8B-Instruct`
- Datasets: wikitext2, gsm8k, qasper, multifieldqa_en, narrativeqa, hotpotqa, musique
- Layers: 10, 20
- Samples per dataset: 10
- Max length: 192
- Threshold definition: `threshold_high` / `threshold_low` computed from normalized attention-derived importance scores using `DiffKVImportanceCalculator + GlobalThresholdManager`.

## Main Findings

- Layer 10 $\tau_h$: median range 0.67120--0.92427, cross-dataset median CV 10.21%.
- Layer 10 $\tau_l$: median range 0.04243--0.06552, cross-dataset median CV 14.37%.
- Layer 20 $\tau_h$: median range 0.35568--0.61690, cross-dataset median CV 19.40%.
- Layer 20 $\tau_l$: median range 0.01732--0.02890, cross-dataset median CV 19.82%.

## Interpretation

These results evaluate the exact threshold type used by the DiffSparseKV prefill-to-decode reuse path: normalized attention-derived importance thresholds. If the cross-dataset CV remains low for a fixed layer, then a threshold estimated during prefill is likely to remain comparable to normalized decode-time scores under the same scaling rule.
