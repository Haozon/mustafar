# Cross-Dataset Threshold Stability Summary

## Experimental Scope

- Model: `/home/zh/model/Meta-Llama-3-8B-Instruct`
- Datasets: wikitext2, gsm8k, qasper, multifieldqa_en, narrativeqa, hotpotqa, musique
- Representative layers: 0, 10, 20
- Samples per dataset: 3
- Max length: 192
- Target sparsity: 0.5

## Main Findings

- The reproduction covers 7 datasets and 3673 threshold vectors per representative layer.
- Cross-dataset threshold medians remain tightly concentrated for all representative layers, with median CVs between 2.27% and 2.99%.
- The most stable representative layer is layer 0 with a cross-dataset median CV of 2.27%, while the largest observed CV is still only 2.99% at layer 20.
- Threshold scale differs substantially across layers, so the correct claim is `cross-dataset stability of per-layer thresholds`, not a single global threshold for all layers.
- The cross-dataset stability is strong but not strictly monotonic across depth in this run.

## Layer-Wise Summary

- Layer 0: median range 0.01309--0.01411, cross-dataset median CV 2.27%, mean range 0.01124--0.01234.
- Layer 10: median range 0.23853--0.25610, cross-dataset median CV 2.37%, mean range 0.23654--0.25470.
- Layer 20: median range 0.25488--0.28296, cross-dataset median CV 2.99%, mean range 0.25466--0.27893.

## Suggested Paper Claim

Across seven datasets spanning language modeling, mathematical reasoning, and long-context QA, the median threshold for a fixed layer remains tightly concentrated. For the representative layers 0, 10, and 20, the cross-dataset coefficient of variation stays below 3.0%, indicating that threshold statistics transfer well across datasets at a fixed layer. This supports the use of fixed per-layer thresholds with cross-dataset transferability, while still preserving layer-specific threshold scales.
