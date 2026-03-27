# Cross-Dataset Threshold Stability Summary

## Experimental Scope

- Model: `/home/zh/model/Llama-2-7b-hf`
- Datasets: wikitext2, gsm8k, qasper, multifieldqa_en, narrativeqa, hotpotqa, musique
- Representative layers: 0, 10, 20
- Samples per dataset: 3
- Max length: 192
- Target sparsity: 0.5

## Main Findings

- The formal reproduction covers 7 datasets and 3780 threshold vectors per representative layer.
- Cross-dataset threshold medians become more stable with depth.
- The strongest evidence is at layers 10 and 20, where the cross-dataset median CV drops below 2%.
- Threshold scale differs substantially across layers, so the correct claim is `cross-dataset stability of per-layer thresholds`, not a single global threshold for all layers.

## Layer-Wise Summary

- Layer 0: median range 0.00568--0.00637, cross-dataset median CV 3.37%, mean range 0.00514--0.00581.
- Layer 10: median range 0.20575--0.21649, cross-dataset median CV 1.83%, mean range 0.20044--0.21055.
- Layer 20: median range 0.27344--0.28223, cross-dataset median CV 0.99%, mean range 0.26712--0.27552.

## Suggested Paper Claim

Across seven datasets spanning language modeling, mathematical reasoning, and long-context QA, the median threshold for a fixed layer remains tightly concentrated. The cross-dataset coefficient of variation decreases from 3.37% at layer 0 to 0.99% at layer 20, indicating that deeper layers exhibit particularly stable threshold statistics. This supports the use of fixed per-layer thresholds with cross-dataset transferability, while still preserving layer-specific threshold scales.
