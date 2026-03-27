# AIDCS Chapter Audit

## Scope

This audit is based on three reproducible experiments saved under `aidcs_repro/results`:

- `layer_type_sensitivity_formal`
- `threshold_stability_formal`
- `massive_token_formal`

The current chapter text contains both empirical inconsistencies and theoretical gaps. The items below are ordered by risk.

## Highest-Risk Problems

### 1. `tab:ppl-k` and the paragraph below it do not match

The original text claims:

- `qkv` tolerates about `60%` sparsity under a `<=1%` PPL increase criterion.
- `up` tolerates about `54%`.
- `out` and `down` tolerate about `80%`.

These claims are not supported even by the original table, and they are also contradicted by the reproducible experiments.

Reproduced safe sparsity under `<=1%` PPL increase:

- WikiText-2:
  - `qkv`: none of `{40, 50, 55, 60, 65}%`
  - `out`: `40%`
  - `up`: none of `{40, 50, 55, 60, 65}%`
  - `down`: `50%`
- RedPajama English sample:
  - `qkv`: none of `{40, 50, 55, 60, 65}%`
  - `out`: `55%`
  - `up`: none of `{40, 50, 55, 60, 65}%`
  - `down`: `55%`

Conclusion:

- `qkv` and `up` are much more sensitive than the chapter currently states.
- `out` and `down` are indeed more robust, but the current `80%` claim is not defensible from available evidence.
- The sentence "仅保留50\%的输入特征即可保持模型性能" must be weakened.

### 2. The large-activation token section is internally inconsistent

The chapter currently says the protected token ratio is about `0.4%--0.6%`, but the later ablation table reports:

- WikiText: `0.0161`
- C4: `0.0164`

That is `1.61%` and `1.64%`, not `0.4%--0.6%`.

The reproduced GSM8K analysis gives:

- layer 2: `1.18%`
- layer 3: `1.18%`
- layer 4: `1.18%`
- layer 10: `1.18%`
- layer 20: `1.18%`
- layer 31: `1.28%`

Conclusion:

- The current protected-ratio statement should be revised to around `1%` scale.
- The section is directionally correct, but the numeric claim is wrong.

### 3. `tab:gsm8k_full_diag_layers` is numerically broken

The columns `prev(Massive)`, `curr(Massive)`, and `提升比例` do not match each other.

Example:

- Layer 2: `0.17920 -> 0.14764`, but the table says `3.02x`
- Layer 31: `0.26421 -> 0.11747`, but the table says `7.00x`

Those ratios cannot be derived from the shown numbers.

The reproduced table should instead compare:

- massive-token diagonal mean
- normal-token diagonal mean
- separation ratio

That form is consistent and interpretable.

### 4. The cross-method comparison is not apples-to-apples

The chapter itself notes:

- `CATS` only sparsifies MLP
- `TEAL` and `AIDCS` sparsify all 7 linear layers

But the main PPL and downstream tables still compare them under the same sparsity labels.

This is methodologically weak unless one of the following is added:

- a matched-scope comparison where all methods act on the same layer set
- a normalized sparsity budget definition that accounts for different affected parameters/FLOPs

### 5. The performance section is not currently reproducible from the available workspace

Problems:

- The source code and figure-generation path for the AIDCS activation kernel are not present in the current repo.
- The sentence `由 0.6 降至 0.` contains an obvious typo.
- The operator and end-to-end results therefore should not remain as strong factual claims unless their pipeline is restored.

Recommendation:

- Keep the system-design description.
- Downgrade the current performance section to a prototype description unless the real kernel/benchmark code is restored.

## Medium-Risk Theory Issues

### 6. Equation `GetMask` uses `sgn` in a problematic way

The chapter writes:

`M = 0.5 * sgn(|x| - t_k) + 0.5`

But the standard sign function returns `0` at zero, which would yield `0.5` rather than a binary mask when `|x| = t_k`.

Recommendation:

- Replace with an indicator function:
  - `M = I(|x| >= t_k)`
- Or explicitly define the non-standard `sgn(0) = 1`.

### 7. The block reconstruction gradient is incomplete

The block objective contains a hard mask `M`, but the text writes gradients through `dM/dt` without stating the surrogate estimator.

Recommendation:

- Add a sentence that a straight-through estimator or a smooth surrogate is used.

### 8. The block sparsity loss is one-sided

`L_c^b = ReLU(target - actual)`

This only penalizes being below target, not above it. That means the optimizer can overshoot the target sparsity if the reconstruction term allows it.

Recommendation:

- Either justify this design explicitly
- Or replace it with a symmetric penalty if exact budget tracking matters

## What Should Be Added

### Necessary additions

1. A matched-scope TEAL/AIDCS comparison

- Same model
- Same dataset
- Same layer scope
- Same sparsity definition

2. A combined-layer additivity test

- To support the "effects are approximately orthogonal" claim
- Example: compare single-type sparsification against two-type joint sparsification

3. Actual sparsity after token protection

- Target sparsity
- realized sparsity
- protected-token ratio

4. Exact calibration statistics

- dataset list
- sample count
- token/vector count
- selected layers
- target sparsity / target `K`

### Recommended reductions

If time is limited, delete or weaken these instead of inventing more numbers:

- multi-model PPL table
- downstream multi-task table
- performance figures without code provenance

## Files Produced

- `aidcs_repro/results/layer_type_sensitivity_formal/layer_type_sensitivity.csv`
- `aidcs_repro/results/layer_type_sensitivity_formal/safe_sparsity_summary.csv`
- `aidcs_repro/results/layer_type_sensitivity_formal/layer_type_sensitivity_plot.pdf`
- `aidcs_repro/results/threshold_stability_formal/threshold_stability.csv`
- `aidcs_repro/results/threshold_stability_formal/layer_0_threshold_boxplot.pdf`
- `aidcs_repro/results/threshold_stability_formal/layer_10_threshold_boxplot.pdf`
- `aidcs_repro/results/threshold_stability_formal/layer_20_threshold_boxplot.pdf`
- `aidcs_repro/results/massive_token_formal/massive_token_analysis.csv`

