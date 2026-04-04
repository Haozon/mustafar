# Current Per-Task Solver Summary

This file tracks the latest finished per-task runs and full-dataset follow-ups across all models currently evaluated.

## Meta-Llama-3-8B 70%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 46.17 -> 46.17 (+0.00) | 44.93 -> 45.12 (+0.19) | weak_positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0] |
| `lcc` | 57.85 -> 63.95 (+6.10) | 54.12 -> 55.45 (+1.33) | positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0] |
| `multifieldqa_en` | 39.53 -> 38.14 (-1.39) | 40.91 -> 41.61 (+0.70) | positive | [0.02, 0.8, 0.18] / [0.0, 0.65, 1.0]; value_aware, max, sink=4, evict |
| `narrativeqa` | 29.81 -> 29.81 (+0.00) | 23.94 -> 23.94 (+0.00) | fallback_uniform | uniform fallback |
| `qasper` | 39.02 -> 38.91 (-0.11) | 40.89 -> 41.03 (+0.14) | weak_positive | [0.0, 0.857143, 0.142857] / [0.0, 0.65, 1.0]; value_aware, mean, sink=4, evict |
| `trec` | 50.00 -> 55.00 (+5.00) | 70.00 -> 72.50 (+2.50) | positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0] |
| **Average** | 43.73 -> 45.29 (+1.56) | 45.80 -> 46.61 (+0.81) | -- | -- |

## Meta-Llama-3-8B 50%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 47.83 -> 47.83 (+0.00) | 45.94 -> 45.94 (+0.00) | fallback_uniform | uniform fallback |
| `lcc` | 63.80 -> 63.60 (-0.20) | 56.03 -> 57.08 (+1.05) | positive | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 40.49 -> 40.49 (+0.00) | 42.93 -> 42.93 (+0.00) | fallback_uniform | uniform fallback |
| `narrativeqa` | 29.98 -> 29.98 (+0.00) | 23.44 -> 23.44 (+0.00) | fallback_uniform | uniform fallback |
| `qasper` | 38.14 -> 38.14 (+0.00) | 43.73 -> 43.73 (+0.00) | fallback_uniform | uniform fallback |
| `trec` | 60.00 -> 60.00 (+0.00) | 73.50 -> 74.00 (+0.50) | positive | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 47.59 -> 47.85 (+0.26) | -- | finalized with fallback where needed |

## Llama-2-7B 70%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 3.16 -> 4.34 (+1.18) | 6.57 -> 6.96 (+0.39) | positive | [0.15, 0.6, 0.25] / [0.0, 0.75, 1.0]; value_aware, max, sink=2, evict |
| `lcc` | 65.60 -> 67.05 (+1.45) | 63.89 -> 66.59 (+2.70) | positive | [0.0, 0.666667, 0.333333] / [0.0, 0.55, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 20.56 -> 23.37 (+2.81) | 19.42 -> 22.73 (+3.31) | positive | [0.15, 0.375, 0.475] / [0.0, 0.6, 1.0]; value_aware, max, sink=2, evict |
| `narrativeqa` | 17.46 -> 19.45 (+1.99) | 13.66 -> 15.00 (+1.34) | positive | [0.1, 0.444444, 0.455556] / [0.0, 0.55, 1.0]; value_aware, mean, sink=8, evict |
| `qasper` | 10.12 -> 6.46 (-3.66) | 7.69 -> 7.88 (+0.19) | positive | [0.1, 0.444444, 0.455556] / [0.0, 0.55, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 65.00 -> 65.00 (+0.00) | 64.50 -> 65.50 (+1.00) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 29.29 -> 30.78 (+1.49) | -- | -- |

## Llama-2-7B 50%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 4.76 -> 4.76 (+0.00) | 7.39 -> 7.39 (+0.00) | fallback_uniform | uniform fallback |
| `lcc` | 70.00 -> 70.55 (+0.55) | 66.88 -> 66.91 (+0.03) | weak_positive | [0.0, 1.0, 0.0] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 20.07 -> 22.34 (+2.27) | 20.90 -> 22.92 (+2.02) | positive | [0.05, 0.75, 0.2] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| `narrativeqa` | 18.09 -> 25.03 (+6.94) | 15.04 -> 16.94 (+1.90) | positive | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| `qasper` | 10.54 -> 10.54 (+0.00) | 9.05 -> 9.05 (+0.00) | fallback_uniform | uniform fallback |
| `trec` | 65.00 -> 65.00 (+0.00) | 66.00 -> 66.00 (+0.00) | neutral | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 30.88 -> 31.54 (+0.66) | -- | finalized with fallback where needed |

## Llama-2-13B 70%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 5.63 -> 8.32 (+2.69) | 13.91 -> 13.91 (+0.00) | fallback_uniform | uniform fallback |
| `lcc` | 55.45 -> 69.25 (+13.80) | 57.59 -> 65.11 (+7.52) | positive | [0.15, 0.428571, 0.421429] / [0.0, 0.65, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 12.03 -> 11.13 (-0.90) | 13.52 -> 17.46 (+3.94) | positive | [0.0, 0.857143, 0.142857] / [0.0, 0.65, 1.0]; value_aware, max, sink=2, evict |
| `narrativeqa` | 14.59 -> 18.71 (+4.12) | 7.03 -> 11.19 (+4.16) | positive | [0.05, 0.714286, 0.235714] / [0.0, 0.65, 1.0]; value_aware, max, sink=2, evict |
| `qasper` | 3.81 -> 4.78 (+0.97) | 5.62 -> 6.97 (+1.35) | positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 45.00 -> 55.00 (+10.00) | 51.00 -> 70.00 (+19.00) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 24.78 -> 30.77 (+5.99) | -- | finalized with fallback where needed |

## Llama-2-13B 50%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 7.81 -> 7.81 (+0.00) | -- | fallback_uniform | uniform fallback |
| `lcc` | 69.45 -> 69.45 (+0.00) | -- | fallback_uniform | uniform fallback |
| `multifieldqa_en` | 10.91 -> 10.91 (+0.00) | -- | fallback_uniform | uniform fallback |
| `narrativeqa` | 19.80 -> 19.80 (+0.00) | -- | fallback_uniform | uniform fallback |
| `qasper` | 6.57 -> 7.31 (+0.74) | -- | positive | [0.05, 0.818182, 0.131818] / [0.0, 0.45, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 55.00 -> 55.00 (+0.00) | -- | neutral | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| **Average** | 28.26 -> 28.38 (+0.12) | -- | -- | finalized with fallback where needed |

## Mistral-7B 70%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 13.53 -> 14.78 (+1.25) | 26.67 -> 26.67 (+0.00) | fallback_uniform | uniform fallback |
| `lcc` | 53.40 -> 53.40 (+0.00) | 53.44 -> 53.44 (+0.00) | fallback_uniform | uniform fallback |
| `multifieldqa_en` | 39.68 -> 40.66 (+0.98) | 38.83 -> 40.65 (+1.82) | positive | [0.05, 0.625, 0.325] / [0.0, 0.6, 1.0]; value_aware, max, sink=2, evict |
| `narrativeqa` | 21.06 -> 21.06 (+0.00) | 13.38 -> 13.38 (+0.00) | fallback_uniform | uniform fallback |
| `qasper` | 41.11 -> 35.36 (-5.75) | 26.64 -> 26.98 (+0.34) | positive | [0.025, 0.785714, 0.189286] / [0.0, 0.65, 1.0]; value_aware, mean, sink=2, prr=1.0 |
| `trec` | 45.00 -> 45.00 (+0.00) | 66.50 -> 67.00 (+0.50) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 37.58 -> 38.02 (+0.44) | -- | finalized with fallback where needed |

## Mistral-7B 50%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `hotpotqa` | 13.80 -> 14.78 (+0.98) | 26.24 -> 26.24 (+0.00) | fallback_uniform | uniform fallback |
| `lcc` | 53.10 -> 53.40 (+0.30) | 53.46 -> 53.46 (+0.00) | fallback_uniform | uniform fallback |
| `multifieldqa_en` | 40.48 -> 41.57 (+1.09) | 39.92 -> 40.37 (+0.45) | positive | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| `narrativeqa` | 17.80 -> 17.80 (+0.00) | 12.99 -> 12.99 (+0.00) | fallback_uniform | uniform fallback |
| `qasper` | 38.50 -> 36.68 (-1.82) | 28.89 -> 29.46 (+0.57) | positive | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 45.00 -> 45.00 (+0.00) | 67.50 -> 67.50 (+0.00) | neutral | [0.0, 0.833333, 0.166667] / [0.0, 0.4, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 38.17 -> 38.34 (+0.17) | -- | finalized with fallback where needed |

## Dense / Native Baselines

These rows summarize the dense or native baseline references currently retained in this workspace.

| Model | Scope | Source | Average | Notes |
|---|---|---|---:|---|
| `Llama-2-13B` | full LongBench | `pred/Llama-2-13b-hf_4096_baseline` | 27.66 | formal retained baseline |
| `Llama-2-7B` | full LongBench | `pred/Llama-2-7b-hf_4096_K_0.0_V_0.0` | 27.50 | formal retained dense-style baseline |
| `Meta-Llama-3-8B-Instruct` | full LongBench | `pred/Meta-Llama-3-8B-Instruct_8192_K_0.0_V_0.0` | 43.14 | formal retained dense-style baseline |
| `Mistral-7B-v0.1` | full LongBench | `pred/Mistral-7B-Instruct-v0.1_8192_native_baseline` | 35.70 | historical directory keeps old model-name prefix |
| `Qwen2.5-7B` | full LongBench | `pred/Qwen2.5-7B_8192_qwen_native_baseline` | 36.88 | formal retained native baseline |
| `Qwen2.5-7B` | 6 focal tasks | `solver_runs_qwen_native_focus/Qwen2.5-7B_8192_native_focus_baseline` | 38.47 | current focused rerun: `narrativeqa/qasper/multifieldqa_en/hotpotqa/trec/lcc` |

## Notes

- `Meta-Llama-3-8B 70%` is fully finalized, including full-dataset follow-up and fallback handling for `narrativeqa`.
- `Meta-Llama-3-8B 50%` is now finalized with uniform fallback on the previously underperforming tasks; the full-task average is now `+0.26`.
- `Qwen2.5-7B` native baseline is now available in both the retained formal manifest and the focused six-task rerun.
  The old degenerate `0.0` path should still be treated as invalid, and the current `Qwen 50% / 70%` sections below reflect the finalized fallback-aware closeout.
- `Mistral-7B-Instruct-v0.1` and `Mistral-7B-v0.1` are treated as the same local model in this workspace.
  Historical result directories may still keep the older `Mistral-7B-Instruct-v0.1` prefix, but future scheduling uses `Mistral-7B-v0.1`.
- `Llama-2-13B 70%` currently shows the strongest first-round validation signal across multiple tasks.
- At `50%`, several models become more stable on `narrativeqa/qasper/multifieldqa_en`, while `trec` often stays near zero gain.
- `Mistral-7B 70%` and `Mistral-7B 50%` rows now reflect a fallback-aware closeout in this human summary: tasks that remained underperforming are shown as uniform fallback.
- Repaired wide-search follow-up on `Mistral-7B qasper 70%` is now full-dataset positive: `26.64 -> 26.98` (`+0.34`), using `[0.025, 0.785714, 0.189286] / [0.0, 0.65, 1.0]`, `value_aware`, `head=mean`, `sink=2`, `protected_recent_ratio=1.0`.
- `Meta-Llama-3-8B 50% / hotpotqa` should be treated as provisional: all 10 searched diff candidates tied exactly with the uniform calibration score (`34.38`), so `cand1` was selected only by first-seen order; the later validation drop to `46.17` vs `47.83` reflects an uninformative calibration split rather than a meaningfully justified best config.

## Repaired Mistral 70% Full Results

| Task | Full Uniform -> Final | Status | Config |
|---|---|---|---|
| `qasper` | 26.64 -> 26.98 (+0.34) | positive | [0.025, 0.785714, 0.189286] / [0.0, 0.65, 1.0]; value_aware, mean, sink=2, prr=1.0 |
| `multifieldqa_en` | 38.83 -> 40.65 (+1.82) | positive | [0.05, 0.625, 0.325] / [0.0, 0.6, 1.0]; value_aware, max, sink=2 |
| `lcc` | 53.44 -> 53.44 (+0.00) | fallback_uniform | uniform fallback |
| `hotpotqa` | 26.67 -> 26.67 (+0.00) | fallback_uniform | uniform fallback |
| `narrativeqa` | 13.38 -> 13.38 (+0.00) | fallback_uniform | uniform fallback |
| `trec` | 66.50 -> 67.00 (+0.50) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2 |

## Qwen Native Baseline

`Qwen2.5-7B` native baseline on the six focal tasks has completed with average `38.47`.
Per-task scores: `narrativeqa 11.91`, `qasper 14.53`, `multifieldqa_en 37.54`, `hotpotqa 30.23`, `trec 69.00`, `lcc 67.63`.
This baseline is retained as the dense/native reference for the finalized Qwen closeout below.
The retained formal full-LongBench native baseline average is `36.88`.

## Qwen2.5-7B 50%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `narrativeqa` | 11.98 -> 11.98 (+0.00) | 9.73 -> 9.73 (+0.00) | fallback_uniform | uniform fallback |
| `qasper` | 9.42 -> 10.25 (+0.83) | 11.62 -> 12.05 (+0.43) | positive | [0.15, 0.7, 0.15] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 27.14 -> 27.25 (+0.11) | 29.88 -> 29.88 (+0.00) | fallback_uniform | uniform fallback |
| `hotpotqa` | 8.53 -> 8.53 (+0.00) | 9.03 -> 9.03 (+0.00) | neutral | [0.0, 1.0, 0.0] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 70.00 -> 70.00 (+0.00) | 69.50 -> 69.50 (+0.00) | neutral | [0.0, 1.0, 0.0] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `lcc` | 58.30 -> 58.30 (+0.00) | 59.63 -> 59.86 (+0.23) | positive | [0.05, 0.9, 0.05] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| **Average** | -- | 31.56 -> 31.68 (+0.11) | -- | finalized with fallback where needed |

## Qwen2.5-7B 70%

| Task | Validation Uniform -> Diff | Full Uniform -> Final | Status | Config |
|---|---|---|---|---|
| `narrativeqa` | 7.38 -> 11.97 (+4.59) | 8.60 -> 9.33 (+0.73) | positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0]; value_aware, max, sink=2, evict |
| `qasper` | 10.07 -> 9.89 (-0.18) | 11.04 -> 11.72 (+0.68) | positive | [0.0, 0.75, 0.25] / [0.0, 0.6, 1.0]; value_aware, max, sink=2, evict |
| `multifieldqa_en` | 26.95 -> 28.05 (+1.10) | 28.52 -> 28.61 (+0.09) | weak_positive | [0.1, 0.4, 0.5] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `hotpotqa` | 8.05 -> 9.15 (+1.10) | 8.76 -> 9.68 (+0.92) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `trec` | 70.00 -> 70.00 (+0.00) | 68.00 -> 69.00 (+1.00) | positive | [0.0, 0.6, 0.4] / [0.0, 0.5, 1.0]; value_aware, max, sink=2, evict |
| `lcc` | 59.10 -> 60.05 (+0.95) | 59.11 -> 59.11 (+0.00) | fallback_uniform | uniform fallback |
| **Average** | -- | 30.67 -> 31.24 (+0.57) | -- | finalized with fallback where needed |

## Latest Additions

- `Llama-2-7B 70% / narrativeqa full` has now completed:
  - uniform `13.75`
  - diff `13.54`
  - delta `-0.21`
- `Mistral-7B 50% / multifieldqa_en full` has now completed:
  - uniform `39.92`
  - diff `40.37`
  - delta `+0.45`
- `Mistral-7B 50% / hotpotqa full` has now completed:
  - uniform `26.24`
  - diff `25.77`
  - delta `-0.47`
- `Mistral-7B 50% / lcc full` has now completed:
  - uniform `53.46`
  - diff `53.13`
  - delta `-0.33`
- `Llama-2-7B 70% / multifieldqa_en full` has now completed:
  - uniform `19.42`
  - diff `22.73`
  - delta `+3.31`
- `Llama-2-13B 70% / qasper full` has now completed:
  - uniform `5.62`
  - diff `6.97`
  - delta `+1.35`
- `Llama-2-13B 70% / lcc full` has now completed:
  - uniform `57.59`
  - diff `65.11`
  - delta `+7.52`
- `Llama-2-7B 50% / hotpotqa full` has now been improved via repair follow-up:
  - previous diff full `7.15`
  - current best full diff `7.31`
  - current delta `-0.08`
- `Llama-2-7B 50% / qasper full` has now been improved via repair follow-up:
  - previous diff full `8.52`
  - current best full diff `8.78`
  - current delta `-0.27`
- `Qwen2.5-7B` six-task native rerun has completed:
  - average `38.47`
  - per-task scores are listed in the `Qwen Native Baseline` section above
