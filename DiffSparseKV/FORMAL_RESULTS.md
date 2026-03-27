# Formal Results Manifest

This file lists the result directories currently retained under `DiffSparseKV/pred/` after cleanup.

## Retained formal results

These directories contain full LongBench-style result files with 16 task scores plus `average`.

| Model | Directory | Average |
|---|---|---:|
| Llama-2-13B | `pred/Llama-2-13b-hf_4096_baseline` | 27.66 |
| Llama-2-7B | `pred/Llama-2-7b-hf_4096_K_0.0_V_0.0` | 27.50 |
| Meta-Llama-3-8B-Instruct | `pred/Meta-Llama-3-8B-Instruct_8192_K_0.0_V_0.0` | 43.14 |
| Mistral-7B-Instruct-v0.1 | `pred/Mistral-7B-Instruct-v0.1_8192_native_baseline` | 35.70 |
| Qwen2.5-7B | `pred/Qwen2.5-7B_8192_qwen_native_baseline` | 36.88 |

## Removed as non-formal / unusable

The following categories were removed:

- `tmp_eval/` search and ablation intermediates
- `pred_main_tmp/`
- `pred_main_tmp_mid30/`
- partial-result directories under `pred/`:
  - `Llama-2-7b-hf_4096_diff_sparse_kv_0.70`
  - `Llama-2-7b-hf_4096_no_eviction`
  - `Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70`
- invalid / broken baseline:
  - `Qwen2.5-7B_8192_baseline`

## Important note

After cleanup, `DiffSparseKV` currently retains only formal baseline-style results.
No formal full-LongBench differential-sparsity result directory remains under `DiffSparseKV/pred/`.

