# JSQKV-lite Small-Sample Comparison

Model:

- `/home/zh/model/Meta-Llama-3-8B-Instruct`

Setting:

- `selected6` tasks:
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `trec`
  - `lcc`
- sample size:
  - `limit = 12`
- context length:
  - `4096`
- target sparsity budget:
  - `70%`

## Methods

1. `uniform + KIVI-align fake`
   - source: `RotateTileKV`
   - `2bit`
   - `residual_length = 128`

2. `DiffSparseKV only`
   - source: `DiffSparseKV`
   - config:
     - `target_distribution = [0.0, 0.75, 0.25]`
     - `sparsity_levels = [0.0, 0.6, 1.0]`
     - `importance_mode = value_aware`
     - `head_aggregation_mode = max`
     - `value_sink_keep = 4`

3. `JSQKV-lite`
   - source: `JSQKV`
   - composition:
     - `DiffSparseKV` token sparsification
     - then `RotateTileKV` fake quantization
   - quantization:
     - `2bit`
     - `per-token-tile`
     - `tile_size = 64`
     - `residual_length = 128`
     - `tile hadamard(64)`

## Results

| Method | narrativeqa | qasper | multifieldqa_en | hotpotqa | trec | lcc | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform + KIVI-align fake | 13.26 | 17.84 | 29.18 | 27.50 | 75.00 | 24.92 | 31.28 |
| DiffSparseKV only | 31.65 | 37.44 | 53.78 | 33.33 | 66.67 | 62.67 | 47.59 |
| JSQKV-lite | 9.42 | 21.36 | 34.06 | 15.33 | 8.33 | 12.92 | 16.90 |

## File Paths

- `uniform + KIVI-align fake`
  - `/home/zh/nas/nas_10g/mustafar/JSQKV_runs/compare_limit12_meta70_len4096/kivi_align_fake_2bit_limit12_len4096/result.json`
- `DiffSparseKV only`
  - `/home/zh/nas/nas_10g/mustafar/JSQKV_runs/compare_limit12_meta70_len4096/Meta-Llama-3-8B-Instruct_4096_diff_sparse_kv_0.70_diffsparse_only_meta70_limit12_len4096/result.json`
- `JSQKV-lite`
  - `/home/zh/nas/nas_10g/mustafar/JSQKV_runs/compare_limit12_meta70_len4096/jsqkv_meta70_2bit_limit12_len4096/result.json`

## Immediate Conclusion

- The current `JSQKV-lite` implementation is **not usable** as a proof-of-concept result.
- It is substantially worse than both:
  - `uniform + KIVI-align fake`
  - `DiffSparseKV only`
- Therefore, the current naive composition:
  - `DiffSparseKV` + post-hoc `RotateTileKV` fake quantization
  does **not** preserve the intended accuracy behavior.

## Likely Reasons

- `RotateTileKV` quantization was originally tuned for dense/uniform-style KV cache layout, not for an already evicted and partially sparse KV cache.
- Applying fake quantization after `DiffSparseKV` changes the value range and token layout, but the current quantization path still assumes the original dense residual-window behavior.
- The current `JSQKV-lite` path quantizes the active KV cache every decode step, which may over-amplify error.
- The Hadamard / quantization path is currently attached without re-tuning the sparse policy and without validating whether the residual-window quantization semantics still match the sparsified cache layout.

## Recommended Next Step

- Do **not** launch a large full benchmark from this version.
- First, simplify the JSQKV path:
  - disable runtime re-quantization in decode
  - only quantize the prefill-compressed KV cache once
  - re-run the same `limit=12` comparison
- If that still fails, the next thing to test is:
  - keep `V` dense
  - quantize `K` only
  - or keep the current `DiffSparseKV` selector but quantize only the evicted-prefix region rather than the full retained cache.
