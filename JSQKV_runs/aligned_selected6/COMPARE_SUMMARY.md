# JSQKV Aligned Selected6 Summary

Date: `2026-04-04`

Model:

- `/home/zh/model/Meta-Llama-3-8B-Instruct`

Protocol:

- `selected6`:
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `trec`
  - `lcc`
- `limit = 12`
- `max_length = 4096`
- evaluation prompt / max_gen loaded from:
  - `config/dataset2prompt.json`
  - `config/dataset2maxlen.json`

## Settings

1. `DiffSparseKV only`
   - `target_distribution = [0.0, 0.75, 0.25]`
   - `sparsity_levels = [0.0, 0.6, 1.0]`
   - `value_aware`, `head=max`, `sink=4`
   - `fp16`

2. `M+K proxy, 4bit`
   - `target_distribution = [0.0, 1.0, 0.0]`
   - `sparsity_levels = [0.0, 0.7, 1.0]`
   - `quant_impl = kivi`
   - `k_quant_scheme = kivi-channel`
   - `v_quant_scheme = per-token-head`

3. `JSQKV, 4bit`
   - same differential sparsity policy as `DiffSparseKV only`
   - `Per-Token-Tile`
   - `4bit`
   - `no hadamard`

4. `M+K proxy, 2bit`
   - same as `M+K proxy, 4bit`
   - `2bit`

5. `JSQKV, 2bit`
   - same differential sparsity policy as `DiffSparseKV only`
   - `Per-Token-Tile`
   - `2bit`
   - `tile hadamard(64)`

## Results

| Method | NarrativeQA | Qasper | MultiFieldQA-En | HotpotQA | TREC | LCC | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiffSparseKV only | 31.65 | 37.60 | 51.45 | 33.33 | 66.67 | 60.42 | 46.85 |
| M+K proxy, 4bit | 27.95 | 37.00 | 53.03 | 33.33 | 66.67 | 60.50 | 46.41 |
| JSQKV, 4bit | 33.90 | 38.52 | 54.84 | 33.33 | 66.67 | 60.42 | 47.95 |
| M+K proxy, 2bit | 21.97 | 32.04 | 54.07 | 31.67 | 66.67 | 58.25 | 44.11 |
| JSQKV, 2bit | 29.80 | 34.94 | 54.49 | 28.33 | 66.67 | 62.33 | 46.09 |

## Delta vs. M+K proxy

### 4bit

- `NarrativeQA`: `+5.95`
- `Qasper`: `+1.52`
- `MultiFieldQA-En`: `+1.81`
- `HotpotQA`: `+0.00`
- `TREC`: `+0.00`
- `LCC`: `-0.08`
- `Average`: `+1.54`

### 2bit

- `NarrativeQA`: `+7.83`
- `Qasper`: `+2.90`
- `MultiFieldQA-En`: `+0.42`
- `HotpotQA`: `-3.34`
- `TREC`: `+0.00`
- `LCC`: `+4.08`
- `Average`: `+1.98`

## Immediate Takeaways

- After aligning the evaluation protocol with `DiffSparseKV`, the `JSQKV` wrapper
  can now reproduce the sparse-only baseline (`46.85` average with `fp16`).
- On this aligned small-sample setup, the current `JSQKV` configurations are now
  already better than the matched `M+K` proxy at both:
  - `4bit` (`47.95 vs 46.41`)
  - `2bit` (`46.09 vs 44.11`)
- `4bit` currently prefers:
  - `no hadamard`
- `2bit` currently prefers:
  - `tile hadamard(64)`
- The main remaining weakness in the current `2bit` setting is:
  - `HotpotQA`

## File Paths

- `DiffSparseKV only`
  - `JSQKV_runs/aligned_selected6/meta70_diffsparse_fp16_limit12/result.json`
- `M+K proxy, 4bit`
  - `JSQKV_runs/aligned_selected6/meta70_uniformkivi_4bit_limit12/result.json`
- `JSQKV, 4bit`
  - `JSQKV_runs/aligned_selected6/meta70_jsqkv_4bit_nohad_limit12/result.json`
- `M+K proxy, 2bit`
  - `JSQKV_runs/aligned_selected6/meta70_uniformkivi_2bit_limit12/result.json`
- `JSQKV, 2bit`
  - `JSQKV_runs/aligned_selected6/meta70_jsqkv_2bit_tilehad_limit12/result.json`
