# KIVI vs RotateTileKV on Llama3-8B

Model:

- `/mnt/home/zh/model/Meta-Llama-3-8B-Instruct`

Protocol:

- datasets: `trec,triviaqa,passage_count,qasper`
- sample limit per dataset: `10`
- same prompt/truncation/scoring protocol as the current RotateTileKV subset evaluation

## KIVI Source

Upstream repo:

- `https://github.com/jy-yuan/KIVI.git`

Upstream commit used:

- `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`

Local clean clone:

- `/mnt/home/zh/mustafar/KIVI_upstream`

Compatibility patch applied for Llama3/GQA decode:

- `/mnt/home/zh/mustafar/KIVI_upstream/models/llama_kivi.py`

Patch summary:

- add `repeat_kv(value_states_full, self.num_key_value_groups)` on the decode branches where upstream directly multiplies attention weights with unrepeated KV heads

Without this patch, upstream `main` crashes on Llama3-8B-Instruct decode because query heads (`32`) and KV heads (`8`) mismatch in the value path.

## KIVI Results

Result directory:

- `/mnt/home/zh/mustafar/KIVI_upstream/exp_llama3_kivi_l10`

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| KIVI 2bit | 80.00 | 100.00 | 20.00 | 36.41 | 59.10 |
| KIVI 4bit | 80.00 | 100.00 | 20.00 | 42.17 | 60.54 |

## RotateTileKV Reference Points

Result directories:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_l10`
- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10`

Selected completed results:

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| per-token 4bit | 80.00 | 100.00 | 10.00 | 30.17 | 55.04 |
| per-token 4bit + full hadamard | 80.00 | 100.00 | 10.00 | 29.21 | 54.80 |
| per-token-head 4bit | 80.00 | 100.00 | 10.00 | 31.25 | 55.31 |
| per-token-head 4bit + full hadamard | 80.00 | 100.00 | 10.00 | 40.65 | 57.66 |
| per-token-tile 4bit + tile hadamard(64) | 70.00 | 100.00 | 20.00 | 43.09 | 58.27 |
| per-token-tile 4bit + residual128 | 80.00 | 100.00 | 10.00 | 42.87 | 58.22 |
| per-token-tile 4bit + residual128 + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 44.30 | 58.58 |
| per-token-tile 3bit + residual128 | 70.00 | 100.00 | 10.00 | 42.41 | 55.60 |
| per-token-tile 3bit + residual128 + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 38.54 | 57.13 |
| per-token-tile 2bit + residual128 | 20.00 | 18.98 | 0.00 | 10.00 | 12.25 |
| per-token-tile 2bit + residual128 + tile hadamard(64) | 70.00 | 94.00 | 0.00 | 22.20 | 46.55 |

## Comparison

### 4bit

| Method | average |
|---|---:|
| KIVI 4bit | 60.54 |
| per-token-tile 4bit + residual128 + tile hadamard(64) | 58.58 |
| per-token-head 4bit + full hadamard | 57.66 |
| per-token-tile 4bit + tile hadamard(64) | 58.27 |
| per-token-head 4bit | 55.31 |
| per-token 4bit | 55.04 |

Observation:

- On the current `limit=10` subset, KIVI 4bit is the best completed 4bit result.
- Adding a residual window improves the RotateTileKV 4bit tile-hadamard result slightly: `58.27 -> 58.58`.
- KIVI 4bit is still ahead, but the gap shrinks to about `+1.96` vs `per-token-tile 4bit + residual128 + tile hadamard(64)`.

### 2bit

| Method | average |
|---|---:|
| KIVI 2bit | 59.10 |
| per-token-tile 2bit + residual128 + tile hadamard(64) | 46.55 |
| per-token-tile 2bit + tile hadamard(64) | 19.84 |
| per-token-tile 2bit + residual128 | 12.25 |
| per-token-tile 2bit | 3.68 |

Observation:

- KIVI 2bit is dramatically stronger than the completed RotateTileKV 2bit runs under the same subset protocol.
- Residual window helps the RotateTileKV 2bit result a lot: `19.84 -> 46.55` for `tile hadamard(64)`.
- KIVI 2bit still leads, but the gap is much smaller after adding the residual window.

## Current Interpretation

- KIVI is a very strong low-bit baseline on this Llama3-8B subset.
- For `2bit`, KIVI still wins, but the residual-window version of RotateTileKV narrows the gap substantially.
- For `4bit`, KIVI also leads the completed RotateTileKV variants, though only by a small margin on this subset.
- RotateTileKV's most promising direction is now `per-token-tile + residual window + tile-aligned hadamard(64)`.
