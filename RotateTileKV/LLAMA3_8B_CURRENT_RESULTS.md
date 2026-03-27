# Llama3-8B Current Results

Model:

- `/mnt/home/zh/model/Meta-Llama-3-8B-Instruct`

Datasets:

- `trec`
- `triviaqa`
- `passage_count`
- `qasper`

## Completed Results

### Pilot Runs: `limit=3`

Result directory:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b`

Pilot here means:

- only a small LongBench subset is used
- only `4` tasks are used: `trec,triviaqa,passage_count,qasper`
- only `3` samples are taken from each task
- the goal is to observe quick trends, not to claim final benchmark numbers

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| fp16 | 100.00 | 100.00 | 0.00 | 26.40 | 56.60 |
| per-token 4bit | 100.00 | 100.00 | 0.00 | 27.51 | 56.88 |
| per-token 4bit + full hadamard | 100.00 | 100.00 | 0.00 | 17.50 | 54.38 |
| per-token 3bit | 33.33 | 29.39 | 0.00 | 0.00 | 15.68 |
| per-token 3bit + full hadamard | 100.00 | 100.00 | 33.33 | 19.79 | 63.28 |
| per-token 2bit | 0.00 | 0.00 | 0.00 | 0.93 | 0.23 |
| per-token 2bit + full hadamard | 0.00 | 4.44 | 0.00 | 5.67 | 2.53 |
| per-token-head 4bit | 100.00 | 100.00 | 0.00 | 27.08 | 56.77 |
| per-token-head 4bit + full hadamard | 100.00 | 100.00 | 0.00 | 24.90 | 56.23 |
| per-token-head 3bit | 100.00 | 100.00 | 33.33 | 20.54 | 63.47 |
| per-token-head 3bit + full hadamard | 100.00 | 100.00 | 0.00 | 15.59 | 53.90 |
| per-token-head 2bit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| per-token-head 2bit + full hadamard | 33.33 | 13.33 | 0.00 | 0.00 | 11.66 |
| per-token-tile 4bit | 100.00 | 100.00 | 33.33 | 26.40 | 64.93 |
| per-token-tile 4bit + full hadamard | 100.00 | 100.00 | 0.00 | 17.50 | 54.38 |
| per-token-tile 4bit + tile hadamard(64) | 100.00 | 100.00 | 33.33 | 26.40 | 64.93 |
| per-token-tile 3bit | 100.00 | 100.00 | 0.00 | 13.68 | 53.42 |
| per-token-tile 3bit + full hadamard | 100.00 | 100.00 | 0.00 | 18.06 | 54.52 |
| per-token-tile 3bit + tile hadamard(64) | 100.00 | 100.00 | 0.00 | 17.25 | 54.31 |
| per-token-tile 2bit | 33.33 | 0.00 | 0.00 | 0.00 | 8.33 |
| per-token-tile 2bit + full hadamard | 33.33 | 33.33 | 33.33 | 3.70 | 25.92 |
| per-token-tile 2bit + tile hadamard(64) | 66.67 | 46.67 | 0.00 | 9.88 | 30.80 |

### Larger Partial Runs: `limit=10`

Result directory:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_l10`

Currently completed:

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| per-token 4bit | 80.00 | 100.00 | 10.00 | 30.17 | 55.04 |
| per-token 4bit + full hadamard | 80.00 | 100.00 | 10.00 | 29.21 | 54.80 |
| per-token-tile 2bit | 10.00 | 3.76 | 0.00 | 0.95 | 3.68 |
| per-token-tile 2bit + tile hadamard(64) | 40.00 | 34.00 | 0.00 | 5.36 | 19.84 |
| per-token-tile 3bit | 70.00 | 100.00 | 0.00 | 28.54 | 49.63 |
| per-token-tile 3bit + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 21.31 | 52.83 |
| per-token-tile 4bit + tile hadamard(64) | 70.00 | 100.00 | 20.00 | 43.09 | 58.27 |

### Larger Partial Runs with Residual Window: `limit=10`, `residual_length=128`

Result directory:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10`

Completed:

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| per-token-tile 4bit + residual128 | 80.00 | 100.00 | 10.00 | 42.87 | 58.22 |
| per-token-tile 4bit + residual128 + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 44.30 | 58.58 |
| per-token-tile 3bit + residual128 | 70.00 | 100.00 | 10.00 | 42.41 | 55.60 |
| per-token-tile 3bit + residual128 + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 38.54 | 57.13 |
| per-token-tile 2bit + residual128 | 20.00 | 18.98 | 0.00 | 10.00 | 12.25 |
| per-token-tile 2bit + residual128 + tile hadamard(64) | 70.00 | 94.00 | 0.00 | 22.20 | 46.55 |

### V-Only Quantization with Residual Window: `limit=10`, `residual_length=128`

Result directory:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10`

Completed:

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| K=fp16, V-only 4bit tile | 80.00 | 100.00 | 20.00 | 39.26 | 59.81 |
| K=fp16, V-only 3bit tile | 80.00 | 100.00 | 10.00 | 42.17 | 58.04 |
| K=fp16, V-only 2bit tile | 80.00 | 100.00 | 20.00 | 30.86 | 57.72 |
| K=fp16, V-only 4bit head | 80.00 | 100.00 | 10.00 | 38.84 | 57.21 |
| K=fp16, V-only 2bit head | 70.00 | 100.00 | 20.00 | 28.69 | 54.67 |

Observation:

- `V-only` is consistently strong even when the bit width is low.
- This indicates that the main bottleneck is not `V` quantization.

### K-Only Quantization with Residual Window: `limit=10`, `residual_length=128`

Result directory:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10`

Completed:

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| V=fp16, K-only tile 4bit | 80.00 | 100.00 | 20.00 | 43.62 | 60.91 |
| V=fp16, K-only tile 4bit + tile hadamard(64) | 80.00 | 90.00 | 10.00 | 43.68 | 55.92 |
| V=fp16, K-only tile 3bit | 80.00 | 100.00 | 10.00 | 26.46 | 54.12 |
| V=fp16, K-only tile 3bit + tile hadamard(64) | 80.00 | 100.00 | 20.00 | 32.14 | 58.03 |
| V=fp16, K-only tile 2bit | 20.00 | 44.66 | 0.00 | 10.00 | 18.66 |
| V=fp16, K-only tile 2bit + tile hadamard(64) | 80.00 | 90.91 | 0.00 | 14.10 | 46.25 |

Observation:

- `K-only` is much more sensitive than `V-only`.
- Tile-sized Hadamard helps a lot at low bit, especially for `2bit` and `3bit`.
- At `4bit`, tile-sized Hadamard is not clearly beneficial on the current subset.

Notes:

- `limit=10` results are still incomplete.
- Previous `limit=10` attempts for `fp16` and plain `per-token-tile 4bit` hit OOM during earlier queue runs.

## Current Background Status

Queue controller:

- `/mnt/home/zh/mustafar/RotateTileKV/queue_longbench_jobs.py`

Active queue:

- no active RotateTileKV queue at the time of this snapshot

Currently running workers at the time of this snapshot:

- no active RotateTileKV workers

Pending in the current queue after those:

- `residual128 per-token / per-token-head` queue can be resumed if needed
- `v-only` queue completed

Queue artifacts:

- job file: `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_token_head_limit10.json`
- job file: `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_vonly_limit10.json`
- logs: `/mnt/home/zh/mustafar/RotateTileKV/bg_logs`

## Short Takeaways

- In the small pilot, `per-token-tile 4bit` is the strongest configuration among the tested 4bit variants.
- Full-head Hadamard hurts `per-token-tile 4bit`, but tile-sized Hadamard(64) removes that regression.
- For `per-token-tile`, both full and tile-sized Hadamard help at `2bit`, and tile-sized Hadamard is currently the best among the two.
- In the current `limit=10` snapshot, `per-token 4bit` and `per-token 4bit + full hadamard` are very close, with a slight edge to no-Hadamard.
- In the pilot, `per-token 3bit + hadamard` improves dramatically over `per-token 3bit`.
- In the pilot, `per-token-head 3bit` works well even without Hadamard, and adding Hadamard is not helpful there.
- Adding `residual_length=128` improves the tile path further, especially at `2bit`.
- `V-only` quantization is very strong even at low bit: `4bit=59.81`, `3bit=58.04`, `2bit=57.72`.
- This strongly suggests that most of the remaining degradation in the current joint-quantized RotateTileKV runs comes from quantizing `K`, not `V`.
