# Llama3-8B Pilot Results

Model: `/mnt/home/zh/model/Meta-Llama-3-8B-Instruct`

Datasets:

- `trec`
- `triviaqa`
- `passage_count`
- `qasper`

Samples per dataset: `3`

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| fp16 | 100.00 | 100.00 | 0.00 | 26.40 | 56.60 |
| per-token 4bit | 100.00 | 100.00 | 0.00 | 27.51 | 56.88 |
| per-token-head 4bit | 100.00 | 100.00 | 0.00 | 27.08 | 56.77 |
| per-token-tile 4bit | 100.00 | 100.00 | 33.33 | 26.40 | 64.93 |
| per-token-tile 4bit + hadamard | 100.00 | 100.00 | 0.00 | 17.50 | 54.38 |
| per-token-tile 4bit + tile-hadamard(64) | 100.00 | 100.00 | 33.33 | 26.40 | 64.93 |
| per-token-tile 3bit | 100.00 | 100.00 | 0.00 | 13.68 | 53.42 |
| per-token-tile 3bit + hadamard | 100.00 | 100.00 | 0.00 | 18.06 | 54.52 |
| per-token-tile 2bit | 33.33 | 0.00 | 0.00 | 0.00 | 8.33 |
| per-token-tile 2bit + hadamard | 33.33 | 33.33 | 33.33 | 3.70 | 25.92 |

Notes:

- This is a pilot matrix, not the full LongBench all-task evaluation.
- A few samples reached the `8192` context boundary warning during generation, but the runs completed and wrote `result.json`.

## Per-Token

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| per-token 4bit | 100.00 | 100.00 | 0.00 | 27.51 | 56.88 |
| per-token 4bit + hadamard | 100.00 | 100.00 | 0.00 | 17.50 | 54.38 |
| per-token 3bit | 33.33 | 29.39 | 0.00 | 0.00 | 15.68 |
| per-token 3bit + hadamard | 100.00 | 100.00 | 33.33 | 19.79 | 63.28 |
| per-token 2bit | 0.00 | 0.00 | 0.00 | 0.93 | 0.23 |
| per-token 2bit + hadamard | 0.00 | 4.44 | 0.00 | 5.67 | 2.53 |

## Per-Token-Head

| Config | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| per-token-head 4bit | 100.00 | 100.00 | 0.00 | 27.08 | 56.77 |
| per-token-head 4bit + hadamard | 100.00 | 100.00 | 0.00 | 24.90 | 56.23 |
| per-token-head 3bit | 100.00 | 100.00 | 33.33 | 20.54 | 63.47 |
| per-token-head 3bit + hadamard | 100.00 | 100.00 | 0.00 | 15.59 | 53.90 |
| per-token-head 2bit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| per-token-head 2bit + hadamard | 33.33 | 13.33 | 0.00 | 0.00 | 11.66 |
