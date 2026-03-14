# Existing Benchmark Results Summary

- Generated at: 2026-03-05 11:29:31
- Data sources:
  - JSQKV raw JSON: `/mnt/home/zh/mustafar/JSQKV_benchmark/results/raw_data` (files: 22, non-empty: 7)
  - Sweep summaries: `/mnt/home/zh/mustafar/benchmark/benchmark_results_bs_output_sweep_*/summary.csv` (dirs: 2)
  - Benchmark outputs: `/mnt/home/zh/mustafar/benchmark/benchmark_results_*/ *_output.txt` (dirs: 3)

## Why Many Lines Are Missing in Plots

- Many result JSON files contain empty dicts for quant configs (`sparse_50_quant_2bit` / `sparse_70_quant_2bit`).
- Some scenarios only have one BS point (e.g., BS=8), not a full BS sweep.
- Some runs only exist in `benchmark_results_*` text outputs (single config/single BS), not in unified JSON.

## Scenario Coverage (5 Lines)

| Scenario | Dense | Sparse50 | Sparse70 | Sparse50+2bit | Sparse70+2bit |
|---|---:|---:|---:|---:|---:|
| `llama2_7b` in=2050, out=2048 | `1,2,4,6,8` | `1,2,4,6,8` | `1,2,4,6,8` | — | — |
| `llama3_8b` in=4096, out=32 | — | — | `1` | — | `1` |
| `llama3_8b` in=4096, out=1024 | — | — | `1,2,4,6,8` | — | `1,2,4,6,8` |
| `llama3_8b` in=4096, out=4096 | — | — | `1,2,4,6,8` | — | `1,2,4,6,8` |
| `llama3_8b` in=4098, out=1024 | `8` | `8` | `8` | `8` | `8` |
| `llama3_8b` in=4098, out=4096 | `1,2,4,6,8` | `1,2,4,6,8` | `1,2,4,6,8` | — | — |

## llama2_7b | input=2050 | output=2048

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1 | 30.16 | 185.09 | 31.31 | 14.59 | 67.91 |
| Dense | 2 | 59.20 | 360.09 | 32.51 | 16.62 | 69.19 |
| Dense | 4 | 99.67 | 708.04 | 32.28 | 20.69 | 82.20 |
| Dense | 6 | 112.50 | 1046.92 | 39.98 | 24.75 | 109.23 |
| Dense | 8 | 118.19 | 1376.90 | 49.64 | 28.82 | 138.62 |
| Sparse50 | 1 | 18.58 | 521.69 | 50.50 | 14.23 | 110.20 |
| Sparse50 | 2 | 35.47 | 997.38 | 53.02 | 15.91 | 115.48 |
| Sparse50 | 4 | 65.20 | 1974.63 | 55.44 | 19.23 | 125.65 |
| Sparse50 | 6 | 90.57 | 2984.55 | 57.92 | 22.53 | 135.67 |
| Sparse50 | 8 | 113.55 | 4039.93 | 60.58 | 25.85 | 144.29 |
| Sparse70 | 1 | 18.42 | 498.07 | 51.75 | 13.88 | 111.21 |
| Sparse70 | 2 | 34.83 | 919.37 | 52.88 | 15.21 | 117.60 |
| Sparse70 | 4 | 63.45 | 1770.80 | 56.26 | 17.82 | 129.12 |
| Sparse70 | 6 | 90.19 | 2787.80 | 60.89 | 20.44 | 136.24 |
| Sparse70 | 8 | 109.62 | 3576.34 | 60.81 | 23.05 | 149.46 |
| Sparse50+2bit | — | — | — | — | — | — |
| Sparse70+2bit | — | — | — | — | — | — |

## llama3_8b | input=4096 | output=32

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | — | — | — | — | — |
| Sparse50 | — | — | — | — | — | — |
| Sparse70 | 1 | 6.52 | 3976.25 | 162.35 | 18.17 | 4.91 |
| Sparse50+2bit | — | — | — | — | — | — |
| Sparse70+2bit | 1 | 6.68 | 2318.00 | 97.12 | 18.05 | 4.79 |

## llama3_8b | input=4096 | output=1024

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | — | — | — | — | — |
| Sparse50 | — | — | — | — | — | — |
| Sparse70 | 1 | 19.08 | 1464.11 | 49.56 | 18.17 | 53.68 |
| Sparse70 | 2 | 31.08 | 3188.20 | 59.01 | 21.40 | 65.89 |
| Sparse70 | 4 | 57.45 | 6013.69 | 57.64 | 27.81 | 71.29 |
| Sparse70 | 6 | 78.89 | 9028.55 | 63.12 | 34.25 | 77.88 |
| Sparse70 | 8 | 97.64 | 11332.78 | 66.79 | 40.66 | 83.90 |
| Sparse50+2bit | — | — | — | — | — | — |
| Sparse70+2bit | 1 | 13.59 | 1076.96 | 69.79 | 18.05 | 75.37 |
| Sparse70+2bit | 2 | 24.71 | 2125.20 | 76.70 | 21.16 | 82.89 |
| Sparse70+2bit | 4 | 40.57 | 4062.29 | 92.36 | 27.33 | 100.96 |
| Sparse70+2bit | 6 | 52.26 | 5932.62 | 103.71 | 33.51 | 117.57 |
| Sparse70+2bit | 8 | 61.60 | 7377.72 | 119.42 | 39.65 | 133.00 |

## llama3_8b | input=4096 | output=4096

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | — | — | — | — | — |
| Sparse50 | — | — | — | — | — | — |
| Sparse70 | 1 | 18.70 | 2524.12 | 50.28 | 18.17 | 218.98 |
| Sparse70 | 2 | 34.61 | 4297.59 | 53.79 | 21.41 | 236.69 |
| Sparse70 | 4 | 57.42 | 6750.50 | 63.43 | 27.81 | 285.35 |
| Sparse70 | 6 | 85.14 | 8289.92 | 65.72 | 34.25 | 288.65 |
| Sparse70 | 8 | 99.04 | 11360.46 | 73.62 | 40.66 | 330.86 |
| Sparse50+2bit | — | — | — | — | — | — |
| Sparse70+2bit | 1 | 12.06 | 1051.37 | 81.52 | 18.05 | 339.57 |
| Sparse70+2bit | 2 | 21.47 | 2064.74 | 90.11 | 21.16 | 381.60 |
| Sparse70+2bit | 4 | 36.13 | 3802.89 | 105.37 | 27.33 | 453.46 |
| Sparse70+2bit | 6 | 46.30 | 5636.98 | 123.51 | 33.51 | 530.79 |
| Sparse70+2bit | 8 | 53.79 | 7320.12 | 144.46 | 39.65 | 609.23 |

## llama3_8b | input=4098 | output=1024

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 8 | 44.13 | 5916.44 | 171.53 | 47.72 | 185.62 |
| Sparse50 | 8 | 90.91 | 11870.48 | 72.59 | 44.72 | 90.11 |
| Sparse70 | 8 | 87.82 | 10948.36 | 73.07 | 43.04 | 93.28 |
| Sparse50+2bit | 8 | 58.31 | 7292.78 | 126.81 | 41.03 | 140.49 |
| Sparse70+2bit | 8 | 62.36 | 7193.62 | 118.24 | 40.87 | 131.36 |

## llama3_8b | input=4098 | output=4096

| Config | BS | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Peak Memory (GB) | Batch Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1 | 37.08 | 402.03 | 23.12 | 18.43 | 110.46 |
| Dense | 2 | 52.86 | 791.34 | 34.10 | 21.90 | 154.99 |
| Dense | 4 | 67.72 | 1566.19 | 44.48 | 28.84 | 241.95 |
| Dense | 6 | 71.93 | 2324.78 | 60.62 | 35.78 | 341.65 |
| Dense | 8 | 74.11 | 3112.23 | 76.84 | 42.72 | 442.16 |
| Sparse50 | 1 | 17.83 | 581.61 | 53.19 | 18.27 | 229.77 |
| Sparse50 | 2 | 34.08 | 1050.87 | 54.22 | 21.61 | 240.37 |
| Sparse50 | 4 | 68.40 | 2081.16 | 54.03 | 28.18 | 239.52 |
| Sparse50 | 6 | 94.35 | 3113.67 | 54.95 | 34.80 | 260.47 |
| Sparse50 | 8 | 127.07 | 4257.26 | 59.89 | 41.40 | 257.87 |
| Sparse70 | 1 | 16.62 | 579.57 | 55.97 | 18.17 | 246.38 |
| Sparse70 | 2 | 33.17 | 1021.49 | 77.20 | 21.42 | 246.99 |
| Sparse70 | 4 | 66.74 | 1962.11 | 55.66 | 27.81 | 245.49 |
| Sparse70 | 6 | 99.11 | 2974.51 | 54.41 | 34.25 | 247.97 |
| Sparse70 | 8 | 131.14 | 4042.86 | 55.43 | 40.66 | 249.88 |
| Sparse50+2bit | — | — | — | — | — | — |
| Sparse70+2bit | — | — | — | — | — | — |

## Notes

- `avg_batch_time` is in seconds (converted from `batch_ms/1000` when source is sweep summary).
- For sweep rows, mapping used: `mustafar -> Sparse70`, `quant -> Sparse70+2bit` (the sweep was run with sparsity=0.7).
- If sweep `throughput_tps` is empty, throughput is derived from `BS * output_length / batch_ms`.
- Latest timestamp wins when same `(scenario, config, BS)` appears in multiple files.
