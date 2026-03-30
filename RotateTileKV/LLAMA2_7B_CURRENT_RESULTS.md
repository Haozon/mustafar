# Llama2-7B 当前结果

日期：

- `2026-03-30`

模型：

- `/home/zh/nas/nas_10g/models/llama-2-7b`

说明：

- 这个文件现在以 `full LongBench` 主线结果为准。
- 跨模型总表在 `ALL_MODELS_CURRENT_RESULTS.md`。
- 工作过程记录在 `工作日志.md`。

## 当前状态

- `llama-2-7b` 的主线 full LongBench 任务已经全部完成。
- 当前没有残留的 `llama-2-7b` 相关运行进程。
- 当前没有待跑的 `llama-2-7b` 主线任务。

## 评测协议

- `full LongBench`
- 16 个任务：
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `2wikimqa`
  - `musique`
  - `gov_report`
  - `qmsum`
  - `multi_news`
  - `trec`
  - `triviaqa`
  - `samsum`
  - `passage_count`
  - `passage_retrieval_en`
  - `lcc`
  - `repobench-p`
- 本地 LongBench 数据：
  - `/data/home/szm/backup_dataset/LongBench/data`
- full 结果目录：
  - `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full`

## KIVI 与我的方法

说明：

- 这里的 `KIVI` 指本地 `KIVI-align fake`，不是 upstream KIVI。
- 这里的“我的方法”指：
  - `Per-Token-Tile + tile Hadamard(64)`
- 两边都使用同一套 residual 配置，所以表里不再重复标 `residual128`。

KIVI 配置：

- `quant_impl = kivi`
- `k_quant_scheme = kivi-channel`
- `v_quant_scheme = per-token-head`
- `group_size = 128`
- `quant_granularity = per-token-tile`
- `residual_length = 128`
- `Hadamard = off`

我的方法配置：

- `quant_granularity = per-token-tile`
- `tile_size = 64`
- `enable_hadamard = true`
- `hadamard_mode = tile`
- `hadamard_group_size = 64`
- `residual_length = 128`

主对比表：

| Bit | KIVI | Average | 我的方法 | Average |
|---|---|---:|---|---:|
| 4bit | KIVI-align fake | 28.09 | Per-Token-Tile + tile Hadamard(64) | 28.33 |
| 3bit | KIVI-align fake | 28.23 | Per-Token-Tile + tile Hadamard(64) | 27.82 |
| 2bit | KIVI-align fake | 24.33 | Per-Token-Tile + tile Hadamard(64) | 24.17 |

## 完整结果

结果目录：

- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/per_token_tile_4bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/per_token_tile_3bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/per_token_tile_2bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/kivi_align_fake_4bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/kivi_align_fake_3bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_llama2_7b_res128_full/kivi_align_fake_2bit`

| 方法 | Bit | narrativeqa | qasper | multifieldqa_en | hotpotqa | 2wikimqa | musique | gov_report | qmsum | multi_news | trec | triviaqa | samsum | passage_count | passage_retrieval_en | lcc | repobench-p | average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Per-Token-Tile + tile Hadamard(64) | 4bit | 15.97 | 8.99 | 22.41 | 7.84 | 10.36 | 4.52 | 27.54 | 20.95 | 2.82 | 64.00 | 88.97 | 42.20 | 2.06 | 7.67 | 67.02 | 59.96 | 28.33 |
| Per-Token-Tile + tile Hadamard(64) | 3bit | 14.55 | 9.07 | 20.68 | 7.92 | 9.50 | 4.04 | 27.14 | 21.11 | 3.57 | 64.00 | 86.42 | 42.04 | 0.50 | 8.85 | 65.53 | 60.18 | 27.82 |
| Per-Token-Tile + tile Hadamard(64) | 2bit | 13.22 | 7.18 | 13.32 | 6.90 | 9.80 | 3.84 | 14.47 | 19.98 | 6.03 | 55.00 | 76.88 | 37.98 | 2.41 | 6.42 | 58.07 | 55.22 | 24.17 |
| KIVI-align fake | 4bit | 15.24 | 9.67 | 22.28 | 7.47 | 9.82 | 4.30 | 27.68 | 20.75 | 2.70 | 63.00 | 87.39 | 41.48 | 1.41 | 7.72 | 67.41 | 61.04 | 28.09 |
| KIVI-align fake | 3bit | 17.16 | 8.95 | 22.43 | 7.43 | 10.23 | 4.01 | 26.18 | 20.33 | 5.86 | 64.50 | 87.57 | 41.80 | 1.75 | 8.00 | 66.17 | 59.26 | 28.23 |
| KIVI-align fake | 2bit | 9.65 | 7.60 | 17.21 | 7.79 | 9.52 | 4.40 | 14.15 | 19.25 | 8.13 | 60.00 | 77.52 | 38.51 | 1.95 | 4.71 | 57.06 | 51.83 | 24.33 |

## 简短结论

- `4bit` 下我的方法略高：`28.33 vs 28.09`
- `3bit` 下 KIVI 略高：`28.23 vs 27.82`
- `2bit` 下 KIVI 略高：`24.33 vs 24.17`
- `4bit/3bit/2bit` 的差距都不大，当前两条方法在 `llama-2-7b` full LongBench 上整体比较接近。
