# Mistral-7B-v0.1 当前结果

日期：

- `2026-04-02`

模型：

- `/home/zh/nas/nas_10g/models/Mistral-7B-v0.1`

说明：

- 这个文件现在以 `selected6` 主线结果为准。
- 跨模型总表在 `ALL_MODELS_CURRENT_RESULTS.md`。
- 工作过程记录在 `工作日志.md`。

## 当前状态

- `Mistral-7B-v0.1` 的 `selected6` 主线任务已经全部完成。
- 当前没有残留的 `Mistral-7B-v0.1` 主线运行进程。
- 当前没有待跑的 `Mistral-7B-v0.1` `selected6` 主线任务。

## 评测协议

- `selected6`
- 6 个任务：
  - `hotpotqa`
  - `lcc`
  - `multifieldqa_en`
  - `narrativeqa`
  - `qasper`
  - `trec`
- 本地 LongBench 数据：
  - `/data/home/szm/backup_dataset/LongBench/data`
- `selected6` 结果目录：
  - `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6`

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
| 4bit | KIVI-align fake | 31.19 | Per-Token-Tile + tile Hadamard(64) | 30.78 |
| 3bit | KIVI-align fake | 30.16 | Per-Token-Tile + tile Hadamard(64) | 29.52 |
| 2bit | KIVI-align fake | 23.91 | Per-Token-Tile + tile Hadamard(64) | 25.72 |

## 完整结果

结果目录：

- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/per_token_tile_4bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/per_token_tile_3bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/per_token_tile_2bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/kivi_align_fake_4bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/kivi_align_fake_3bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_mistral_7b_v01_res128_selected6/kivi_align_fake_2bit`

| 方法 | Bit | hotpotqa | lcc | multifieldqa_en | narrativeqa | qasper | trec | average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Per-Token-Tile + tile Hadamard(64) | 4bit | 9.74 | 66.79 | 26.22 | 4.99 | 8.42 | 68.50 | 30.78 |
| Per-Token-Tile + tile Hadamard(64) | 3bit | 9.42 | 65.63 | 24.81 | 4.55 | 7.72 | 65.00 | 29.52 |
| Per-Token-Tile + tile Hadamard(64) | 2bit | 9.59 | 57.59 | 17.70 | 2.38 | 6.06 | 61.00 | 25.72 |
| KIVI-align fake | 4bit | 9.95 | 66.27 | 25.97 | 6.56 | 8.39 | 70.00 | 31.19 |
| KIVI-align fake | 3bit | 9.61 | 64.53 | 24.53 | 4.54 | 8.26 | 69.50 | 30.16 |
| KIVI-align fake | 2bit | 8.29 | 56.61 | 15.55 | 2.26 | 5.24 | 55.50 | 23.91 |

## 简短结论

- `4bit` 和 `3bit` 下，KIVI 略高。
- `2bit` 下，我的方法更高：`25.72 vs 23.91`。
- 当前 `selected6` 主线已经跑完，可以直接用于跨模型总表对比。
