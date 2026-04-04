# Qwen2.5-7B-instruct 当前结果

日期：

- `2026-04-02`

模型：

- `/home/zh/nas/nas_10g/models/Qwen2.5-7B-instruct`

说明：

- 这个文件现在以 `selected6` 主线结果为准。
- 跨模型总表在 `ALL_MODELS_CURRENT_RESULTS.md`。
- 工作过程记录在 `工作日志.md`。

## 当前状态

- `Qwen2.5-7B-instruct` 的 `selected6` 主线任务已经全部完成。
- 当前没有残留的 `Qwen2.5-7B-instruct` 主线运行进程。
- 当前没有待跑的 `Qwen2.5-7B-instruct` `selected6` 主线任务。

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
  - `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6`

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
| 4bit | KIVI-align fake | 14.47 | Per-Token-Tile + tile Hadamard(64) | 3.53 |
| 3bit | KIVI-align fake | 10.94 | Per-Token-Tile + tile Hadamard(64) | 1.76 |
| 2bit | KIVI-align fake | 8.13 | Per-Token-Tile + tile Hadamard(64) | 2.04 |

## 完整结果

结果目录：

- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/per_token_tile_4bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/per_token_tile_3bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/per_token_tile_2bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/kivi_align_fake_4bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/kivi_align_fake_3bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6/kivi_align_fake_2bit`

| 方法 | Bit | hotpotqa | lcc | multifieldqa_en | narrativeqa | qasper | trec | average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Per-Token-Tile + tile Hadamard(64) | 4bit | 0.16 | 18.74 | 1.90 | 0.13 | 0.25 | 0.00 | 3.53 |
| Per-Token-Tile + tile Hadamard(64) | 3bit | 0.00 | 9.65 | 0.59 | 0.00 | 0.31 | 0.00 | 1.76 |
| Per-Token-Tile + tile Hadamard(64) | 2bit | 0.12 | 11.22 | 0.37 | 0.19 | 0.37 | 0.00 | 2.04 |
| KIVI-align fake | 4bit | 16.30 | 31.18 | 13.81 | 1.65 | 6.37 | 17.50 | 14.47 |
| KIVI-align fake | 3bit | 9.00 | 27.50 | 11.58 | 2.73 | 4.34 | 10.50 | 10.94 |
| KIVI-align fake | 2bit | 3.02 | 28.16 | 7.30 | 0.47 | 4.07 | 5.75 | 8.13 |

## 简短结论

- `4bit/3bit/2bit` 三档下，KIVI 都明显高于我的方法。
- 当前 `Qwen2.5-7B-instruct` 的 `selected6` 主线对比已经完整结束。
