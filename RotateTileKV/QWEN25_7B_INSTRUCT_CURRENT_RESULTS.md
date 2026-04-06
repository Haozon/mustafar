# Qwen2.5-7B-instruct 当前结果

日期：

- `2026-04-06`

模型：

- `/home/zh/nas/nas_10g/models/Qwen2.5-7B-instruct`

说明：

- 这个文件现在以 `selected6` 主线结果为准。
- 跨模型总表在 `ALL_MODELS_CURRENT_RESULTS.md`。
- 工作过程记录在 `工作日志.md`。

## 当前状态

- 旧版 `selected6` 结果存在实现问题，不能继续作为正式结论引用。
- 问题已经定位并修复，当前以 `fix4096` 重跑结果为准。
- 当前没有残留的 `Qwen2.5-7B-instruct` 修复版主线运行进程。

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
- 旧版结果目录：
  - `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6`
- 修复版结果目录：
  - `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096`

## 已修复的问题

- 旧实现会在 decode 过程中反复对同一段已经量化过的 KV cache 重复 fake-quant。
- 该问题在 `Qwen2.5-7B-instruct` 上会迅速累积误差，并导致回答退化成明显的乱码 / 重复垃圾串。
- 修复后改为：
  - 只对新进入 prefix 的那一小段 KV 做增量量化
  - 不再每一步重质量化整段旧 prefix

因此，旧版结果中出现的极端异常分数不能再直接引用。

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

主对比表（修复后）：

| Bit | KIVI | Average | 我的方法 | Average |
|---|---|---:|---|---:|
| 4bit | KIVI-align fake | 17.62 | Per-Token-Tile + tile Hadamard(64) | 8.05 |
| 3bit | KIVI-align fake | 13.52 | Per-Token-Tile + tile Hadamard(64) | 2.88 |
| 2bit | KIVI-align fake | 9.13 | Per-Token-Tile + tile Hadamard(64) | 2.19 |

## 完整结果

结果目录（修复后）：

- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/per_token_tile_4bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/per_token_tile_3bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/per_token_tile_2bit_tile_hadamard64`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/kivi_align_fake_4bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/kivi_align_fake_3bit`
- `/mnt/nas/nas_192.168.7.2/zh/mustafar/RotateTileKV/exp_qwen2_5_7b_instruct_res128_selected6_fix4096/kivi_align_fake_2bit`

| 方法 | Bit | hotpotqa | lcc | multifieldqa_en | narrativeqa | qasper | trec | average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Per-Token-Tile + tile Hadamard(64) | 4bit | 1.32 | 23.71 | 5.59 | 0.51 | 7.16 | 10.00 | 8.05 |
| Per-Token-Tile + tile Hadamard(64) | 3bit | 0.14 | 12.89 | 1.96 | 0.37 | 1.93 | 0.00 | 2.88 |
| Per-Token-Tile + tile Hadamard(64) | 2bit | 0.00 | 11.36 | 0.84 | 0.06 | 0.85 | 0.00 | 2.19 |
| KIVI-align fake | 4bit | 8.36 | 31.97 | 12.44 | 11.14 | 15.05 | 26.75 | 17.62 |
| KIVI-align fake | 3bit | 8.58 | 27.35 | 9.92 | 8.16 | 10.10 | 17.00 | 13.52 |
| KIVI-align fake | 2bit | 0.88 | 28.19 | 6.44 | 1.79 | 4.97 | 12.50 | 9.13 |

## 简短结论

- 修复后，Qwen 结果已经从“明显异常坏掉”恢复到“正常但仍显著弱于 KIVI”。
- 这说明旧结果里确实有实现 bug，但 bug 不是全部原因。
- 当前更合理的结论是：
  - `Qwen2.5-7B-instruct` 对纯 `Per-Token-Tile + tile Hadamard(64)` 的纯量化路径仍然非常敏感；
  - 即使修复实现后，`4bit/3bit/2bit` 三档下，KIVI 仍明显优于该路径。
