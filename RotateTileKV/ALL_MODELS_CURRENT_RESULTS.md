# 全部模型当前结果

日期：

- `2026-03-30`

用途：

- 这是当前目录的总结果入口文件。
- 以后新增模型时，继续往这个文件里追加。
- 各模型自己的详细结果保存在对应的单模型文档里。

统一说明：

- 默认主线对比为：
  - `KIVI-align fake`
  - `Per-Token-Tile + tile Hadamard(64)`
- 这里的 `KIVI` 指本地 `KIVI-align fake`，不是 upstream KIVI。
- 当前 KIVI 统一配置：
  - `quant_impl = kivi`
  - `k_quant_scheme = kivi-channel`
  - `v_quant_scheme = per-token-head`
  - `group_size = 128`
  - `quant_granularity = per-token-tile`
  - `residual_length = 128`
  - `Hadamard = off`

## 当前模型索引

| 模型 | 当前状态 | 详细文档 |
|---|---|---|
| `Meta-Llama-3-8B-Instruct` | full LongBench 主线已完成 | `LLAMA3_8B_CURRENT_RESULTS.md` |
| `llama-2-7b` | full LongBench 主线已完成 | `LLAMA2_7B_CURRENT_RESULTS.md` |

## Meta-Llama-3-8B-Instruct

模型：

- `/home/zh/nas/nas_10g/models/Meta-Llama-3-8B-Instruct`

当前主线协议：

- `full LongBench`
- 16 个任务

详细文档：

- `LLAMA3_8B_CURRENT_RESULTS.md`

KIVI 与我的方法：

| Bit | KIVI | Average | 我的方法 | Average |
|---|---|---:|---|---:|
| 4bit | KIVI-align fake | 42.87 | Per-Token-Tile + tile Hadamard(64) | 42.72 |
| 3bit | KIVI-align fake | 41.80 | Per-Token-Tile + tile Hadamard(64) | 41.48 |
| 2bit | KIVI-align fake | 26.94 | Per-Token-Tile + tile Hadamard(64) | 30.59 |

简短结论：

- `4bit` 和 `3bit` 下，KIVI 略高。
- `2bit` 下，我的方法更高：`30.59 vs 26.94`。

## llama-2-7b

模型：

- `/home/zh/nas/nas_10g/models/llama-2-7b`

当前主线协议：

- `full LongBench`
- 16 个任务

详细文档：

- `LLAMA2_7B_CURRENT_RESULTS.md`

KIVI 与我的方法：

| Bit | KIVI | Average | 我的方法 | Average |
|---|---|---:|---|---:|
| 4bit | KIVI-align fake | 28.09 | Per-Token-Tile + tile Hadamard(64) | 28.33 |
| 3bit | KIVI-align fake | 28.23 | Per-Token-Tile + tile Hadamard(64) | 27.82 |
| 2bit | KIVI-align fake | 24.33 | Per-Token-Tile + tile Hadamard(64) | 24.17 |

简短结论：

- `4bit` 下，我的方法略高。
- `3bit` 和 `2bit` 下，KIVI 略高。
- 三个 bit 的差距都不大。

## 后续追加模板

新增模型时，按下面格式继续追加：

```md
## <模型名>

模型：

- <模型路径>

当前主线协议：

- <full LongBench / 子集说明>

详细文档：

- <单模型文档>

KIVI 与我的方法：

| Bit | KIVI | Average | 我的方法 | Average |
|---|---|---:|---|---:|
| 4bit | ... | ... | ... | ... |
| 3bit | ... | ... | ... | ... |
| 2bit | ... | ... | ... | ... |
```
