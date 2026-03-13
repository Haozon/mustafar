# Quant 优化现状总结（2026-03-11）

## 1. 当前代码状态

当前量化路径已经完成以下改造：

1. `K` / `V` 分别支持独立 `dequant_mode`
- `K`: 默认 `dequant_mode=0 (speed)`
- `V`: 默认 `dequant_mode=0 (speed)`

2. `V` 独立支持 `Split-K`
- 默认 `quant_v_split_k=4`
- 已加入安全裁剪，避免 `split_k > K_Global / 64` 导致非法访问

3. `V` 独立支持 tile 配置分流
- `quant_v_tile_config=0`: auto
- `quant_v_tile_config=1`: tile64
- `quant_v_tile_config=2`: tile128

4. `Value` 独立接入 `counts / units_per_tile` 元数据
- kernel 直接使用压缩阶段计算好的非零计数和 uint32 数量
- 避免在 `Value` 热路径里重复 `popcount + units_needed`

5. `Value` 2-bit 解包循环已改成按实际 `tile_units` 精确迭代
- 不再固定遍历 4 个 `uint32` 再在内层 `break`
- `nnz=0` 的 tile 直接跳过

6. `Value` fused `tile_config=3` 实验分支已加入
- 当前原型可跑 benchmark，但实测为负收益
- 保留为实验入口，不作为默认路径

7. local window 已改为固定容量缓冲
- 去掉 decode 阶段每步 `torch.cat`

## 2. 独立 Value benchmark 结论

### 2.1 原因

之前阶段性优化的结论主要来自 Key 路径；但 Value 路径的：

- `K_Global`
- `Split_K`
- GQA 结构
- grid 规模

都和 Key 不同，因此不能继续用 Key 的最优结果替代 Value。

### 2.2 新增的独立测试入口

- 脚本: `kernel_quant/kernel_bench/benchmark_spmv_value_detailed.py`
- 跟踪: `kernel_quant/kernel_bench/output/VALUE_OPTIMIZATION_TRACKING.md`

### 2.3 当前最佳 Value 配置

GQA 口径（Llama-3-8B 风格，`Heads=32, KV Heads=8`）下：

- `dequant_mode=0`
- `split_k=4`
- `tile_config=0(auto)` 或 `1(tile64)`

代表数据：

- `kernel_quant/kernel_bench/output/value_spmv_detailed_20260311_145634.json`
- `kernel_quant/kernel_bench/output/value_spmv_detailed_20260311_160756.json`
- `kernel_quant/kernel_bench/output/value_spmv_detailed_20260311_182559.json`
- `kernel_quant/kernel_bench/output/value_spmv_detailed_20260312_183132.json`

关键结果：

- `Value sparse`: `0.0418 ~ 0.0624 ms/token`
- `Value quant (old split_k=1)`: `0.2400 ms/token`
- `Value quant (split_k=4, tile auto/tile64)`: `0.0670 ~ 0.0738 ms/token`
- `Value quant (split_k=4, tile auto/tile64, counts/units enabled)`: `~0.0620 ms/token`
- `Value fused experimental tile3`: 明显慢于当前最优路径

结论：

- `Value` 单项已有约 `3x+` 提升
- 但仍未稳定全面打赢 sparse-only

## 3. benchmark/ 目录已接入新核心

`benchmark/run_controlled_benchmark.sh` 已更新，量化实验现在会自动传递：

- `QUANT_K_DEQUANT_MODE`
- `QUANT_V_DEQUANT_MODE`
- `QUANT_V_SPLIT_K`
- `QUANT_V_TILE_CONFIG`

新的 `summary.csv` 也会记录这些字段。

## 4. 本轮 controlled benchmark 结果

结果目录：

- `benchmark/benchmark_results_20260312_142024/`

核心配置：

- `K_mode=0`
- `V_mode=0`
- `V_split_k=4`
- `V_tile=0(auto)`
- `bs=1`
- `prompt_length=1024`
- `output_length=8`
- `repeats=1`

### 4.1 summary.csv

| Config | TTFT(ms) | TPOT(ms) | Peak GB |
|---|---:|---:|---:|
| dense | 98.72 | 22.37 | 15.83 |
| sparse_50 | 212.21 | 51.80 | 15.80 |
| sparse_70 | 202.57 | 46.61 | 15.78 |
| sparse_50_quant_2bit | 134.55 | 36.67 | 15.80 |
| sparse_70_quant_2bit | 134.75 | 36.87 | 15.80 |

来源：

- `benchmark/benchmark_results_20260312_142024/summary.csv`

### 4.2 当前观察

1. `sparse_50_quant_2bit` 已优于本轮 `sparse_50`：
   - `36.67 ms` vs `51.80 ms`
2. `sparse_70_quant_2bit` 已优于本轮 `sparse_70`：
   - `36.87 ms` vs `46.61 ms`
3. 这说明：
   - `Value` 的参数级优化有效
   - 经过 `Value` 的 `counts/units + split_k + tile + exact-unit loop` 组合优化后，当前统一 benchmark 下量化版本已经打败纯 sparse

## 5. 当前默认建议

建议默认使用：

```text
quant_k_dequant_mode = 0
quant_v_dequant_mode = 0
quant_v_split_k = 4
quant_v_tile_config = 0
```

原因：

- 这是当前在真实模型中可稳定运行的最优保守配置
- `split_k=8` 在部分独立基准更快，但在真实 decode/GQA 下当前不稳定

## 6. 下一步优化方向

在现有文件基础上，接下来最值得做的是：

1. 围绕 `Value_Kernel_Quant` 继续做结构性 fused 优化
2. 尝试减少“反量化到 shared 再算”的完整 materialization 开销
3. 保持 `Value` 独立 benchmark 和 tracking，不再回退到用 Key 结果替代

不再优先考虑的方向：

- 继续做 Key 导向微优化
- 继续做 wrapper / `torch.cat` / `flatten` 类小修补
- 再次复用 Key benchmark 作为 Value 的配置依据
