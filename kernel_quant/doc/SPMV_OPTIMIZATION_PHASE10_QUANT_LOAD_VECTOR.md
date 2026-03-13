# SPMV 优化阶段10：量化数据加载向量化（`uint2/uint4`）实验记录

## 1. 背景与目标

- 目标函数：`SpMM_CopyFromGlobalToReg_Quant`
- 代码位置：`csrc/SpMM_Kernel_Quant.cuh`
- 当前主干逻辑：每个 tile 按 `for (j=0..3)` 标量读取 `uint32`（不足部分补 0）

本阶段假设：
- 量化值读取路径存在一定的全局内存事务开销；
- 若按对齐条件使用 `uint2/uint4` 向量加载，可能减少指令数和访存事务数，从而提升吞吐。

## 2. 优化原理

### 2.1 原始标量路径（主干）

- 先计算 `units_needed = ceil(nnz / capacity)`；
- 再执行 4 次展开循环：
  - `j < units_needed` 时读取 `GlobalPTR_quant[uint32_offset + j]`
  - 否则写 0 到寄存器

特点：
- 优点：分支简单、行为稳定、无额外对齐要求；
- 缺点：固定 4 次标量 load，理论上有向量化空间。

### 2.2 实验向量化路径（已回滚）

实验版设计思路：
- 在 `units_needed` 较大时，尝试使用 `uint2`（8B）或 `uint4`（16B）加载；
- 通过地址对齐检查决定是否走向量路径；
- 对齐不满足时回退到标量加载；
- 保留尾部补零语义，保证与原始逻辑等价。

预期收益来源：
- 读取指令减少；
- 合并事务概率提升；
- 降低 copy 阶段的指令开销。

潜在风险：
- 对齐判断与分支增加的控制流开销；
- 短路径（`units_needed` 小）不一定获益；
- 不同配置下 L2/调度行为可能变化，带来抖动。

## 3. A/B 测试设计

- 编译命令：
  - `cd /mnt/home/zh/mustafar/kernel_quant/kernel_wrapper`
  - `python setup.py build_ext --inplace`
- 基准命令：
  - `cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench`
  - `python benchmark_spmv_detailed.py --dequant-mode both`
- 指标口径：
  - 重点看 Large 配置 `Quant TPOT(ms/token)`；
  - 同时观察 `speed` 与 `memory` 两模式一致性；
  - 若无稳定净收益，按保守策略回滚。

## 4. 实测结果（Large）

| 运行 | 版本 | speed TPOT (ms) | memory TPOT (ms) | 备注 |
|---|---|---:|---:|---|
| `20260302_135415` | 向量化加载实验 | 0.1517 | 0.1570 | speed 略好，memory 略差 |
| `20260302_135958` | 回滚对照（标量） | 0.1583 | 0.1574 | 对照样本 |
| `20260302_140344` | 向量化加载复测 | 0.1578 | 0.1586 | 与对照接近/略差 |
| `20260302_140837` | 回滚后锚点复测 | 0.1601 | 0.1582 | 确认回滚后稳定可跑 |

数据文件：
- `kernel_bench/output/spmv_detailed_20260302_135415.json`
- `kernel_bench/output/spmv_detailed_20260302_135958.json`
- `kernel_bench/output/spmv_detailed_20260302_140344.json`
- `kernel_bench/output/spmv_detailed_20260302_140837.json`

## 5. 结论

- 该优化在当前 workload 下没有给出稳定可复现的净收益；
- 不同轮次中 speed/memory 模式呈现“有时小幅好、有时持平或回退”的波动；
- 按“保守路线”原则，本阶段结论为：**不并入主干，保持标量加载实现**。

## 6. 回滚后主干状态

- 当前主干已恢复标量加载循环（安全版本）：
  - 文件：`csrc/SpMM_Kernel_Quant.cuh`
  - 函数：`SpMM_CopyFromGlobalToReg_Quant`
- 已重新编译并完成回滚后 benchmark 验证，无功能异常。

## 7. 后续可选方向（保守优先）

- 优先考虑“减少波动”的改造而非引入更多条件分支：
  - Shared memory 初始化路径精简（只清零必要区间）；
  - 更稳定的 launch/stream 侧基准规范化（降低测量噪声）；
  - 针对固定 `nnz` 分布做编译期分支特化（先做离线分布统计再决定）。
