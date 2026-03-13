# SpMV 优化阶段 1 实现说明（除法/取模消除 + 寄存器瘦身）

**日期**: 2026-03-01  
**状态**: 已实现（待性能回归）  
**目标函数**: `SpMM_DecompressFromRegisterToShared_Quant`

---

## 1. 背景与问题

在量化 SpMV 的解包与反量化内层循环中，原始实现对每个非零元素都会执行：

- `unit_idx = j / capacity`
- `bit_offset = (j % capacity) * bit`

对应位置：

- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh`（原逻辑位置约在解压函数内层循环）

对于当前基准配置（`bit=2`, `capacity=16`），这两步属于固定模式计算，存在明显优化空间。

---

## 2. 本次优化内容

### 2.1 优化 A：2-bit 快路径（消除除法/取模）

在 `bit == 2 && capacity == 16` 时，使用位运算替代除法与取模：

- `unit_idx = j >> 4`（等价于 `j / 16`）
- `bit_offset = (j & 15) << 1`（等价于 `(j % 16) * 2`）

实现位置：

- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh:154`
- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh:165`
- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh:166`

同时保留通用回退路径（非 2-bit/16 容量时仍走原始算法）：

- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh:176`

这保证了接口与行为兼容，不会破坏未来不同位宽实验。

### 2.2 优化 B：寄存器数组瘦身

原代码使用 `Registers_quant[64]`，但每个 tile 最多只需要 4 个 `uint32`（64 个值、2-bit、每 `uint32` 存 16 个）。  
每个线程一次处理 2 个 tile，因此理论上仅需 `2 * 4 = 8` 个 `uint32`。

本次将其改为紧凑布局：

- 常量定义：
  - `QUANT_TILES_PER_THREAD = 2`
  - `MAX_UINT32_PER_TILE = 4`
  - `QUANT_REG_UNITS = 8`
- `Registers_quant` 改为 `Registers_quant[QUANT_REG_UNITS]`
- 索引由旧布局 `i * 32 + j` 改为 `i * MAX_UINT32_PER_TILE + j`

实现位置：

- 常量定义：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh:13`
- 加载函数签名：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh:35`
- 索引写入：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh:84`
- Key kernel 寄存器声明：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh:264`
- Value kernel 寄存器声明：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh:483`

---

## 3. 原理说明（为什么等价）

在当前配置 `bit=2, capacity=16` 下：

1. 每 16 个量化值对应一个 `uint32` 存储单元。  
2. `j / 16` 给出第几个 `uint32`，可写成右移 4 位 `j >> 4`。  
3. `j % 16` 给出该值在单元内的槽位，可写成 `j & 15`。  
4. 2-bit 量化下每槽位偏移为 `2 * slot`，即 `(j & 15) << 1`。  

因此位运算版本与除法/取模版本计算结果逐项一致，只是指令开销更低。

---

## 4. 预期收益与边界

### 4.1 预期收益

- 降低解包路径的整数运算开销（特别是热点内层循环）。
- 降低寄存器压力，潜在提升 occupancy，减少 local spill 风险。
- 对 Quant SpMV 的单次与批量 TPOT 均有望带来收益。

### 4.2 边界与风险

- 本次没有改变反量化公式与写回路径，数值行为应保持一致。
- 2-bit 快路径仅在 `bit=2, capacity=16` 激活；其他配置走通用路径。
- 实际加速幅度仍需用基准脚本实测确认。

---

## 5. 验证状态

### 5.1 编译验证

已通过本地扩展构建命令验证：

```bash
cd kernel_wrapper
python setup.py build_ext --inplace
```

说明：`build_quant_kernel.sh` 在当前环境会因安装目录权限/CUDA runtime 可见性报错，但不影响本次代码本身的编译正确性判断。

### 5.2 建议性能回归

建议直接复用现有脚本进行 A/B 对比：

```bash
python kernel_bench/benchmark_spmv_detailed.py
```

重点关注 `Large` 配置下 `quant.batch_spmv_avg`（即 Quant TPOT）。

---

## 6. 对论文可用的简述模板

> 在量化 SpMV 解包阶段，我们针对 2-bit（capacity=16）场景引入了位运算快路径，将内层循环中的除法和取模替换为移位与按位与运算。同时，将量化寄存器缓存从过度分配的 64 单元压缩为理论最小 8 单元，降低寄存器压力并改善执行效率。在不改变反量化公式与接口兼容性的前提下，该优化为后续性能提升提供了低风险基础。

---

## 7. 追踪表与数据索引

为便于后续优化阶段管理，本阶段对应记录已同步到跟踪表：

- 跟踪总表：`kernel_bench/output/SPMV_OPTIMIZATION_TRACKING.md`
- 本阶段实测结果文件：`kernel_bench/output/spmv_detailed_20260301_212304.json`
- 本阶段实测报告：`kernel_bench/output/spmv_detailed_20260301_212304.md`

当前跟踪表中的对应策略名为：

- `索引计算优化（位运算快路径+寄存器瘦身，实测）`

---

## 8. 详细实现补充（前后对照）

### 8.1 解包索引计算前后对照

旧路径（通用）：

```cpp
int unit_idx = j / capacity;
int bit_offset = (j % capacity) * bit;
uint32_t q = (quant_units[unit_idx] >> bit_offset) & mask;
```

新路径（2-bit 快路径）：

```cpp
int unit_idx = j >> 4;          // j / 16
int bit_offset = (j & 15) << 1; // (j % 16) * 2
uint32_t q = (quant_units[unit_idx] >> bit_offset) & 0x3u;
```

触发条件：

- `bit == 2`
- `capacity == 16`

回退条件：

- 任一条件不满足时，自动走通用路径。

### 8.2 寄存器缓存布局前后对照

旧布局（过度分配）：

- `Registers_quant[64]`
- 实际最多只会用到前 `8` 个逻辑单元（2 tile x 4 uint32）

新布局（紧凑分配）：

- `Registers_quant[8]`
- 索引规则：`i * MAX_UINT32_PER_TILE + j`

预期影响：

- 降低线程常驻寄存器占用压力
- 降低编译器 spill 风险
- 为 SM occupancy 留出更大空间

---

## 9. 正确性检查清单

### 9.1 索引映射一致性

在 `bit=2, capacity=16` 下，对任意 `j in [0,63]`：

- `j / 16 == j >> 4`
- `(j % 16) * 2 == (j & 15) << 1`

因此 `(unit_idx, bit_offset)` 与旧实现逐项一致。

### 9.2 数据访问边界

- 每 tile 最多 `64` 个非零值
- 每 `uint32` 存 `16` 个 2-bit 值
- 所需 `uint32` 数上限为 `4`
- 两个 tile 上限共 `8`，与 `QUANT_REG_UNITS` 一致

### 9.3 行为不变项

- `q_value` 位提取方式不变
- 反量化公式不变
- shared memory 写回坐标不变
- Python/C++ 调用签名不变

---

## 10. 复现与回滚建议

### 10.1 复现实验

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py
```

建议至少记录：

- `Large quant.single_spmv`
- `Large quant.batch_spmv_avg`
- `Large quant.memory_mb`

### 10.2 单点回滚

若需要仅回退阶段 1：

1. 将快路径索引计算改回除法/取模版本
2. 将 `Registers_quant[8]` 改回旧布局
3. 保持其他阶段优化不动

这样可隔离验证阶段 1 的独立贡献。
