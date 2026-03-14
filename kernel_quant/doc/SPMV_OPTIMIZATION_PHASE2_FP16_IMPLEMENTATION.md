# SpMV 优化阶段 2 实现说明（FP16 反量化 + FP16 Scale/Zero + 双模式/2-bit特化）

**日期**: 2026-03-01  
**状态**: 已实现（首轮测试已记录）；新增双模式与2-bit特化后待复测  
**目标函数**: `SpMM_DecompressFromRegisterToShared_Quant`

---

## 1. 优化目标

将当前“FP32 scale/zero 存储 + FP32 反量化 + FP32->FP16 转换”的路径改为：

- `scale/zero_point` 以 FP16 存储（压缩阶段直接输出 FP16）
- kernel 内直接 FP16 反量化

以减少显存带宽、寄存器压力和循环内类型转换开销。

原路径（每个非零值）：

- `float dequant = (float(q) - zero_point) * scale`
- `SharedPTR[...] = __float2half(dequant)`

新路径（每个非零值）：

- `half q_h = __int2half_rn(q)`
- `half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h)`
- `SharedPTR[...] = dequant_h`

---

## 2. 实现内容

### 2.1 压缩阶段改为 FP16 存储 scale/zero

`compression_quant.py` 中，`convert_key_batched_quant / convert_value_batched_quant` 的 `scales/zeros` 输出类型由 `float32` 改为 `float16`。

在 Triton 压缩核中，读取 `scales/zeros` 后显式转 `float32` 参与量化计算（避免量化计算本身受半精度算术影响）：

- `scale = tl.load(...).to(tl.float32)`
- `zero_point = tl.load(...).to(tl.float32)`

### 2.2 C++/CUDA 接口链路改为 half 指针

以下文件的 `scales/zeros` 指针类型统一从 `float*` 改为 `half*`：

- `kernel_wrapper/mustafar_wrapper_quant.cu`
- `csrc/SpMM_API_Quant.cuh`
- `csrc/SpMM_API_Quant.cu`
- `csrc/SpMM_Kernel_Quant.cuh`

### 2.3 两条解压路径统一改为 FP16 反量化

1. **2-bit 快路径**（`bit=2, capacity=16`）  
2. **通用回退路径**（其他参数配置）

两条路径都已替换为 FP16 反量化，不再执行 `float dequant + __float2half`。

### 2.4 新增双模式（速度/内存）

新增 `dequant_mode` 参数：

- `0`：速度优先（使用 float 反量化算术，再写回 half）
- `1`：内存优先（使用 half 反量化算术）

绑定层默认 `dequant_mode=0`，兼容旧调用（不传该参数时行为保持可用）。

### 2.5 新增 2-bit 编译期特化（Fast2Bit）

在 API 启动器中按 `(bit==2 && capacity==16)` 选择 `Fast2Bit=true/false` 模板实例，  
并在解压函数中使用 `if constexpr` 固化解包路径：

- `Fast2Bit=true`：`unit_idx = j >> 4`, `bit_offset = (j & 15) << 1`
- `Fast2Bit=false`：保留通用 `j / capacity`, `(j % capacity) * bit`

---

## 3. 代码位置

文件：

- `kernel_quant/compression_quant.py`
- `kernel_quant/kernel_wrapper/mustafar_wrapper_quant.cu`
- `kernel_quant/kernel_wrapper/pybind_quant.cpp`
- `kernel_quant/csrc/SpMM_API_Quant.cuh`
- `kernel_quant/csrc/SpMM_API_Quant.cu`
- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh`

关键位置：

- 压缩输出改为 FP16：`183-184`, `456-457`（`compression_quant.py`）
- 包装层 dtype 检查与指针：`49-53`, `83-85`, `153-157`, `188-190`（`mustafar_wrapper_quant.cu`）
- 双模式绑定默认参数：`dequant_mode=0`（`pybind_quant.cpp`）
- API 启动器 Fast2Bit 特化与 dequant_mode 透传：`SpMM_API_Quant.cu/.cuh`
- kernel 参数链路与寄存器：`SpMM_Kernel_Quant.cuh`
- 反量化算术（速度/内存双模式）：`SpMM_Kernel_Quant.cuh`

---

## 4. 兼容性与风险

### 4.1 兼容性

- Python 调用签名未变，但 `scales/zeros` Tensor dtype 现在要求为 `float16`。
- 新增可选参数 `dequant_mode`（默认 0），旧调用无需修改。
- 位图遍历、解包逻辑、写回位置均未改。
- 非 2-bit 配置仍可通过通用回退路径执行。

### 4.2 风险点

- 数值精度：FP16 反量化可能引入轻微误差（通常可接受，需实测验证）。
- 性能不确定性：若瓶颈不在该算术段，提升可能有限。

---

## 5. 编译验证

已通过：

```bash
cd kernel_wrapper
python setup.py build_ext --inplace
```

---

## 6. 性能回归建议

使用现有脚本：

```bash
cd kernel_bench
python benchmark_spmv_detailed.py
```

对比双模式（推荐一次跑完）：

```bash
cd kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

仅测试内存模式：

```bash
cd kernel_bench
python benchmark_spmv_detailed.py --dequant-mode 1
```

建议重点关注（Large）：

- `quant_by_mode["0"].single_spmv`
- `quant_by_mode["0"].batch_spmv_avg`（速度模式 Quant TPOT）
- `quant_by_mode["1"].batch_spmv_avg`（内存模式 Quant TPOT）

并与阶段 1 最优结果 `spmv_detailed_20260301_212304.json` 对比。

---

## 7. 首轮实测结果（20260301_215521，特化前）

数据文件：

- `kernel_bench/output/spmv_detailed_20260301_215521.json`
- `kernel_bench/output/spmv_detailed_20260301_215521.md`

Large 配置（`seq_len=2048, heads=32, decode_steps=1024`）：

- Sparse TPOT: `0.1961 ms/token`
- Quant TPOT: `0.1920 ms/token`
- Quant single_spmv: `0.0687 ms`
- Quant 内存: `3.2233 MB`（Sparse: `10.3632 MB`，约 `3.22x` 更省）

对比结论：

- 相对最初基线（`0.2903 ms/token`）：提升到 `0.1920 ms/token`，约 `1.51x` 加速。  
- 相对阶段1最优（`0.1220 ms/token`）：出现约 `57.3%` 回退。  
- 内存侧：相对阶段1（约 `3.72 MB`）进一步下降到 `3.22 MB`，约 `13.4%` 下降。

说明：以上数据采集于“双模式/2-bit特化”引入前版本。  
当前代码已加入这两项改造，需要用相同脚本重新回归并更新结论。

---

## 8. 最新回归结果（20260301_222428，speed模式）

数据文件：

- `kernel_bench/output/spmv_detailed_20260301_222428.json`
- `kernel_bench/output/spmv_detailed_20260301_222428.md`

Large 配置（`dequant_mode=0`）：

- Sparse TPOT: `0.0936 ms/token`
- Quant TPOT: `0.1884 ms/token`
- Quant single_spmv: `0.0597 ms`
- Quant 内存: `3.2230 MB`（Sparse: `10.36 MB`，约 `3.22x` 更省）

结论：

- 相对最初基线（`0.2903 ms/token`）：约 `1.54x` 加速。  
- 相对阶段1最优（`0.1220 ms/token`）：仍慢约 `54.5%`。  
- 该轮仅包含 speed 模式结果，memory 模式待补测。

---

## 9. 详细链路补充（dtype/接口）

### 9.1 端到端 dtype 变化表

| 链路位置 | 旧实现 | 新实现 | 备注 |
|---|---|---|---|
| `compression_quant.py` 输出 `scales/zeros` | `float32` | `float16` | 存储压缩，减少内存占用 |
| wrapper 输入检查（`mustafar_wrapper_quant.cu`） | `at::kFloat` | `at::kHalf` | 防止混 dtype 误传 |
| API 指针（`SpMM_API_Quant.cu/.cuh`） | `const float*` | `const half*` | 保持链路一致 |
| kernel 寄存器缓存 | `float` | `half` | 减少寄存器宽度 |
| 反量化内层 | `float` | `half/float(双模式)` | `dequant_mode` 控制 |

### 9.2 双模式语义（设计目的）

- `dequant_mode=0`（speed）：
  - 反量化算术用 `float`
  - 目标是更高算术吞吐与稳态性能
- `dequant_mode=1`（memory）：
  - 反量化算术用 `half`
  - 目标是更低带宽/寄存器压力

说明：

- 两种模式都是正确实现，不是“一个正确一个实验”。
- 是否更快与输入规模和 GPU 架构相关，需要实测决定。

---

## 10. 精度与正确性补充

### 10.1 为什么可接受

- `2-bit` 量化本身已引入主导误差项。
- 将反量化算术从 `float` 改为 `half`，额外误差通常小于量化误差主量级。
- 写回目标本来就是 `half`，所以路径中存在 `float->half` 的归一化过程。

### 10.2 需要重点关注的回归点

- logits 一致性（统计误差，不要求 bitwise 相同）
- attention 分数分布漂移（均值/方差对齐）
- 长序列场景下稳定性（累积误差是否放大）

---

## 11. 复现实验建议（阶段2口径）

建议固定以下参数进行阶段2回归：

- 稀疏度 `50%`
- `bit=2, capacity=16`
- 测试脚本 `benchmark_spmv_detailed.py`
- 模式：先 `--dequant-mode both`，再按 Large 最优模式单独复测一次

建议最少记录字段：

- `results[].quant_by_mode["0"].batch_spmv_avg`
- `results[].quant_by_mode["1"].batch_spmv_avg`
- `results[].quant_by_mode[*].memory_mb`

---

## 12. 单点回滚策略（只回退阶段2）

若阶段2需要独立回滚，可按以下顺序：

1. `compression_quant.py` 恢复 `scales/zeros` 为 `float32`
2. wrapper/API/kernel 的 `half* scales/zeros` 恢复 `float*`
3. 保留阶段1索引优化不动（避免交叉污染）
4. 重新 benchmark 对比确认差异来源

该策略可把“dtype 链路影响”与“索引/解包优化影响”分离验证。
