# SpMV 优化阶段 5 详细说明（Fast2Bit full-unit/tail 解包分支收敛）

**日期**: 2026-03-02  
**状态**: 已实测，建议回滚  
**目标**: 在 2-bit 快路径中去掉“每个量化值都做一次尾部边界判断”的控制开销。

---

## 1. 背景与动机

阶段 3 的 Fast2Bit 路径已经把解包改为顺序消费 `uint32`，但仍存在一类细粒度控制开销：

- 内层循环中按值处理时，需要持续判断是否到达有效 `nnz` 边界。
- 这类判断在 `nnz` 较大（中/大配置）时会重复很多次，影响热路径吞吐。

本阶段不改数据结构、不改接口，只重排 Fast2Bit 解包循环结构。

---

## 2. 原理

2-bit 量化且 `capacity=16` 时：

- 每个 `uint32` 固定存 16 个量化值。
- 对任意 `nnz_tile`，总可以拆成：
  - `full_units = nnz_tile / 16`
  - `tail_values = nnz_tile % 16`

因此可将解包拆成两段：

1. **完整单元段**：每个单元固定解包 16 次，不需要每值边界判断。  
2. **尾部段**：仅在 `tail_values > 0` 时处理最后一个 `uint32` 的前 `tail_values` 个值。

收益点：

- 减少热路径中的动态分支频率
- 让完整单元段更利于编译器展开与调度

---

## 3. 代码改动

改动文件：

- `csrc/SpMM_Kernel_Quant.cuh`

改动位置（同一函数的两条模式分支）：

- `SpMM_DecompressFromRegisterToShared_Quant` 中 Fast2Bit + `DequantMode=memory`
- `SpMM_DecompressFromRegisterToShared_Quant` 中 Fast2Bit + `DequantMode=speed`

核心变化：

- 新增 `values_to_unpack / full_units / tail_values` 分解。
- 完整单元循环固定处理 16 个值（`bit_lane: 0..15`）。
- 尾部循环仅处理剩余值。

备注：

- 本阶段同时确认已回滚 FMA 路径，反量化恢复为：
  - speed: `(q - zero_f) * scale_f`
  - memory: `__hmul(__hsub(q_h, zero_h), scale_h)`

---

## 4. 正确性说明

该优化只改变“循环组织方式”，不改变以下语义：

- 位图弹位顺序：仍是 `brev + ffs + x&(x-1)`。
- 量化值解包顺序：仍按 packed `uint32` 的低位到高位顺序。
- 反量化公式：与阶段 3 保持一致（FMA 已回退）。

因此数值结果应与回退后的阶段 3 路径一致（允许浮点常规微小波动）。

---

## 5. 编译验证

已通过：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_wrapper
python setup.py build_ext --inplace
```

---

## 6. 性能验证建议

建议使用双模式一次性复测：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

重点对比：

- 基准对照：`kernel_bench/output/spmv_detailed_20260302_093621.json`（阶段3最优）
- 观察指标（Large）：
  - `quant_by_mode["0"].batch_spmv_avg`
  - `quant_by_mode["1"].batch_spmv_avg`

判定标准（建议）：

- 若 Large 任一模式较 `0.1057/0.1088` 继续下降，阶段 5 生效。
- 若出现回退，可单点回滚 Fast2Bit full-unit/tail 拆分，不影响其余阶段优化。

---

## 7. 单点回滚说明

回滚范围仅限：

- `SpMM_DecompressFromRegisterToShared_Quant` 的 Fast2Bit 两条分支（speed/memory）

回滚后恢复到“顺序解包 + 每值边界判断”的前一版，不影响：

- API 模板分发
- 位图解码优化
- FP16 scale/zero 存储链路

---

## 8. 实测结果回填（20260302_102135）

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_102135.json`
- `kernel_bench/output/spmv_detailed_20260302_102135.md`

Large 配置（`seq=2048, heads=32, decode=1024`）：

- Sparse TPOT: `0.1089 ms/token`
- Quant TPOT (`speed`): `0.1544 ms/token`
- Quant TPOT (`memory`): `0.1518 ms/token`

对比阶段3最优（20260302_093621）：

- `speed`: `0.1057 -> 0.1544`（约 `+46.1%` 回退）
- `memory`: `0.1088 -> 0.1518`（约 `+39.5%` 回退）

对比阶段4（FMA 回退版本，20260302_101112）：

- `speed`: `0.1669 -> 0.1544`（约 `7.5%` 改善）
- `memory`: `0.1631 -> 0.1518`（约 `6.9%` 改善）

结论：

- 本阶段相对阶段4有所恢复，但仍显著劣于阶段3最优版本。  
- 建议将该优化标记为“实验分支”，主干回到阶段3最优后再进行下一轮优化。
