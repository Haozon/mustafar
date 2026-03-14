# SpMV 优化阶段 7 详细说明（Fast2Bit remaining-counter 解包）

**日期**: 2026-03-02  
**状态**: 已实测，结果波动大，暂不保留  
**目标**: 去掉 Fast2Bit 内层 `j = unit_idx * 16 + bit_lane` 的乘加索引，改为剩余计数器边界控制。

---

## 1. 原理

原实现（Fast2Bit）每次解包都计算：

- `j = unit_idx * QUANT_CAPACITY_2BIT + bit_lane`
- `if (j >= values_to_unpack) break`

阶段7改为：

- `remaining = nnz_tile`
- 每处理一个量化值 `remaining--`
- `if (remaining <= 0) break`

目的：减少热路径里的整数乘加与比较开销。

---

## 2. 改动文件

- `csrc/SpMM_Kernel_Quant.cuh`
  - 函数：`SpMM_DecompressFromRegisterToShared_Quant`
  - 分支：`Fast2Bit + DequantMode=speed/memory`

---

## 3. 实测结果

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_111029.json`
- `kernel_bench/output/spmv_detailed_20260302_111055.json`

Large 配置（`seq=2048, heads=32, decode=1024`）：

- Run#1: `speed=0.1401`, `memory=0.1355 ms/token`
- Run#2: `speed=0.1547`, `memory=0.1553 ms/token`

结果在两次复测间波动较大，未形成稳定正收益结论。

---

## 4. 结论

- 阶段7具备潜在收益，但当前 benchmark 波动较大，无法稳定证明有效。
- 按保守策略，不并入主干，代码已回滚到阶段3稳定写法。

