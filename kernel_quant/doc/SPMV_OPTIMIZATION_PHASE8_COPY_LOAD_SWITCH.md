# SpMV 优化阶段 8 详细说明（Quant Copy switch/fallthrough 加载）

**日期**: 2026-03-02  
**状态**: 已实测，A/B 回退  
**目标**: 将 `SpMM_CopyFromGlobalToReg_Quant` 中的 `for+if` 加载改为固定清零 + `switch/fallthrough`，减少分支与索引开销。

---

## 1. 原理

旧逻辑：

- 对 `j in [0,3]` 循环
- 每次判断 `j < units_needed` 决定加载或清零

阶段8尝试：

- 先固定将 4 个寄存器槽清零
- 用 `switch(units_needed)` + `[[fallthrough]]` 只加载需要的项

理论期望：降低循环分支开销。

---

## 2. 改动文件

- `csrc/SpMM_Kernel_Quant.cuh`
  - 函数：`SpMM_CopyFromGlobalToReg_Quant`

---

## 3. 实测结果

实验版：

- `kernel_bench/output/spmv_detailed_20260302_111511.json`

回滚对照（同一时段，恢复旧实现后）：

- `kernel_bench/output/spmv_detailed_20260302_111612.json`

Large 配置：

- 实验版：`speed=0.1651`, `memory=0.1658 ms/token`
- 对照版：`speed=0.1482`, `memory=0.1464 ms/token`

相对对照版：

- `speed` 回退约 `+11.4%`
- `memory` 回退约 `+13.3%`

---

## 4. 结论

- 阶段8在当前实现上为负收益。
- 已按流程回滚，不并入主干。

