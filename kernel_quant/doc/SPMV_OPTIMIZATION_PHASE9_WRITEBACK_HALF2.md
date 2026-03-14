# SpMV 优化阶段 9 详细说明（Shared 写回 half2 向量化）

**日期**: 2026-03-02  
**状态**: 已实测，A/B 回退  
**目标**: 在结果写回阶段，将 `smem_CFrag(float)` 到 `global(half)` 的标量写回改为 `half2` 向量化写回。

---

## 1. 原理

原实现在 Key/Value kernel 尾声使用标量循环：

- `__float2half_rn` 每值转换
- `half` 每值写回全局内存

阶段9尝试在满足偶数长度条件时改为：

- 每线程处理 2 个相邻元素
- `float2 -> half2`（`__float22half2_rn`）
- `half2` 向量化写回

并保留标量回退分支用于不满足条件的场景。

---

## 2. 改动文件

- `csrc/SpMM_Kernel_Quant.cuh`
  - `Key_Kernel_Quant` 尾声写回环节
  - `Value_Kernel_Quant` 尾声写回环节

---

## 3. A/B 测试

同一时段对照：

- 基线（优化前）：`kernel_bench/output/spmv_detailed_20260302_112309.json`
- 实验版（half2写回）：`kernel_bench/output/spmv_detailed_20260302_112500.json`

命令一致：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

---

## 4. 实测结果（Large）

- 基线：`speed=0.1472`, `memory=0.1525 ms/token`
- 实验：`speed=0.1478`, `memory=0.1537 ms/token`

相对基线：

- `speed`: `+0.4%`（回退）
- `memory`: `+0.8%`（回退）

---

## 5. 结论

- 阶段9未带来收益，且在 Large 主指标上轻微回退。
- 已按流程回滚，不并入主干。

