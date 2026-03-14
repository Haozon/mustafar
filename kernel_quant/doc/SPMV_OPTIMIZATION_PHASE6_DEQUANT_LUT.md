# SpMV 优化阶段 6 详细说明（2-bit Dequant LUT）

**日期**: 2026-03-02  
**状态**: 已实测，A/B 回退  
**目标**: 用 tile 内 4 项查表替代每值反量化算术，减少内层 `sub/mul/int2half`。

---

## 1. 原理

当前量化位宽为 2-bit，量化值 `q` 仅有 `{0,1,2,3}` 四种取值。  
在每个 tile 内，`scale/zero` 固定，因此可先构造：

- memory 路径：`lut_h[q] = (half(q) - zero_h) * scale_h`
- speed 路径：`lut_h[q] = half((float(q) - zero_f) * scale_f)`

解包时直接：

```cpp
q_value = packed & 0x3u;
SharedPTR[out_idx] = lut_h[q_value];
```

理论上可降低热路径算术密度。

---

## 2. 改动位置

- `csrc/SpMM_Kernel_Quant.cuh`
  - `SpMM_DecompressFromRegisterToShared_Quant`
  - 仅 Fast2Bit 分支（`DequantMode=speed/memory`）加入 LUT 预计算与索引写回

---

## 3. A/B 测试设计

为排除环境漂移，使用同时段对照：

1. **LUT 版本**
- `kernel_bench/output/spmv_detailed_20260302_110703.json`

2. **非 LUT 对照版本（回滚后）**
- `kernel_bench/output/spmv_detailed_20260302_110817.json`

命令一致：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

---

## 4. 实测结果（Large）

- LUT 版本：`speed=0.1586`, `memory=0.1604 ms/token`
- 非 LUT：`speed=0.1500`, `memory=0.1478 ms/token`

相对非 LUT 对照：

- `speed`: `+5.7%`（回退）
- `memory`: `+8.5%`（回退）

---

## 5. 结论与处理

- 阶段6 在当前实现与硬件上为**负收益**。
- 已按流程回滚，不并入主干。
- 可能原因：LUT 增加寄存器压力，抵消了算术减少带来的收益。

---

## 6. 回滚说明

回滚范围仅：

- `SpMM_DecompressFromRegisterToShared_Quant` Fast2Bit 分支

恢复为逐值反量化公式：

- memory: `__hmul(__hsub(q_h, zero_h), scale_h)`
- speed: `(float(q)-zero_f)*scale_f`

