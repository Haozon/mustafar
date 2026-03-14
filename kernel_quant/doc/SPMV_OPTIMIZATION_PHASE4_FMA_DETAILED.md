# SpMV 优化阶段 4 详细说明（FMA 反量化融合）

**日期**: 2026-03-02  
**状态**: 已回滚（实测回退）  
**目标**: 在不改动整体 kernel 结构的前提下，进一步压缩反量化热路径指令开销。

---

## 1. 背景与动机

在阶段 3 后，量化路径的主要热点已经集中在解压 + 反量化内层循环。  
旧实现中每个量化值都执行：

- speed 模式：`(q - zero) * scale`
- memory 模式：`__hmul(__hsub(q_h, zero_h), scale_h)`

这本质是“减法 + 乘法”两步。  
本阶段采用融合乘加（FMA）形式，把它改为“一步完成”，并把与 `q` 无关的常量提前到 tile 级预计算。

---

## 2. 数学原理（等价变换）

原式：

`dequant = (q - zero) * scale`

展开：

`dequant = q * scale + (-zero * scale)`

令：

`bias = -zero * scale`

则可写为：

`dequant = fma(q, scale, bias)`

其中 `zero` 和 `scale` 在 tile 内恒定，`bias` 可在 tile 级别只计算一次。  
这样把每元素“减+乘”变为“fma”，降低热路径指令数和依赖链长度。

---

## 3. 代码改动总览

改动文件：

- `csrc/SpMM_Kernel_Quant.cuh`

核心位置：

- `SpMM_DecompressFromRegisterToShared_Quant`（约 `#L154` 起）

---

## 4. 具体实现细节

### 4.1 memory 模式（`DequantMode == DEQUANT_MODE_MEMORY`）

tile 级预计算：

- `scale_f_tile = __half2float(scale_h)`
- `zero_f_tile = __half2float(zero_h)`
- `bias_h = __float2half(-zero_f_tile * scale_f_tile)`

内层反量化由：

- `__hmul(__hsub(q_h, zero_h), scale_h)`

改为：

- `__hfma(q_h, scale_h, bias_h)`

说明：

- `bias_h` 的预计算放在 tile 循环内，不在每个 `q` 上重复计算。
- 使用 `__hfma` 保持半精度路径一致性和吞吐友好性。

### 4.2 speed 模式（`DequantMode == DEQUANT_MODE_SPEED`）

tile 级预计算：

- `scale_f = __half2float(scale_h)`
- `zero_f = __half2float(zero_h)`
- `bias_f = -zero_f * scale_f`

内层反量化由：

- `(static_cast<float>(q_value) - zero_f) * scale_f`

改为：

- `__fmaf_rn(static_cast<float>(q_value), scale_f, bias_f)`

说明：

- 与 memory 模式一致，减少每元素常量相关运算。
- `__fmaf_rn` 显式使用 round-to-nearest-even 语义。

---

## 5. 正确性说明

该优化属于代数等价变换，不改变理论值。  
可能出现的差异仅来自浮点舍入路径变化（正常、可接受）：

- 旧路径：先减再乘（两次舍入机会）
- 新路径：融合乘加（一次融合舍入）

在当前任务（2-bit 反量化）中，这类差异通常远小于量化本身误差。

---

## 6. 风险与边界

主要风险：

- 寄存器压力变化：新增 `bias_f/bias_h` 可能带来轻微寄存器占用变化。
- 架构相关收益差异：不同 GPU 上 FMA 吞吐与调度特性不同，收益幅度可能波动。

非风险项：

- 不修改数据布局
- 不修改调用接口
- 不修改 bitmap 解码逻辑
- 不修改 `Fast2Bit` / 通用回退分支语义

---

## 7. 与前几轮优化关系

本阶段不替代前面优化，而是叠加：

- 编译期 `dequant_mode` 特化（阶段 3）
- Fast2Bit 顺序解包（阶段 3）
- 位图解码 `brev+ffs`（阶段 3）
- 本阶段 FMA 融合反量化（阶段 4）

这保证了改动范围小、可逐轮回归。

---

## 8. 验证与回归建议

编译验证：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_wrapper
python setup.py build_ext --inplace
```

性能回归：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

重点指标（Large）：

- `quant_by_mode["0"].batch_spmv_avg`（speed）
- `quant_by_mode["1"].batch_spmv_avg`（memory）
- 与 `spmv_detailed_20260302_093621.json` 对比变化

判定建议：

- 若 Large speed 比 0.1057 更低，说明阶段 4 生效。
- 若出现回退，优先检查寄存器压力和 occupancy 变化（Nsight Compute）。

---

## 9. 回滚说明（单点回退）

若阶段 4 不稳定，可仅回退 FMA 逻辑，不影响阶段 3：

- speed 路径：把 `__fmaf_rn(...)` 改回 `(q - zero) * scale`
- memory 路径：把 `__hfma(...)` 改回 `__hmul(__hsub(...), ...)`

回退范围仅 `csrc/SpMM_Kernel_Quant.cuh` 的反量化内层。

---

## 10. 实测结果回填（20260302_101112）

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_101112.json`
- `kernel_bench/output/spmv_detailed_20260302_101112.md`

Large 配置（`seq=2048, heads=32, decode=1024`）：

- Sparse TPOT: `0.0707 ms/token`
- Quant TPOT (`speed`): `0.1669 ms/token`
- Quant TPOT (`memory`): `0.1631 ms/token`

对比阶段3最优（20260302_093621）：

- `speed`: `0.1057 -> 0.1669`（约 `+57.9%` 回退）
- `memory`: `0.1088 -> 0.1631`（约 `+49.8%` 回退）

对比里程碑：

- 相对基线 `0.2903`：仅 `1.74x~1.78x`（低于阶段3的 `2.75x`）
- 相对阶段1最优 `0.1220`：慢约 `33.7%~36.8%`

结论：

- 本轮 FMA 融合在当前实现与当前硬件上出现明显性能回退。  
- 建议按第 9 节回滚阶段4改动，恢复阶段3最优版本作为主干。
