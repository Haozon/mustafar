# SpMV 优化阶段 3 实现说明（编译期 dequant 特化 + Fast2Bit 顺序解包）

**日期**: 2026-03-01  
**状态**: 已实现，待实测  
**目标**: 降低量化解压内层循环的控制开销与位操作开销

---

## 1. 背景问题

在阶段2实现后，`dequant_mode` 由运行时参数控制，kernel 内层循环仍包含模式分支；同时 `Fast2Bit` 虽然去掉了除法/取模，但每个量化值仍执行一次 `unit_idx/bit_offset` 计算思路。

这两点会在 `SpMM_DecompressFromRegisterToShared_Quant` 热路径上引入额外指令和控制流。

---

## 2. 本轮优化内容

### 2.1 `dequant_mode` 编译期特化

- 将 `DequantMode` 提升为模板参数（`DEQUANT_MODE_SPEED` / `DEQUANT_MODE_MEMORY`）。
- API 启动层按 `dequant_mode` 在 launch 前分发到对应模板实例。
- kernel 内层去掉运行时模式判断，改为 `if constexpr`。

预期收益：
- 减少热循环中的分支开销
- 让编译器更充分内联和优化两条反量化路径

### 2.2 Fast2Bit 顺序解包（向量化解包落地）

- 对 `Fast2Bit` 路径改为“按 `uint32` 为单位批量顺序解包”：
  - `q = packed & 0x3`
  - `packed >>= 2`
- 每个 `uint32` 一次加载后顺序消费 16 个 2-bit 值，不再为每个 `j` 计算 `unit_idx` 与 `bit_offset`。
- 通用路径（非2-bit或非`capacity=16`）保留原有逻辑。

预期收益：
- 降低位偏移计算与索引计算开销
- 提高内层循环指令局部性

### 2.3 清理内层死操作

- 移除解压循环内无效的 `pos1++`（下一轮会被 `pos1 = __clzll(bmp)` 直接覆盖）。
- 避免无意义指令进入热路径。

### 2.4 减少常驻寄存器

- 删除 `Registers_tile_offset[2]` 常驻寄存器数组（仅在加载阶段瞬时使用）。
- `tile_offset` 改为在 `SpMM_CopyFromGlobalToReg_Quant` 内按 `globalTileIdx` 直接读取局部变量。
- 目标是降低寄存器压力，改善 occupancy 空间。

### 2.5 位图解码路径优化（`clz` -> `brev+ffs`）

- 原路径每个非零值使用：
  - `pos1 = __clzll(bmp)`
  - `bmp &= ~(0x8000000000000000ULL >> pos1)`
- 新路径改为：
  - 先一次 `bmp_rev = __brevll(bmp)`
  - 每个非零值 `pos1 = __ffsll(bmp_rev) - 1`
  - 弹出最低位 `bmp_rev &= (bmp_rev - 1)`

收益目标：

- 减少每个非零值的位图定位与清位开销
- 在高 `nnz_tile` 场景下进一步降低解压热路径指令数
- 并将循环边界改为 `j < nnz_tile`，减少无效迭代判断

---

## 3. 改动文件

- `csrc/SpMM_Kernel_Quant.cuh`
  - `SpMM_DecompressFromRegisterToShared_Quant` 增加模板参数 `DequantMode`
  - `Key_Kernel_Quant` / `Value_Kernel_Quant` 增加模板参数 `DequantMode`
  - `Fast2Bit` 路径改为顺序解包

- `csrc/SpMM_API_Quant.cu`
  - 启动器模板增加 `DequantMode`
  - 按 `Fast2Bit x DequantMode` 进行编译期实例分发
  - kernel 启动参数移除运行时 `dequant_mode` 传递

---

## 4. 编译验证

已通过：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_wrapper
python setup.py build_ext --inplace
```

---

## 5. 建议实测方式

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

重点观察（Large）：

- `quant_by_mode["0"].batch_spmv_avg`
- `quant_by_mode["1"].batch_spmv_avg`
- 相对 `spmv_detailed_20260301_222428.json` 的变化

---

## 6. 首轮实测结果（20260302_092021，speed模式）

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_092021.json`
- `kernel_bench/output/spmv_detailed_20260302_092021.md`

Large 配置（`dequant_mode=0`）：

- Sparse TPOT: `0.0928 ms/token`
- Quant TPOT: `0.1482 ms/token`
- Quant single_spmv: `0.0592 ms`
- Quant 内存: `3.2235 MB`

对比阶段2最近一次（`20260301_222428`）：

- Quant TPOT：`0.1884 -> 0.1482 ms/token`（约 `21.3%` 提升）
- Quant single_spmv：`0.0597 -> 0.0592 ms`（基本持平）

对比基线与阶段1：

- 相对初始基线 `0.2903 ms/token`：累计约 `1.96x` 加速
- 相对阶段1最优 `0.1220 ms/token`：仍慢约 `21.5%`

---

## 7. 双模式复测结果（20260302_093007，speed vs memory）

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_093007.json`
- `kernel_bench/output/spmv_detailed_20260302_093007.md`

Large 配置（`seq=2048, heads=32, decode=1024`）：

- Sparse TPOT: `0.0920 ms/token`
- Quant TPOT (`speed`): `0.1419 ms/token`
- Quant TPOT (`memory`): `0.1390 ms/token`

关键结论：

- `memory` 模式在 Large 上优于 `speed`，约 `2.0%`（`0.1419 -> 0.1390`）。
- 相对 20260302_092021（`0.1482`）进一步提升约 `6.2%`。
- 相对初始基线 `0.2903`：达到约 `2.09x` 加速。
- 相对阶段1最优 `0.1220`：仍慢约 `14.0%`。

---

## 8. 后续增量（待复测）

已在当前分支实现位图解码优化（2.5），并通过编译：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_wrapper
python setup.py build_ext --inplace
```

建议下一轮优先复测：

```bash
cd /mnt/home/zh/mustafar/kernel_quant/kernel_bench
python benchmark_spmv_detailed.py --dequant-mode both
```

---

## 9. 位图解码优化实测结果（20260302_093621）

数据文件：

- `kernel_bench/output/spmv_detailed_20260302_093621.json`
- `kernel_bench/output/spmv_detailed_20260302_093621.md`

Large 配置（`seq=2048, heads=32, decode=1024`）：

- Sparse TPOT: `0.0949 ms/token`
- Quant TPOT (`speed`): `0.1057 ms/token`
- Quant TPOT (`memory`): `0.1088 ms/token`

对比上一轮（20260302_093007）：

- `speed`: `0.1419 -> 0.1057`（约 `25.5%` 提升）
- `memory`: `0.1390 -> 0.1088`（约 `21.7%` 提升）
- Large 下最优模式从 `memory` 切换为 `speed`

对比关键里程碑：

- 相对基线 `0.2903`：`speed` 达到约 `2.75x` 加速
- 相对阶段1最优 `0.1220`：`speed` 再快约 `13.4%`
- 相对 Sparse `0.0949`：Quant 仍慢约 `11.4%`

---

## 10. FMA 反量化融合（已实现，待复测）

实现内容：

- speed 模式：将
  - `(float(q) - zero_f) * scale_f`
  - 改为 `__fmaf_rn(float(q), scale_f, bias_f)`，其中 `bias_f = -zero_f * scale_f`
- memory 模式：将
  - `__hmul(__hsub(q_h, zero_h), scale_h)`
  - 改为 `__hfma(q_h, scale_h, bias_h)`，其中 `bias_h = half(-zero*scale)`（tile 级预计算）

目的：

- 将反量化内层从“减+乘”收敛为“融合乘加”，减少热路径指令并提高流水线利用率。

状态：

- 已完成代码改造并通过编译，待 benchmark 回填实际收益。

---

## 11. 阶段3分项拆解（便于论文写作）

阶段3实际包含三类“低侵入”优化，目标是让热路径更线性：

1. 编译期分发：`Fast2Bit x DequantMode`
- 去掉运行时分支判断
- 让编译器针对具体路径内联/展开

2. 解包重排：`uint32` 顺序消费 16 个 2-bit
- 从“每值计算偏移”改成“每单元批量弹出”
- 减少位偏移计算与索引计算

3. 位图弹位：`brev + ffs + x&(x-1)`
- 从 `clz + mask_clear` 改为低位弹出模式
- 改善每个非零值的位置定位开销

---

## 12. 前后逻辑对照（伪代码）

### 12.1 旧思路（概念上）

```cpp
for each nonzero j:
  pos = clz(bmp)
  clear bit by mask
  unit = j / capacity
  off  = (j % capacity) * bit
  q = (packed[unit] >> off) & mask
  dequant = ...
```

### 12.2 新思路（阶段3后）

```cpp
bmp_rev = brev(bmp)
for each packed uint32:
  repeat 16 lanes:
    pos = ffs(bmp_rev) - 1
    bmp_rev &= (bmp_rev - 1)
    q = packed & 0x3
    packed >>= 2
    dequant = ...
```

编译期再固定：

- `Fast2Bit=true/false`
- `DequantMode=speed/memory`

这样内层不会再走运行时分支选择。

---

## 13. 正确性不变量

阶段3虽然改了“怎么取值”，但保持了以下不变量：

1. 非零元素访问顺序不变  
- `brev + ffs` 等价于原先“从 MSB 到 LSB”的位图扫描顺序

2. 解包值不变  
- `packed & 0x3; packed >>= 2` 与 `(packed >> bit_offset) & 0x3` 逐值等价

3. 写回坐标不变  
- `output_idx = base + (pos << 6)` 公式保持一致

4. 接口不变  
- Python 调用与参数签名兼容（`dequant_mode` 仍是可选参数）

---

## 14. 分项收益归因建议

如果后续要在论文里写“每项贡献”，建议按下面顺序做增量 A/B：

1. 固定阶段2代码，打开编译期分发（只改分发，不改解包）
2. 在 1 基础上打开顺序解包
3. 在 2 基础上打开位图弹位

每一步都记录 Large 的：

- `quant.batch_spmv_avg`
- `quant.single_spmv`
- `memory_mb`

这样可避免“多个优化叠加后无法归因”的问题。

---

## 15. 回滚边界（阶段3）

如出现回退，建议按“从新到旧”的顺序回滚：

1. 回退位图弹位（恢复 `clz` 方案）
2. 回退顺序解包（恢复 `unit_idx/bit_offset` 方案）
3. 回退编译期分发（恢复运行时模式判断）

每次只回退一项并重新 benchmark，便于锁定回退来源。
