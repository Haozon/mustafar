# 新 Session 任务总结

## 📋 任务概述

**目标：** Profile 并优化量化版本的 Kernel，将 TPOT 从 82.78ms 降低到 25-30ms

**当前问题：** 量化版本在 decode 阶段（TPOT）比无量化版本慢 55%

---

## 🎯 核心问题

### 性能对比

| 指标 | 无量化版本 | 量化版本 | 差异 |
|------|-----------|---------|------|
| **TTFT** | 6798 ms | 3469 ms ✅ | **快 49%** |
| **TPOT** | 53.41 ms ✅ | 82.78 ms ❌ | **慢 55%** |
| **内存** | 44.72 GB | 41.28 GB ✅ | **节省 7.7%** |

### 根本原因

**反量化开销导致 TPOT 变慢**

量化版本的数据流：
```
Query × [Compressed Key (2-bit) + Scales + Zeros]
  ↓
解包 2-bit 值
  ↓
反量化：(q - zero) * scale
  ↓
转换为 FP16
  ↓
SpMM 计算
```

**额外开销：~30ms（反量化 + 类型转换）**

---

## 📊 已知的 Profiling 数据

从之前的 Nsys 分析（0211 测试）：

| Kernel | 调用次数 | 总时间 | 平均时间 | 时间占比 |
|--------|---------|--------|---------|---------|
| `Key_Kernel_Quant` | 229,440 | 210.4s | 0.92ms | 34.5% |
| `Value_Kernel_Quant` | 229,440 | 187.6s | 0.82ms | 30.7% |
| **合计** | **458,880** | **398s** | **1.74ms** | **65.2%** |

**关键发现：**
- 量化 kernel 占用了 65% 的总时间
- 单次调用时间：1.74ms（需要优化到 0.5-0.6ms）

---

## 🔍 瓶颈分析

### 反量化代码（主要瓶颈）

位置：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh`

函数：`SpMM_DecompressFromRegisterToShared_Quant`

```cuda
for (int j = 0; j < 64; j++) {
    // 1. 从 bitmap 找到非零位置
    pos1 = __clzll(bmp);
    
    // 2. 计算索引（慢！）
    int unit_idx = j / capacity;           // 整数除法
    int bit_offset = (j % capacity) * bit; // 整数取模
    
    // 3. 解包 2-bit 值
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    
    // 4. 反量化（慢！）
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    
    // 5. 类型转换（慢！）
    SharedPTR[output_idx] = __float2half(dequant_value);
}
```

**性能开销：**
- 整数除法/取模：~5-10 cycles
- 反量化计算：~10-15 cycles
- 类型转换：~5-10 cycles
- **总计：~20-35 cycles / 非零值**

---

## 🎯 优化策略

### 优先级排序

| 优化方案 | 难度 | 预期加速 | 累积加速 | 优化后 TPOT |
|---------|------|---------|---------|------------|
| **当前** | - | 1.0x | 1.0x | 82.78 ms |
| **1. 预计算索引** | 低 | 1.3-1.5x | 1.3-1.5x | 55-64 ms |
| **2. Half 精度计算** | 低 | 1.1-1.2x | 1.4-1.8x | 46-59 ms |
| **3. 向量化解包** | 中 | 1.2-1.3x | 1.7-2.3x | 36-49 ms |
| **4. SIMD 指令** | 中 | 1.1-1.2x | 1.9-2.8x | 30-44 ms |
| **5. 共享内存优化** | 低 | 1.05-1.1x | 2.0-3.1x | 27-41 ms |

**目标：TPOT < 30 ms（比无量化快 1.8x）**

### 优化方案详解

#### 方案 1：预计算索引（最重要，立即实施）

**问题：** 每个非零值都需要计算 `j / capacity` 和 `j % capacity`

**解决：** 使用编译时常量数组

```cuda
// 添加到文件开头
constexpr int unit_indices[64] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  // 0-15: unit 0
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  // 16-31: unit 1
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  // 32-47: unit 2
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3   // 48-63: unit 3
};

constexpr int bit_offsets[64] = {
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // unit 0
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // unit 1
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // unit 2
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30   // unit 3
};

// 在循环中使用
int unit_idx = unit_indices[j];    // 查表，快！
int bit_offset = bit_offsets[j];   // 查表，快！
```

**预期收益：消除整数除法/取模，快 1.3-1.5x**

#### 方案 2：Half 精度计算（简单，立即实施）

**问题：** FP32 → FP16 转换有开销

**解决：** 直接在 Half 精度下计算

```cuda
// 将 scale 和 zero 转换为 half（只需一次）
half scale_h = __float2half(scale);
half zero_h = __float2half(zero_point);

// 在循环中使用 half 精度
half q_h = __int2half_rn(q_value);
half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
SharedPTR[output_idx] = dequant_h;  // 无需转换
```

**预期收益：减少类型转换，快 1.1-1.2x**

#### 方案 3：向量化解包（中等难度）

**问题：** 逐个解包 2-bit 值效率低

**解决：** 一次性解包整个 uint32

```cuda
// 一次性解包 16 个 2-bit 值
uint32_t packed = quant_units[unit_idx];
uint8_t values[16];

#pragma unroll
for (int i = 0; i < 16; i++) {
    values[i] = (packed >> (i * 2)) & 0x3;
}

// 批量反量化
#pragma unroll
for (int i = 0; i < 16; i++) {
    half q_h = __int2half_rn(values[i]);
    half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
    SharedPTR[...] = dequant_h;
}
```

**预期收益：减少位操作，快 1.2-1.3x**

---

## 📁 关键文件位置

### 需要修改的文件

1. **Kernel 实现（主要修改）**
   ```
   /home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh
   ```
   - 修改 `SpMM_DecompressFromRegisterToShared_Quant` 函数
   - 添加预计算索引数组
   - 优化反量化计算

2. **编译脚本**
   ```
   /home/zh/mustafar/kernel_quant/setup.py
   ```
   - 重新编译 kernel

3. **测试脚本**
   ```
   /home/zh/mustafar/benchmark/mem_spd_test_quant.py
   ```
   - 验证性能

### 参考文档

1. **优化计划**
   ```
   /home/zh/mustafar/benchmark/PROFILING_AND_OPTIMIZATION_PLAN.md
   ```
   - 完整的优化路线图
   - 3 周实施计划

2. **快速开始**
   ```
   /home/zh/mustafar/benchmark/QUICK_START_PROFILING.md
   ```
   - Profiling 命令
   - 分析方法

3. **Kernel 对比**
   ```
   /home/zh/mustafar/benchmark/KERNEL_COMPARISON_ANALYSIS.md
   ```
   - 无量化 vs 量化的详细对比

4. **任务总结**
   ```
   /home/zh/mustafar/benchmark/OPTIMIZATION_TASK_SUMMARY.md
   ```
   - 之前的分析结果

---

## 🚀 立即开始的步骤

### Step 1: 运行 Profiling（5-10 分钟）

```bash
cd /home/zh/mustafar/benchmark
conda activate mustar

# 运行详细 profiling
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    -o quant_detailed_profile \
    python mem_spd_test_quant.py

# 导出统计
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output quant_kernel_stats.csv
```

### Step 2: 分析结果（5 分钟）

```bash
# 查看 kernel 统计
grep "Key_Kernel_Quant\|Value_Kernel_Quant" quant_kernel_stats.csv

# 确认瓶颈类型
# - Memory-Bound: 内存带宽利用率 > 80%
# - Compute-Bound: 计算利用率 > 80%
```

### Step 3: 实施优化 1（30 分钟）

```bash
# 编辑 kernel 文件
vim /home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh

# 添加预计算索引数组（见上面的代码）
# 修改 SpMM_DecompressFromRegisterToShared_Quant 函数

# 重新编译
cd /home/zh/mustafar/kernel_quant
python setup.py install

# 测试性能
cd /home/zh/mustafar/benchmark
python mem_spd_test_quant.py
# 查看 TPOT 是否降低到 55-64 ms
```

### Step 4: 实施优化 2（20 分钟）

```bash
# 继续编辑 kernel 文件
vim /home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh

# 添加 Half 精度计算（见上面的代码）

# 重新编译
cd /home/zh/mustafar/kernel_quant
python setup.py install

# 测试性能
cd /home/zh/mustafar/benchmark
python mem_spd_test_quant.py
# 查看 TPOT 是否降低到 46-59 ms
```

### Step 5: 验证正确性（10 分钟）

```bash
# 运行完整测试
cd /home/zh/mustafar/benchmark
bash run_complete_benchmark.sh

# 对比结果
# 确保精度没有明显下降
```

---

## ✅ 成功标准

### 短期目标（1-2 天）

- [ ] 完成 Profiling 分析
- [ ] 实施优化 1（预计算索引）
- [ ] 实施优化 2（Half 精度）
- [ ] TPOT 降低到 50-60 ms（快 1.4-1.7x）

### 中期目标（1 周）

- [ ] 实施优化 3（向量化解包）
- [ ] 实施优化 4（SIMD 指令）
- [ ] TPOT 降低到 35-45 ms（快 1.8-2.4x）

### 最终目标（2-3 周）

- [ ] 实施优化 5（共享内存）
- [ ] 综合调优
- [ ] TPOT 降低到 25-30 ms（快 2.8-3.3x）
- [ ] 在 LongBench 上验证精度
- [ ] 生成性能报告

---

## 📊 预期结果

### 优化前

```
Performance Metrics:
TTFT: 3681.42 ms
TPOT: 82.78 ms ❌
Total generation time: 88152 ms
Peak memory: 41.28 GB
```

### 优化后（目标）

```
Performance Metrics:
TTFT: 3681.42 ms (保持不变)
TPOT: 25-30 ms ✅ (快 2.8-3.3x)
Total generation time: 29000-34000 ms (快 2.6-3.0x)
Peak memory: 41.28 GB (保持不变)
```

### 与无量化版本对比

| 指标 | 无量化 | 量化（优化前） | 量化（优化后） |
|------|--------|--------------|--------------|
| TTFT | 6798 ms | 3469 ms ✅ | 3469 ms ✅ |
| TPOT | 53.41 ms | 82.78 ms ❌ | **25-30 ms** ✅✅ |
| 内存 | 44.72 GB | 41.28 GB ✅ | 41.28 GB ✅ |

**最终效果：量化版本在所有指标上都优于无量化版本！**

---

## 🆘 遇到问题？

### 编译错误

```bash
# 清理旧的编译文件
cd /home/zh/mustafar/kernel_quant
rm -rf build/ dist/ *.egg-info
python setup.py clean --all

# 重新编译
python setup.py install
```

### 性能没有提升

```bash
# 检查代码是否真的被修改
grep "unit_indices" /home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh

# 检查是否真的重新编译
ls -lt /home/zh/mustafar/kernel_quant/build/

# 重新运行测试
python mem_spd_test_quant.py
```

### 精度下降

```bash
# 如果使用 Half 精度导致精度下降
# 可以只使用优化 1（预计算索引）
# 跳过优化 2（Half 精度）
```

---

## 📚 相关资源

### CUDA 优化技巧

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsys Profiling Guide](https://docs.nvidia.com/nsight-systems/)

### 项目文档

- `benchmark/PROFILING_AND_OPTIMIZATION_PLAN.md` - 完整优化计划
- `benchmark/QUICK_START_PROFILING.md` - 快速开始指南
- `benchmark/KERNEL_COMPARISON_ANALYSIS.md` - Kernel 对比分析
- `benchmark/OPTIMIZATION_TASK_SUMMARY.md` - 任务总结

---

## 💡 关键洞察

1. **TTFT 已经很好**（快 49%）
   - Prefill 阶段的量化是成功的
   - 不需要优化 prefill

2. **TPOT 是主要问题**（慢 55%）
   - Decode 阶段的反量化开销太大
   - 需要优化反量化计算

3. **优化空间很大**
   - 当前实现没有充分优化
   - 简单的优化就能带来显著提升

4. **目标是可达成的**
   - 预期优化后比无量化版本快 1.8x
   - 同时保持内存节省

---

**准备好了吗？让我们开始优化！** 🚀

**第一步：运行 Profiling**
```bash
cd /home/zh/mustafar/benchmark
conda activate mustar
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --gpu-metrics-device=all -o quant_detailed python mem_spd_test_quant.py
```
