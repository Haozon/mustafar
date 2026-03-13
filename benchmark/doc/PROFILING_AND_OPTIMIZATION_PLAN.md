# 量化 Kernel 性能分析与优化计划

## 📋 任务概述

**目标：** 优化量化版本的 TPOT 性能，从 82.78ms 降低到 25-30ms（快 2.8-3.3x）

**当前状态：**
- TPOT: 82.78 ms（比无量化版本慢 55%）
- TTFT: 3469 ms（比无量化版本快 49%）✅
- 内存: 41.28 GB（节省 7.7%）✅

**核心问题：** Decode 阶段的反量化开销导致性能下降

---

## 🔍 Phase 1: 深度 Profiling（当前阶段）

### 1.1 已知的性能数据

从之前的 Nsys 分析：

| Kernel | 调用次数 | 总时间 | 平均时间 | 时间占比 |
|--------|---------|--------|---------|---------|
| `Key_Kernel_Quant` | 229,440 | 210.4s | 0.92ms | 34.5% |
| `Value_Kernel_Quant` | 229,440 | 187.6s | 0.82ms | 30.7% |
| **合计** | **458,880** | **398s** | **1.74ms** | **65.2%** |

### 1.2 需要深入分析的指标

运行详细的 Nsys profiling：

```bash
cd /home/zh/mustafar/benchmark

# 运行 profiling（生成详细的 kernel 指标）
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    -o quant_detailed_profile \
    python mem_spd_test_quant.py

# 导出 kernel 统计
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output quant_kernel_stats.csv

# 查看内存带宽利用率
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_mem_time_sum \
    --format csv \
    --output quant_memory_stats.csv
```

**关键指标：**
1. **Memory Bandwidth Utilization** - 内存带宽利用率
   - 预期：量化版本应该更低（因为反量化增加了计算）
   - 目标：找出是否 memory-bound

2. **Compute Utilization** - 计算单元利用率
   - 预期：量化版本可能更高（反量化计算）
   - 目标：找出是否 compute-bound

3. **Occupancy** - SM 占用率
   - 预期：应该接近 100%
   - 目标：确认没有资源限制

4. **Warp Efficiency** - Warp 执行效率
   - 预期：应该 > 90%
   - 目标：确认没有分支发散

### 1.3 对比无量化版本

同时 profile 无量化版本：

```bash
# Profile 无量化版本
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    -o non_quant_detailed_profile \
    python mem_spd_test.py

# 导出统计
nsys stats non_quant_detailed_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output non_quant_kernel_stats.csv
```

**对比分析：**
- 内存访问模式差异
- 计算强度差异
- Kernel launch overhead 差异

---

## 🔬 Phase 2: 代码分析（根本原因定位）

### 2.1 反量化代码分析

从 `kernel_quant/csrc/SpMM_Kernel_Quant.cuh` 中的关键代码：

```cuda
// 在 SpMM_DecompressFromRegisterToShared_Quant 函数中
for (int j = 0; j < 64; j++) {
    // 1. 从 bitmap 找到非零位置
    pos1 = __clzll(bmp);
    bmp &= ~(0x8000000000000000ULL >> pos1);
    
    // 2. 从打包的 uint32 中提取量化值
    int unit_idx = j / capacity;           // 第几个 uint32
    int bit_offset = (j % capacity) * bit; // 在 uint32 内的位偏移
    uint32_t packed_unit = quant_units[unit_idx];
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    
    // 3. 反量化 ⚠️ 这是主要瓶颈
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    
    // 4. 转换为 half 并写入共享内存
    int output_idx = fuk + (pos1 << 6);
    SharedPTR[output_idx] = __float2half(dequant_value);
}
```

**性能瓶颈分析：**

1. **内存访问开销：**
   ```
   无量化：读取 1 个 FP16 值（2 bytes）
   量化：读取 1 个 uint32（4 bytes）+ 1 个 scale（4 bytes）+ 1 个 zero（4 bytes）
        = 12 bytes 总访问量
   
   内存带宽增加：6x
   ```

2. **计算开销：**
   ```
   无量化：0 次额外计算
   量化：每个非零值需要：
        - 1 次整数除法（unit_idx）
        - 1 次整数取模（bit_offset）
        - 1 次位移（packed_unit >> bit_offset）
        - 1 次位与（& mask）
        - 1 次类型转换（static_cast<float>）
        - 1 次减法（- zero_point）
        - 1 次乘法（* scale）
        - 1 次 FP32 → FP16 转换
   
   总计：8 次额外操作 / 非零值
   ```

3. **Scale/Zero 访问模式：**
   ```cuda
   // 当前：每个 tile 读取一次（已经优化）
   float scale = Registers_scale[i];
   float zero_point = Registers_zero[i];
   
   // 但是：scale/zero 存储在寄存器中，访问效率已经很高
   // 问题不在这里！
   ```

### 2.2 根本原因总结

**主要瓶颈：反量化计算本身**

量化版本的流程：
```
1. 从全局内存加载 uint32 打包数据 → 寄存器
2. 从全局内存加载 scale/zero → 寄存器
3. 在寄存器中解包 2-bit 值
4. 反量化：(q - zero) * scale
5. 转换为 FP16
6. 写入共享内存
7. 从共享内存读取进行 SpMM 计算
```

无量化版本的流程：
```
1. 从全局内存加载 FP16 值 → 寄存器
2. 写入共享内存
3. 从共享内存读取进行 SpMM 计算
```

**额外开销来源：**
- 步骤 3-5：解包 + 反量化 + 类型转换（~30ms）
- 内存访问量增加 6x（但可能不是主要瓶颈，因为数据在寄存器中）

---

## 🎯 Phase 3: 优化策略

### 策略 1：融合反量化到 SpMM Kernel（最重要）

**当前问题：**
- 反量化在 `SpMM_DecompressFromRegisterToShared_Quant` 中完成
- 数据流：寄存器 → 反量化 → 共享内存 → SpMM 计算
- 中间结果（FP16）需要写入共享内存

**优化方案：**
- 在 SpMM 计算时直接从量化数据读取并反量化
- 数据流：寄存器 → 反量化 → 直接用于计算（无需写入共享内存）

**实现难点：**
1. SpMM 计算使用 Tensor Core（需要 FP16 输入）
2. 共享内存用于 Tensor Core 的数据加载
3. 无法直接在 Tensor Core 中进行反量化

**可行性分析：**
- ❌ 无法完全消除共享内存写入（Tensor Core 需要）
- ✅ 可以优化反量化的计算效率
- ✅ 可以优化内存访问模式

**结论：融合反量化的收益有限（预期 1.2-1.3x）**

### 策略 2：优化反量化计算（重点）

**当前实现：**
```cuda
// 每个非零值都需要这些操作
int unit_idx = j / capacity;           // 整数除法（慢）
int bit_offset = (j % capacity) * bit; // 整数取模（慢）
uint32_t packed_unit = quant_units[unit_idx];
uint32_t q_value = (packed_unit >> bit_offset) & mask;
float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
```

**优化方案 A：预计算索引**
```cuda
// 预计算所有的 unit_idx 和 bit_offset
constexpr int unit_indices[64] = {0,0,0,...,3,3,3};  // 编译时常量
constexpr int bit_offsets[64] = {0,2,4,...,28,30};   // 编译时常量

#pragma unroll
for (int j = 0; j < 64; j++) {
    int unit_idx = unit_indices[j];      // 查表（快）
    int bit_offset = bit_offsets[j];     // 查表（快）
    uint32_t q_value = (quant_units[unit_idx] >> bit_offset) & 0x3;
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    SharedPTR[output_idx] = __float2half(dequant_value);
}
```

**预期收益：消除整数除法和取模，快 1.3-1.5x**

**优化方案 B：向量化解包**
```cuda
// 一次性解包整个 uint32（16 个 2-bit 值）
uint32_t packed = quant_units[0];
uint8_t values[16];

#pragma unroll
for (int i = 0; i < 16; i++) {
    values[i] = (packed >> (i * 2)) & 0x3;
}

// 批量反量化
#pragma unroll
for (int i = 0; i < 16; i++) {
    float dequant = (static_cast<float>(values[i]) - zero_point) * scale;
    SharedPTR[...] = __float2half(dequant);
}
```

**预期收益：减少位操作次数，快 1.2-1.3x**

**优化方案 C：使用 SIMD 指令**
```cuda
// 使用 __shfl_sync 在 warp 内广播 scale/zero
float scale = __shfl_sync(0xffffffff, Registers_scale[i], 0);
float zero = __shfl_sync(0xffffffff, Registers_zero[i], 0);

// 使用向量化加载
uint4 packed_vec = *reinterpret_cast<const uint4*>(&quant_units[0]);
// 批量处理 4 个 uint32（64 个 2-bit 值）
```

**预期收益：提高内存访问效率，快 1.1-1.2x**

### 策略 3：减少数据类型转换

**当前实现：**
```cuda
float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);
```

**优化方案：使用 half 精度计算**
```cuda
// 将 scale 和 zero 存储为 half
half scale_h = __float2half(scale);
half zero_h = __float2half(zero_point);

// 直接在 half 精度下计算
half q_h = __int2half_rn(q_value);
half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
SharedPTR[output_idx] = dequant_h;
```

**预期收益：减少类型转换，快 1.1-1.2x**

### 策略 4：优化内存访问模式（次要）

**当前实现：**
```cuda
// 每个线程独立访问 quant_units
uint32_t packed_unit = quant_units[unit_idx];
```

**优化方案：使用共享内存缓存**
```cuda
// 将 quant_units 加载到共享内存
__shared__ uint32_t shared_quant[MAX_UNITS];

// 协作加载
if (threadIdx.x < num_units) {
    shared_quant[threadIdx.x] = quant_units[threadIdx.x];
}
__syncthreads();

// 从共享内存读取
uint32_t packed_unit = shared_quant[unit_idx];
```

**预期收益：提高缓存命中率，快 1.05-1.1x**

---

## 📊 优化效果预测

| 优化方案 | 实现难度 | 预期加速 | 累积加速 | 优化后 TPOT |
|---------|---------|---------|---------|------------|
| **当前** | - | 1.0x | 1.0x | 82.78 ms |
| **方案 A：预计算索引** | 低 | 1.3-1.5x | 1.3-1.5x | 55-64 ms |
| **+ 方案 B：向量化解包** | 中 | 1.2-1.3x | 1.6-2.0x | 41-52 ms |
| **+ 方案 C：SIMD 指令** | 中 | 1.1-1.2x | 1.8-2.4x | 34-46 ms |
| **+ 方案 D：Half 精度** | 低 | 1.1-1.2x | 2.0-2.9x | 29-41 ms |
| **+ 方案 E：共享内存** | 低 | 1.05-1.1x | 2.1-3.2x | 26-39 ms |

**目标：TPOT < 30 ms（比无量化快 1.8x）**

---

## 🛠️ 实施计划

### Week 1: Profiling & 基础优化

**Day 1-2: 深度 Profiling**
- [ ] 运行详细的 Nsys profiling
- [ ] 分析内存带宽和计算利用率
- [ ] 对比无量化版本的差异
- [ ] 确认主要瓶颈

**Day 3-4: 实现方案 A（预计算索引）**
- [ ] 修改 `SpMM_DecompressFromRegisterToShared_Quant`
- [ ] 使用编译时常量数组替代除法/取模
- [ ] 测试正确性
- [ ] 测试性能（预期：55-64 ms）

**Day 5: 实现方案 D（Half 精度）**
- [ ] 修改 scale/zero 的数据类型
- [ ] 使用 half 精度进行反量化计算
- [ ] 测试正确性（可能有精度损失）
- [ ] 测试性能

### Week 2: 高级优化

**Day 1-3: 实现方案 B（向量化解包）**
- [ ] 重构解包逻辑
- [ ] 批量处理 uint32
- [ ] 优化循环展开
- [ ] 测试性能（预期：41-52 ms）

**Day 4-5: 实现方案 C（SIMD 指令）**
- [ ] 使用 warp shuffle 指令
- [ ] 使用向量化加载（uint4）
- [ ] 优化 warp 内协作
- [ ] 测试性能（预期：34-46 ms）

### Week 3: 精细调优

**Day 1-2: 实现方案 E（共享内存优化）**
- [ ] 添加共享内存缓存
- [ ] 优化协作加载
- [ ] 测试性能（预期：26-39 ms）

**Day 3-4: 综合测试**
- [ ] 在 LongBench 上完整测试
- [ ] 验证精度（与无量化版本对比）
- [ ] 测试不同 batch size 和序列长度
- [ ] 生成性能报告

**Day 5: 文档和总结**
- [ ] 更新文档
- [ ] 总结优化经验
- [ ] 准备下一步工作

---

## 📝 下一步行动

### 立即开始：

1. **运行详细 Profiling**
   ```bash
   cd /home/zh/mustafar/benchmark
   
   # Profile 量化版本
   nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
       -o quant_detailed python mem_spd_test_quant.py
   
   # 导出统计
   nsys stats quant_detailed.nsys-rep \
       --report cuda_gpu_kern_sum --format csv \
       --output quant_kernel_stats.csv
   ```

2. **分析 Profiling 结果**
   ```bash
   # 查看 kernel 统计
   cat quant_kernel_stats.csv | grep "Key_Kernel_Quant\|Value_Kernel_Quant"
   
   # 查看内存统计
   nsys stats quant_detailed.nsys-rep \
       --report cuda_gpu_mem_time_sum --format csv
   ```

3. **开始实现方案 A（预计算索引）**
   - 修改 `kernel_quant/csrc/SpMM_Kernel_Quant.cuh`
   - 在 `SpMM_DecompressFromRegisterToShared_Quant` 函数中
   - 添加编译时常量数组

---

## 🎯 成功标准

1. **性能目标：**
   - TPOT: < 30 ms（当前 82.78 ms）
   - 加速比: > 2.7x
   - 相比无量化版本：快 1.8x

2. **正确性目标：**
   - LongBench 评测结果与无量化版本差异 < 1%
   - 无数值溢出或精度问题

3. **内存目标：**
   - 保持内存节省（< 42 GB）
   - 无内存泄漏

---

## 📚 参考资料

### 相关文件
- Kernel 实现：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh`
- API 接口：`kernel_quant/csrc/SpMM_API_Quant.cu`
- 模型集成：`models/llama_mustafar_quant_kernel.py`
- 压缩函数：`kernel_quant/compression_quant.py`

### 文档
- 算法总结：`benchmark/OPTIMIZATION_TASK_SUMMARY.md`
- Kernel 对比：`benchmark/KERNEL_COMPARISON_ANALYSIS.md`
- Profiling 结果：`benchmark/benchmark_results_20260211_092254/`

### CUDA 优化技巧
- 使用 `#pragma unroll` 展开循环
- 使用 `__restrict__` 提示编译器优化
- 使用 `__forceinline__` 强制内联
- 使用 warp shuffle 减少共享内存访问
- 使用向量化加载（uint2, uint4）提高带宽利用率

---

## ✅ Checklist

### Phase 1: Profiling
- [ ] 运行 Nsys profiling（量化版本）
- [ ] 运行 Nsys profiling（无量化版本）
- [ ] 分析内存带宽利用率
- [ ] 分析计算利用率
- [ ] 确认主要瓶颈

### Phase 2: 基础优化
- [ ] 实现预计算索引（方案 A）
- [ ] 实现 Half 精度计算（方案 D）
- [ ] 测试正确性
- [ ] 测试性能

### Phase 3: 高级优化
- [ ] 实现向量化解包（方案 B）
- [ ] 实现 SIMD 指令（方案 C）
- [ ] 实现共享内存优化（方案 E）
- [ ] 综合测试

### Phase 4: 验证
- [ ] LongBench 完整测试
- [ ] 精度验证
- [ ] 性能报告
- [ ] 文档更新

---

**准备好了吗？让我们开始优化！** 🚀
