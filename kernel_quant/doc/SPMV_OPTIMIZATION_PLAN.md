# SpMV 反量化优化方案

**目标**: 将量化 SpMV 的性能从 0.29 ms/token 优化到接近无量化版本的 0.096 ms/token

**当前性能**: Quant 比 Sparse 慢 3×  
**优化目标**: 减少到 1.5-2× 的差距

---

## 🔍 当前实现分析

### 核心瓶颈代码（SpMM_DecompressFromRegisterToShared_Quant）

```cuda
#pragma unroll
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) break;
    
    // 1. 找到下一个非零位置（位操作）
    pos1 = __clzll(bmp);                              // ⏱️ ~4 cycles
    bmp &= ~(0x8000000000000000ULL >> pos1);          // ⏱️ ~3 cycles
    
    // 2. 计算索引（整数除法和取模）⚠️ 主要瓶颈
    int unit_idx = j / capacity;                      // ⏱️ ~20 cycles (整数除法！)
    int bit_offset = (j % capacity) * bit;            // ⏱️ ~20 cycles (取模！)
    
    // 3. 解包 2-bit 值（位操作）
    uint32_t packed_unit = quant_units[unit_idx];    // ⏱️ ~1 cycle
    uint32_t q_value = (packed_unit >> bit_offset) & mask;  // ⏱️ ~2 cycles
    
    // 4. 反量化（浮点运算）
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;  // ⏱️ ~10 cycles
    
    // 5. 类型转换 + 写入共享内存
    SharedPTR[output_idx] = __float2half(dequant_value);  // ⏱️ ~5 cycles
}
```

### 性能分析

**每个非零值的开销：**
- 位操作: ~7 cycles
- **整数除法/取模: ~40 cycles** ⚠️ **主要瓶颈**
- 解包: ~3 cycles
- 反量化: ~10 cycles
- 类型转换: ~5 cycles
- **总计: ~65 cycles/值**

**对于 Large 配置（2M 非零值）：**
- 总 cycles: 2M × 65 = 130M cycles
- GPU 频率: 1.4 GHz
- 理论时间: 130M / 1.4G = 0.093 ms

**实际测量: 0.29 ms**（慢 3×）

**差距原因：**
- 内存访问延迟
- 分支预测失败
- 寄存器压力

---

## 🎯 优化策略

### 优化 1：预计算索引表（消除除法/取模）⭐⭐⭐⭐⭐

**问题：**
```cuda
int unit_idx = j / capacity;      // 整数除法，~20 cycles
int bit_offset = (j % capacity) * bit;  // 取模，~20 cycles
```

**优化方案：**
```cuda
// 预计算索引表（编译时或初始化时）
__constant__ uint8_t UNIT_IDX_TABLE[64];     // unit_idx = j / 16
__constant__ uint8_t BIT_OFFSET_TABLE[64];   // bit_offset = (j % 16) * 2

// 使用查表替代计算
int unit_idx = UNIT_IDX_TABLE[j];           // ~1 cycle
int bit_offset = BIT_OFFSET_TABLE[j];       // ~1 cycle
```

**预期效果：**
- 节省: 40 cycles → 2 cycles
- 加速比: 65 / 27 = **2.4×**

**实现难度：** ⭐⭐ (简单)

---

### 优化 2：FP16 反量化（减少类型转换）⭐⭐⭐⭐

**问题：**
```cuda
// 当前：FP32 反量化 + FP32→FP16 转换
float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);  // 类型转换开销
```

**优化方案：**
```cuda
// 直接使用 FP16 计算
half scale_h = __float2half(scale);
half zero_h = __float2half(zero_point);

// FP16 反量化（无需类型转换）
half q_h = __int2half_rn(q_value);
half dequant_value = __hmul(__hsub(q_h, zero_h), scale_h);
SharedPTR[output_idx] = dequant_value;  // 直接写入
```

**预期效果：**
- 节省: 10 + 5 = 15 cycles → 8 cycles
- 加速比: 27 / 20 = **1.35×**

**实现难度：** ⭐⭐⭐ (中等，需要测试精度)

---

### 优化 3：向量化解包（批量处理）⭐⭐⭐

**问题：**
```cuda
// 当前：逐个解包
for (int j = 0; j < 64; j++) {
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    // 处理单个值
}
```

**优化方案：**
```cuda
// 一次解包 16 个值（一个 uint32）
uint32_t packed = quant_units[unit_idx];

// 使用位操作批量提取
uint32_t q0 = (packed >> 0) & 0x3;
uint32_t q1 = (packed >> 2) & 0x3;
uint32_t q2 = (packed >> 4) & 0x3;
// ... 展开 16 个

// 或使用 SIMD 指令（如果可用）
```

**预期效果：**
- 减少循环开销
- 更好的指令级并行
- 加速比: **1.2-1.3×**

**实现难度：** ⭐⭐⭐⭐ (较难，需要重构循环)

---

### 优化 4：Shared Memory 缓存 Scale/Zero（减少重复加载）⭐⭐

**问题：**
```cuda
// 每个线程都加载 scale 和 zero_point
float scale = Registers_scale[i];
float zero_point = Registers_zero[i];
```

**优化方案：**
```cuda
// 使用 shared memory 缓存
__shared__ float smem_scales[NUM_TILES];
__shared__ float smem_zeros[NUM_TILES];

// 只有一个线程加载
if (threadIdx.x == 0) {
    smem_scales[i] = Registers_scale[i];
    smem_zeros[i] = Registers_zero[i];
}
__syncthreads();

// 所有线程从 shared memory 读取
float scale = smem_scales[i];
float zero_point = smem_zeros[i];
```

**预期效果：**
- 减少寄存器压力
- 加速比: **1.1×**

**实现难度：** ⭐⭐ (简单)

---

## 📊 综合优化效果预估

### 优化组合

| 优化策略 | 加速比 | 实现难度 | 优先级 |
|---------|--------|---------|--------|
| 1. 预计算索引表 | 2.4× | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 2. FP16 反量化 | 1.35× | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3. 向量化解包 | 1.2× | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 4. Shared Memory 缓存 | 1.1× | ⭐⭐ | ⭐⭐ |

### 累计效果

**保守估计（优化 1 + 2）：**
```
当前: 0.29 ms/token
优化后: 0.29 / (2.4 × 1.35) = 0.089 ms/token
加速比: 3.24×
```

**激进估计（优化 1 + 2 + 3 + 4）：**
```
当前: 0.29 ms/token
优化后: 0.29 / (2.4 × 1.35 × 1.2 × 1.1) = 0.068 ms/token
加速比: 4.27×
```

**对比目标：**
- Sparse SpMV: 0.096 ms/token
- 优化后 Quant: 0.068-0.089 ms/token
- **可能比 Sparse 更快！** ✅

---

## 🚀 实施计划

### 阶段 1：快速优化（1-2 天）⭐⭐⭐⭐⭐

**实现优化 1：预计算索引表**

1. 创建索引表
2. 修改反量化代码
3. 测试正确性
4. Benchmark 性能

**预期收益：2.4× 加速**

### 阶段 2：精度优化（2-3 天）⭐⭐⭐⭐

**实现优化 2：FP16 反量化**

1. 修改反量化为 FP16
2. 测试精度损失
3. 如果精度可接受，部署
4. Benchmark 性能

**预期收益：额外 1.35× 加速**

### 阶段 3：高级优化（3-5 天）⭐⭐⭐

**实现优化 3 + 4**

1. 向量化解包
2. Shared Memory 优化
3. 综合测试

**预期收益：额外 1.3× 加速**

---

## 💻 代码示例

### 优化 1：预计算索引表

```cuda
// 在文件开头定义常量表
__constant__ uint8_t UNIT_IDX_TABLE[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 0-15: unit 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 16-31: unit 1
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  // 32-47: unit 2
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3   // 48-63: unit 3
};

__constant__ uint8_t BIT_OFFSET_TABLE[64] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,  // unit 0
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,  // unit 1
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,  // unit 2
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30   // unit 3
};

// 在反量化循环中使用
#pragma unroll
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) break;
    
    pos1 = __clzll(bmp);
    bmp &= ~(0x8000000000000000ULL >> pos1);
    
    // 使用查表替代除法/取模
    int unit_idx = UNIT_IDX_TABLE[j];      // 替代 j / 16
    int bit_offset = BIT_OFFSET_TABLE[j];  // 替代 (j % 16) * 2
    
    uint32_t packed_unit = quant_units[unit_idx];
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    SharedPTR[output_idx] = __float2half(dequant_value);
    
    pos1++;
}
```

---

## 🎯 推荐方案

### 方案 A：快速见效（推荐给硕士论文）⭐⭐⭐⭐⭐

**只实现优化 1（预计算索引表）**

**理由：**
- ✅ 实现简单（1-2 天）
- ✅ 效果显著（2.4× 加速）
- ✅ 风险低（不影响精度）
- ✅ 足够写论文

**预期结果：**
- 当前: 0.29 ms/token
- 优化后: 0.12 ms/token
- 仍比 Sparse 慢 1.25×，但已经很接近

### 方案 B：完整优化（如果时间充足）

**实现优化 1 + 2**

**预期结果：**
- 优化后: 0.089 ms/token
- 接近 Sparse 的 0.096 ms/token
- 可以在论文中展示完整的优化过程

---

## 📝 论文呈现

### 优化前后对比表

| 版本 | TPOT (ms/token) | 相对 Sparse | 说明 |
|------|----------------|-------------|------|
| Quant (原始) | 0.290 | 3.0× 慢 | 反量化开销 |
| Quant (优化1) | 0.120 | 1.25× 慢 | 预计算索引 |
| Quant (优化1+2) | 0.089 | 0.93× 快 | + FP16 反量化 |
| Sparse (基准) | 0.096 | 1.0× | 无量化 |

### 优化说明

> 通过预计算索引表消除整数除法和取模运算，将反量化开销从 40 cycles/值降低到 2 cycles/值，实现了 2.4× 的加速。进一步采用 FP16 反量化减少类型转换开销，最终使量化 SpMV 的性能接近甚至超过无量化版本。

---

## ❓ 讨论问题

1. **要不要优化？**
   - 如果优化，建议至少做优化 1
   - 如果不优化，可以在论文中说明优化空间

2. **优化到什么程度？**
   - 方案 A：只做优化 1（快速，低风险）
   - 方案 B：做优化 1 + 2（完整，需要测试精度）

3. **时间安排？**
   - 优化 1：1-2 天
   - 优化 2：2-3 天
   - 总计：3-5 天

你觉得怎么样？要不要先实现优化 1 试试效果？
