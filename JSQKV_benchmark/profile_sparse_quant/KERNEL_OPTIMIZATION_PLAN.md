# Kernel 优化方案

## 🔍 当前实现分析

### 反量化代码（第 136-174 行）

```cuda
for (int i = 0; i < 2; i++) {  // 处理 2 个 tile
    float scale = Registers_scale[i];
    float zero_point = Registers_zero[i];
    uint32_t* quant_units = Registers_quant + i * 32;
    
    for (int j = 0; j < 64; j++) {  // 最多 64 个非零元素
        // 1. 找到下一个非零位置（bitmap 操作）
        pos1 = __clzll(bmp);
        bmp &= ~(0x8000000000000000ULL >> pos1);
        
        // 2. 从 uint32 中提取 2-bit 值
        int unit_idx = j / 16;              // 第几个 uint32
        int bit_offset = (j % 16) * 2;      // 位偏移
        uint32_t packed_unit = quant_units[unit_idx];
        uint32_t q_value = (packed_unit >> bit_offset) & 0x3;
        
        // 3. 反量化
        float dequant_value = (q_value - zero_point) * scale;
        
        // 4. 转换并写入 shared memory
        SharedPTR[output_idx] = __float2half(dequant_value);
    }
}
```

## ⚠️ 性能问题

### 问题1：逐个元素处理（串行）⭐⭐⭐⭐⭐

**当前**：
```cuda
for (int j = 0; j < 64; j++) {
    // 每次处理 1 个元素
    q_value = (packed >> offset) & 0x3;
    dequant = (q_value - zero) * scale;
}
```

**问题**：
- 没有利用 GPU 的并行性
- 每个元素都要：移位 → 掩码 → 减法 → 乘法 → 类型转换
- 64 次循环，每次都是串行操作

**影响**：这是最大的性能瓶颈！

### 问题2：重复加载 packed_unit ⭐⭐⭐⭐

**当前**：
```cuda
for (int j = 0; j < 64; j++) {
    int unit_idx = j / 16;
    uint32_t packed_unit = quant_units[unit_idx];  // 每次都加载
    ...
}
```

**问题**：
- 同一个 uint32 被加载 16 次（因为包含 16 个值）
- 寄存器压力大
- 编译器可能无法优化

**影响**：额外的内存访问开销

### 问题3：分支预测失败 ⭐⭐⭐

**当前**：
```cuda
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) break;  // 动态退出
    ...
}
```

**问题**：
- 每个 tile 的非零元素数量不同
- 导致 warp 内线程分歧
- 部分线程空闲

**影响**：GPU 利用率降低

### 问题4：类型转换开销 ⭐⭐

**当前**：
```cuda
float dequant_value = (q_value - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);
```

**问题**：
- uint32 → float → half，两次转换
- 每个元素都要转换

**影响**：额外的计算开销

## 🚀 优化方案

### 优化1：向量化处理（最重要）⭐⭐⭐⭐⭐

#### 方案A：一次处理 4 个元素

```cuda
// 当前：逐个处理
for (int j = 0; j < 64; j++) {
    uint32_t q = (packed >> (j%16)*2) & 0x3;
    float dq = (q - zero) * scale;
}

// 优化：一次处理 4 个
for (int j = 0; j < 64; j += 4) {
    uint32_t packed = quant_units[j / 16];
    
    // 一次提取 4 个 2-bit 值（8 bits）
    uint32_t q0 = (packed >> ((j%16)*2 + 0)) & 0x3;
    uint32_t q1 = (packed >> ((j%16)*2 + 2)) & 0x3;
    uint32_t q2 = (packed >> ((j%16)*2 + 4)) & 0x3;
    uint32_t q3 = (packed >> ((j%16)*2 + 6)) & 0x3;
    
    // 向量化反量化
    float4 q_vec = make_float4(q0, q1, q2, q3);
    float4 dq_vec = (q_vec - zero_point) * scale;
    
    // 向量化写入
    half4 result = make_half4(dq_vec.x, dq_vec.y, dq_vec.z, dq_vec.w);
    *(half4*)(&SharedPTR[output_idx]) = result;
}
```

**预期提升**：30-40%

#### 方案B：使用 SIMD 指令

```cuda
// 使用 PTX 内联汇编进行向量化
asm volatile (
    "prmt.b32 %0, %1, %2, %3;" 
    : "=r"(result) 
    : "r"(packed), "r"(mask), "r"(selector)
);
```

**预期提升**：40-50%

### 优化2：预加载 packed_unit ⭐⭐⭐⭐

```cuda
// 当前：每次都加载
for (int j = 0; j < 64; j++) {
    uint32_t packed = quant_units[j / 16];
    ...
}

// 优化：预加载到局部变量
uint32_t packed0 = quant_units[0];
uint32_t packed1 = quant_units[1];
uint32_t packed2 = quant_units[2];
uint32_t packed3 = quant_units[3];

for (int j = 0; j < 64; j++) {
    uint32_t packed = (j < 16) ? packed0 :
                      (j < 32) ? packed1 :
                      (j < 48) ? packed2 : packed3;
    ...
}
```

**预期提升**：10-15%

### 优化3：使用 Shared Memory 缓存 scale/zero ⭐⭐⭐⭐

```cuda
// 当前：从寄存器读取（已经很快）
float scale = Registers_scale[i];
float zero_point = Registers_zero[i];

// 优化：如果多个线程需要相同的 scale/zero
__shared__ float shared_scales[NUM_TILES];
__shared__ float shared_zeros[NUM_TILES];

// 协作加载
if (threadIdx.x < NUM_TILES) {
    shared_scales[threadIdx.x] = GlobalPTR_scale[...];
    shared_zeros[threadIdx.x] = GlobalPTR_zero[...];
}
__syncthreads();

// 从 shared memory 读取
float scale = shared_scales[tile_idx];
float zero_point = shared_zeros[tile_idx];
```

**预期提升**：5-10%（如果多个线程共享）

### 优化4：减少类型转换 ⭐⭐⭐

```cuda
// 当前：uint32 → float → half
float dequant_value = (q_value - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);

// 优化：使用 half 精度计算
half scale_h = __float2half(scale);
half zero_h = __float2half(zero_point);
half q_h = __uint2half_rn(q_value);
half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
SharedPTR[output_idx] = dequant_h;
```

**预期提升**：5-10%

### 优化5：循环展开 ⭐⭐⭐

```cuda
// 当前：动态循环
#pragma unroll
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) break;  // 动态退出
    ...
}

// 优化：完全展开（如果 nnz_tile 已知）
#pragma unroll 64
for (int j = 0; j < 64; j++) {
    ...
}
```

**预期提升**：10-15%

## 📊 综合优化方案

### 方案A：快速优化（1-2天）

**实现**：
1. 向量化处理（一次 4 个元素）
2. 预加载 packed_unit
3. 循环展开

**预期提升**：40-50%
**难度**：中等
**风险**：低

### 方案B：深度优化（3-5天）

**实现**：
1. 方案A 的所有优化
2. 使用 SIMD 指令
3. Shared Memory 优化
4. Half 精度计算

**预期提升**：60-70%
**难度**：高
**风险**：中等

### 方案C：终极优化（1-2周）

**实现**：
1. 完全重写 kernel
2. Fused dequant + matmul
3. 使用 Tensor Core
4. 优化内存访问模式

**预期提升**：80-100%
**难度**：很高
**风险**：高

## 🎯 推荐执行顺序

### Phase 1：快速验证（今天）
```bash
# 1. 修改 kernel，添加向量化处理
# 2. 重新编译
cd kernel_quant/kernel_wrapper
python setup.py install

# 3. 测试性能
cd ../../JSQKV_benchmark/profile_sparse_quant
python test_quant_bits.py
```

### Phase 2：迭代优化（1-2天）
1. 实现向量化（预期 +30%）
2. 测试验证
3. 实现预加载（预期 +10%）
4. 测试验证
5. 实现循环展开（预期 +10%）
6. 最终测试

### Phase 3：性能调优（2-3天）
1. Profile 新版本
2. 找出剩余瓶颈
3. 针对性优化

## 📝 代码模板

### 优化后的反量化循环

```cuda
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegToSharedMem_Quant_Optimized(
    half* __restrict__ SharedPTR,
    uint32_t Registers_quant[64],
    uint64_t* Registers_bmp,
    float* Registers_scale,
    float* Registers_zero,
    uint32_t* nnz_tile0, 
    uint32_t* nnz_tile1,
    int TB_ROW, 
    int TB_COL,
    int bit,
    int capacity
)
{
    int tile_element_start = TB_ROW * 64 * 64 + TB_COL * 2;
    
#pragma unroll
    for (int i = 0; i < 2; i++) {
        uint64_t bmp = Registers_bmp[i];
        float scale = Registers_scale[i];
        float zero_point = Registers_zero[i];
        uint32_t nnz_tile = i ? *nnz_tile1 : *nnz_tile0;
        
        // 预加载所有 packed units
        uint32_t packed0 = Registers_quant[i * 32 + 0];
        uint32_t packed1 = Registers_quant[i * 32 + 1];
        uint32_t packed2 = Registers_quant[i * 32 + 2];
        uint32_t packed3 = Registers_quant[i * 32 + 3];
        
        int pos1 = 0;
        int fuk = tile_element_start + i;
        
        // 向量化处理：一次 4 个元素
#pragma unroll
        for (int j = 0; j < 64; j += 4) {
            if (j >= nnz_tile) break;
            
            // 选择对应的 packed unit
            int unit_group = j / 16;
            uint32_t packed = (unit_group == 0) ? packed0 :
                             (unit_group == 1) ? packed1 :
                             (unit_group == 2) ? packed2 : packed3;
            
            int base_offset = (j % 16) * 2;
            
            // 一次提取 4 个 2-bit 值
            uint32_t q0 = (packed >> (base_offset + 0)) & 0x3;
            uint32_t q1 = (packed >> (base_offset + 2)) & 0x3;
            uint32_t q2 = (packed >> (base_offset + 4)) & 0x3;
            uint32_t q3 = (packed >> (base_offset + 6)) & 0x3;
            
            // 向量化反量化
            float dq0 = (static_cast<float>(q0) - zero_point) * scale;
            float dq1 = (static_cast<float>(q1) - zero_point) * scale;
            float dq2 = (static_cast<float>(q2) - zero_point) * scale;
            float dq3 = (static_cast<float>(q3) - zero_point) * scale;
            
            // 找到 4 个非零位置并写入
            for (int k = 0; k < 4 && (j + k) < nnz_tile; k++) {
                pos1 = __clzll(bmp);
                bmp &= ~(0x8000000000000000ULL >> pos1);
                
                int output_idx = fuk + (pos1 << 6);
                float dq = (k == 0) ? dq0 : (k == 1) ? dq1 : (k == 2) ? dq2 : dq3;
                SharedPTR[output_idx] = __float2half(dq);
                
                pos1++;
            }
        }
    }
}
```

## 🎯 预期最终效果

| 阶段 | 量化 Kernel 时间 | 总时间 | 吞吐量 | 提升 |
|------|-----------------|--------|--------|------|
| **当前** | 12.95s (49.5%) | 26.17s | 35.68 tok/s | - |
| **Phase 1** | 7.77s (35%) | 19.99s | 46.7 tok/s | +31% |
| **Phase 2** | 5.18s (25%) | 17.40s | 53.7 tok/s | +50% |
| **Phase 3** | 3.89s (20%) | 16.11s | 58.0 tok/s | +63% |

**目标**：达到或超过 Sparse-50% FP16 的性能（55.25 tok/s）
