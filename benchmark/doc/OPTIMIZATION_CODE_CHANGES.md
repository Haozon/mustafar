# 优化代码修改指南

## 📝 文件位置

```
/home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh
```

---

## 🔧 优化 1：预计算索引（最重要）

### 修改位置

在文件开头（`#include` 之后，函数定义之前）添加：

```cuda
/***************************************************************************
 * 优化：预计算的索引数组，避免运行时除法和取模运算
 * 
 * 对于 2-bit 量化，capacity = 16（每个 uint32 存储 16 个 2-bit 值）
 * - unit_indices[j]: 第 j 个值在第几个 uint32 中
 * - bit_offsets[j]: 第 j 个值在 uint32 中的位偏移
 ***************************************************************************/

// 每个 uint32 存储 16 个 2-bit 值，所以 64 个值需要 4 个 uint32
__device__ __constant__ int unit_indices_2bit[64] = {
    // Unit 0: values 0-15
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Unit 1: values 16-31
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Unit 2: values 32-47
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // Unit 3: values 48-63
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

__device__ __constant__ int bit_offsets_2bit[64] = {
    // Unit 0: bit offsets 0, 2, 4, ..., 30
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    // Unit 1: bit offsets 0, 2, 4, ..., 30
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    // Unit 2: bit offsets 0, 2, 4, ..., 30
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    // Unit 3: bit offsets 0, 2, 4, ..., 30
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
};
```

### 修改函数

找到 `SpMM_DecompressFromRegisterToShared_Quant` 函数，修改解包循环：

**原始代码（约在第 150-180 行）：**
```cuda
#pragma unroll
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) {
        break;
    }
    
    // 找到下一个非零位置
    pos1 = __clzll(bmp);
    bmp &= ~(0x8000000000000000ULL >> pos1);
    
    // 从打包的 uint32 中提取量化值
    int unit_idx = j / capacity;           // ⚠️ 整数除法（慢）
    int bit_offset = (j % capacity) * bit; // ⚠️ 整数取模（慢）
    uint32_t packed_unit = quant_units[unit_idx];
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    
    // 反量化：dequant_value = (q - zero_point) * scale
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    
    // 转换为 half 并写入共享内存
    int output_idx = fuk + (pos1 << 6);
    SharedPTR[output_idx] = __float2half(dequant_value);
    
    pos1++;
}
```

**优化后的代码：**
```cuda
#pragma unroll
for (int j = 0; j < 64; j++) {
    if (j == nnz_tile) {
        break;
    }
    
    // 找到下一个非零位置
    pos1 = __clzll(bmp);
    bmp &= ~(0x8000000000000000ULL >> pos1);
    
    // ✅ 使用预计算的索引（快！）
    int unit_idx = unit_indices_2bit[j];
    int bit_offset = bit_offsets_2bit[j];
    uint32_t packed_unit = quant_units[unit_idx];
    uint32_t q_value = (packed_unit >> bit_offset) & mask;
    
    // 反量化：dequant_value = (q - zero_point) * scale
    float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
    
    // 转换为 half 并写入共享内存
    int output_idx = fuk + (pos1 << 6);
    SharedPTR[output_idx] = __float2half(dequant_value);
    
    pos1++;
}
```

**预期效果：TPOT 从 82.78ms 降低到 55-64ms（快 1.3-1.5x）**

---

## 🔧 优化 2：Half 精度计算（简单）

### 修改函数

继续修改 `SpMM_DecompressFromRegisterToShared_Quant` 函数：

**原始代码：**
```cuda
#pragma unroll
for (int i = 0; i < 2; i++) {
    // 获取当前 tile 的 bitmap, scale, zero_point
    uint64_t bmp = Registers_bmp[i];
    float scale = Registers_scale[i];
    float zero_point = Registers_zero[i];
    uint32_t nnz_tile = i ? *nnz_tile1 : *nnz_tile0;
    
    // ... 解包循环 ...
    
    for (int j = 0; j < 64; j++) {
        // ...
        
        // ⚠️ FP32 计算 + 类型转换（慢）
        float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
        SharedPTR[output_idx] = __float2half(dequant_value);
    }
}
```

**优化后的代码：**
```cuda
#pragma unroll
for (int i = 0; i < 2; i++) {
    // 获取当前 tile 的 bitmap, scale, zero_point
    uint64_t bmp = Registers_bmp[i];
    float scale = Registers_scale[i];
    float zero_point = Registers_zero[i];
    uint32_t nnz_tile = i ? *nnz_tile1 : *nnz_tile0;
    
    // ✅ 转换为 half 精度（只需一次）
    half scale_h = __float2half(scale);
    half zero_h = __float2half(zero_point);
    
    // ... 解包循环 ...
    
    for (int j = 0; j < 64; j++) {
        // ...
        
        // ✅ 直接在 half 精度下计算（快！）
        half q_h = __int2half_rn(q_value);
        half dequant_h = __hfma(scale_h, __hsub(q_h, zero_h), __float2half(0.0f));
        // 或者更简单：
        // half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
        SharedPTR[output_idx] = dequant_h;
    }
}
```

**注意：** 使用 `__hfma` (fused multiply-add) 可能更快：
```cuda
// dequant = (q - zero) * scale
// 等价于：dequant = scale * (q - zero) + 0
half dequant_h = __hfma(scale_h, __hsub(q_h, zero_h), __float2half(0.0f));
```

**预期效果：TPOT 从 55-64ms 降低到 46-59ms（快 1.1-1.2x）**

---

## 🔧 优化 3：向量化解包（可选，中等难度）

### 修改函数

如果前两个优化还不够，可以尝试向量化解包：

**优化后的代码：**
```cuda
#pragma unroll
for (int i = 0; i < 2; i++) {
    uint64_t bmp = Registers_bmp[i];
    float scale = Registers_scale[i];
    float zero_point = Registers_zero[i];
    uint32_t nnz_tile = i ? *nnz_tile1 : *nnz_tile0;
    
    half scale_h = __float2half(scale);
    half zero_h = __float2half(zero_point);
    
    uint32_t* quant_units = Registers_quant + i * 32;
    
    int pos1 = 0;
    int fuk = tile_element_start + i;
    uint32_t mask = 0x3;  // 2-bit mask
    
    // ✅ 向量化解包：一次处理 16 个值（1 个 uint32）
    int j = 0;
    while (j < nnz_tile) {
        // 计算当前 uint32 索引
        int unit_idx = j / 16;
        uint32_t packed = quant_units[unit_idx];
        
        // 一次性解包 16 个值（或剩余的值）
        int values_in_unit = min(16, nnz_tile - j);
        
        #pragma unroll
        for (int k = 0; k < 16 && j < nnz_tile; k++, j++) {
            // 找到下一个非零位置
            pos1 = __clzll(bmp);
            bmp &= ~(0x8000000000000000ULL >> pos1);
            
            // 从已加载的 packed 中提取值
            uint32_t q_value = (packed >> (k * 2)) & mask;
            
            // 反量化
            half q_h = __int2half_rn(q_value);
            half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
            
            // 写入共享内存
            int output_idx = fuk + (pos1 << 6);
            SharedPTR[output_idx] = dequant_h;
            
            pos1++;
        }
    }
}
```

**预期效果：TPOT 从 46-59ms 降低到 36-49ms（快 1.2-1.3x）**

---

## 📋 完整的修改步骤

### Step 1: 备份原文件

```bash
cd /home/zh/mustafar/kernel_quant/csrc
cp SpMM_Kernel_Quant.cuh SpMM_Kernel_Quant.cuh.backup
```

### Step 2: 编辑文件

```bash
vim SpMM_Kernel_Quant.cuh
```

或者使用你喜欢的编辑器。

### Step 3: 应用优化 1

1. 在文件开头（第 10 行左右）添加预计算索引数组
2. 找到 `SpMM_DecompressFromRegisterToShared_Quant` 函数（约第 120 行）
3. 修改解包循环，使用 `unit_indices_2bit[j]` 和 `bit_offsets_2bit[j]`

### Step 4: 应用优化 2

1. 在同一个函数中
2. 在外层循环开始处添加 half 精度转换
3. 修改反量化计算，使用 half 精度

### Step 5: 编译测试

```bash
cd /home/zh/mustafar/kernel_quant
python setup.py install

# 测试
cd /home/zh/mustafar/benchmark
python mem_spd_test_quant.py
```

### Step 6: 验证结果

查看输出中的 TPOT：

```
Performance Metrics:
TTFT: 3681.42 ms
TPOT: XX.XX ms  ← 应该降低到 46-59 ms
```

---

## 🔍 调试技巧

### 如果编译失败

```bash
# 查看编译错误
cd /home/zh/mustafar/kernel_quant
python setup.py install 2>&1 | tee compile.log

# 常见错误：
# 1. 语法错误：检查括号、分号
# 2. 类型不匹配：检查 half/float 转换
# 3. 未定义的函数：检查 CUDA intrinsics
```

### 如果性能没有提升

```bash
# 1. 确认代码真的被修改了
grep "unit_indices_2bit" /home/zh/mustafar/kernel_quant/csrc/SpMM_Kernel_Quant.cuh

# 2. 确认重新编译了
ls -lt /home/zh/mustafar/kernel_quant/build/lib*/

# 3. 清理并重新编译
cd /home/zh/mustafar/kernel_quant
rm -rf build/ dist/ *.egg-info
python setup.py clean --all
python setup.py install

# 4. 重新测试
cd /home/zh/mustafar/benchmark
python mem_spd_test_quant.py
```

### 如果精度下降

```bash
# 如果使用 half 精度导致精度问题
# 可以只保留优化 1，移除优化 2

# 或者使用更高精度的中间计算：
float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);
```

---

## 📊 预期性能提升

| 优化阶段 | TPOT | 加速比 | 累积加速 |
|---------|------|--------|---------|
| **原始** | 82.78 ms | 1.0x | 1.0x |
| **+ 优化 1** | 55-64 ms | 1.3-1.5x | 1.3-1.5x |
| **+ 优化 2** | 46-59 ms | 1.1-1.2x | 1.4-1.8x |
| **+ 优化 3** | 36-49 ms | 1.2-1.3x | 1.7-2.3x |

**目标：TPOT < 50 ms（第一阶段）**

---

## ✅ Checklist

### 优化 1：预计算索引
- [ ] 添加 `unit_indices_2bit` 数组
- [ ] 添加 `bit_offsets_2bit` 数组
- [ ] 修改解包循环使用预计算索引
- [ ] 编译测试
- [ ] 验证 TPOT 降低到 55-64 ms

### 优化 2：Half 精度
- [ ] 添加 half 精度转换
- [ ] 修改反量化计算
- [ ] 编译测试
- [ ] 验证 TPOT 降低到 46-59 ms
- [ ] 验证精度没有明显下降

### 优化 3：向量化解包（可选）
- [ ] 重构解包逻辑
- [ ] 批量处理 uint32
- [ ] 编译测试
- [ ] 验证 TPOT 降低到 36-49 ms

---

## 🆘 需要帮助？

如果遇到问题，可以：

1. 查看编译日志：`compile.log`
2. 对比原始文件：`diff SpMM_Kernel_Quant.cuh.backup SpMM_Kernel_Quant.cuh`
3. 恢复备份：`cp SpMM_Kernel_Quant.cuh.backup SpMM_Kernel_Quant.cuh`
4. 查看相关文档：
   - `benchmark/PROFILING_AND_OPTIMIZATION_PLAN.md`
   - `benchmark/QUICK_START_PROFILING.md`

---

**准备好了吗？开始修改代码！** 🚀

```bash
cd /home/zh/mustafar/kernel_quant/csrc
cp SpMM_Kernel_Quant.cuh SpMM_Kernel_Quant.cuh.backup
vim SpMM_Kernel_Quant.cuh
```
