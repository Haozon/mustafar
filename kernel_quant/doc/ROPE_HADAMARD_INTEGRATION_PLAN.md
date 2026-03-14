# RoPE + Hadamard 集成方案

## 目标
在稀疏量化 kernel 中集成 Hadamard 变换和 RoPE，使其在解压阶段在线计算。

## 数据流设计

### 存储阶段（compression_quant.py）
```
原始 Key: [B, H, S, D] (float16)
    ↓
稀疏化: 保留非零元素
    ↓
量化: 2-bit per value
    ↓
存储: bitmaps + packed_quant + scales + zeros
```

**注意**：存储的是原始 Key，未经 Hadamard 和 RoPE 处理。

### 推理阶段（SpMM_Kernel_Quant.cuh）
```
加载量化数据
    ↓
解包 + 反量化 → float16
    ↓
应用 Hadamard 变换 (可选)
    ↓
应用 RoPE
    ↓
写入共享内存
    ↓
Tensor Core 计算 (Q @ K^T)
```

## 实现细节

### 1. Hadamard 变换

#### 分组 Head 旋转
```cuda
// 输入: key_states [B, H, S, D]
// 参数: head_group_num (G)
// 
// 步骤:
// 1. Reshape: [B, H, S, D] → [B, S, H//G, D*G]
// 2. Hadamard: 在最后一维 (D*G) 上应用
// 3. Scale: 乘以 1/sqrt(D*G)
// 4. Reshape: [B, S, H//G, D*G] → [B, H, S, D]
```

#### Hadamard 矩阵
```
H_2 = [1  1]
      [1 -1]

H_4 = [1  1  1  1]
      [1 -1  1 -1]
      [1  1 -1 -1]
      [1 -1 -1  1]

H_n = H_2 ⊗ H_{n/2}  (递归定义)
```

#### 快速算法
使用 Fast Walsh-Hadamard Transform (FWHT)：
- 时间复杂度: O(n log n)
- 空间复杂度: O(1) (in-place)
- 适合 GPU 并行

### 2. RoPE 实现

#### 标准 RoPE 公式
```cuda
// 对于维度 i (i 为偶数):
k_rot[i]   = k[i] * cos(pos * theta_i) - k[i+1] * sin(pos * theta_i)
k_rot[i+1] = k[i] * sin(pos * theta_i) + k[i+1] * cos(pos * theta_i)

// 其中:
// theta_i = 10000^(-2i/d)
// pos = position_ids[token_idx]
```

#### 预计算 cos/sin 表
```python
# 在 Python 层预计算
max_seq_len = 32768
head_dim = 128
theta = 10000.0

inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(max_seq_len).float()
freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]

cos_table = freqs.cos()  # [max_seq_len, head_dim/2]
sin_table = freqs.sin()  # [max_seq_len, head_dim/2]
```

### 3. 稀疏性处理

#### 挑战
- Hadamard 是全局变换，每个输出依赖所有输入
- 但我们只存储了稀疏的非零元素
- 需要特殊处理

#### 解决方案 A：假设零元素
```cuda
// 对于稀疏元素，假设缺失的配对维度为 0
// 这会引入近似误差，但保持稀疏性
if (is_sparse_element) {
    // 只对非零元素应用 Hadamard
    // 缺失的元素假设为 0
}
```

#### 解决方案 B：分组 Hadamard
```cuda
// 如果 head_group_num 较小（如 2, 4），可以在小范围内应用
// 例如 D*G = 128*2 = 256，在 256 维度内做 Hadamard
// 这样可以减少稀疏元素之间的依赖
```

#### 解决方案 C：存储前应用 Hadamard（唯一可行方案）
```python
# 在 compression_quant.py 中，存储前先做 Hadamard
def convert_key_batched_quant_with_hadamard(inputs, head_group_num):
    # 1. 应用 Hadamard 变换
    inputs_hadamard = apply_hadamard_transform(inputs, head_group_num)
    
    # 2. 稀疏化 + 量化
    bitmaps, tile_offsets, packed_quant, scales, zeros = \
        convert_key_batched_quant(inputs_hadamard)
    
    return bitmaps, tile_offsets, packed_quant, scales, zeros

# 推理时只需要应用 RoPE
```

**必须使用方案 C**，因为：
- ❌ Hadamard 无法在 tile-based kernel 中实现
  - Hadamard 需要跨越整个 D*G 维度（如 256 维）
  - 但每个 tile 只有 64 个元素，且在 seq_len 维度上连续
  - 同一个 token 的 256 个维度分散在多个 tile 中
  - Tile-based kernel 无法访问其他 tile 的数据
- ✅ Hadamard 是位置无关的，可以预先计算
- ✅ RoPE 是位置相关的，但可以逐维度独立计算，适合在 kernel 中融合
- ✅ 这样可以减少推理时的计算开销

## 修改清单

### Python 层修改

#### 1. compression_quant.py
```python
# 添加 Hadamard 变换函数
def hadamard_transform_batched(x, head_group_num, scale=None):
    """
    对分组 head 应用 Hadamard 变换
    x: [B, H, S, D]
    """
    pass

# 修改压缩函数
def convert_key_batched_quant_with_hadamard(inputs, head_group_num):
    """
    先应用 Hadamard，再压缩
    """
    pass
```

#### 2. 添加 RoPE 参数传递
```python
# 在调用 kernel 时传入 cos/sin 表
mustafar_package_quant.mustafar_key_formulation_quant(
    bitmaps,
    packed_quant,
    tile_offsets,
    scales,
    zeros,
    query,
    cos_table,  # 新增
    sin_table,  # 新增
    position_ids,  # 新增
    M_Global=256,
    K_Global=128,
    ...
)
```

### CUDA Kernel 修改

#### 1. SpMM_Kernel_Quant.cuh

##### 修改解压函数签名
```cuda
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared_Quant_WithRoPE(
    half* __restrict__ SharedPTR,
    uint32_t Registers_quant[64],
    uint64_t* Registers_bmp,
    float* Registers_scale,
    float* Registers_zero,
    const half* cos_table,     // [max_seq_len, head_dim/2]
    const half* sin_table,     // [max_seq_len, head_dim/2]
    const int* position_ids,   // [batch_size, seq_len]
    int current_seq_offset,    // 当前 tile 的序列偏移
    int head_dim,
    uint32_t* nnz_tile0, 
    uint32_t* nnz_tile1,
    int TB_ROW, 
    int TB_COL,
    int bit,
    int capacity
)
```

##### 在解压循环中融合 RoPE
```cuda
// 反量化
float dequant_value = (q_value - zero_point) * scale;

// 计算当前元素的位置和维度
int token_pos = current_seq_offset + (pos1 / head_dim);
int dim_idx = pos1 % head_dim;

// 应用 RoPE（成对处理）
if (dim_idx % 2 == 0 && dim_idx < head_dim) {
    int freq_idx = dim_idx / 2;
    
    // 加载 cos/sin
    half cos_val = cos_table[token_pos * (head_dim/2) + freq_idx];
    half sin_val = sin_table[token_pos * (head_dim/2) + freq_idx];
    
    // 需要配对的下一个维度
    // 这里需要特殊处理稀疏性...
}
```

#### 2. SpMM_API_Quant.cu
```cuda
// 添加 RoPE 参数
void mustafar_key_formulation_quant(
    // ... 现有参数 ...
    const half* cos_table,
    const half* sin_table,
    const int* position_ids,
    // ...
)
```

## 性能估算

### 计算开销
- **Hadamard (离线)**: 0 ms (预计算)
- **RoPE (在线)**: ~0.1-0.2 ms per layer
  - 每个元素 2 次乘法 + 2 次加法
  - 访问 cos/sin 表（缓存友好）

### 内存开销
- **cos/sin 表**: 2 * max_seq_len * (head_dim/2) * 2 bytes
  - 例如: 2 * 32768 * 64 * 2 = 8 MB (可接受)

### 总体影响
- 预计增加 5-10% 的推理延迟
- 内存增加 < 10 MB
- 精度损失：Hadamard 无损，RoPE 无损

## 测试计划

1. **单元测试**
   - Hadamard 变换正确性
   - RoPE 应用正确性
   - 与 PyTorch 参考实现对比

2. **集成测试**
   - 端到端 attention 计算
   - 不同序列长度
   - 不同 head_group_num

3. **性能测试**
   - 延迟对比（有/无 RoPE）
   - 吞吐量测试
   - 内存占用

## 风险和挑战

1. **稀疏性与 RoPE 的配对问题**
   - RoPE 需要成对的维度 (2i, 2i+1)
   - 如果只有一个维度非零，需要特殊处理
   - 可能需要存储额外信息或做近似

2. **Hadamard 与稀疏性的冲突**
   - Hadamard 是全局变换，破坏稀疏性
   - 必须在稀疏化之前应用
   - 或者使用近似方法

3. **性能开销**
   - RoPE 增加计算量
   - cos/sin 表访问可能成为瓶颈
   - 需要优化内存访问模式

## 下一步

1. 实现 Hadamard 变换（Python + Triton）
2. 修改 CUDA kernel 添加 RoPE 支持
3. 编写测试用例验证正确性
4. 性能优化和调优
