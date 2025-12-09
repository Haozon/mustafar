# RoPE + Hadamard 集成最终方案

## 核心结论

基于 tile-based kernel 的架构分析：

✅ **Hadamard 变换：必须在压缩前（Python 层）完成**
✅ **RoPE：可以在解压时（CUDA Kernel）在线计算**

## 为什么 Hadamard 无法融合到 Kernel？

### Tile 布局分析

```
Key Cache: [B, M, N]  例如 [8, 256, 128]
转置后:    [B, N, M]  即   [8, 128, 256]

Tile 划分（每个 tile 64 元素）：
┌─────────────────────────────────┐
│  N=128 (head_dim)               │
│  ┌───┬───┬───┬───┐              │
│  │ 0 │ 1 │ 2 │ 3 │ ← tile 0-3   │ M=256
│  ├───┼───┼───┼───┤              │ (seq_len)
│  │ 4 │ 5 │ 6 │ 7 │ ← tile 4-7   │
│  ├───┼───┼───┼───┤              │
│  │...│...│...│...│              │
│  └───┴───┴───┴───┘              │
│  每个 tile: 64 个元素            │
│  (在 M/seq_len 维度上连续)      │
└─────────────────────────────────┘
```

### Hadamard 的维度需求

```python
# 你的算法：
key_states: [B, H, S, D] → [B, S, H//G, D*G]
# 例如：[8, 32, 256, 128] → [8, 256, 16, 256]
# Hadamard 在最后一维 (D*G=256) 上操作

# 对于 Token 0，需要访问：
Token_0_data = [dim_0, dim_1, ..., dim_255]  # 256 个维度

# 但在 tile 布局中：
Token_0_dim_0_to_63:    在 tile 0
Token_0_dim_64_to_127:  在 tile 1  
Token_0_dim_128_to_191: 在 tile 2
Token_0_dim_192_to_255: 在 tile 3

# Kernel 每次只处理 2 个 tile，无法访问所有需要的维度！
```

### 根本冲突

| 维度 | Hadamard 需求 | Tile 提供 | 是否匹配 |
|------|--------------|----------|---------|
| 数据范围 | 同一 token 的所有维度 (D*G) | 同一维度的多个 token (64) | ❌ |
| 访问模式 | 跨 tile 访问 | 单 tile 内访问 | ❌ |
| 依赖关系 | 全局依赖 (所有维度) | 局部独立 (tile 内) | ❌ |

**结论**：Hadamard 的全局依赖性与 tile-based 的局部性根本冲突。

## 为什么 RoPE 可以融合到 Kernel？

### RoPE 的特性

```cuda
// RoPE 公式（成对旋转）
k_rot[2i]   = k[2i] * cos(θ) - k[2i+1] * sin(θ)
k_rot[2i+1] = k[2i] * sin(θ) + k[2i+1] * cos(θ)

// 关键：只需要相邻的两个维度 (2i, 2i+1)
```

### 在 Tile 中的可行性

```
假设 head_dim = 128，一个 tile 包含 64 个 token 的 dim_0：

Tile 0: [Token_0_dim_0, Token_1_dim_0, ..., Token_63_dim_0]

对于 Token_0_dim_0 应用 RoPE：
- 需要配对维度：Token_0_dim_1
- Token_0_dim_1 在哪里？在同一个 tile 的下一个位置？

❌ 不对！Tile 是在 seq_len 维度连续的，不是在 head_dim 维度！
```

### RoPE 的挑战

**问题**：配对维度 (2i, 2i+1) 不在同一个 tile 中！

**解决方案**：
1. **方案 A**：在压缩时保证配对维度的完整性
   - 如果 dim_2i 非零，强制保留 dim_2i+1（即使为零）
   - 代价：降低 5-10% 压缩比
   
2. **方案 B**：在 kernel 中加载配对维度
   - 从全局内存额外加载配对维度
   - 代价：增加内存访问
   
3. **方案 C**：近似处理
   - 假设缺失的配对维度为 0
   - 代价：精度损失

## 最终实现方案

### 阶段 1：存储时（Python 层）

```python
# compression_quant.py

from fast_hadamard_transform import hadamard_transform
import math

def convert_key_batched_quant_with_hadamard(
    inputs: torch.Tensor,      # [B, H, S, D]
    head_group_num: int,       # G
    apply_hadamard: bool = True
):
    """
    对 Key Cache 应用 Hadamard 变换后进行稀疏量化压缩
    
    Args:
        inputs: [B, H, S, D] Key states
        head_group_num: 分组数量 G
        apply_hadamard: 是否应用 Hadamard 变换
    
    Returns:
        bitmaps, tile_offsets, packed_quant, scales, zeros
    """
    B, H, S, D = inputs.shape
    dtype = inputs.dtype
    
    if apply_hadamard:
        # 1. Reshape for grouped heads
        # [B, H, S, D] → [B, S, H//G, D*G]
        inputs = inputs.transpose(1, 2).reshape(
            B, S, H // head_group_num, D * head_group_num
        )
        
        # 2. Apply Hadamard transform on last dimension
        inputs = hadamard_transform(
            inputs.float(), 
            scale=1.0 / math.sqrt(D * head_group_num)
        ).to(dtype)
        
        # 3. Reshape back to [B, H, S, D]
        inputs = inputs.reshape(B, S, H, D).transpose(1, 2)
    
    # 4. 稀疏化 + 量化（使用现有函数）
    bitmaps, tile_offsets, packed_quant, scales, zeros = \
        convert_key_batched_quant(inputs)
    
    return bitmaps, tile_offsets, packed_quant, scales, zeros
```

### 阶段 2：推理时（CUDA Kernel）

#### 2.1 修改 Python 接口

```python
# 在调用 kernel 时传入 RoPE 参数
output = mustafar_package_quant.mustafar_key_formulation_quant(
    bitmaps,
    packed_quant,
    tile_offsets,
    scales,
    zeros,
    query,
    cos_table,      # 新增：[max_seq_len, head_dim/2]
    sin_table,      # 新增：[max_seq_len, head_dim/2]
    position_ids,   # 新增：[batch_size, seq_len]
    M_Global=256,
    K_Global=128,
    Batch_Size=8,
    num_key_value_groups=1,
    apply_rope=True,  # 新增：是否应用 RoPE
    bit=2,
    capacity=16
)
```

#### 2.2 修改 CUDA Kernel

**关键挑战**：RoPE 需要配对维度，但它们不在同一个 tile 中。

**解决方案**：采用**方案 A - 保证配对完整性**

在压缩时修改稀疏化策略：

```python
# 在 calculate_bitmap_and_scale_key_batched 中
# 如果维度 2i 非零，强制保留 2i+1

@triton.jit
def calculate_bitmap_with_rope_pairing(
    input_ptr,
    bitmaps_ptr,
    counts_ptr,
    scales_ptr,
    zeros_ptr,
    head_dim: tl.constexpr,
    # ... 其他参数
):
    # ... 现有逻辑 ...
    
    # 修改 bitmap 计算，保证 RoPE 配对
    bit_mask = tl.where(vals != 0.0, 1, 0)
    
    # 对于每个偶数维度，如果非零，强制其奇数配对也标记为非零
    for i in range(0, 64, 2):
        if bit_mask[i] == 1:
            bit_mask[i+1] = 1  # 强制保留配对维度
    
    # ... 继续现有逻辑 ...
```

**注意**：这个修改需要在 tile 内部完成，但由于 tile 是在 seq_len 维度连续的，实际上需要更复杂的处理。

### 更实际的方案：RoPE 也在存储前应用

考虑到实现复杂度，**最简单的方案**是：

```python
def convert_key_batched_quant_with_hadamard_rope(
    inputs: torch.Tensor,      # [B, H, S, D]
    head_group_num: int,
    cos: torch.Tensor,         # RoPE cos table
    sin: torch.Tensor,         # RoPE sin table
    position_ids: torch.Tensor
):
    """
    应用 Hadamard + RoPE 后再压缩
    """
    # 1. Hadamard 变换
    inputs = apply_grouped_hadamard(inputs, head_group_num)
    
    # 2. RoPE
    inputs = apply_rotary_pos_emb_k(inputs, cos, sin, position_ids)
    
    # 3. 稀疏化 + 量化
    return convert_key_batched_quant(inputs)
```

**优点**：
- 实现简单，无需修改 CUDA kernel
- 无精度损失
- 无额外计算开销

**缺点**：
- RoPE 的 position_ids 必须在压缩时确定
- 如果 position_ids 动态变化（如 KV Cache 复用），需要重新压缩

## 方案对比

| 方案 | Hadamard | RoPE | 实现难度 | 灵活性 | 性能 |
|------|----------|------|---------|--------|------|
| **A. 全部离线** | 存储前 | 存储前 | ⭐ 简单 | ❌ position_ids 固定 | ⭐⭐⭐ 最快 |
| **B. RoPE 在线** | 存储前 | 解压时 | ⭐⭐⭐ 复杂 | ✅ 动态 position_ids | ⭐⭐ 较快 |
| **C. 混合方案** | 存储前 | 部分在线 | ⭐⭐ 中等 | ⚠️ 部分灵活 | ⭐⭐ 较快 |

## 推荐方案

### 如果 position_ids 固定（推荐）

使用**方案 A**：Hadamard + RoPE 都在存储前完成

```python
# 在模型的 forward 中
if use_cache:
    # 应用 Hadamard + RoPE
    key_states = apply_grouped_hadamard(key_states, head_group_num)
    key_states = apply_rotary_pos_emb_k(key_states, cos, sin, position_ids)
    
    # 压缩存储
    compressed_key = convert_key_batched_quant(key_states)
    
    # 推理时直接使用
    attn_output = mustafar_attention(query, compressed_key, ...)
```

### 如果需要动态 position_ids

使用**方案 B**：需要修改 CUDA kernel 支持在线 RoPE

这需要：
1. 修改压缩策略，保证 RoPE 配对维度的完整性
2. 修改解压 kernel，在解压时应用 RoPE
3. 传入 cos/sin 表和 position_ids

## 下一步实现

1. ✅ 实现 Hadamard 变换的 Python 接口（使用 fast_hadamard_transform）
2. ✅ 修改 compression_quant.py，添加 Hadamard 支持
3. ⚠️ 决定 RoPE 的处理方式（离线 vs 在线）
4. 如果选择在线 RoPE：
   - 修改 CUDA kernel
   - 处理配对维度问题
   - 性能优化
5. 编写测试用例
6. 性能基准测试

## 测试计划

```python
# test_hadamard_rope_integration.py

def test_hadamard_transform():
    """测试 Hadamard 变换的正确性"""
    pass

def test_compression_with_hadamard():
    """测试带 Hadamard 的压缩"""
    pass

def test_rope_application():
    """测试 RoPE 应用"""
    pass

def test_end_to_end_attention():
    """端到端 attention 测试"""
    pass
```
