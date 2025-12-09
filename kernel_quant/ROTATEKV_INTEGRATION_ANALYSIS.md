# RotateKV 算法与稀疏量化算子的集成分析

## RotateKV 算法理解

基于你提供的代码片段：

```python
if args.Grouped_Head_Key_Rotation:
    # 只对 Key 做 Hadamard 变换
    key_states = key_states.transpose(1, 2).reshape(
        bsz, seq, head_num // head_group_num, head_dim * head_group_num
    )
    key_states = hadamard_transform(
        key_states.float(), 
        scale=1/math.sqrt(key_states.shape[-1])
    ).to(dtype)
    key_states = key_states.reshape(bsz, seq, head_num, head_dim).transpose(1, 2)
    
    # RoPE（Query 和 Key 都做）
    query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)
    key_states = apply_rotary_pos_emb_k(key_states, cos, sin, position_ids)
    
    # 计算 Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    attn_scores = query_states @ key_states.transpose(-2, -1)
```

## 关键发现

**RotateKV 的设计**：
1. ✅ **只对 Key 做 Hadamard 变换**（Query 不做）
2. ✅ **Hadamard 用于改变 Key 的表示，增加稀疏性**
3. ❌ **在计算 attention 前，Key 必须恢复到原始空间**

**这意味着**：
```
存储: Key → Hadamard → 稀疏化 + 量化 → 存储
推理: 加载 → 解压 → Hadamard^(-1) → RoPE → 与 Query 计算
```

**必须做 Hadamard 反变换！**

## 问题重述

你的稀疏量化算子流程：
```
加载量化数据 → 解压 + 反量化 → 写入共享内存 → Tensor Core 计算
```

**需要插入 Hadamard 反变换的位置**：
```
加载量化数据 → 解压 + 反量化 → [Hadamard^(-1)] → 写入共享内存 → Tensor Core 计算
```

## 为什么之前的方案不可行

**方案 1（Query 也做 Hadamard）**：
- ❌ 违背了 RotateKV 的设计
- ❌ RotateKV 只对 Key 做 Hadamard，Query 保持原样
- ❌ 改变算法语义

**方案 2（不做反变换）**：
- ❌ Query 在原始空间，Key 在 Hadamard 空间
- ❌ `Q @ K^T` 的结果没有意义
- ❌ Attention 计算错误

## 核心挑战

### 挑战 1：Hadamard 反变换需要完整的 D*G 维度

```
Hadamard 变换维度：D*G = 128 * 2 = 256
Tile 提供的维度：TILE_K = 64

问题：需要 256 维，但 tile 只有 64 维
```

### 挑战 2：跨 Tile 的数据依赖

```
Token 0 的 256 维数据分布：
- 维度 0-63:   Tile 0
- 维度 64-127: Tile 1
- 维度 128-191: Tile 2
- 维度 192-255: Tile 3

Hadamard 反变换需要所有 256 维 → 需要 4 个 tile 的数据
```

## 可能的解决方案

### 方案 A：在全局内存中做 Hadamard 反变换（推荐）

**流程**：
```
1. 解压到全局内存（临时 buffer）
2. 在全局内存中应用 Hadamard 反变换
3. 应用 RoPE（可选，也可以在 kernel 中）
4. 加载到共享内存
5. Tensor Core 计算
```

**实现**：
```cuda
// 新增一个预处理 kernel
__global__ void decompress_and_inverse_hadamard_kernel(
    const uint32_t* packed_quant,
    const uint64_t* bitmaps,
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    half* output,              // 输出到全局内存
    int M, int N,
    int head_group_num,
    int bit, int capacity
) {
    // 1. 解压 + 反量化（与现有逻辑类似）
    // 2. 应用 Hadamard 反变换（在 D*G 维度上）
    // 3. 写入全局内存
}

// 主 kernel 从全局内存加载已经反变换的数据
__global__ void Key_Kernel_With_Preprocessed_Data(
    const half* preprocessed_key,  // 已经做过 Hadamard^(-1) 的数据
    const half* B,
    half* C,
    ...
) {
    // 直接从全局内存加载到共享内存
    // 应用 RoPE（如果需要）
    // Tensor Core 计算
}
```

**优点**：
- ✅ 可以访问完整的 D*G 维度
- ✅ 逻辑清晰，易于实现
- ✅ 不改变现有 kernel 的核心逻辑

**缺点**：
- ❌ 需要额外的全局内存空间（临时 buffer）
- ❌ 增加一次全局内存读写
- ❌ 性能开销较大

### 方案 B：修改 Tile 划分策略

**核心思想**：改变 tile 的划分方式，使每个 tile 包含同一 token 的所有维度。

**当前 Tile 划分**：
```
Tile 在 seq_len 维度连续
Tile 0: [Token 0-63 的 dim 0]
Tile 1: [Token 0-63 的 dim 1]
...
```

**新的 Tile 划分**：
```
Tile 在 head_dim 维度连续
Tile 0: [Token 0 的 dim 0-255]  ← 包含完整的 D*G 维度
Tile 1: [Token 1 的 dim 0-255]
...
```

**问题**：
- ❌ 需要完全重写压缩和解压逻辑
- ❌ 改变 Tensor Core 的数据访问模式
- ❌ 可能影响性能
- ❌ 工作量巨大

### 方案 C：分组 Hadamard 反变换（近似方案）

**核心思想**：如果 `D*G` 太大，分组进行反变换。

```
假设 D=128, G=2, D*G=256

分组方案：
- 将 256 维分成 4 组，每组 64 维
- 在每组内独立做 Hadamard 反变换
```

**问题**：
- ❌ 这不是真正的 Hadamard 反变换
- ❌ 无法恢复原始数据
- ❌ 破坏了 RotateKV 的算法语义

### 方案 D：两阶段 Kernel（推荐的折中方案）

**核心思想**：使用两个 kernel，第一个做解压和反变换，第二个做 attention 计算。

**阶段 1：解压 + Hadamard 反变换 Kernel**
```cuda
__global__ void decompress_and_inverse_hadamard(
    // 输入：量化数据
    const uint32_t* packed_quant,
    const uint64_t* bitmaps,
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    // 输出：反变换后的 Key
    half* decompressed_key,  // [B, H, S, D]
    // 参数
    int M, int N, int head_group_num, int bit, int capacity
) {
    // 每个 thread block 处理一个或多个 token
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    
    // 1. 收集该 token 的所有维度数据（D*G 维）
    __shared__ half token_data[256];  // D*G = 256
    
    // 从多个 tile 中收集数据
    for (int dim_group = 0; dim_group < 4; dim_group++) {
        // 计算对应的 tile
        int tile_idx = calculate_tile_index(token_idx, head_idx, dim_group);
        
        // 解压该 tile 的数据
        decompress_tile(tile_idx, token_data + dim_group * 64, ...);
    }
    __syncthreads();
    
    // 2. 应用 Hadamard 反变换（Fast Walsh-Hadamard Transform）
    fast_hadamard_inverse(token_data, 256);
    __syncthreads();
    
    // 3. 写回全局内存（只写需要的 D 维）
    int head_group_idx = head_idx / head_group_num;
    int offset_in_group = head_idx % head_group_num;
    
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int src_idx = offset_in_group * D + d;
        int dst_idx = batch_idx * (H * S * D) + head_idx * (S * D) + token_idx * D + d;
        decompressed_key[dst_idx] = token_data[src_idx];
    }
}
```

**阶段 2：标准 Attention Kernel**
```cuda
// 使用现有的 attention kernel，但输入是已经反变换的 Key
Key_Kernel(
    query,
    decompressed_key,  // 已经做过 Hadamard^(-1)
    value,
    output,
    ...
);
```

**优点**：
- ✅ 可以正确实现 Hadamard 反变换
- ✅ 不改变 attention kernel 的核心逻辑
- ✅ 逻辑清晰，易于调试

**缺点**：
- ⚠️ 需要额外的全局内存（存储反变换后的 Key）
- ⚠️ 两个 kernel 调用，增加 launch overhead
- ⚠️ 性能不如单 kernel 方案

### 方案 E：Hadamard 反变换融合到加载阶段（最优但最复杂）

**核心思想**：在加载数据到共享内存时，协调多个 thread block 完成 Hadamard 反变换。

**实现思路**：
```cuda
// 使用 cooperative groups 或 grid-level synchronization
__global__ void Key_Kernel_With_Inline_Hadamard_Inverse(
    // ... 参数 ...
) {
    // 1. 多个 block 协作加载同一 token 的数据
    // 2. 使用全局内存作为临时 buffer
    // 3. Grid-level sync
    // 4. 应用 Hadamard 反变换
    // 5. 继续正常的 attention 计算
}
```

**问题**：
- ❌ 实现极其复杂
- ❌ 需要 CUDA 11+ 的 cooperative groups
- ❌ 性能调优困难
- ❌ 可能不如两阶段方案

## 推荐方案：方案 D（两阶段 Kernel）

### 实现步骤

1. **创建预处理 kernel**：
   - 输入：量化的稀疏数据
   - 输出：解压 + Hadamard 反变换后的 Key
   - 存储在全局内存中

2. **修改主 kernel**：
   - 输入：预处理后的 Key（全局内存）
   - 可选：在共享内存中应用 RoPE
   - 执行 Tensor Core 计算

3. **Python 接口**：
```python
# 两步调用
decompressed_key = mustafar_decompress_with_hadamard_inverse(
    bitmaps, packed_quant, tile_offsets, scales, zeros,
    M_Global, K_Global, head_group_num, bit, capacity
)

output = mustafar_attention_with_rope(
    query, decompressed_key, value,
    cos_table, sin_table, position_ids,
    ...
)
```

### 性能估算

**额外开销**：
- 预处理 kernel：~0.2-0.5 ms
- 额外全局内存：M * N * 2 bytes (如 256 * 128 * 2 = 64 KB per batch)
- Kernel launch overhead：~0.01 ms

**总开销**：约 5-10% 的额外延迟

### 优化方向

1. **融合 RoPE 到预处理 kernel**：
   - 在 Hadamard 反变换后立即应用 RoPE
   - 减少一次全局内存读写

2. **使用 pinned memory**：
   - 加速 CPU-GPU 数据传输（如果需要）

3. **Stream 并行**：
   - 预处理和计算可以在不同 stream 中 pipeline

## 下一步

需要你确认：

1. **RotateKV 的具体要求**：
   - Query 是否做 Hadamard？（从代码看似乎没有）
   - Hadamard 反变换必须在哪个阶段完成？
   - head_group_num 的典型值是多少？

2. **性能要求**：
   - 能接受多少额外延迟？
   - 内存预算是多少？

3. **实现偏好**：
   - 倾向于简单方案（两阶段）还是高性能方案（单 kernel）？

请提供更多 RotateKV 的细节，我可以给出更精确的实现方案。
