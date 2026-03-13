# 在共享内存中应用 Hadamard 反变换的可行性分析

## 问题

算子的流程是：
```
解量化 + 解稀疏 → 写入共享内存 → Tensor Core 计算
```

**关键问题**：能否在写入共享内存后、Tensor Core 计算前，对共享内存中的数据应用 Hadamard 反变换？

## 共享内存布局分析

### 当前的共享内存结构

```cuda
extern __shared__ __align__(128) half smem[];

// 共享内存布局：
// [0 ... TILE_M * TILE_K)           : A 矩阵 (Key Cache 解压后)
// [TILE_M * TILE_K ... end)         : B 矩阵 (Query)
```

### A 矩阵（Key Cache）在共享内存中的布局

```
A 矩阵维度：[TILE_M, TILE_K]
- TILE_M: 例如 64 (序列长度方向)
- TILE_K: 固定 64 (head_dim 方向)

内存布局：行主序（row-major）
┌─────────────────────────────────┐
│  TILE_K = 64 (head_dim)         │
│  ┌──────────────────────────┐   │
│  │ Token 0: [d0...d63]      │   │ ← 64 个维度连续
│  │ Token 1: [d0...d63]      │   │
│  │ Token 2: [d0...d63]      │   │
│  │ ...                      │   │
│  │ Token 63: [d0...d63]     │   │
│  └──────────────────────────┘   │
│  TILE_M = 64 (tokens)           │
└─────────────────────────────────┘

地址计算：
smem[token_idx * TILE_K + dim_idx]
```

## Hadamard 变换的需求

### 原始 Hadamard 变换

```python
# 在 Python 层：
key_states: [B, H, S, D] → [B, S, H//G, D*G]
# 例如：[8, 32, 256, 128] → [8, 256, 16, 256]
# Hadamard 在最后一维 (D*G=256) 上操作
```

### Hadamard 反变换

如果在存储前应用了 Hadamard 变换，那么在使用前需要应用**反变换**：

```
H^(-1) = (1/n) * H^T = (1/n) * H  (Hadamard 矩阵是对称的)
```

对于归一化的 Hadamard：
```
H_normalized = (1/√n) * H
H_normalized^(-1) = √n * H
```

## 关键发现：共享内存中的数据布局适合 Hadamard！

### ✅ 可行性分析

**重要发现**：在共享内存中，每个 token 的所有维度是**连续存储**的！

```
Token 0: smem[0:64]     ← 64 个维度连续
Token 1: smem[64:128]   ← 64 个维度连续
Token 2: smem[128:192]  ← 64 个维度连续
...
```

**但是**，Hadamard 需要的维度数是 `D*G`（如 256），而共享内存中只有 `TILE_K=64` 个维度。

### 问题：维度不匹配

| 项目 | Hadamard 需求 | 共享内存提供 | 匹配？ |
|------|--------------|-------------|--------|
| 维度数 | D*G = 256 | TILE_K = 64 | ❌ |
| 数据连续性 | 需要连续 | 连续存储 | ✅ |
| 访问模式 | 同一 token | 同一 token | ✅ |

### 深入分析：TILE_K 与 head_dim 的关系

让我们看看实际情况：

```
原始 Key: [B, H, S, D]
- H: num_heads (如 32)
- S: seq_len (如 256)
- D: head_dim (如 128)

Hadamard 后: [B, S, H//G, D*G]
- 如果 G=2: [B, S, 16, 256]
- 每个 token 有 16 个 group，每个 group 256 维

但是！在 attention 计算时：
Q @ K^T 的维度是 [B, H, S_q, S_k]
- Q: [B, H, S_q, D]
- K: [B, H, S_k, D]  ← 这里的 D 是 head_dim，不是 D*G！
```

### 关键问题：Hadamard 的作用域

**重要理解**：Hadamard 变换是在**存储前**对整个 Key Cache 进行的，但在**计算时**，每个 head 独立处理。

```python
# 存储前（Python 层）
key_states: [B, H, S, D]
# 1. Reshape
key_states = key_states.transpose(1,2).reshape(B, S, H//G, D*G)
# 2. Hadamard (在 D*G 维度)
key_states = hadamard_transform(key_states)  # [B, S, H//G, D*G]
# 3. Reshape back
key_states = key_states.reshape(B, S, H, D).transpose(1,2)  # [B, H, S, D]

# 现在每个 head 的数据已经被 Hadamard 变换过了
# 但维度仍然是 D (128)，不是 D*G (256)
```

**关键洞察**：Hadamard 变换后，数据被 reshape 回 `[B, H, S, D]`，所以：
- 每个 head 的维度仍然是 `D` (如 128)
- Hadamard 的效果是**跨 head 混合**了信息
- 在共享内存中，每个 token 有 `D` 个维度（不是 `D*G`）

## 重新评估：是否需要 Hadamard 反变换？

### 场景 1：如果 Hadamard 是可逆的

如果算法要求在计算前恢复原始数据：

```
存储: 原始 Key → Hadamard → 压缩 → 存储
推理: 加载 → 解压 → Hadamard^(-1) → 计算
```

**问题**：Hadamard 反变换需要 `D*G` 维度，但共享内存只有 `D` 维度。

### 场景 2：如果 Hadamard 不需要反变换

如果算法设计是：

```
存储: 原始 Key → Hadamard → 压缩 → 存储
推理: 加载 → 解压 → 直接计算（Query 也做了 Hadamard）
```

**这种情况下不需要反变换！**

## 实际可行的方案

### 方案 A：部分 Hadamard 反变换（如果 TILE_K = D）

如果 `TILE_K = head_dim = D`（如都是 128），可以在共享内存中对每个 token 的 `D` 维度做反变换：

```cuda
__device__ void apply_hadamard_inverse_in_shared(
    half* smem,
    int TILE_M,
    int TILE_K  // = head_dim
) {
    // 对每个 token 应用 Hadamard 反变换
    for (int token_idx = threadIdx.x; token_idx < TILE_M; token_idx += blockDim.x) {
        half* token_data = smem + token_idx * TILE_K;
        
        // 应用 Fast Walsh-Hadamard Transform
        // 在 TILE_K 维度上（如 128 维）
        fast_hadamard_transform_inplace(token_data, TILE_K);
    }
    __syncthreads();
}
```

**限制**：
- 只能在 `D` 维度上做反变换，不是 `D*G` 维度
- 这**不是完整的 Hadamard 反变换**
- 可能无法完全恢复原始数据

### 方案 B：跨 Tile 协作（理论可行，实现复杂）

如果需要完整的 `D*G` 维度反变换：

1. **多个 tile 协作**：
   - 同一个 token 的不同 head 在不同 tile 中
   - 需要跨 tile 通信

2. **使用全局内存作为中转**：
   - 将多个 tile 的数据收集到全局内存
   - 应用完整的 Hadamard 反变换
   - 再分发回各个 tile

**问题**：
- 实现极其复杂
- 性能开销巨大（全局内存访问）
- 违背了 tile-based 设计的初衷

### 方案 C：不做反变换（推荐）

**最佳方案**：在存储前应用 Hadamard，推理时直接使用变换后的数据。

```python
# 存储前
key_states = apply_grouped_hadamard(key_states, head_group_num)
key_states = apply_rotary_pos_emb_k(key_states, cos, sin, position_ids)
compressed = convert_key_batched_quant(key_states)

# 推理时
# Query 也需要做相同的 Hadamard 变换
query_states = apply_grouped_hadamard(query_states, head_group_num)
query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

# 然后直接计算 attention
attn_scores = query_states @ key_states.T  # 都是 Hadamard 变换后的
```

**关键**：Query 和 Key 都在同一个变换空间中，所以点积仍然有意义。

## 数学验证

### Hadamard 变换的性质

```
设 H 为 Hadamard 矩阵（归一化）
Q' = H @ Q  (变换后的 Query)
K' = H @ K  (变换后的 Key)

Attention scores:
S' = Q' @ K'^T 
   = (H @ Q) @ (H @ K)^T
   = (H @ Q) @ (K^T @ H^T)
   = H @ Q @ K^T @ H^T
   = H @ S @ H^T

其中 S = Q @ K^T 是原始的 attention scores
```

**结论**：变换后的 attention scores 是原始 scores 经过双边 Hadamard 变换的结果。

**如果 H 是正交矩阵**（Hadamard 矩阵是正交的）：
```
H @ H^T = I
所以 S' = H @ S @ H^T ≠ S
```

**这意味着**：
- 如果只对 Key 做 Hadamard，Query 不做，结果会不同
- 必须 Query 和 Key 都做相同的变换
- 或者都不做变换

## 结论

### 回答你的问题

**能否在共享内存中应用 Hadamard 反变换？**

1. **技术上部分可行**：
   - ✅ 可以在 `head_dim` (D) 维度上做反变换
   - ❌ 无法在 `D*G` 维度上做完整反变换（需要跨 tile）

2. **实际上不推荐**：
   - ❌ 部分反变换无法恢复原始数据
   - ❌ 完整反变换需要跨 tile 协作，性能开销巨大
   - ✅ 更好的方案是不做反变换，Query 和 Key 都在变换空间中计算

3. **推荐方案**：
   - 存储前：Key 做 Hadamard + RoPE
   - 推理时：Query 也做 Hadamard + RoPE
   - 直接在变换空间中计算 attention
   - 不需要反变换

### 实现建议

```python
class MustafarAttention:
    def forward(self, query, key, value, position_ids):
        # 1. 对 Query 和 Key 都应用 Hadamard
        if self.use_grouped_hadamard:
            query = apply_grouped_hadamard(query, self.head_group_num)
            key = apply_grouped_hadamard(key, self.head_group_num)
        
        # 2. 应用 RoPE
        query = apply_rotary_pos_emb_q(query, cos, sin, position_ids)
        key = apply_rotary_pos_emb_k(key, cos, sin, position_ids)
        
        # 3. Key 压缩（如果是新的 key）
        if is_new_key:
            compressed_key = convert_key_batched_quant(key)
        
        # 4. Attention 计算（在变换空间中）
        attn_output = mustafar_attention(query, compressed_key, value)
        
        return attn_output
```

### 性能考虑

如果在共享内存中做 Hadamard 反变换（即使只是部分的）：
- **额外延迟**：~0.1-0.2 ms per layer
- **同步开销**：需要 `__syncthreads()`
- **寄存器压力**：Hadamard 需要临时存储

**不如**：
- 在 Python 层对 Query 做 Hadamard：~0.05 ms
- 一次性完成，无需在每个 tile 中重复

## 最终建议

**不要在共享内存中做 Hadamard 反变换**，而是：

1. ✅ 存储前对 Key 做 Hadamard
2. ✅ 推理时对 Query 也做 Hadamard
3. ✅ 在变换空间中计算 attention
4. ✅ 如果需要 RoPE，在 kernel 中融合（因为 RoPE 是逐维度的）

这样既简单又高效！
