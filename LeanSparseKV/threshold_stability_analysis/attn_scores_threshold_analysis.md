# Attention Scores 用于 Threshold 计算的详细解读

## 概述

本文档详细解读了 vLLM 工程中如何计算 attention scores 并将其用于 KV cache 剪枝的 threshold 计算。这是 DiffKV 稀疏注意力机制的核心组成部分。

## 1. 基本概念

### 1.1 核心参数

- **`kv_score_prune_thresh`**: 要保留的总 softmax score 的比例，范围 [0, 1]
  - `1.0`: 不进行任何剪枝
  - `0.0`: 最激进的剪枝
  - 实际使用中通常为 `0.02` - `0.04`

- **`kv_buffer_size`**: 最近的 N 个 token 永远不会被剪枝（局部缓冲区）

### 1.2 设计理念

基于 **累积 softmax 分数比例** 而非简单的分数阈值，确保：
- 自适应地保留最重要的 token
- 根据实际注意力分布动态调整
- 保护最近的 token（局部性原理）

## 2. Attention Scores 计算流程

### 2.1 第一步：获取原始注意力分数

```python
# 在 sparse_attention_small_kernel.py 中
score = F.softmax(score, dim=-1, dtype=torch.float32)  
# shape: [batch_size, num_heads, seq_len, seq_len]
```

**关键点**：
- 使用 `float32` 精度确保数值稳定性
- 经过 softmax 归一化，每行和为 1

### 2.2 第二步：聚合 GQA (Grouped Query Attention) 分数

```python
if self.num_heads > self.num_kv_heads:
    # 将多个 query head 的分数聚合到对应的 KV head
    scores = scores.view(
        batch_size, self.num_kv_heads, self.num_queries_per_kv, 
        max_seq_len, max_seq_len)
    kv_scores, _ = torch.max(scores, dim=2)  # 使用最大值聚合
```

**设计选择**：
- 使用 `max` 聚合而非 `mean`，保留最强的注意力信号
- 支持 GQA 架构（多个 query head 共享一个 KV head）

### 2.3 第三步：计算每个 token 的重要性分数

```python
# 对每个 token，计算它被所有后续 token 关注的总分数
kv_scores = torch.sum(kv_scores, dim=2, dtype=scores.dtype)  
# shape: [batch_size, num_kv_heads, seq_len]
```

**重要性定义**：
- Token i 的重要性 = 所有后续 token 对它的注意力分数之和
- 体现了该 token 对后续生成的影响力

## 3. 归一化和平均化处理

### 3.1 按查询次数归一化

```python
# 每个 token 被查询的次数 = seq_len - token_position
num_queries = seq_len - kv_index[:seq_len]
mean_kv_scores[batch_id, :, :seq_len] = kv_scores[batch_id, :, :seq_len] / num_queries
```

**原理**：
- 早期 token 被更多后续 token 查询，需要归一化
- 避免位置偏差，确保公平比较

### 3.2 归一化到概率分布

```python
# 确保每个 head 的所有 token 分数和为 1
sum_kv_scores = torch.sum(mean_kv_scores[batch_id, :, :seq_len], dim=-1, keepdim=True)
mean_kv_scores[batch_id, :, :seq_len] /= sum_kv_scores
```

**目的**：
- 将重要性分数转换为概率分布
- 便于后续的累积分数计算

## 4. Threshold 应用逻辑

### 4.1 累积分数计算

```python
# 按重要性排序（降序）
sorted_scores, sorted_indices = torch.sort(
    mean_kv_scores[batch_id, :, :buffer_start], 
    dim=-1, descending=True)

# 计算累积分数
sorted_scores = torch.cumsum(sorted_scores, dim=-1)
# 加上不可剪枝的 buffer 部分的分数
sorted_scores += buffer_sum_kv_scores
```

**策略**：
1. 将可剪枝的 token 按重要性排序
2. 计算累积重要性分数
3. 加上局部缓冲区的分数

### 4.2 Threshold 判断

```python
# 找到累积分数超过 threshold 的位置
masked_scores = torch.where(
    sorted_scores >= self.kv_score_thresh,  # 关键的 threshold 比较
    sorted_scores,
    0)

# 找到第一个超过 threshold 的位置
cut_indices = torch.nonzero(masked_scores[head])
if cut_indices.shape[0] > 0:
    cut_index = cut_indices[0][0]
    # 保留前 cut_index + 1 个最重要的 token
```

**决策逻辑**：
- 找到累积分数首次超过 threshold 的位置
- 保留该位置之前的所有 token
- 如果没有超过 threshold，保留所有 token

## 5. 实际配置示例

### 5.1 不同模型的 Threshold 设置

```bash
# 从 benchmark_throughput.sh 中的配置
--kv-prune-thresh 0.02   # Llama3-8B: 保留 2% 的总分数
--kv-prune-thresh 0.04   # Qwen2.5-7B: 保留 4% 的总分数  
--kv-prune-thresh 0.0    # Llama3-70B: 不进行剪枝
```

### 5.2 参数调优脚本示例

```python
# 从 param_tuning 脚本中
compress_configs = [kv_prune_thresh, kv_quant_thresh]
# kv_prune_thresh: 剪枝阈值
# kv_quant_thresh: 量化阈值
```

## 6. 算法特点分析

### 6.1 优势

1. **自适应性**：根据实际注意力分布动态调整
2. **公平性**：通过归一化消除位置偏差
3. **保护机制**：局部缓冲区保护最近 token
4. **精确控制**：通过累积分数精确控制保留比例

### 6.2 设计考量

1. **Per-head 处理**：每个 attention head 独立计算
2. **数值稳定性**：使用 float32 精度
3. **内存效率**：只保留必要的 token
4. **计算效率**：批量处理和并行化

## 7. 数学公式总结

整个过程的数学表示：

```
1. 原始分数: score_ij = softmax(QK^T / √d)_ij

2. Token 重要性: importance_i = Σ_j score_ji / (seq_len - i)

3. 归一化: norm_importance_i = importance_i / Σ_k importance_k

4. 累积分数: cumsum_i = Σ_{k=0}^i sorted_norm_importance_k

5. 保留条件: cumsum_i ≥ kv_score_prune_thresh
```

## 8. 代码位置索引

### 8.1 核心实现文件

- **主要逻辑**: `vllm/model_executor/layers/sparse_attention_small_kernel.py`
  - `_prune_prompt_kv_cache()` 函数 (第 307-450 行)
  
- **配置定义**: `vllm/config.py`
  - `CacheConfig` 类 (第 287-350 行)

### 8.2 参数传递路径

```
命令行参数 → compress_configs → CacheConfig → SparsePagedAttention → _prune_prompt_kv_cache
```

## 9. 调试和监控

### 9.1 关键调试信息

```python
# 在代码中可以添加的调试信息
print(f'Layer: {self.layer}, saved_kv_len: {saved_kv_len}')
print(f'Threshold: {self.kv_score_thresh}, Cut index: {cut_index}')
```

### 9.2 性能监控

- 监控实际保留的 token 比例
- 观察不同 head 的剪枝行为差异
- 跟踪累积分数的分布特征

## 10. 总结

这个 threshold 计算机制是 DiffKV 的核心创新，通过：

1. **智能的重要性评估**：基于实际注意力分布
2. **精确的比例控制**：通过累积分数确保准确性
3. **灵活的参数调节**：支持不同模型和任务的需求
4. **高效的实现方式**：优化的 CUDA 内核和批处理

实现了在保持模型性能的同时，显著减少 KV cache 的内存占用。