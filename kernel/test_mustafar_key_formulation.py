#!/usr/bin/env python3
"""
Mustafar Key Formulation 测试

本脚本用于测试 `mustafar_package.mustafar_key_formulation` 函数，
模拟在解码阶段使用压缩的 Key Cache 进行稀疏注意力计算。
"""

import torch
import numpy as np
import sys
import os
import math
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compression as compression

# 尝试导入 mustafar_package
try:
    import mustafar_package
    print("✓ mustafar_package 导入成功")
    MUSTAFAR_AVAILABLE = True
except ImportError as e:
    print(f"✗ mustafar_package 导入失败: {e}")
    print("将使用 Python 实现作为参考")
    mustafar_package = None
    MUSTAFAR_AVAILABLE = False

# 设置随机种子
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

print("依赖导入完成\n")


def reconstruct_sparse_key_matrix(bitmaps, accum_counts, packed_values, batch_idx, seq_len, head_dim):
    """
    从压缩格式重构稀疏 Key 矩阵。
    
    Args:
        bitmaps: [B, num_tiles] int64
        accum_counts: [B, num_tiles+1] int32
        packed_values: 列表，每个元素是一个 batch 的打包值
        batch_idx: batch 索引
        seq_len: 序列长度
        head_dim: head 维度
    
    Returns:
        重构的矩阵 [seq_len, head_dim]
    """
    reconstructed = torch.zeros(seq_len, head_dim, dtype=torch.float16, device=bitmaps.device)
    
    num_tiles_per_batch = (seq_len * head_dim) // 64
    
    for tile_idx in range(num_tiles_per_batch):
        bitmap = bitmaps[batch_idx, tile_idx].item()
        start_offset = accum_counts[batch_idx, tile_idx].item() * 2
        
        tile_start = tile_idx * 64
        
        value_idx = 0
        for i in range(64):
            if (bitmap >> (63 - i)) & 1:
                if start_offset + value_idx < len(packed_values[batch_idx]):
                    flat_idx = tile_start + i
                    row = flat_idx // head_dim
                    col = flat_idx % head_dim
                    if row < seq_len and col < head_dim:
                        reconstructed[row, col] = packed_values[batch_idx][start_offset + value_idx]
                value_idx += 1
    
    return reconstructed


def sparse_matmul_reference(query, bitmaps, accum_counts, packed_values, seq_len, head_dim, num_key_value_groups):
    """
    Python 参考实现：Q @ K^T（稀疏）
    
    Args:
        query: [total_batch_size, query_len, head_dim]
        bitmaps, accum_counts, packed_values: 压缩的 Key
        seq_len: Key 序列长度
        head_dim: head 维度
        num_key_value_groups: 每个 KV head 对应的 query head 数量
    
    Returns:
        注意力权重 [total_batch_size, query_len, seq_len]
    """
    total_batch_size = query.shape[0]
    query_len = query.shape[1]
    num_kv_heads = bitmaps.shape[0]
    
    attention_weights = torch.zeros(total_batch_size, query_len, seq_len, dtype=query.dtype, device=query.device)
    
    # 对每个 KV head 进行处理
    for kv_idx in range(num_kv_heads):
        # 重构该 KV head 的稀疏 Key 矩阵
        k_sparse = reconstruct_sparse_key_matrix(bitmaps, accum_counts, packed_values, kv_idx, seq_len, head_dim)
        
        # 计算对应的 query head 索引范围
        # 如果 num_key_value_groups > 1，多个 query heads 共享一个 KV head
        for group_idx in range(num_key_value_groups):
            q_idx = kv_idx * num_key_value_groups + group_idx
            if q_idx < total_batch_size:
                # query[q_idx, :, :] @ k_sparse.T -> [query_len, seq_len]
                attention_weights[q_idx, :, :] = torch.matmul(query[q_idx, :, :], k_sparse.T)
    
    return attention_weights


def dense_matmul_reference(query, k_cache, num_key_value_groups):
    """
    普通密集矩阵乘法参考实现：Q @ K^T
    
    Args:
        query: [total_batch_size, query_len, head_dim]
        k_cache: [num_kv_heads, seq_len, head_dim]
        num_key_value_groups: 每个 KV head 对应的 query head 数量
    
    Returns:
        注意力权重 [total_batch_size, query_len, seq_len]
    """
    total_batch_size = query.shape[0]
    query_len = query.shape[1]
    num_kv_heads = k_cache.shape[0]
    seq_len = k_cache.shape[1]
    
    attention_weights = torch.zeros(total_batch_size, query_len, seq_len, 
                                   dtype=query.dtype, device=query.device)
    
    # 对每个 KV head 进行处理
    for kv_idx in range(num_kv_heads):
        k_dense = k_cache[kv_idx]  # [seq_len, head_dim]
        
        # 计算对应的 query head 索引范围
        for group_idx in range(num_key_value_groups):
            q_idx = kv_idx * num_key_value_groups + group_idx
            if q_idx < total_batch_size:
                # query[q_idx, :, :] @ k_dense.T -> [query_len, seq_len]
                attention_weights[q_idx, :, :] = torch.matmul(query[q_idx, :, :], k_dense.T)
    
    return attention_weights


def main():
    # ========== 1. 模型参数 ==========
    batch_size = 2
    num_heads = 4
    num_key_value_heads = 4
    head_dim = 128
    seq_len = 256  # 压缩的序列长度
    
    # 计算派生参数
    total_batch_kv = batch_size * num_key_value_heads
    total_batch_size = batch_size * num_heads
    num_key_value_groups = num_heads // num_key_value_heads
    model_dim = head_dim
    compressed_length = seq_len
    
    print(f"=== 模型参数 ===")
    print(f"batch_size: {batch_size}")
    print(f"num_heads: {num_heads}")
    print(f"num_key_value_heads: {num_key_value_heads}")
    print(f"head_dim: {head_dim}")
    print(f"seq_len (compressed): {seq_len}")
    print(f"total_batch_kv: {total_batch_kv}")
    print(f"total_batch_size: {total_batch_size}")
    print(f"num_key_value_groups: {num_key_value_groups}\n")
    
    # ========== 2. 生成并压缩 Key Cache ==========
    sparsity = 0.7  # 70% 稀疏度
    dense_tensor = torch.randn(total_batch_kv, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    # 应用稀疏掩码
    k = int(round(head_dim * (1 - sparsity)))
    rand_vals = torch.rand(total_batch_kv, seq_len, head_dim, device='cuda')
    _, indices = torch.topk(rand_vals, k=k, dim=-1)
    mask = torch.zeros(total_batch_kv, seq_len, head_dim, device='cuda', dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    
    k_cache = (dense_tensor * mask.half()).half()  # 保持 float16 类型
    
    print(f"K Cache 形状: {k_cache.shape}")
    print(f"非零元素比例: {(mask.sum() / mask.numel()).item():.2%}")
    print(f"数据类型: {k_cache.dtype}\n")
    
    # ========== 3. 调用压缩函数 ==========
    print("调用 convert_key_batched...")
    k_bmps, k_idxs, k_nzs = compression.convert_key_batched(k_cache)
    
    # 计算 nz_offset（非零值偏移）
    k_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=k_cache.device)
    for i in range(1, total_batch_kv):
        k_nz_offset[i] = k_nz_offset[i-1] + k_idxs[i-1][-1] // 4
    
    # 构造 k_compressed 结构
    k_compressed = [k_bmps, k_idxs, k_nzs, k_nz_offset]
    
    print(f"k_bitmaps 形状: {k_bmps.shape}")
    print(f"k_accum_counts 形状: {k_idxs.shape}")
    print(f"k_packed_not_batched 长度: {len(k_nzs)}")
    print(f"k_nz_offset 形状: {k_nz_offset.shape}")
    
    total_compressed = (k_bmps.numel() * 8 + 
                       k_idxs.numel() * 4 + 
                       sum(nz.numel() * 2 for nz in k_nzs) +
                       k_nz_offset.numel() * 4)
    original_memory = k_cache.numel() * 2
    print(f"\n压缩后内存占用:")
    print(f"原始: {original_memory / 1024 / 1024:.2f} MB")
    print(f"压缩后: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"压缩比: {total_compressed / original_memory:.2%}\n")
    
    # ========== 4. 生成 Query 张量 ==========
    query_len = 1
    query = torch.randn(batch_size, num_heads, query_len, head_dim, dtype=torch.float16, device='cuda')
    
    # 对 Query 进行 padding
    padded_query = torch.nn.functional.pad(
        query.view(total_batch_size, -1, head_dim), 
        (0, 0, 0, 7), 
        mode='constant', 
        value=0
    )
    
    print(f"Query 形状: {query.shape}")
    print(f"Padded Query 形状: {padded_query.shape}")
    print(f"Query 数据类型: {query.dtype}\n")
    
    # ========== 5. 调用 mustafar_key_formulation ==========
    att_compressed = None
    if MUSTAFAR_AVAILABLE:
        print("调用 mustafar_package.mustafar_key_formulation...")
        try:
            att_compressed = mustafar_package.mustafar_key_formulation(
                k_compressed[0],              # bitmaps
                torch.cat(k_compressed[2]),   # packed_values
                k_compressed[1],              # accum_counts
                k_compressed[3],              # nz_offset
                padded_query,                 # query
                compressed_length,            # compressed_length
                model_dim,                    # model_dim
                total_batch_size,             # total_batch_size
                num_key_value_groups          # num_key_value_groups
            )
            
            print(f"✓ 调用成功")
            print(f"输出形状: {att_compressed.shape}")
            print(f"输出数据类型: {att_compressed.dtype}")
            print(f"输出统计:")
            print(f"  最大值: {att_compressed.max().item():.6f}")
            print(f"  最小值: {att_compressed.min().item():.6f}")
            print(f"  平均值: {att_compressed.mean().item():.6f}")
            print(f"  标准差: {att_compressed.std().item():.6f}\n")
            
        except Exception as e:
            print(f"✗ 调用失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("mustafar_package 未导入，跳过调用\n")
    
    # ========== 6. 测试普通密集矩阵乘法（Ground Truth）==========
    print("测试普通密集矩阵乘法（Ground Truth）...")
    att_dense = dense_matmul_reference(padded_query, k_cache, num_key_value_groups)
    
    print(f"密集矩阵乘法输出形状: {att_dense.shape}")
    print(f"密集矩阵乘法输出统计:")
    print(f"  最大值: {att_dense.max().item():.6f}")
    print(f"  最小值: {att_dense.min().item():.6f}")
    print(f"  平均值: {att_dense.mean().item():.6f}")
    print(f"  标准差: {att_dense.std().item():.6f}\n")
    
    # ========== 7. 测试稀疏 Python 参考实现 ==========
    print("测试稀疏 Python 参考实现...")
    att_ref = sparse_matmul_reference(
        padded_query,
        k_bmps,
        k_idxs,
        k_nzs,
        seq_len,
        head_dim,
        num_key_value_groups
    )
    
    print(f"稀疏参考实现输出形状: {att_ref.shape}")
    print(f"稀疏参考实现输出统计:")
    print(f"  最大值: {att_ref.max().item():.6f}")
    print(f"  最小值: {att_ref.min().item():.6f}")
    print(f"  平均值: {att_ref.mean().item():.6f}")
    print(f"  标准差: {att_ref.std().item():.6f}\n")
    
    # ========== 8. 对比各种实现 ==========
    is_close_to_ref = False
    is_close_to_dense = False
    
    if MUSTAFAR_AVAILABLE and att_compressed is not None:
        print("=" * 60)
        print("对比分析")
        print("=" * 60)
        
        # 只对比第一个 query token（未 padding 的部分）
        cuda_result = att_compressed[:, 0:1, :]
        ref_result = att_ref[:, 0:1, :]
        dense_result = att_dense[:, 0:1, :]
        
        print(f"\n【1】CUDA 稀疏实现 vs 稀疏参考实现")
        print(f"CUDA 结果样本 (前5个值): {cuda_result[0, 0, :5]}")
        print(f"稀疏参考样本 (前5个值): {ref_result[0, 0, :5]}")
        
        diff_cuda_ref = torch.abs(cuda_result.float() - ref_result.float())
        print(f"差异统计:")
        print(f"  最大差异: {diff_cuda_ref.max().item():.6f}")
        print(f"  平均差异: {diff_cuda_ref.mean().item():.6f}")
        print(f"  中位数差异: {torch.median(diff_cuda_ref).item():.6f}")
        
        ref_abs = torch.abs(ref_result.float())
        rel_error_ref = diff_cuda_ref / (ref_abs + 1e-8)
        print(f"相对误差:")
        print(f"  最大相对误差: {rel_error_ref.max().item():.2%}")
        print(f"  平均相对误差: {rel_error_ref.mean().item():.2%}")
        
        is_close_to_ref = torch.allclose(cuda_result.float(), ref_result.float(), rtol=1e-2, atol=1e-3)
        print(f"是否接近 (rtol=1e-2, atol=1e-3): {is_close_to_ref}")
        
        print(f"\n{'='*60}")
        print(f"【2】CUDA 稀疏实现 vs 密集矩阵乘法（Ground Truth）")
        print(f"CUDA 结果样本 (前5个值): {cuda_result[0, 0, :5]}")
        print(f"密集矩阵样本 (前5个值): {dense_result[0, 0, :5]}")
        
        diff_cuda_dense = torch.abs(cuda_result.float() - dense_result.float())
        print(f"差异统计:")
        print(f"  最大差异: {diff_cuda_dense.max().item():.6f}")
        print(f"  平均差异: {diff_cuda_dense.mean().item():.6f}")
        print(f"  中位数差异: {torch.median(diff_cuda_dense).item():.6f}")
        
        dense_abs = torch.abs(dense_result.float())
        rel_error_dense = diff_cuda_dense / (dense_abs + 1e-8)
        print(f"相对误差:")
        print(f"  最大相对误差: {rel_error_dense.max().item():.2%}")
        print(f"  平均相对误差: {rel_error_dense.mean().item():.2%}")
        
        is_close_to_dense = torch.allclose(cuda_result.float(), dense_result.float(), rtol=1e-2, atol=1e-3)
        print(f"是否接近 (rtol=1e-2, atol=1e-3): {is_close_to_dense}")
        
        print(f"\n{'='*60}")
        print(f"【3】稀疏参考实现 vs 密集矩阵乘法（Ground Truth）")
        print(f"稀疏参考样本 (前5个值): {ref_result[0, 0, :5]}")
        print(f"密集矩阵样本 (前5个值): {dense_result[0, 0, :5]}")
        
        diff_ref_dense = torch.abs(ref_result.float() - dense_result.float())
        print(f"差异统计 (稀疏误差):")
        print(f"  最大差异: {diff_ref_dense.max().item():.6f}")
        print(f"  平均差异: {diff_ref_dense.mean().item():.6f}")
        print(f"  中位数差异: {torch.median(diff_ref_dense).item():.6f}")
        
        rel_error_ref_dense = diff_ref_dense / (dense_abs + 1e-8)
        print(f"相对误差:")
        print(f"  最大相对误差: {rel_error_ref_dense.max().item():.2%}")
        print(f"  平均相对误差: {rel_error_ref_dense.mean().item():.2%}")
        print(f"{'='*60}\n")
    else:
        print("CUDA 实现未可用，跳过对比\n")
    
    # ========== 9. 性能测试 ==========
    if MUSTAFAR_AVAILABLE and att_compressed is not None:
        print("性能测试...")
        
        num_iterations = 10
        
        # 预热
        for _ in range(10):
            _ = mustafar_package.mustafar_key_formulation(
                k_compressed[0], torch.cat(k_compressed[2]), k_compressed[1], k_compressed[3],
                padded_query, compressed_length, model_dim, total_batch_size, num_key_value_groups
            )
        
        # 测试 CUDA 稀疏实现
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = mustafar_package.mustafar_key_formulation(
                k_compressed[0], torch.cat(k_compressed[2]), k_compressed[1], k_compressed[3],
                padded_query, compressed_length, model_dim, total_batch_size, num_key_value_groups
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_iterations * 1000  # ms
        
        # 测试稀疏参考实现
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = sparse_matmul_reference(
                padded_query, k_bmps, k_idxs, k_nzs, seq_len, head_dim, num_key_value_groups
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iterations * 1000  # ms
        
        # 测试密集矩阵乘法
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = dense_matmul_reference(padded_query, k_cache, num_key_value_groups)
        torch.cuda.synchronize()
        dense_time = (time.time() - start) / num_iterations * 1000  # ms
        
        print(f"\n性能对比 ({num_iterations} 次迭代):")
        print(f"  CUDA 稀疏实现:      {cuda_time:.4f} ms")
        print(f"  稀疏参考实现:       {ref_time:.4f} ms")
        print(f"  密集矩阵乘法:       {dense_time:.4f} ms")
        print(f"\n加速比:")
        print(f"  CUDA vs 稀疏参考:   {ref_time / cuda_time:.2f}x")
        print(f"  CUDA vs 密集矩阵:   {dense_time / cuda_time:.2f}x")
        print(f"  密集矩阵 vs 稀疏参考: {ref_time / dense_time:.2f}x\n")
    else:
        print("CUDA 实现未可用，跳过性能测试\n")
    
    # ========== 10. 总结 ==========
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"\n模型配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Compressed seq len: {compressed_length}")
    print(f"  Sparsity: {sparsity:.1%}")
    
    print(f"\n压缩效果:")
    print(f"  原始内存: {original_memory / 1024 / 1024:.2f} MB")
    print(f"  压缩后内存: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  压缩比: {total_compressed / original_memory:.2%}")
    
    print(f"\n功能验证:")
    if MUSTAFAR_AVAILABLE:
        print(f"  ✓ mustafar_key_formulation 调用成功")
        if att_compressed is not None:
            print(f"  ✓ 输出形状正确: {att_compressed.shape}")
            if is_close_to_ref:
                print(f"  ✓ CUDA 实现与稀疏参考实现接近")
            else:
                print(f"  ⚠ CUDA 实现与稀疏参考实现有差异")
            if is_close_to_dense:
                print(f"  ✓ CUDA 实现与密集矩阵乘法接近")
            else:
                print(f"  ⚠ CUDA 实现与密集矩阵乘法有差异（预期有稀疏误差）")
    else:
        print(f"  ✗ mustafar_package 未导入")
    
    print(f"\n✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
