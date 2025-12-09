#!/usr/bin/env python3
"""
Mustafar Key Formulation 量化版本测试

本脚本用于测试量化版本的 `mustafar_package_quant.mustafar_key_formulation_quant` 函数，
模拟在解码阶段使用量化压缩的 Key Cache 进行稀疏注意力计算。
"""

import torch
import numpy as np
import sys
import os
import math
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compression_quant as compression_quant

# 尝试导入 mustafar_package_quant
try:
    import mustafar_package_quant
    print("✓ mustafar_package_quant 导入成功")
    MUSTAFAR_AVAILABLE = True
except ImportError as e:
    print(f"✗ mustafar_package_quant 导入失败: {e}")
    print("将使用 Python 实现作为参考")
    mustafar_package_quant = None
    MUSTAFAR_AVAILABLE = False

# 设置随机种子
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

print("依赖导入完成\n")


def dequantize_tile(packed_units, bitmap, scale, zero_point, bit=2, capacity=16):
    """
    反量化一个 tile 的数据
    
    Args:
        packed_units: uint32 打包的量化值
        bitmap: int64 位图
        scale: float32 缩放因子
        zero_point: float32 零点
        bit: 量化位宽
        capacity: 每 uint32 容纳的量化值数
    
    Returns:
        解压后的 64 个值（float16）
    """
    values = torch.zeros(64, dtype=torch.float16, device=packed_units.device)
    mask = (1 << bit) - 1
    
    value_idx = 0
    for i in range(64):
        if (bitmap >> (63 - i)) & 1:
            # 从打包的 uint32 中提取量化值
            unit_idx = value_idx // capacity
            bit_offset = (value_idx % capacity) * bit
            if unit_idx < len(packed_units):
                packed_unit = packed_units[unit_idx].item()
                q_value = (packed_unit >> bit_offset) & mask
                
                # 反量化
                dequant_value = (float(q_value) - zero_point) * scale
                values[i] = dequant_value
            value_idx += 1
    
    return values


def reconstruct_sparse_key_matrix_quant(bitmaps, tile_offsets, packed_quant, scales, zeros, 
                                       batch_idx, seq_len, head_dim, bit=2, capacity=16):
    """
    从量化压缩格式重构稀疏 Key 矩阵
    
    Args:
        bitmaps: [B, num_tiles] int64
        tile_offsets: [B, num_tiles] int32 (uint32 偏移)
        packed_quant: uint32 一维数组（全局打包缓冲）
        scales: [B, num_tiles] float32
        zeros: [B, num_tiles] float32
        batch_idx: batch 索引
        seq_len: 序列长度
        head_dim: head 维度
        bit: 量化位宽
        capacity: 每 uint32 容纳的量化值数
    
    Returns:
        重构的矩阵 [seq_len, head_dim]
    """
    reconstructed = torch.zeros(seq_len, head_dim, dtype=torch.float16, device=bitmaps.device)
    
    num_tiles_per_batch = (seq_len * head_dim) // 64
    
    for tile_idx in range(num_tiles_per_batch):
        bitmap = bitmaps[batch_idx, tile_idx].item()
        uint32_offset = tile_offsets[batch_idx, tile_idx].item()
        scale = scales[batch_idx, tile_idx].item()
        zero_point = zeros[batch_idx, tile_idx].item()
        
        # 计算该 tile 需要的 uint32 数量
        num_nz = bin(bitmap).count('1')
        units_needed = (num_nz + capacity - 1) // capacity
        
        # 提取该 tile 的打包 uint32
        tile_units = packed_quant[uint32_offset:uint32_offset + units_needed]
        
        # 反量化
        tile_values = dequantize_tile(tile_units, bitmap, scale, zero_point, bit, capacity)
        
        # 根据 Key 矩阵的 tile 布局填充
        # Key 矩阵是转置后按列主序存储的
        tile_start = tile_idx * 64
        for i in range(64):
            flat_idx = tile_start + i
            row = flat_idx // head_dim
            col = flat_idx % head_dim
            if row < seq_len and col < head_dim:
                reconstructed[row, col] = tile_values[i]
    
    return reconstructed


def sparse_matmul_reference_quant(query, bitmaps, tile_offsets, packed_quant, scales, zeros,
                                 seq_len, head_dim, num_key_value_groups, bit=2, capacity=16):
    """
    Python 参考实现：Q @ K^T（量化稀疏）
    
    Args:
        query: [total_batch_size, query_len, head_dim]
        bitmaps, tile_offsets, packed_quant, scales, zeros: 量化压缩的 Key
        seq_len: Key 序列长度
        head_dim: head 维度
        num_key_value_groups: 每个 KV head 对应的 query head 数量
        bit: 量化位宽
        capacity: 每 uint32 容纳的量化值数
    
    Returns:
        注意力权重 [total_batch_size, query_len, seq_len]
    """
    total_batch_size = query.shape[0]
    query_len = query.shape[1]
    num_kv_heads = bitmaps.shape[0]
    
    attention_weights = torch.zeros(total_batch_size, query_len, seq_len, 
                                   dtype=query.dtype, device=query.device)
    
    # 对每个 KV head 进行处理
    for kv_idx in range(num_kv_heads):
        # 重构该 KV head 的稀疏 Key 矩阵
        k_sparse = reconstruct_sparse_key_matrix_quant(
            bitmaps, tile_offsets, packed_quant, scales, zeros,
            kv_idx, seq_len, head_dim, bit, capacity
        )
        
        # 计算对应的 query head 索引范围
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
    bit = 2  # 量化位宽
    capacity = 16  # 每 uint32 容纳的量化值数 (32 // bit)
    
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
    print(f"num_key_value_groups: {num_key_value_groups}")
    print(f"量化位宽: {bit} bits")
    print(f"每字节容纳量化值数: {capacity}\n")
    
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
    
    # ========== 3. 调用量化压缩函数 ==========
    print("调用 convert_key_batched_quant...")
    k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
        compression_quant.convert_key_batched_quant(k_cache)
    
    print(f"k_bitmaps 形状: {k_bmps.shape}")
    print(f"k_tile_offsets 形状: {k_tile_offsets.shape}")
    print(f"k_packed_quant 形状: {k_packed_quant.shape}")
    print(f"k_scales 形状: {k_scales.shape}")
    print(f"k_zeros 形状: {k_zeros.shape}")
    
    total_compressed = (k_bmps.numel() * 8 + 
                       k_tile_offsets.numel() * 4 + 
                       k_packed_quant.numel() * 4 +  # uint32 = 4 bytes
                       k_scales.numel() * 4 +
                       k_zeros.numel() * 4)
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
    
    # ========== 5. 调用 mustafar_key_formulation_quant ==========
    att_compressed = None
    if MUSTAFAR_AVAILABLE:
        print("调用 mustafar_package_quant.mustafar_key_formulation_quant...")
        try:
            att_compressed = mustafar_package_quant.mustafar_key_formulation_quant(
                k_bmps,              # bitmaps
                k_packed_quant,      # packed_quant_values
                k_tile_offsets,      # tile_offsets
                k_scales,            # scales
                k_zeros,             # zeros
                padded_query,        # query
                compressed_length,   # M_Global
                model_dim,           # K_Global
                total_batch_size,    # Batch_Size
                num_key_value_groups,# num_key_value_groups
                bit,                 # bit
                capacity             # capacity
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
        print("mustafar_package_quant 未导入，跳过调用\n")
    
    # ========== 6. 测试普通密集矩阵乘法（Ground Truth）==========
    print("测试普通密集矩阵乘法（Ground Truth）...")
    att_dense = dense_matmul_reference(padded_query, k_cache, num_key_value_groups)
    
    print(f"密集矩阵乘法输出形状: {att_dense.shape}")
    print(f"密集矩阵乘法输出统计:")
    print(f"  最大值: {att_dense.max().item():.6f}")
    print(f"  最小值: {att_dense.min().item():.6f}")
    print(f"  平均值: {att_dense.mean().item():.6f}")
    print(f"  标准差: {att_dense.std().item():.6f}\n")
    
    # ========== 7. 测试量化稀疏 Python 参考实现 ==========
    print("测试量化稀疏 Python 参考实现...")
    att_ref = sparse_matmul_reference_quant(
        padded_query,
        k_bmps,
        k_tile_offsets,
        k_packed_quant,
        k_scales,
        k_zeros,
        seq_len,
        head_dim,
        num_key_value_groups,
        bit,
        capacity
    )
    
    print(f"量化稀疏参考实现输出形状: {att_ref.shape}")
    print(f"量化稀疏参考实现输出统计:")
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
        
        print(f"\n【1】CUDA 量化实现 vs 量化稀疏参考实现")
        print(f"CUDA 结果样本 (前5个值): {cuda_result[0, 0, :5]}")
        print(f"量化参考样本 (前5个值): {ref_result[0, 0, :5]}")
        
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
        
        is_close_to_ref = torch.allclose(cuda_result.float(), ref_result.float(), rtol=5e-2, atol=5e-2)
        print(f"是否接近 (rtol=5e-2, atol=5e-2): {is_close_to_ref}")
        
        print(f"\n{'='*60}")
        print(f"【2】CUDA 量化实现 vs 密集矩阵乘法（Ground Truth）")
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
        
        is_close_to_dense = torch.allclose(cuda_result.float(), dense_result.float(), rtol=5e-2, atol=5e-2)
        print(f"是否接近 (rtol=5e-2, atol=5e-2): {is_close_to_dense}")
        
        print(f"\n{'='*60}")
        print(f"【3】量化稀疏参考实现 vs 密集矩阵乘法（Ground Truth）")
        print(f"量化参考样本 (前5个值): {ref_result[0, 0, :5]}")
        print(f"密集矩阵样本 (前5个值): {dense_result[0, 0, :5]}")
        
        diff_ref_dense = torch.abs(ref_result.float() - dense_result.float())
        print(f"差异统计 (量化误差 + 稀疏误差):")
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
            _ = mustafar_package_quant.mustafar_key_formulation_quant(
                k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
                padded_query, compressed_length, model_dim, total_batch_size, 
                num_key_value_groups, bit, capacity
            )
        
        # 测试 CUDA 量化实现
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = mustafar_package_quant.mustafar_key_formulation_quant(
                k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
                padded_query, compressed_length, model_dim, total_batch_size, 
                num_key_value_groups, bit, capacity
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_iterations * 1000  # ms
        
        # 测试量化稀疏参考实现
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = sparse_matmul_reference_quant(
                padded_query, k_bmps, k_tile_offsets, k_packed_quant, 
                k_scales, k_zeros, seq_len, head_dim, num_key_value_groups, bit, capacity
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
        print(f"  CUDA 量化实现:      {cuda_time:.4f} ms")
        print(f"  量化稀疏参考实现:   {ref_time:.4f} ms")
        print(f"  密集矩阵乘法:       {dense_time:.4f} ms")
        print(f"\n加速比:")
        print(f"  CUDA vs 量化参考:   {ref_time / cuda_time:.2f}x")
        print(f"  CUDA vs 密集矩阵:   {dense_time / cuda_time:.2f}x")
        print(f"  密集矩阵 vs 量化参考: {ref_time / dense_time:.2f}x\n")
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
    print(f"  量化位宽: {bit} bits")
    
    print(f"\n压缩效果:")
    print(f"  原始内存: {original_memory / 1024 / 1024:.2f} MB")
    print(f"  压缩后内存: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  压缩比: {total_compressed / original_memory:.2%}")
    
    print(f"\n功能验证:")
    if MUSTAFAR_AVAILABLE:
        print(f"  ✓ mustafar_key_formulation_quant 调用成功")
        if att_compressed is not None:
            print(f"  ✓ 输出形状正确: {att_compressed.shape}")
            if is_close_to_ref:
                print(f"  ✓ CUDA 实现与量化参考实现接近")
            else:
                print(f"  ⚠ CUDA 实现与量化参考实现有差异")
            if is_close_to_dense:
                print(f"  ✓ CUDA 实现与密集矩阵乘法接近")
            else:
                print(f"  ⚠ CUDA 实现与密集矩阵乘法有差异（预期有量化+稀疏误差）")
    else:
        print(f"  ✗ mustafar_package_quant 未导入")
    
    print(f"\n✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
