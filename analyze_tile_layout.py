#!/usr/bin/env python3
"""
分析 Key 矩阵中 tile 的获取方式和 Nonzeros 排布
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import kernel.compression as compression

def analyze_key_tile_layout():
    """分析 Key 矩阵的 tile 布局"""
    print("=== 分析 Key 矩阵的 tile 布局 ===\n")

    # 创建一个小的测试张量，方便追踪元素位置
    B = 1  # batch
    M = 128  # seq_len (必须是64的倍数)
    N = 64   # head_dim (必须是64的倍数)

    print(f"输入形状: [B={B}, M={M}, N={N}]")
    print(f"转置后形状: [B={B}, N={N}, M={M}]")

    # 创建顺序编号的张量，便于追踪元素位置
    total_elements = B * M * N
    indices = torch.arange(total_elements, dtype=torch.float16, device='cuda')
    test_tensor = indices.view(B, M, N)

    print(f"\n原始张量扁平化索引 (前100个):")
    print(test_tensor.view(-1)[:100].cpu().numpy())

    # 调用压缩函数
    print("\n调用 convert_key_batched...")
    bitmaps, accum_counts, packed = compression.convert_key_batched(test_tensor)

    num_tiles_per_batch = (M * N) // 64
    print(f"\n每个 batch 的 tile 数量: {num_tiles_per_batch}")

    # 分析前几个 tile 的索引
    print("\n=== 分析前几个 tile 的索引 ===")

    # 获取转置后的张量（压缩函数内部进行转置）
    test_tensor_t = test_tensor.transpose(1, 2).contiguous()
    print(f"转置后张量形状: {test_tensor_t.shape}")
    print(f"转置后扁平化索引 (前100个):")
    print(test_tensor_t.view(-1)[:100].cpu().numpy())

    # 手动计算前几个 tile 的索引
    stride_batch = num_tiles_per_batch * 64

    print(f"\nstride_batch = {stride_batch}")
    print("\n手动计算 tile 索引 (按 calculate_bitmap_key_batched 逻辑):")

    for tile_id in range(10):
        # 内核中的计算
        block_row = tile_id % N  # N = head_dim = 64
        block_col = tile_id // N

        base_idx = block_row * M + block_col * 64
        tile_indices = list(range(base_idx, base_idx + 64))

        print(f"\nTile {tile_id}:")
        print(f"  block_row = tile_id % N = {tile_id} % {N} = {block_row}")
        print(f"  block_col = tile_id // N = {tile_id} // {N} = {block_col}")
        print(f"  base_idx = block_row * M + block_col * 64 = {block_row} * {M} + {block_col} * 64 = {base_idx}")
        print(f"  索引范围: [{base_idx}, {base_idx+63}]")

        # 将这些索引映射回原始张量
        print(f"  对应转置张量中的元素:")
        print(f"    位置: {tile_indices[:5]}...")

        # 获取实际值
        values = test_tensor_t.view(-1)[base_idx:base_idx+64].cpu().numpy()
        print(f"    值: {values[:5]}...")

        # 检查位图
        bitmap = bitmaps[0, tile_id].item()
        print(f"  位图值: {bitmap} (十六进制: {hex(bitmap)})")

        # 解析位图
        bits = [(bitmap >> (63 - i)) & 1 for i in range(64)]
        nonzero_positions = [i for i, bit in enumerate(bits) if bit == 1]
        print(f"  非零位置: {nonzero_positions[:10]}{'...' if len(nonzero_positions) > 10 else ''}")

    print("\n=== 验证 tile 在原始 Key 矩阵中的位置 ===")
    print("原始 Key 矩阵形状: [M=seq_len=128, N=head_dim=64]")
    print("每个 tile 覆盖 64 个连续的行（seq_len 维度）和 1 个列（head_dim 维度）")

    # 显示 tile_id=0 和 tile_id=1 在原始矩阵中的位置
    for tile_id in [0, 1, 64]:
        block_row = tile_id % N
        block_col = tile_id // N

        print(f"\nTile {tile_id}:")
        print(f"  在转置矩阵 [N={N}, M={M}] 中:")
        print(f"    行: {block_row}")
        print(f"    列范围: [{block_col*64}, {block_col*64+63}]")

        print(f"  在原始矩阵 [M={M}, N={N}] 中:")
        print(f"    行范围: [{block_col*64}, {block_col*64+63}]")
        print(f"    列: {block_row}")

    print("\n=== 遍历顺序总结 ===")
    print("tile_id 从 0 到 (M*N/64-1):")
    print("1. 固定 block_col (列组)，遍历所有 block_row (行)")
    print("2. block_col 递增，重复步骤1")
    print("\n这对应原始 Key 矩阵中:")
    print("- 按列优先遍历 (head_dim 维度)")
    print("- 每列按64行一组进行分块")
    print("- 先遍历所有列的第0-63行，再遍历所有列的第64-127行，以此类推")

def visualize_small_example():
    """可视化一个小例子"""
    print("\n\n=== 小型可视化示例 ===")

    # 创建一个更小的张量
    B = 1
    M = 32  # 简化，实际应为64的倍数
    N = 32

    print(f"\n小型张量形状: [B={B}, M={M}, N={N}]")
    print("注意: 实际实现要求 M 和 N 是64的倍数，这里为演示简化")

    # 创建有意义的模式
    test_tensor = torch.zeros(B, M, N, dtype=torch.float16, device='cuda')

    # 设置一些非零值形成模式
    for i in range(M):
        for j in range(N):
            if (i + j) % 8 == 0:
                test_tensor[0, i, j] = 1.0

    # 计算非零位置
    nonzero_mask = test_tensor[0] != 0
    nonzero_positions = torch.nonzero(nonzero_mask)

    print(f"\n非零位置 (原始矩阵 [M={M}, N={N}]):")
    for pos in nonzero_positions[:20]:
        print(f"  ({pos[0]}, {pos[1]})")

    # 模拟 tile 划分 (假设 tile_size=8 简化)
    tile_size = 8
    print(f"\n假设 tile_size={tile_size} 的划分:")

    for col in range(0, N, tile_size):
        for row in range(0, M, tile_size):
            tile_nz = []
            for i in range(row, min(row+tile_size, M)):
                for j in range(col, min(col+tile_size, N)):
                    if test_tensor[0, i, j] != 0:
                        tile_nz.append((i, j))

            if tile_nz:
                print(f"Tile (行{row}-{min(row+tile_size, M)-1}, 列{col}-{min(col+tile_size, N)-1}): {len(tile_nz)} 个非零")

if __name__ == "__main__":
    analyze_key_tile_layout()
    visualize_small_example()