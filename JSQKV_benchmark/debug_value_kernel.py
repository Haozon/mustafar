#!/usr/bin/env python3
"""
调试Value kernel
"""
import torch
import sys
sys.path.append('kernel_quant')
import mustafar_package_quant

# 最简单的测试
batch_size = 2
num_heads = 8
compressed_length = 256
model_dim = 128

print(f"测试参数:")
print(f"  batch_size: {batch_size}")
print(f"  num_heads: {num_heads}")
print(f"  compressed_length: {compressed_length}")
print(f"  model_dim: {model_dim}")

# 计算tile数量
# Value矩阵: [compressed_length, model_dim] = [64, 64]
# 每64个元素一个bitmap
num_elements = compressed_length * model_dim  # 64 * 64 = 4096
num_bitmaps = num_elements // 64  # 4096 / 64 = 64

print(f"\n计算:")
print(f"  总元素数: {num_elements}")
print(f"  bitmap数量: {num_bitmaps}")

# 创建输入
bmp = torch.ones((batch_size * num_heads, num_bitmaps), dtype=torch.int64, device='cuda')  # 全1表示全部非零
NZ_quant = torch.randint(0, 256, (batch_size * num_heads * num_elements // 16,), dtype=torch.int32, device='cuda')
tile_offsets = torch.arange(0, num_bitmaps, dtype=torch.int32, device='cuda') * 16  # 每个bitmap 64个元素，量化后64/16*4=16字节
scales = torch.ones(num_bitmaps, dtype=torch.float32, device='cuda') * 0.1
zeros = torch.zeros(num_bitmaps, dtype=torch.float32, device='cuda')
tile_counts = torch.full((batch_size * num_heads, num_bitmaps), 64, dtype=torch.int32, device='cuda')
tile_units = torch.full((batch_size * num_heads, num_bitmaps), 4, dtype=torch.int32, device='cuda')
B = torch.randn(batch_size * num_heads, 8, compressed_length, dtype=torch.float16, device='cuda')
Reduction_Workspace = torch.zeros(1, dtype=torch.float16, device='cuda')

print(f"\n输入shapes:")
print(f"  bmp: {bmp.shape}")
print(f"  NZ_quant: {NZ_quant.shape}")
print(f"  tile_offsets: {tile_offsets.shape}")
print(f"  tile_counts: {tile_counts.shape}")
print(f"  tile_units: {tile_units.shape}")
print(f"  scales: {scales.shape}")
print(f"  zeros: {zeros.shape}")
print(f"  B: {B.shape}")
print(f"\n参数:")
print(f"  M_Global (model_dim): {model_dim}")
print(f"  K_Global (compressed_length): {compressed_length}")
print(f"  Batch_Size: {batch_size * num_heads}")
print(f"  num_key_value_groups: {num_heads // 4}")

try:
    output = mustafar_package_quant.mustafar_quant_sparse_value_forward(
        bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales.half(), zeros.half(), B, Reduction_Workspace,
        model_dim,          # M_Global
        compressed_length,  # K_Global
        batch_size * num_heads,
        num_heads // 4,  # num_key_value_groups
        2,  # bit
        16  # capacity
    )
    print(f"\n✅ 成功! 输出shape: {output.shape}")
    
    # 尝试访问数据
    try:
        first_elem = output[0, 0, 0].item()
        print(f"第一个元素: {first_elem}")
        min_val = output.min().item()
        print(f"最小值: {min_val}")
    except Exception as e:
        print(f"访问数据失败: {e}")
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
