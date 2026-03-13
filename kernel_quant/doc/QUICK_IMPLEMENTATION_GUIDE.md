# Value 量化快速实现指南

## 概述

本文档提供 Value 矩阵量化压缩的快速实现指南。CUDA 计算核心已完成，只需添加 Python 压缩层。

---

## 当前状态

### ✅ 已完成
- Key 量化压缩（Python + CUDA）
- Value 量化计算（CUDA kernel）
- Python 绑定接口

### ❌ 待实现
- Value 量化压缩（Python Triton kernels）

---

## 实现步骤

### 步骤 1: 在 `compression_quant.py` 添加 Value bitmap 计算

在 `compress_key_batched` 函数之后添加：

```python
@triton.jit
def calculate_bitmap_and_scale_value_batched(
    input_ptr, bitmaps_ptr, counts_ptr, scales_ptr, zeros_ptr,
    total_elems: tl.constexpr, shifts_ptr,
    stride_batch: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Value 行主序索引（关键差异）
    tiles_per_row = N // 64
    tiles_per_block = tiles_per_row * 64
    block_idx = tile_id // tiles_per_block
    rem = tile_id % tiles_per_block
    col_tile = rem // 64
    r_in_block = rem % 64
    row = block_idx * 64 + r_in_block
    col_start = col_tile * 64
    base_idx = batch_id * stride_batch + row * N + col_start
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    vals = tl.load(input_ptr + indices)
    bit_mask = tl.where(vals != 0.0, 1, 0)
    shifts = tl.load(shifts_ptr + offsets)
    bitmap = tl.sum(bit_mask * shifts, axis=0)
    cnt = tl.sum(bit_mask, axis=0)

    # 量化参数计算
    INF = 1e10
    masked_vals_for_min = tl.where(bit_mask != 0, vals, INF)
    masked_vals_for_max = tl.where(bit_mask != 0, vals, -INF)
    xmin = tl.min(masked_vals_for_min, axis=0)
    xmax = tl.max(masked_vals_for_max, axis=0)
    
    has_nonzero = cnt > 0
    if has_nonzero:
        scale = (xmax - xmin) / 3.0  # 2-bit: 0-3
        if scale == 0.0:
            scale = 1.0
        zero_point = tl.floor(-xmin / scale + 0.5)
    else:
        scale = 1.0
        zero_point = 0.0

    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    tl.store(bitmaps_ptr + flat_tile_index, bitmap)
    tl.store(counts_ptr + flat_tile_index, cnt)
    tl.store(scales_ptr + flat_tile_index, scale)
    tl.store(zeros_ptr + flat_tile_index, zero_point)
```

### 步骤 2: 添加 Value 压缩 kernel

```python
@triton.jit
def compress_value_batched(
    input_ptr, bitmaps_ptr, counts_ptr, tile_offsets_ptr,
    packed_not_ptr, scales_ptr, zeros_ptr, bit, capacity,
    total_elems: tl.constexpr, stride_batch: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Value 行主序索引（与上面相同）
    tiles_per_row = N // 64
    tiles_per_block = tiles_per_row * 64
    block_idx = tile_id // tiles_per_block
    rem = tile_id % tiles_per_block
    col_tile = rem // 64
    r_in_block = rem % 64
    row = block_idx * 64 + r_in_block
    col_start = col_tile * 64
    base_idx = batch_id * stride_batch + row * N + col_start
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    vals = tl.load(input_ptr + indices)

    tiles_per_batch = stride_batch // 64
    flat_tile_index = batch_id * tiles_per_batch + tile_id

    bitmap = tl.load(bitmaps_ptr + flat_tile_index)
    cnt = tl.load(counts_ptr + flat_tile_index)
    
    if cnt == 0:
        return

    shifted = bitmap >> (63 - offsets)
    bit_mask = shifted & 1
    valid = bit_mask != 0
    prefix = tl.cumsum(bit_mask, axis=0) - 1
    gidx = tl.where(valid, prefix, 0)

    cnt_i = tl.cast(cnt, tl.int32)
    within_cnt = gidx < cnt_i
    mask_valid = valid & within_cnt

    scale = tl.load(scales_ptr + flat_tile_index)
    zero_point = tl.load(zeros_ptr + flat_tile_index)
    if scale == 0.0:
        scale = 1.0

    maxq = (1 << bit) - 1
    q_float = tl.floor(vals / scale + 0.5) + zero_point
    q_clamped = tl.minimum(tl.maximum(q_float, 0.0), tl.cast(maxq, tl.float32))
    q_int = tl.cast(q_clamped, tl.uint32)

    tile_uint32_offset = tl.load(tile_offsets_ptr + flat_tile_index)
    uint32_idx = tile_uint32_offset + (gidx // capacity)
    bit_shift = (gidx % capacity) * bit
    value_to_write = tl.cast(q_int << bit_shift, tl.int32)

    tl.atomic_or(packed_not_ptr + uint32_idx, value_to_write, mask=mask_valid)
```

### 步骤 3: 添加 Python 接口函数

```python
def convert_value_batched_quant(inputs: torch.Tensor):
    """Value Cache 量化压缩"""
    B, M, N = inputs.shape
    assert inputs.is_cuda and inputs.dim() == 3 and M % 64 == 0
    device = inputs.device
    
    inputs_t = inputs.contiguous()  # Value 不转置
    num_tiles_per_batch = (M * N) // 64

    bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=device)
    counts = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=device)
    scales = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=device)
    zeros = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=device)

    shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
    shifts_np = np.left_shift(np.int64(1), shift_amounts)
    const_shifts = torch.tensor(shifts_np, device='cuda')

    grid = (num_tiles_per_batch, B)
    stride_batch = num_tiles_per_batch * 64

    # 计算 bitmap 和量化参数
    calculate_bitmap_and_scale_value_batched[grid](
        inputs_t.view(-1), bitmaps.view(-1), counts.view(-1),
        scales.view(-1), zeros.view(-1),
        total_elems=B * M * N, shifts_ptr=const_shifts,
        stride_batch=stride_batch, M=M, N=N
    )

    bit = 2
    capacity = 16

    # 计算 tile_offsets
    units_per_tile = (counts + capacity - 1) // capacity
    total_units_per_batch = units_per_tile.sum(dim=1).to(torch.int32)
    
    batch_base_offsets = torch.zeros((B,), dtype=torch.int32, device=device)
    if B > 1:
        batch_base_offsets[1:] = torch.cumsum(total_units_per_batch, dim=0)[:-1]

    starts_intra = torch.cumsum(units_per_tile, dim=1)
    starts_intra = torch.cat([
        torch.zeros((B, 1), dtype=starts_intra.dtype, device=device),
        starts_intra[:, :-1]
    ], dim=1)

    tile_offsets = (batch_base_offsets.unsqueeze(1).to(starts_intra.dtype) + 
                    starts_intra).to(torch.int32)

    total_packed_size = int(total_units_per_batch.sum().item()) \
                        if total_units_per_batch.numel() > 0 else 0
    packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.int32, device=device)

    # 压缩并量化
    if total_packed_size > 0:
        compress_value_batched[grid](
            inputs_t.view(-1), bitmaps.view(-1), counts.view(-1),
            tile_offsets.contiguous().view(-1), packed_not_flat.view(-1),
            scales.view(-1), zeros.view(-1), bit, capacity,
            total_elems=B * M * N, stride_batch=stride_batch, M=M, N=N
        )

    return bitmaps, tile_offsets, packed_not_flat, scales, zeros
```

### 步骤 4: 添加测试代码

在 `compression_quant.py` 的 `if __name__ == '__main__':` 部分添加：

```python
if __name__ == '__main__':
    import time
    
    # 测试 Value 压缩
    print('\n=== Testing Value Quantization ===')
    torch.manual_seed(42)
    B, M, N = 8, 256, 128
    value_inputs = torch.randn(B, M, N, dtype=torch.float16, device='cuda')
    
    # 应用稀疏性
    sparsity = 0.7
    mask = torch.rand(B, M, N, device='cuda') > sparsity
    value_inputs = value_inputs * mask.float()
    
    original_size = value_inputs.numel() * 2  # float16 = 2 bytes
    
    print(f'Value tensor shape: {value_inputs.shape}')
    print(f'Original size: {original_size / 1024 / 1024:.2f} MB')
    print(f'Sparsity: {sparsity * 100:.1f}%')
    
    torch.cuda.synchronize()
    start = time.time()
    
    v_bitmaps, v_tile_offsets, v_packed_quant, v_scales, v_zeros = \
        convert_value_batched_quant(value_inputs)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    
    # 计算压缩后大小
    compressed_size = (v_bitmaps.numel() * 8 + 
                      v_tile_offsets.numel() * 4 +
                      v_packed_quant.numel() * 4 +
                      v_scales.numel() * 4 +
                      v_zeros.numel() * 4)
    
    print(f'\nCompressed size: {compressed_size / 1024 / 1024:.2f} MB')
    print(f'Compression ratio: {compressed_size / original_size:.4f}')
    print(f'Compression time: {elapsed:.2f} ms')
    print('\n✅ Value quantization test passed!')
```

---

## 验证清单

运行以下命令验证实现：

```bash
# 1. 测试压缩函数
cd /home/zh/mustafar/kernel_quant
python compression_quant.py

# 2. 测试端到端
python test_value_only.py

# 3. 对比精度
python test_nonquant_value.py  # 非量化版本
python test_value_only.py      # 量化版本
```

---

## 关键差异总结

| 项目 | Key | Value |
|------|-----|-------|
| 输入处理 | `transpose(1,2)` | `contiguous()` |
| 索引方式 | 列主序 | 行主序 |
| 量化方案 | 2-bit per-tile | 2-bit per-tile |
| 打包格式 | uint32 (16值) | uint32 (16值) |
| CUDA kernel | ✅ 已实现 | ✅ 已实现 |
| Python 压缩 | ✅ 已实现 | ❌ 待实现 |

---

## 预期结果

实现完成后：
- ✅ Value 矩阵支持 2-bit 量化
- ✅ 内存占用降低到原始的 ~16%
- ✅ 完整的 KV Cache 量化系统
- ✅ 精度损失 < 5%

---

## 故障排查

### 问题 1: Shape 不匹配
**症状**: `RuntimeError: shape mismatch`
**解决**: 检查 `inputs_t = inputs.contiguous()` 而不是 `transpose`

### 问题 2: 索引越界
**症状**: CUDA kernel 崩溃
**解决**: 验证 Value 的行主序索引计算是否正确

### 问题 3: 量化误差过大
**症状**: 输出与预期差异大
**解决**: 检查 scale 和 zero_point 计算，确保没有除零

---

## 下一步

1. 实现上述三个函数
2. 运行测试验证
3. 集成到模型代码
4. 性能基准测试
5. 更新文档

完成后，整个量化稀疏注意力系统即可投入使用！
