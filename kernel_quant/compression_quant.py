import torch
import numpy as np
import triton
import triton.language as tl


@triton.jit
def calculate_bitmap_and_scale_key_batched(
    input_ptr,        # [B * num_tiles_per_batch * 64]
    bitmaps_ptr,      # [B * num_tiles_per_batch]
    counts_ptr,       # [B * num_tiles_per_batch]
    scales_ptr,       # [B * num_tiles_per_batch]  # 新增：存储每个 tile 的 scale
    zeros_ptr,        # [B * num_tiles_per_batch]  # 新增：存储每个 tile 的 zero_point
    total_elems: tl.constexpr,
    shifts_ptr,       # [64]
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    block_row = tile_id % N
    block_col = tile_id // N
    base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    vals = tl.load(input_ptr + indices)
    bit_mask = tl.where(vals != 0.0, 1, 0)
    shifts = tl.load(shifts_ptr + offsets)  # shifts_ptr[0:64]
    bitmap = tl.sum(bit_mask * shifts, axis=0)

    cnt = tl.sum(bit_mask, axis=0)
    # 存储 raw 非零元素数量（0..64），Python 层按 bit 计算字节数/偏移

    # 计算 scale 和 zero_point
    # 使用 where 来处理非零值的 min/max，避免动态索引
    # 对于零值，用一个很大/很小的数来避免影响 min/max
    INF = 1e10
    masked_vals_for_min = tl.where(bit_mask != 0, vals, INF)
    masked_vals_for_max = tl.where(bit_mask != 0, vals, -INF)
    
    xmin = tl.min(masked_vals_for_min, axis=0)
    xmax = tl.max(masked_vals_for_max, axis=0)
    
    # 如果 cnt > 0，计算 scale 和 zero_point；否则使用默认值
    has_nonzero = cnt > 0
    if has_nonzero:
        scale = (xmax - xmin) / (2**2 - 1)  # 假设量化为 2-bit
        # 避免除以零
        if scale == 0.0:
            scale = 1.0
        zero_point = tl.floor(-xmin / scale + 0.5)
    else:
        scale = 1.0
        zero_point = 0.0

    # 存储 bitmap、counts、scale 和 zero_point
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    tl.store(bitmaps_ptr + flat_tile_index, bitmap)
    tl.store(counts_ptr + flat_tile_index, cnt)
    tl.store(scales_ptr + flat_tile_index, scale)  # 存储 scale
    tl.store(zeros_ptr + flat_tile_index, zero_point)  # 存储 zero_point

    
@triton.jit
def compress_key_batched(
    input_ptr,          # flattened [B * num_tiles_per_batch * 64]
    bitmaps_ptr,        # flattened [B * num_tiles_per_batch]
    counts_ptr,         # flattened [B * num_tiles_per_batch]  (raw counts per tile)
    tile_offsets_ptr,   # flattened [B * num_tiles_per_batch]  # 每 tile 的全局 uint32 偏移
    packed_not_ptr,     # flattened output buffer (uint32)
    scales_ptr,         # flattened [B * num_tiles_per_batch] float32, per-tile scale
    zeros_ptr,          # flattened [B * num_tiles_per_batch] float32, per-tile zero_point
    bit,  # 量化比特宽度 (2)
    capacity,  # 每 uint32 可容纳量化值数 = 32//bit = 16
    total_elems: tl.constexpr,
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):
    """
    Triton kernel: 对每个 tile(64 lanes)：
      - 读取 bitmap、raw count、per-tile scale/zero_point、tile_uint32_offset
      - 对非零 lane 做量化并打包写入全局 uint32 buffer (packed_not_ptr)
    说明：
      - tile_offsets_ptr 已在 host 侧计算好（每 tile 的起始 uint32 偏移）
      - counts_ptr 是 raw per-tile 非零数量 (0..64)
      - 使用 atomic_or 进行原子写入，避免并发覆盖
      - 每个 uint32 存储 16 个 2-bit 量化值
    """
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # tile -> 64 lanes 索引
    block_row = tile_id % N
    block_col = tile_id // N
    base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    # 读取值（假设完整块）
    vals = tl.load(input_ptr + indices)

    # flat tile index (用于访问 per-tile metadata)
    tiles_per_batch = stride_batch // 64
    flat_tile_index = batch_id * tiles_per_batch + tile_id

    # 读取 per-tile 元数据
    bitmap = tl.load(bitmaps_ptr + flat_tile_index)
    cnt = tl.load(counts_ptr + flat_tile_index)                # raw non-zero count
    tile_uint32_offset = tl.load(tile_offsets_ptr + flat_tile_index)  # uint32 offset (int)

    # 如果没有非零，直接返回，节省计算
    if cnt == 0:
        return

    # 哪些 lane 有有效值（根据 bitmap 的位序：offset 0 对应高位 63）
    shifted = bitmap >> (63 - offsets)
    bit_mask = shifted & 1                # 0/1 per lane
    valid = bit_mask != 0

    # tile 内的顺序索引 gidx 对应每个非零元素（0..cnt-1）
    prefix = tl.cumsum(bit_mask, axis=0) - 1
    gidx = tl.where(valid, prefix, 0)

    # 在 cnt 范围内的 mask，防止 bitmap/计数不一致时越界写入
    cnt_i = tl.cast(cnt, tl.int32)
    within_cnt = gidx < cnt_i
    mask_valid = valid & within_cnt

    # per-tile scale / zero_point（calculate_bitmap_and_scale_key_batched 已按 tile 存储）
    scale = tl.load(scales_ptr + flat_tile_index)
    zero_point = tl.load(zeros_ptr + flat_tile_index)
    # 对标量使用条件表达式而不是 tl.where
    if scale == 0.0:
        scale = 1.0

    # 量化
    maxq = (1 << bit) - 1

    q_float = tl.floor(vals / scale + 0.5) + zero_point
    q_clamped = tl.minimum(tl.maximum(q_float, 0.0), tl.cast(maxq, tl.float32))
    q_int = tl.cast(q_clamped, tl.uint32)    # 每 lane 的量化值

    # 计算目标 uint32 索引与位移（每个 uint32 存 16 个 2-bit 值）
    uint32_idx = tile_uint32_offset + (gidx // capacity)
    bit_shift = (gidx % capacity) * bit

    # 构造要写入的 uint32 值（将 2-bit 量化值放在正确的位置）
    value_to_write = tl.cast(q_int << bit_shift, tl.int32)

    # 使用原子 OR 操作写入，避免并发写入覆盖
    # 多个 lane 可能写入同一个 uint32 的不同位
    # 注意：虽然 buffer 是 int32，但位操作在 int32 和 uint32 上是等价的
    tl.atomic_or(packed_not_ptr + uint32_idx, value_to_write, mask=mask_valid)

def convert_key_batched_quant(inputs: torch.Tensor):
    """
    对 Key Cache 进行稀疏压缩并应用 2-bit per-token-head 量化（Triton kernel 实现打包）。
    返回：
        bitmaps: [B, num_tiles_per_batch] int64
        tile_offsets: [B, num_tiles_per_batch] int32 (uint32 偏移量)
        packed_quant_values: uint32 一维数组（全局打包缓冲）
        scales: [B, num_tiles_per_batch] float32 每个 tile 的缩放因子
        zeros: [B, num_tiles_per_batch] float32 每个 tile 的零点
    """
    # B: batch_size * num_kv_heads
    # M: seq_length
    # N: head_dim
    B, M, N = inputs.shape
    assert inputs.is_cuda
    assert inputs.dim() == 3
    assert M % 64 == 0
    device = inputs.device 
    inputs_t = inputs.transpose(1, 2).contiguous()  # [B, N, M]    
    num_tiles_per_batch = (M * N) // 64
    #total_tiles = B * num_tiles_per_batch

    bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=inputs.device)
    counts  = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=inputs.device)
    scales = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=inputs.device)
    zeros = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=inputs.device)

    # PPrecomputed shifts for bitmap assembly
    shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
    shifts_np = np.left_shift(np.int64(1), shift_amounts)
    const_shifts = torch.tensor(shifts_np, device='cuda')

    #grid = (B, num_tiles_per_batch)
    grid = (num_tiles_per_batch, B) # flip grid to escape tigher limit on y dim. 
    stride_batch = num_tiles_per_batch * 64

    # 1) 计算 bitmap / counts / scale / zero_point（Triton）
    calculate_bitmap_and_scale_key_batched[grid](
        inputs_t.view(-1),
        bitmaps.view(-1),
        counts.view(-1),
        scales.view(-1),
        zeros.view(-1),
        total_elems=B * M * N,
        shifts_ptr=const_shifts,
        stride_batch=stride_batch,
        M=M,
        N=N
    )

    # capacity = 32 // bit (使用 uint32 存储)
    bit = 2
    capacity = 16  # 每个 uint32 能存放多少个量化值 (32 bits / 2 bits = 16)

    '''
    使用 uint32 存储量化值的说明：
    - 每个 uint32 存储 16 个 2-bit 量化值
    - 天然 4 字节对齐，无需额外 padding
    - 支持 Triton 的 atomic_or 操作，避免并发写入覆盖问题
    - 每个 tile 最多需要 4 个 uint32 (64 / 16 = 4)
    '''
    # 每个 tile 需要的 uint32 数量 = ceil(raw_count / capacity)
    # counts 是该 batch 中每个 tile 中非零元素的个数
    units_per_tile = (counts + capacity - 1) // capacity  # [B, T], 单位是 uint32 数量
    
    # 每 batch 总 uint32 数量
    total_units_per_batch = units_per_tile.sum(dim=1).to(torch.int32)  # [B]
    
    # 每 batch 在全局 buffer 的基址（单位：uint32）
    batch_base_offsets = torch.zeros((B,), dtype=torch.int32, device=device)
    if B > 1:
        batch_base_offsets[1:] = torch.cumsum(total_units_per_batch, dim=0)[:-1]

    # 每个 tile 在 batch 内的起始偏移（单位：uint32，row-wise prefix, 左移）
    starts_intra = torch.cumsum(units_per_tile, dim=1)
    starts_intra = torch.cat([torch.zeros((B, 1), dtype=starts_intra.dtype, device=device), starts_intra[:, :-1]], dim=1)

    # 每个 tile 的全局 uint32 起始偏移 (B, T)
    tile_offsets = (batch_base_offsets.unsqueeze(1).to(starts_intra.dtype) + starts_intra).to(torch.int32)

    # 全局总 uint32 数量用于分配 packed buffer
    total_packed_size = int(total_units_per_batch.sum().item()) if total_units_per_batch.numel() > 0 else 0
    # 使用 int32 创建 buffer（PyTorch 没有 uint32 类型）
    # 在位操作层面，int32 和 uint32 是等价的
    packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.int32, device=device)

    # 2) 调用 Triton kernel 将每个 tile 的非零量化并写入全局 packed_not_flat
    if total_packed_size > 0:
        compress_key_batched[grid](
            inputs_t.view(-1),
            bitmaps.view(-1),
            counts.view(-1),
            tile_offsets.contiguous().view(-1),
            packed_not_flat.view(-1),  # 直接使用 int32，Triton 会处理
            scales.view(-1),
            zeros.view(-1),
            bit,
            capacity,
            total_elems=B * M * N,
            stride_batch=stride_batch,
            M=M,
            N=N
        )

    return bitmaps, tile_offsets, packed_not_flat, scales, zeros


if __name__ == '__main__':
    import time
    
    # 简单测试
    torch.manual_seed(42)
    B, M, N = 8, 256, 128
    inputs = torch.randn(B, M, N, dtype=torch.float16, device='cuda')
    # 应用稀疏性
    sparsity = 0.7
    mask = torch.rand(B, M, N, device='cuda') > sparsity
    inputs = inputs * mask.float()
    
    # 计算原始大小（float16 = 2 bytes）
    original_size_bytes = inputs.numel() * 2  # float16 is 2 bytes
    
    print('Testing quantized compression...')
    print(f'Original tensor shape: {inputs.shape}')
    print(f'Original size: {original_size_bytes / 1024 / 1024:.2f} MB')
    print(f'Sparsity: {sparsity * 100:.1f}%')
    print()
    
    # 计时压缩过程
    torch.cuda.synchronize()
    start_time = time.time()
    
    bitmaps, tile_offsets, packed_quant, scales, zeros = convert_key_batched_quant(inputs)
    torch.cuda.synchronize()
    end_time = time.time()
    compression_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 计算压缩后的大小
    bitmaps_size = bitmaps.numel() * 8  # int64 = 8 bytes
    tile_offsets_size = tile_offsets.numel() * 4  # int32 = 4 bytes
    packed_quant_size = packed_quant.numel() * 4  # uint32 = 4 bytes
    scales_size = scales.numel() * 4  # float32 = 4 bytes
    zeros_size = zeros.numel() * 4  # float32 = 4 bytes
    
    compressed_size_bytes = bitmaps_size + tile_offsets_size + packed_quant_size + scales_size + zeros_size
    
    # 打印结果
    print('=== 压缩结果 ===')
    print(f'Bitmaps shape: {bitmaps.shape}, size: {bitmaps_size / 1024:.2f} KB')
    print(f'Tile offsets shape: {tile_offsets.shape}, size: {tile_offsets_size / 1024:.2f} KB')
    print(f'Packed quant values shape: {packed_quant.shape}, size: {packed_quant_size / 1024:.2f} KB')
    print(f'Scales shape: {scales.shape}, size: {scales_size / 1024:.2f} KB')
    print(f'Zeros shape: {zeros.shape}, size: {zeros_size / 1024:.2f} KB')
    print()
    
    print('=== 存储占比统计 ===')
    print(f'原始大小: {original_size_bytes / 1024 / 1024:.2f} MB')
    print(f'压缩后大小: {compressed_size_bytes / 1024 / 1024:.2f} MB')
    compression_ratio = compressed_size_bytes / original_size_bytes
    print(f'压缩比: {compression_ratio:.4f} ({(1 - compression_ratio) * 100:.2f}% 节省)')
    print()
    
    print('=== 时间统计 ===')
    print(f'压缩耗时: {compression_time:.2f} ms')
    print(f'吞吐量: {original_size_bytes / 1024 / 1024 / (compression_time / 1000):.2f} MB/s')