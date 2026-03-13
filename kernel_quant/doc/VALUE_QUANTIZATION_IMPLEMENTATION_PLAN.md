# Value 量化实现技术方案

## 文档信息

- **创建日期**: 2026-02-06
- **目标**: 完成 Value 矩阵的量化压缩实现
- **状态**: 规划阶段

---

## 1. 当前实现状况分析

### 1.1 已完成的模块

#### ✅ Key 矩阵量化（完整实现）

**Python 层 (compression_quant.py)**:
- `calculate_bitmap_and_scale_key_batched`: 计算 bitmap、counts、scale、zero_point
- `compress_key_batched`: 量化并打包 Key 数据到 uint32
- `convert_key_batched_quant`: 完整的 Key 压缩接口

**CUDA 层 (csrc/)**:
- `Key_Kernel_Quant`: Key 矩阵的量化稀疏矩阵乘法
- `SpMM_CopyFromGlobalToReg_Quant`: 加载量化数据到寄存器
- `SpMM_DecompressFromRegisterToShared_Quant`: 解压并反量化到共享内存
- `Key_SplitK_API_Quant`: Key 矩阵 API 接口

**Python 绑定**:
- `mustafar_key_formulation_quant`: Key 前向传播
- `mustafar_quant_sparse_forward`: 别名接口

#### ✅ Value 矩阵量化（CUDA 层已实现）

**CUDA 层 (csrc/)**:
- `Value_Kernel_Quant`: Value 矩阵的量化稀疏矩阵乘法（已实现）
- `Value_SplitK_API_Quant`: Value 矩阵 API 接口（已实现）

**Python 绑定**:
- `mustafar_value_formulation_quant`: Value 前向传播（已实现）
- `mustafar_quant_sparse_value_forward`: 别名接口（已实现）

### 1.2 缺失的模块

#### ❌ Value 矩阵量化（Python 压缩层未实现）

**Python 层 (compression_quant.py)** - **需要实现**:
- `calculate_bitmap_and_scale_value_batched`: 计算 Value 的 bitmap 和量化参数
- `compress_value_batched`: 量化并打包 Value 数据
- `convert_value_batched_quant`: 完整的 Value 压缩接口

---

## 2. Key vs Value 的关键差异

### 2.1 矩阵布局差异

```
Key 矩阵:  [B, M, N] -> transpose -> [B, N, M] (列主序访问)
Value 矩阵: [B, M, N] -> contiguous -> [B, M, N] (行主序访问)
```

### 2.2 索引计算差异

#### Key 矩阵索引（列主序）
```python
block_row = tile_id % N
block_col = tile_id // N
base_idx = batch_id * stride_batch + block_row * M + block_col * 64
```

#### Value 矩阵索引（行主序）
```python
tiles_per_row = N // 64
tiles_per_block = tiles_per_row * 64
block_idx = tile_id // tiles_per_block
rem = tile_id % tiles_per_block
col_tile = rem // 64
r_in_block = rem % 64
row = block_idx * 64 + r_in_block
col_start = col_tile * 64
base_idx = batch_id * stride_batch + row * N + col_start
```

### 2.3 相同的量化方案

- **量化位宽**: 2-bit per value
- **量化方法**: Per-tile min-max 量化
- **打包格式**: 16 个 2-bit 值打包到 1 个 uint32
- **数据结构**: 
  - `bitmaps`: [B, num_tiles] int64
  - `tile_offsets`: [B, num_tiles] int32 (uint32 偏移)
  - `packed_quant`: int32 一维数组
  - `scales`: [B, num_tiles] float32
  - `zeros`: [B, num_tiles] float32

---

## 3. 实现方案

### 3.1 需要添加的函数

#### 函数 1: `calculate_bitmap_and_scale_value_batched`

**功能**: 计算 Value 矩阵的 bitmap、counts、scale、zero_point

**输入参数**:
```python
input_ptr        # [B * num_tiles_per_batch * 64] float16
bitmaps_ptr      # [B * num_tiles_per_batch] int64 (输出)
counts_ptr       # [B * num_tiles_per_batch] int32 (输出)
scales_ptr       # [B * num_tiles_per_batch] float32 (输出)
zeros_ptr        # [B * num_tiles_per_batch] float32 (输出)
total_elems      # 总元素数
shifts_ptr       # [64] 预计算的位移量
stride_batch     # num_tiles_per_batch * 64
M, N             # 矩阵维度
```

**核心逻辑**:
1. 使用 Value 的行主序索引计算
2. 加载 64 个元素
3. 计算 bitmap（非零位置）
4. 计算非零元素数量
5. 计算 per-tile 的 min/max
6. 计算 scale 和 zero_point
7. 存储结果

**关键代码片段**:
```python
@triton.jit
def calculate_bitmap_and_scale_value_batched(...):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Value 矩阵的行主序索引
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
    
    # 计算 bitmap
    shifts = tl.load(shifts_ptr + offsets)
    bitmap = tl.sum(bit_mask * shifts, axis=0)
    cnt = tl.sum(bit_mask, axis=0)

    # 计算量化参数
    INF = 1e10
    masked_vals_for_min = tl.where(bit_mask != 0, vals, INF)
    masked_vals_for_max = tl.where(bit_mask != 0, vals, -INF)
    
    xmin = tl.min(masked_vals_for_min, axis=0)
    xmax = tl.max(masked_vals_for_max, axis=0)
    
    has_nonzero = cnt > 0
    if has_nonzero:
        scale = (xmax - xmin) / (2**2 - 1)
        if scale == 0.0:
            scale = 1.0
        zero_point = tl.floor(-xmin / scale + 0.5)
    else:
        scale = 1.0
        zero_point = 0.0

    # 存储结果
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    tl.store(bitmaps_ptr + flat_tile_index, bitmap)
    tl.store(counts_ptr + flat_tile_index, cnt)
    tl.store(scales_ptr + flat_tile_index, scale)
    tl.store(zeros_ptr + flat_tile_index, zero_point)
```

#### 函数 2: `compress_value_batched`

**功能**: 量化并打包 Value 矩阵数据

**输入参数**:
```python
input_ptr          # [B * num_tiles_per_batch * 64] float16
bitmaps_ptr        # [B * num_tiles_per_batch] int64
counts_ptr         # [B * num_tiles_per_batch] int32
tile_offsets_ptr   # [B * num_tiles_per_batch] int32
packed_not_ptr     # 输出缓冲区 (int32)
scales_ptr         # [B * num_tiles_per_batch] float32
zeros_ptr          # [B * num_tiles_per_batch] float32
bit                # 量化位宽 (2)
capacity           # 每 uint32 容纳的量化值数 (16)
total_elems        # 总元素数
stride_batch       # num_tiles_per_batch * 64
M, N               # 矩阵维度
```

**核心逻辑**:
1. 使用 Value 的行主序索引计算
2. 加载 64 个元素和对应的 bitmap
3. 加载 scale 和 zero_point
4. 对非零元素进行量化
5. 计算打包位置
6. 使用 atomic_or 写入打包数据

**关键代码片段**:
```python
@triton.jit
def compress_value_batched(...):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Value 矩阵的行主序索引（与上面相同）
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

    # 提取非零位置
    shifted = bitmap >> (63 - offsets)
    bit_mask = shifted & 1
    valid = bit_mask != 0

    prefix = tl.cumsum(bit_mask, axis=0) - 1
    gidx = tl.where(valid, prefix, 0)

    cnt_i = tl.cast(cnt, tl.int32)
    within_cnt = gidx < cnt_i
    mask_valid = valid & within_cnt

    # 加载量化参数
    scale = tl.load(scales_ptr + flat_tile_index)
    zero_point = tl.load(zeros_ptr + flat_tile_index)
    if scale == 0.0:
        scale = 1.0

    # 量化
    maxq = (1 << bit) - 1
    q_float = tl.floor(vals / scale + 0.5) + zero_point
    q_clamped = tl.minimum(tl.maximum(q_float, 0.0), tl.cast(maxq, tl.float32))
    q_int = tl.cast(q_clamped, tl.uint32)

    # 计算打包位置
    tile_uint32_offset = tl.load(tile_offsets_ptr + flat_tile_index)
    uint32_idx = tile_uint32_offset + (gidx // capacity)
    bit_shift = (gidx % capacity) * bit
    value_to_write = tl.cast(q_int << bit_shift, tl.int32)

    # 原子写入
    tl.atomic_or(packed_not_ptr + uint32_idx, value_to_write, mask=mask_valid)
```

#### 函数 3: `convert_value_batched_quant`

**功能**: 完整的 Value 压缩接口

**函数签名**:
```python
def convert_value_batched_quant(inputs: torch.Tensor) -> Tuple[
    torch.Tensor,  # bitmaps: [B, num_tiles] int64
    torch.Tensor,  # tile_offsets: [B, num_tiles] int32
    torch.Tensor,  # packed_quant: [total_size] int32
    torch.Tensor,  # scales: [B, num_tiles] float32
    torch.Tensor   # zeros: [B, num_tiles] float32
]:
```

**输入**:
- `inputs`: [B, M, N] float16，Value 矩阵

**输出**:
- `bitmaps`: [B, num_tiles_per_batch] int64，每个 tile 的非零位图
- `tile_offsets`: [B, num_tiles_per_batch] int32，每个 tile 在打包数组中的 uint32 偏移
- `packed_quant_values`: int32 一维数组，打包的量化值
- `scales`: [B, num_tiles_per_batch] float32，每个 tile 的缩放因子
- `zeros`: [B, num_tiles_per_batch] float32，每个 tile 的零点

**实现流程**:
```python
def convert_value_batched_quant(inputs: torch.Tensor):
    B, M, N = inputs.shape
    assert inputs.is_cuda
    assert inputs.dim() == 3
    assert M % 64 == 0
    device = inputs.device
    
    # Value 矩阵不需要转置
    inputs_t = inputs.contiguous()
    num_tiles_per_batch = (M * N) // 64

    # 分配输出缓冲区
    bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=device)
    counts = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=device)
    scales = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=device)
    zeros = torch.empty((B, num_tiles_per_batch), dtype=torch.float, device=device)

    # 预计算位移量
    shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
    shifts_np = np.left_shift(np.int64(1), shift_amounts)
    const_shifts = torch.tensor(shifts_np, device='cuda')

    grid = (num_tiles_per_batch, B)
    stride_batch = num_tiles_per_batch * 64

    # 步骤 1: 计算 bitmap / counts / scale / zero_point
    calculate_bitmap_and_scale_value_batched[grid](
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

    bit = 2
    capacity = 16  # 每个 uint32 存 16 个 2-bit 值

    # 步骤 2: 计算 tile_offsets
    units_per_tile = (counts + capacity - 1) // capacity  # 每个 tile 需要的 uint32 数
    total_units_per_batch = units_per_tile.sum(dim=1).to(torch.int32)
    
    # 计算每个 batch 的基址偏移
    batch_base_offsets = torch.zeros((B,), dtype=torch.int32, device=device)
    if B > 1:
        batch_base_offsets[1:] = torch.cumsum(total_units_per_batch, dim=0)[:-1]

    # 计算每个 tile 在 batch 内的偏移
    starts_intra = torch.cumsum(units_per_tile, dim=1)
    starts_intra = torch.cat([
        torch.zeros((B, 1), dtype=starts_intra.dtype, device=device), 
        starts_intra[:, :-1]
    ], dim=1)

    # 全局 tile_offsets
    tile_offsets = (batch_base_offsets.unsqueeze(1).to(starts_intra.dtype) + 
                    starts_intra).to(torch.int32)

    # 步骤 3: 分配打包缓冲区
    total_packed_size = int(total_units_per_batch.sum().item()) \
                        if total_units_per_batch.numel() > 0 else 0
    packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.int32, device=device)

    # 步骤 4: 压缩并量化
    if total_packed_size > 0:
        compress_value_batched[grid](
            inputs_t.view(-1),
            bitmaps.view(-1),
            counts.view(-1),
            tile_offsets.contiguous().view(-1),
            packed_not_flat.view(-1),
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
```

---

## 4. 集成到模型

### 4.1 修改 `llama_mustafar_quant_kernel.py`

在 prefill 阶段，将 Value 压缩从非量化版本切换到量化版本：

```python
# 当前代码（使用非量化压缩）
v_bmps, v_idxs, v_nzs = compression.convert_value_batched(
    value_states[:, :, :(compressed_length), :].reshape(total_batch_kv, -1, self.head_dim)
)

# 修改为（使用量化压缩）
from kernel_quant.compression_quant import convert_value_batched_quant

v_bmps, v_tile_offsets, v_nzs, v_scales, v_zeros = convert_value_batched_quant(
    value_states[:, :, :(compressed_length), :].reshape(total_batch_kv, -1, self.head_dim)
)
```

### 4.2 更新数据结构

确保 `v_compressed` 包含正确的量化参数：

```python
# 旧格式（非量化）
v_compressed = [v_bmps, v_idxs, v_nzs, v_nz_offset]

# 新格式（量化）
v_compressed = [v_bmps, v_tile_offsets, v_nzs, v_nz_offset, v_scales, v_zeros]
```

---

## 5. 测试验证

### 5.1 单元测试

创建 `test_value_quant_compression.py`:

```python
import torch
from kernel_quant.compression_quant import convert_value_batched_quant

def test_value_compression():
    # 测试参数
    B, M, N = 8, 256, 128
    inputs = torch.randn(B, M, N, dtype=torch.float16, device='cuda')
    
    # 应用稀疏性
    sparsity = 0.7
    mask = torch.rand(B, M, N, device='cuda') > sparsity
    inputs = inputs * mask.float()
    
    # 压缩
    bitmaps, tile_offsets, packed_quant, scales, zeros = \
        convert_value_batched_quant(inputs)
    
    # 验证形状
    num_tiles = (M * N) // 64
    assert bitmaps.shape == (B, num_tiles)
    assert tile_offsets.shape == (B, num_tiles)
    assert scales.shape == (B, num_tiles)
    assert zeros.shape == (B, num_tiles)
    
    print("✅ Value compression test passed!")

if __name__ == '__main__':
    test_value_compression()
```

### 5.2 端到端测试

修改 `test_value_only.py` 使用量化压缩：

```python
from kernel_quant.compression_quant import convert_value_batched_quant
import mustafar_package_quant

# 创建测试数据
value_cache = torch.randn(B, M, N, dtype=torch.float16, device='cuda')

# 压缩
bitmaps, tile_offsets, packed_quant, scales, zeros = \
    convert_value_batched_quant(value_cache)

# 调用 kernel
output = mustafar_package_quant.mustafar_quant_sparse_value_forward(
    bitmaps, packed_quant, tile_offsets, scales, zeros,
    attention_scores, Reduction_Workspace,
    model_dim, compressed_length, batch_size,
    num_key_value_groups, 2, 16
)
```

### 5.3 精度验证

对比量化前后的输出差异：

```python
def test_quantization_accuracy():
    # 非量化版本
    from kernel.compression import convert_value_batched
    v_bmps_fp16, v_idxs_fp16, v_nzs_fp16 = convert_value_batched(value_cache)
    output_fp16 = mustafar_package.mustafar_value_formulation(...)
    
    # 量化版本
    v_bmps_q, v_offsets_q, v_nzs_q, v_scales, v_zeros = \
        convert_value_batched_quant(value_cache)
    output_quant = mustafar_package_quant.mustafar_quant_sparse_value_forward(...)
    
    # 计算误差
    mse = torch.mean((output_fp16 - output_quant) ** 2)
    relative_error = mse / torch.mean(output_fp16 ** 2)
    
    print(f"MSE: {mse.item():.6f}")
    print(f"Relative Error: {relative_error.item():.6f}")
    
    assert relative_error < 0.05, "Quantization error too large!"
```

---

## 6. 性能优化建议

### 6.1 内存优化

- 使用 `torch.cuda.empty_cache()` 及时释放中间变量
- 考虑使用 in-place 操作减少内存分配

### 6.2 计算优化

- 确保 Triton kernel 的 grid 配置合理
- 使用 `@triton.autotune` 自动调优 block size

### 6.3 数值稳定性

- 在量化时添加 epsilon 避免除零
- 使用 clamp 确保量化值在有效范围内

---

## 7. 实现检查清单

- [ ] 实现 `calculate_bitmap_and_scale_value_batched`
- [ ] 实现 `compress_value_batched`
- [ ] 实现 `convert_value_batched_quant`
- [ ] 添加单元测试
- [ ] 集成到模型代码
- [ ] 端到端测试
- [ ] 精度验证
- [ ] 性能基准测试
- [ ] 文档更新

---

## 8. 预期效果

### 8.1 内存节省

以 `[8, 256, 128]` 的 Value Cache 为例（70% 稀疏度）：

| 方案 | 内存占用 | 压缩比 |
|------|----------|--------|
| 原始 float16 | 0.50 MB | 100% |
| 稀疏压缩 (float16) | 0.15 MB | 30% |
| 量化压缩 (2-bit) | 0.08 MB | 16% |

### 8.2 精度损失

- 2-bit 量化引入的量化误差
- 预期相对误差 < 5%
- 对最终输出的影响较小

### 8.3 性能影响

- 解压开销：~10-20% 额外延迟
- 内存带宽节省：~6x
- 适用场景：内存受限的长序列推理

---

## 9. 参考资料

### 9.1 相关文件

- `kernel_quant/compression_quant.py`: Key 量化压缩实现（参考）
- `kernel/compression.py`: 非量化压缩实现（参考）
- `kernel_quant/csrc/SpMM_Kernel_Quant.cuh`: CUDA kernel 实现
- `models/llama_mustafar_quant_kernel.py`: 模型集成

### 9.2 技术文档

- `kernel_quant/README.md`: 量化内核总体说明
- `kernel_quant/doc/COMPRESSION_QUANT_TECHNICAL_REPORT.md`: 量化技术报告

---

## 10. 附录

### 10.1 数据格式说明

#### Bitmap 格式
- 类型: int64 (uint64)
- 含义: 64-bit 位图，bit=1 表示该位置非零
- 位序: MSB 对应 offset 0，LSB 对应 offset 63

#### Tile Offsets 格式
- 类型: int32 (uint32)
- 含义: 每个 tile 在 packed_quant 数组中的起始 uint32 索引
- 单位: uint32 数量（不是字节）

#### Packed Quant 格式
- 类型: int32 (作为 uint32 使用)
- 含义: 打包的 2-bit 量化值
- 布局: 每个 uint32 存储 16 个 2-bit 值
  ```
  uint32: [q15|q14|...|q1|q0]
  每个 qi 占 2 bits
  ```

#### Scale/Zero 格式
- 类型: float32
- 含义: Per-tile 量化参数
- 反量化公式: `value = (q - zero_point) * scale`

### 10.2 常见问题

**Q: 为什么使用 int32 而不是 uint32？**
A: PyTorch 没有 uint32 类型，使用 int32 在位操作层面是等价的。

**Q: 为什么 Value 不需要转置？**
A: Value 矩阵在注意力计算中是按行访问的，不需要转置。Key 矩阵需要转置是因为要计算 Q @ K^T。

**Q: tile_offsets 和 accum_counts 有什么区别？**
A: 
- `accum_counts`: 非量化版本使用，存储累积的 float16 数量
- `tile_offsets`: 量化版本使用，存储累积的 uint32 数量（偏移）

**Q: 如何验证实现正确性？**
A: 
1. 检查输出形状是否正确
2. 对比量化前后的数值误差
3. 与非量化版本的输出对比
4. 检查内存占用是否符合预期

---

## 结论

Value 矩阵的量化实现方案已经明确，主要工作是在 Python 层添加三个 Triton kernel 函数。CUDA 层的计算核心已经完整实现，只需要完成数据压缩部分即可实现完整的量化稀疏注意力系统。

预期实现后可以获得：
- **6x 内存节省**（相比非量化稀疏）
- **< 5% 精度损失**
- **完整的 KV Cache 量化压缩系统**
