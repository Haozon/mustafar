# Compression 模块计时差异分析（Sparse vs Quant）

## 1. 问题背景

当前对比的是两个 Python 接口函数的耗时：

- `compression.convert_key_batched`
- `compression_quant.convert_key_batched_quant`

观察现象是：

1. Quant 压缩时间显著低于 Sparse；
2. 在 Small/Medium/Large 下，Quant 时间变化不大（例如约 `9.66 -> 9.61 -> 10.32 ms`）。

该现象看起来“过好”，需要确认是否是测试脚本问题，或实现路径差异导致。

---

## 2. 关键结论（先给结论）

这个结果在当前实现下是**可解释且可复现**的，主要原因不是“算错”，而是：

1. 两条路径并非完全同构输出格式，Sparse 有额外后处理和搬运；
2. Quant 的调用中固定开销（分配/常量构造）占比较大，掩盖了规模增长；
3. 当前 benchmark 是 API 端到端计时，而非纯 kernel microbenchmark。

因此，Quant 显著更快且“随规模变化不大”是有工程原因的，不是单纯异常值。

---

## 3. 代码级证据

### 3.1 Sparse 路径有对齐填充（会放大写回量）

来源：`kernel/compression.py:48`

```python
cnt = tl.sum(bit_mask, axis=0)
#padding happens here.
cnt = ((cnt + 7) & ~7) >> 1  # padded to nearest multiple of 8, then halved
```

这一步使 tile 计数按规则对齐，后续写回量会比“纯 nnz 数”更大。

---

### 3.2 Sparse 路径有显式后处理/搬运（cumsum + 切片 + clone）

来源：`kernel/compression.py:297-338`

```python
accum_counts = torch.cumsum(counts, dim=1).to(torch.int32)
accum_counts = torch.cat([
    torch.zeros((B, 1), dtype=counts.dtype, device=counts.device),
    accum_counts
], dim=1).contiguous()

total = 2 * accum_counts[:, -1]
offsets = torch.cumsum(total, dim=0)
batch_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=inputs.device), offsets[:-1]])
total_packed_size = offsets[-1].item()
packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.float16, device=inputs.device)

# Step 2: Slice `packed_not_flat` into per-batch tensors
start_offsets = torch.zeros_like(offsets)
start_offsets[1:] = offsets[:-1]
packed_not_batched = []
for b in range(B):
    start = start_offsets[b].item()
    end = offsets[b].item()
    packed_not_batched.append(packed_not_flat[start:end].clone())
```

这里的 `item()`（同步点）和 per-batch `clone()`（额外复制）会增加 API 端到端耗时，且规模大时更明显。

---

### 3.3 Quant 路径返回平铺结构，不做 per-batch clone

来源：`kernel_quant/compression_quant.py:222-263`

```python
units_per_tile = (counts + capacity - 1) // capacity
total_units_per_batch = units_per_tile.sum(dim=1).to(torch.int32)

starts_intra = torch.cumsum(units_per_tile, dim=1)
starts_intra = torch.cat([torch.zeros((B, 1), dtype=starts_intra.dtype, device=device), starts_intra[:, :-1]], dim=1)
tile_offsets = (batch_base_offsets.unsqueeze(1).to(starts_intra.dtype) + starts_intra).to(torch.int32)

total_packed_size = int(total_units_per_batch.sum().item()) if total_units_per_batch.numel() > 0 else 0
packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.int32, device=device)

return bitmaps, tile_offsets, packed_not_flat, scales, zeros
```

Quant 的输出是 `flat buffer + offsets`，避免了 Sparse 那类“拆 batch + clone 列表”的额外路径。

---

### 3.4 Quant 中有明显固定开销（解释为何不同规模差距小）

来源：`kernel_quant/compression_quant.py:181-190`, `:243`

```python
bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=inputs.device)
counts  = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=inputs.device)
scales = torch.empty((B, num_tiles_per_batch), dtype=torch.float16, device=inputs.device)
zeros = torch.empty((B, num_tiles_per_batch), dtype=torch.float16, device=inputs.device)

shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
shifts_np = np.left_shift(np.int64(1), shift_amounts)
const_shifts = torch.tensor(shifts_np, device='cuda')

packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.int32, device=device)
```

这部分每次调用都会执行，属于“固定成本”。当输入规模从 Small 到 Large 增长时，总时间会被固定成本部分压平。

---

## 4. 实测现象（median 口径）

### 4.1 Small/Medium/Large（50% 稀疏）

一次复测结果（median）：

- Sparse: `25.3348 -> 78.8055 -> 130.6061 ms`
- Quant:  `9.4797 -> 9.6635 -> 10.1581 ms`

说明：

1. Sparse 随规模上升明显；
2. Quant 增长较小，但不是不增长。

---

### 4.2 Batch sweep（heads=32, seq=2048, dim=128）

复测结果（median）：

- `batch=1`：Sparse `137.888 ms`，Quant `10.092 ms`
- `batch=2`：Sparse `214.382 ms`，Quant `11.043 ms`
- `batch=4`：Sparse `489.837 ms`，Quant `12.337 ms`

说明 Quant 也随 batch 增长，但由于固定开销占比高，斜率显著小于 Sparse。

---

## 5. “测试脚本有没有问题”的结论

结论分两层：

1. **没有明显“算错”**：函数调用、CUDA Event 计时流程是正常的；
2. **存在口径差异**：这是 API 端到端时间，不是纯 kernel 公平对比。

因此，结果“看起来非常好”并不必然代表问题，而是当前实现差异（输出组织 + 后处理 + 固定成本）被完整计入后的真实表现。

---

## 6. 建议的报告口径（避免误解）

建议同时报告两组时间：

1. `API end-to-end`（当前口径）：反映实际调用成本；
2. `Kernel-only`（拆分口径）：只统计 Triton 核心计算段。

并在图表说明中明确：

- “当前图表为 Compression API 时间，不等同于纯 kernel 时间”。

---

## 7. 附：可复现实验命令

```bash
# 固定口径重跑 compression benchmark
python kernel_bench/benchmark_compression_detailed.py \
  --num-warmup 20 \
  --num-iters 50 \
  --report-stat median \
  --seed 20260304

# 根据该次 json 生成图表（避免硬编码旧数据）
python kernel_bench/plot_compression_results.py --input <compression_detailed_xxx.json>
```

