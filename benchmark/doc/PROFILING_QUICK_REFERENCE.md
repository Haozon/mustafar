# Nsight Compute Profiling 快速参考

## 🚀 一键运行命令

```bash
cd /home/zh/mustafar/benchmark

# 快速 profiling（5 分钟）
ncu --kernel-name "Key_Kernel_Quant" \
    --launch-skip 10 \
    --launch-count 5 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,l2_cache_hit_rate,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    python profile_kernel.py > profiling_results/metrics.csv
```

---

## 📊 关键指标速查

| 指标 | 含义 | 好的值 | 坏的值 |
|------|------|--------|--------|
| **DRAM Bytes Read** | 全局内存读取量 | 越少越好 | - |
| **L2 Cache Hit Rate** | L2 缓存命中率 | > 80% | < 60% |
| **Memory Bandwidth Util** | 内存带宽利用率 | < 50% (Compute-bound) | > 80% (Memory-bound) |
| **Compute Utilization** | 计算单元利用率 | > 70% (Compute-bound) | < 30% (Memory-bound) |

---

## 🎯 结果判断

### 如果 Memory Bandwidth Util > 70%
→ **Memory-bound**（内存瓶颈）
→ 优化：Shared Memory, 合并访问

### 如果 Compute Utilization > 70%
→ **Compute-bound**（计算瓶颈）
→ 优化：减少计算，向量化

### 如果两者都 < 50%
→ **Latency-bound**（延迟瓶颈）
→ 优化：增加并行度，减少同步

---

## 📝 测试脚本模板

保存为 `profile_kernel.py`：

```python
import torch
import sys
sys.path.insert(0, '/home/zh/mustafar/kernel_quant')
import compression_quant
import mustafar_package_quant

# 配置
batch, heads, seq_len, head_dim = 1, 32, 2048, 128
total_batch_kv = batch * heads

# 数据
k_cache = torch.randn(total_batch_kv, seq_len, head_dim, 
                     dtype=torch.float16, device='cuda')
mask = torch.rand_like(k_cache) > 0.5
k_cache_sparse = k_cache * mask
query = torch.randn(batch, heads, 1, head_dim, 
                   dtype=torch.float16, device='cuda')

# Compression
k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
    compression_quant.convert_key_batched_quant(k_cache_sparse)

# Pad query
padded_query = torch.nn.functional.pad(
    query.view(total_batch_kv, -1, head_dim),
    (0, 0, 0, 7), mode='constant', value=0
)

# Warmup
for _ in range(10):
    _ = mustafar_package_quant.mustafar_key_formulation_quant(
        k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
        padded_query, seq_len, head_dim, total_batch_kv, 1, 2, 16
    )
torch.cuda.synchronize()

# Test
for _ in range(5):
    result = mustafar_package_quant.mustafar_key_formulation_quant(
        k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
        padded_query, seq_len, head_dim, total_batch_kv, 1, 2, 16
    )
torch.cuda.synchronize()
print("✓ Done")
```

---

## 🔧 常用命令

```bash
# 查看所有可用指标
ncu --query-metrics

# 查看指标详情
ncu --query-metrics-details dram__bytes_read.sum

# 使用 GUI 查看结果
ncu-ui profiling_results/metrics.ncu-rep

# 导出为 CSV
ncu --csv ... > output.csv
```

---

## ⚠️ 注意事项

1. Profiling 会让程序慢 10-100x
2. 只 profile 少量迭代（5-10 次）
3. 使用 `--launch-skip` 跳过 warmup
4. 先用基础指标，再用完整指标

---

**详细文档：** `NSIGHT_COMPUTE_PROFILING_GUIDE.md`
