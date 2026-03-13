# Nsight Compute 深度 Profiling 指南

## 📋 目标

使用 Nsight Compute 对量化 kernel 进行详细的性能分析，获取关键指标以理解性能瓶颈的根本原因。

---

## 🎯 关键问题

### 问题 1：为什么量化版本的 SpMV 慢 32 倍？

**假设 A：反量化计算开销大**
- 预期：Compute Utilization 高
- 预期：DRAM Bytes Read 少（因为 2-bit 数据小）

**假设 B：访存模式不友好**
- 预期：L2 Cache Hit Rate 低
- 预期：Memory Bandwidth Utilization 低

### 问题 2：量化的访存优势在哪里？

**理论：**
- 量化数据小（2-bit vs FP16）
- 应该减少内存访问量
- 应该提高带宽利用率

**实际：**
- 需要额外读取 scale 和 zero
- 总访存量可能更大
- 需要测量验证

---

## 🔧 Nsight Compute 基础使用

### 安装和环境

```bash
# 检查是否安装
ncu --version

# 如果未安装，通常随 CUDA Toolkit 一起安装
# 位置：/usr/local/cuda/bin/ncu
```

### 基本命令

```bash
# 1. 简单 profiling（所有 kernel）
ncu python your_script.py

# 2. 只 profile 特定 kernel
ncu --kernel-name "Key_Kernel_Quant" python your_script.py

# 3. 指定输出文件
ncu -o profile_output python your_script.py

# 4. 使用 GUI 查看
ncu-ui profile_output.ncu-rep
```

---

## 📊 关键指标详解

### 1. DRAM Bytes Read/Write

**含义：** 从全局内存（DRAM）读取/写入的字节数

**为什么重要：**
- 反映实际的内存访问量
- 量化应该减少这个值（2-bit vs FP16）

**如何测量：**
```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --kernel-name "Key_Kernel_Quant" \
    python benchmark_test.py
```

**预期结果：**
```
无量化版本：
  DRAM Bytes Read: ~X GB (FP16 数据)
  
量化版本：
  DRAM Bytes Read: ~X/8 GB (2-bit 数据 + scale/zero)
  
如果量化版本的 DRAM Bytes Read 更少 → 访存优势存在
如果相近或更多 → scale/zero 访问抵消了优势
```

### 2. L2 Cache Hit Rate

**含义：** L2 缓存命中率

**为什么重要：**
- 高命中率 → 数据局部性好
- 低命中率 → 频繁访问全局内存

**如何测量：**
```bash
ncu --metrics l2_cache_hit_rate \
    --kernel-name "Key_Kernel_Quant" \
    python benchmark_test.py
```

**预期结果：**
```
无量化版本：
  L2 Hit Rate: ~80-90% (连续访问 FP16 数据)
  
量化版本：
  L2 Hit Rate: ~60-70%? (scale/zero 访问可能不连续)
  
如果量化版本命中率更低 → 访存模式不友好
```

### 3. Memory Bandwidth Utilization

**含义：** 内存带宽利用率（实际使用 / 理论峰值）

**为什么重要：**
- 高利用率 → Memory-bound（内存瓶颈）
- 低利用率 → Compute-bound（计算瓶颈）

**如何测量：**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "Key_Kernel_Quant" \
    python benchmark_test.py
```

**预期结果：**
```
无量化版本：
  Bandwidth Utilization: ~70-80% (Memory-bound)
  
量化版本：
  Bandwidth Utilization: ~30-40%? (Compute-bound，因为反量化)
  
如果量化版本利用率更低 → 计算是瓶颈，不是内存
```

### 4. Compute Utilization

**含义：** 计算单元（SM）利用率

**为什么重要：**
- 高利用率 → Compute-bound
- 低利用率 → Memory-bound 或其他瓶颈

**如何测量：**
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "Key_Kernel_Quant" \
    python benchmark_test.py
```

**预期结果：**
```
无量化版本：
  Compute Utilization: ~30-40% (主要是内存访问)
  
量化版本：
  Compute Utilization: ~60-70%? (反量化增加计算)
  
如果量化版本利用率更高 → 反量化计算是瓶颈
```

---

## 🚀 完整的 Profiling 流程

### Step 1: 准备测试脚本

创建一个简单的测试脚本 `profile_kernel.py`：

```python
#!/usr/bin/env python3
"""
用于 Nsight Compute profiling 的简化测试脚本
只运行少量迭代以减少 profiling 时间
"""
import torch
import sys
import os

# 添加路径
sys.path.insert(0, '/home/zh/mustafar/kernel_quant')

import compression_quant
import mustafar_package_quant

# 配置
batch = 1
heads = 32
seq_len = 2048
head_dim = 128
sparsity = 0.5
total_batch_kv = batch * heads

# 准备数据
k_cache = torch.randn(total_batch_kv, seq_len, head_dim, 
                     dtype=torch.float16, device='cuda')
mask = torch.rand_like(k_cache) > sparsity
k_cache_sparse = k_cache * mask

query = torch.randn(batch, heads, 1, head_dim, 
                   dtype=torch.float16, device='cuda')

# Compression
k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
    compression_quant.convert_key_batched_quant(k_cache_sparse)

# Pad query
padded_query = torch.nn.functional.pad(
    query.view(total_batch_kv, -1, head_dim),
    (0, 0, 0, 7),
    mode='constant',
    value=0
)

# Warmup
for _ in range(10):
    _ = mustafar_package_quant.mustafar_key_formulation_quant(
        k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
        padded_query, seq_len, head_dim, total_batch_kv, 1, 2, 16
    )

torch.cuda.synchronize()

# 测试（只运行 5 次，减少 profiling 时间）
for _ in range(5):
    result = mustafar_package_quant.mustafar_key_formulation_quant(
        k_bmps, k_packed_quant, k_tile_offsets, k_scales, k_zeros,
        padded_query, seq_len, head_dim, total_batch_kv, 1, 2, 16
    )

torch.cuda.synchronize()
print("✓ Profiling test completed")
```

### Step 2: 运行基础 Profiling

```bash
cd /home/zh/mustafar/benchmark

# 创建输出目录
mkdir -p profiling_results

# 运行 profiling（只 profile Key_Kernel_Quant）
ncu --kernel-name "Key_Kernel_Quant" \
    --launch-skip 10 \
    --launch-count 5 \
    -o profiling_results/key_kernel_quant_basic \
    python profile_kernel.py
```

**参数说明：**
- `--kernel-name`: 只 profile 指定的 kernel
- `--launch-skip 10`: 跳过前 10 次调用（warmup）
- `--launch-count 5`: 只 profile 5 次调用
- `-o`: 输出文件名

### Step 3: 收集关键指标

```bash
# 收集所有关键指标
ncu --kernel-name "Key_Kernel_Quant" \
    --launch-skip 10 \
    --launch-count 5 \
    --metrics \
        dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        l2_cache_hit_rate,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
        smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    --csv \
    -o profiling_results/key_kernel_quant_metrics \
    python profile_kernel.py > profiling_results/key_kernel_quant_metrics.csv
```

### Step 4: 对比无量化版本

如果有无量化版本的 kernel，也进行 profiling：

```bash
# 修改 profile_kernel.py 使用无量化版本
# 然后运行相同的 profiling

ncu --kernel-name "mustafar_key_formulation" \
    --launch-skip 10 \
    --launch-count 5 \
    --metrics \
        dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        l2_cache_hit_rate,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    -o profiling_results/key_kernel_nonquant_metrics \
    python profile_kernel_nonquant.py > profiling_results/key_kernel_nonquant_metrics.csv
```

### Step 5: 分析结果

```bash
# 查看 CSV 结果
cat profiling_results/key_kernel_quant_metrics.csv

# 或使用 GUI 查看详细报告
ncu-ui profiling_results/key_kernel_quant_metrics.ncu-rep
```

---

## 📊 结果解读

### 场景 1：Memory-Bound（内存瓶颈）

**指标特征：**
```
Memory Bandwidth Utilization: > 70%
Compute Utilization: < 40%
L2 Cache Hit Rate: < 70%
```

**结论：** 内存访问是瓶颈

**优化方向：**
- 使用 Shared Memory 缓存数据
- 优化内存访问模式（合并访问）
- 减少全局内存访问次数

### 场景 2：Compute-Bound（计算瓶颈）

**指标特征：**
```
Memory Bandwidth Utilization: < 50%
Compute Utilization: > 70%
```

**结论：** 计算是瓶颈（反量化开销大）

**优化方向：**
- 减少反量化计算（预计算索引）
- 使用更快的指令（Half 精度）
- 向量化处理

### 场景 3：Latency-Bound（延迟瓶颈）

**指标特征：**
```
Memory Bandwidth Utilization: < 50%
Compute Utilization: < 50%
```

**结论：** 数据依赖或同步问题

**优化方向：**
- 增加并行度
- 减少同步点
- 优化 warp 调度

---

## 🎯 针对量化 Kernel 的具体分析

### 预期的 Profiling 结果

**量化版本（Key_Kernel_Quant）：**
```
DRAM Bytes Read: ~X MB
  - 2-bit 量化数据: ~Y MB (小)
  - Scale/Zero: ~Z MB (额外开销)
  - 总计: 可能比 FP16 少，也可能相近

Memory Bandwidth Utilization: ~30-40%
  - 低于无量化版本
  - 说明不是 Memory-bound

Compute Utilization: ~60-70%
  - 高于无量化版本
  - 说明反量化计算是瓶颈

L2 Cache Hit Rate: ~60-70%
  - 可能低于无量化版本
  - Scale/Zero 访问可能不连续
```

### 关键发现（预期）

1. **访存优势存在但有限**
   - DRAM Bytes Read 减少 2-4x（不是 8x，因为 scale/zero）
   - 但被反量化计算抵消

2. **计算是主要瓶颈**
   - Compute Utilization 高
   - 反量化每个值需要 10-20 cycles

3. **优化方向明确**
   - 优先优化反量化计算
   - 其次优化 scale/zero 访问模式

---

## 📝 完整的测试命令汇总

### 快速测试（推荐）

```bash
cd /home/zh/mustafar/benchmark

# 1. 创建测试脚本（见上面的 profile_kernel.py）

# 2. 运行 profiling
ncu --kernel-name "Key_Kernel_Quant" \
    --launch-skip 10 \
    --launch-count 5 \
    --metrics \
        dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        l2_cache_hit_rate,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    python profile_kernel.py > profiling_results/metrics.csv

# 3. 查看结果
cat profiling_results/metrics.csv
```

### 详细分析（如果有时间）

```bash
# 收集更多指标
ncu --kernel-name "Key_Kernel_Quant" \
    --launch-skip 10 \
    --launch-count 5 \
    --set full \
    -o profiling_results/key_kernel_full \
    python profile_kernel.py

# 使用 GUI 查看
ncu-ui profiling_results/key_kernel_full.ncu-rep
```

---

## ⚠️ 注意事项

### 1. Profiling 会显著降低性能

**原因：**
- Nsight Compute 会插入大量监控代码
- 每个 kernel 调用会慢 10-100x

**建议：**
- 只 profile 少量迭代（5-10 次）
- 使用 `--launch-skip` 跳过 warmup
- 使用 `--launch-count` 限制次数

### 2. 选择合适的指标集

**基础指标（快速）：**
```bash
--metrics dram__bytes_read.sum,dram__bytes_write.sum,l2_cache_hit_rate
```

**完整指标（慢）：**
```bash
--set full
```

**建议：** 先用基础指标快速测试，再用完整指标深入分析

### 3. 对比测试要公平

**确保：**
- 相同的输入数据
- 相同的配置（batch, seq_len, etc.）
- 相同的 GPU 状态（温度、频率）

---

## 📚 参考资料

### Nsight Compute 文档

- [官方文档](https://docs.nvidia.com/nsight-compute/)
- [Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)
- [CLI 参考](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

### 常用指标列表

```bash
# 查看所有可用指标
ncu --query-metrics

# 查看指标说明
ncu --query-metrics-details dram__bytes_read.sum
```

### 推荐的指标组合

**内存分析：**
```
dram__bytes_read.sum
dram__bytes_write.sum
l2_cache_hit_rate
lts__t_sectors_op_read.sum
lts__t_sectors_op_write.sum
```

**计算分析：**
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
```

**带宽分析：**
```
dram__throughput.avg.pct_of_peak_sustained_elapsed
l2_cache_throughput.avg.pct_of_peak_sustained_elapsed
```

---

## 🎯 总结

### 为什么需要 Nsight Compute？

**当前问题：**
- 只知道量化版本慢 32 倍
- 不知道是计算慢还是内存慢
- 不知道访存优势是否存在

**Nsight Compute 能告诉我们：**
1. 实际的内存访问量（DRAM Bytes Read）
2. 内存带宽利用率（是否 Memory-bound）
3. 计算单元利用率（是否 Compute-bound）
4. 缓存命中率（访存模式是否友好）

### 预期结论

**基于理论分析，预期：**
1. 量化版本是 **Compute-bound**（反量化计算瓶颈）
2. 访存优势存在但有限（scale/zero 抵消部分优势）
3. 优化方向：减少反量化计算开销

**但需要 Profiling 验证！**

---

**文档创建时间：** 2026-03-01  
**适用版本：** CUDA 11.0+, Nsight Compute 2020.1+  
**测试环境：** NVIDIA A100, Ubuntu 20.04
