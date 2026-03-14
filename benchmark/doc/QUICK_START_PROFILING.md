# 快速开始：量化 Kernel Profiling

## 🎯 目标

找出量化版本 TPOT 慢 55% 的根本原因，并开始优化。

---

## 📊 当前性能数据

| 配置 | TTFT | TPOT | 总时间 | 内存 |
|------|------|------|--------|------|
| Mustafar 50% | 6798 ms | **53.41 ms** ✅ | 61437 ms | 44.72 GB |
| Mustafar-Quant 50% | 3469 ms ✅ | **82.78 ms** ❌ | 88152 ms | 41.28 GB ✅ |

**问题：量化版本的 TPOT 慢了 55%（82.78 ms vs 53.41 ms）**

---

## 🚀 立即执行的命令

### Step 1: 进入工作目录

```bash
cd /home/zh/mustafar/benchmark
conda activate mustar
```

### Step 2: 运行详细 Profiling（量化版本）

```bash
# 运行 Nsys profiling（约 5-10 分钟）
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    -o quant_detailed_profile \
    python mem_spd_test_quant.py

echo "✅ Profiling 完成！"
```

### Step 3: 导出 Kernel 统计

```bash
# 导出 kernel 执行时间统计
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output quant_kernel_stats.csv

# 导出内存访问统计
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_mem_time_sum \
    --format csv \
    --output quant_memory_stats.csv

# 导出 GPU 指标统计
nsys stats quant_detailed_profile.nsys-rep \
    --report cuda_gpu_trace \
    --format csv \
    --output quant_gpu_trace.csv

echo "✅ 统计数据导出完成！"
```

### Step 4: 查看关键指标

```bash
# 查看量化 kernel 的执行时间
echo "=== Key_Kernel_Quant 统计 ==="
grep "Key_Kernel_Quant" quant_kernel_stats.csv | head -5

echo ""
echo "=== Value_Kernel_Quant 统计 ==="
grep "Value_Kernel_Quant" quant_kernel_stats.csv | head -5

# 查看内存带宽利用率
echo ""
echo "=== 内存访问统计 ==="
head -20 quant_memory_stats.csv
```

### Step 5: 对比无量化版本（可选）

```bash
# Profile 无量化版本
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    -o non_quant_detailed_profile \
    python mem_spd_test.py

# 导出统计
nsys stats non_quant_detailed_profile.nsys-rep \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output non_quant_kernel_stats.csv

echo "✅ 无量化版本 Profiling 完成！"
```

---

## 📈 分析 Profiling 结果

### 关键指标说明

1. **Total Time (ns)** - Kernel 总执行时间
   - 量化版本应该更长（因为有反量化开销）

2. **Avg Time (ns)** - 平均每次调用时间
   - 这是最重要的指标
   - 目标：找出为什么量化版本慢

3. **Instances** - 调用次数
   - 量化和无量化应该相同（都是 229,440 次）

4. **Memory Bandwidth (GB/s)** - 内存带宽
   - 量化版本可能更低（因为数据更小）
   - 但反量化增加了计算

5. **Compute Utilization (%)** - 计算单元利用率
   - 量化版本可能更高（反量化计算）

### 预期发现

**如果是 Memory-Bound（内存瓶颈）：**
- Memory Bandwidth Utilization > 80%
- Compute Utilization < 50%
- 优化方向：减少内存访问，使用共享内存

**如果是 Compute-Bound（计算瓶颈）：**
- Compute Utilization > 80%
- Memory Bandwidth Utilization < 50%
- 优化方向：优化反量化计算，减少指令数

**如果是 Latency-Bound（延迟瓶颈）：**
- 两者都不高（< 50%）
- 优化方向：优化数据依赖，增加并行度

---

## 🔍 深入分析（使用 Nsys UI）

如果需要更详细的分析，可以使用 Nsys UI：

```bash
# 在本地机器上（如果有 GUI）
nsys-ui quant_detailed_profile.nsys-rep

# 或者导出到本地查看
# 1. 将 .nsys-rep 文件下载到本地
# 2. 在本地安装 Nsys
# 3. 使用 nsys-ui 打开
```

**在 Nsys UI 中查看：**
1. Timeline 视图 - 查看 kernel 执行时间线
2. GPU Metrics - 查看 SM 利用率、内存带宽
3. Kernel Details - 查看每个 kernel 的详细指标
4. Memory Operations - 查看内存访问模式

---

## 📝 记录分析结果

创建一个分析报告：

```bash
cat > profiling_analysis_results.txt << 'EOF'
# 量化 Kernel Profiling 分析结果

## 执行时间
- Key_Kernel_Quant 平均时间: ___ ms
- Value_Kernel_Quant 平均时间: ___ ms
- 总时间占比: ___ %

## 内存指标
- Memory Bandwidth Utilization: ___ %
- Memory Throughput: ___ GB/s
- L2 Cache Hit Rate: ___ %

## 计算指标
- Compute Utilization: ___ %
- SM Efficiency: ___ %
- Occupancy: ___ %

## 瓶颈分析
主要瓶颈是：[ ] Memory-Bound  [ ] Compute-Bound  [ ] Latency-Bound

原因：
___

## 优化建议
1. ___
2. ___
3. ___

EOF

echo "✅ 请填写 profiling_analysis_results.txt"
```

---

## 🎯 下一步行动

根据 Profiling 结果，选择优化方案：

### 如果是 Compute-Bound（最可能）

**立即实施：优化反量化计算**

修改文件：`kernel_quant/csrc/SpMM_Kernel_Quant.cuh`

**优化 1：预计算索引（预期快 1.3-1.5x）**

在 `SpMM_DecompressFromRegisterToShared_Quant` 函数中：

```cuda
// 当前代码（慢）：
int unit_idx = j / capacity;           // 整数除法
int bit_offset = (j % capacity) * bit; // 整数取模

// 优化后（快）：
constexpr int unit_indices[64] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  // 0-15
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  // 16-31
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  // 32-47
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3   // 48-63
};
constexpr int bit_offsets[64] = {
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // 0-15
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // 16-31
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,  // 32-47
    0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30   // 48-63
};

int unit_idx = unit_indices[j];    // 查表
int bit_offset = bit_offsets[j];   // 查表
```

**优化 2：使用 Half 精度（预期快 1.1-1.2x）**

```cuda
// 当前代码（慢）：
float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
SharedPTR[output_idx] = __float2half(dequant_value);

// 优化后（快）：
half scale_h = __float2half(scale);
half zero_h = __float2half(zero_point);
half q_h = __int2half_rn(q_value);
half dequant_h = __hmul(__hsub(q_h, zero_h), scale_h);
SharedPTR[output_idx] = dequant_h;
```

### 如果是 Memory-Bound

**立即实施：优化内存访问**

```cuda
// 使用共享内存缓存量化数据
__shared__ uint32_t shared_quant[MAX_UNITS];

// 协作加载
if (threadIdx.x < num_units) {
    shared_quant[threadIdx.x] = quant_units[threadIdx.x];
}
__syncthreads();

// 从共享内存读取
uint32_t packed_unit = shared_quant[unit_idx];
```

---

## ✅ Checklist

- [ ] 进入工作目录 `/home/zh/mustafar/benchmark`
- [ ] 激活环境 `conda activate mustar`
- [ ] 运行 Nsys profiling（量化版本）
- [ ] 导出 kernel 统计
- [ ] 导出内存统计
- [ ] 查看关键指标
- [ ] 分析瓶颈类型
- [ ] 记录分析结果
- [ ] 选择优化方案
- [ ] 开始实施优化

---

## 🆘 常见问题

### Q1: Nsys 命令找不到？

```bash
# 检查 Nsys 是否安装
which nsys

# 如果没有，安装 CUDA Toolkit
# 或者使用完整路径
/usr/local/cuda/bin/nsys profile ...
```

### Q2: Profiling 太慢？

```bash
# 减少测试规模（修改 mem_spd_test_quant.py）
# 将 repeat 从 3 改为 1
# 将 output_length 从 1024 改为 256
```

### Q3: 内存不足？

```bash
# 减少 batch size
# 修改 mem_spd_test_quant.py 中的 batch_size
```

### Q4: 如何快速验证优化效果？

```bash
# 修改代码后，重新编译
cd /home/zh/mustafar/kernel_quant
python setup.py install

# 快速测试（只跑 1 次）
cd /home/zh/mustafar/benchmark
python mem_spd_test_quant.py  # 查看 TPOT 是否降低
```

---

**准备好了吗？开始 Profiling！** 🚀

```bash
cd /home/zh/mustafar/benchmark
conda activate mustar
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --gpu-metrics-device=all -o quant_detailed python mem_spd_test_quant.py
```
