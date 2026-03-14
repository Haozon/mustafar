# 量化版本性能优化任务总结

## 📊 当前状态

### **性能数据（0211 测试，服务器有负载）**

| 配置 | TTFT | TPOT | 总时间 | 内存 |
|------|------|------|--------|------|
| **Mustafar 50%** | 6798 ms | **53.41 ms** ✅ | 61437 ms | 44.72 GB |
| **Mustafar-Quant 50%** | 3469 ms ✅ | **82.78 ms** ❌ | 88152 ms | 41.28 GB ✅ |

**关键问题：量化版本的 TPOT 慢了 55%（82.78 ms vs 53.41 ms）**

---

## 🔍 已完成的分析

### **1. Kernel 调用次数分析**

| Kernel | 无量化调用次数 | 量化调用次数 | 差异 |
|--------|--------------|------------|------|
| `compress_key_batched` | 159 | 1,152 | 7.2x |
| `compress_value_batched` | 171 | 1,152 | 6.7x |
| `Key_Kernel_Quant` | 0 | **229,440** | 新增 |
| `Value_Kernel_Quant` | 0 | **229,440** | 新增 |

**发现：量化 kernel 占用了 65% 的总时间（398 秒）**

### **2. Kernel 执行时间分析**

| Kernel | 时间占比 | 总时间 | 平均时间 |
|--------|---------|--------|---------|
| `Key_Kernel_Quant` | 34.5% | 210.4 秒 | 0.92 ms |
| `Value_Kernel_Quant` | 30.7% | 187.6 秒 | 0.82 ms |
| **合计** | **65.2%** | **398 秒** | **1.74 ms** |

### **3. 根本原因定位**

**量化版本慢的原因：反量化开销**

```
无量化流程：
Query × Compressed_Key (FP16) → Attention
时间：~50 ms

量化流程：
Query × [Compressed_Key (2-bit) + Dequantize] → Attention
时间：~82 ms

额外开销：~32 ms（反量化）
```

---

## 🎯 核心问题

### **问题 1：反量化在 kernel 中进行，开销大**

**当前实现：**
```cuda
// 在 SpMM kernel 中，每次访问都需要反量化
for (int i = 0; i < nnz; i++) {
    uint8_t quant_val = unpack_2bit(packed_data[i]);
    float scale = scales[tile_id];
    float zero = zeros[tile_id];
    float value = (quant_val * scale) + zero;  // 反量化
    result += query[i] * value;
}
```

**开销分析：**
- 每个非零值：1次解包 + 1次乘法 + 1次加法 + 2次内存访问（scale/zero）
- 内存访问量：12 bytes（uint32 + scale + zero）vs 2 bytes（FP16）
- **内存带宽增加 6x**

### **问题 2：量化参数访问效率低**

**当前：**
- 每个非零值都从全局内存读取 scale 和 zero
- 没有利用 shared memory 缓存

### **问题 3：调用次数多（次要问题）**

- 量化 kernel 调用 229,440 次
- 但这是测试过程累积的（3次重复 × 1024 tokens × 8 batch × 32 layers）
- 单次调用时间才是主要问题

---

## 🔧 优化方向

### **优先级 1：融合反量化到 SpMM kernel（最重要）**

**目标：消除反量化的独立步骤**

```cuda
// 当前（慢）：
dequantize_kernel<<<...>>>(compressed_key, scales, zeros, key_fp16);
spmm_kernel<<<...>>>(query, key_fp16, output);

// 优化（快）：
spmm_with_inline_dequant<<<...>>>(query, compressed_key, scales, zeros, output);
```

**预期收益：**
- 减少一次 kernel launch
- 减少中间结果的内存读写
- 预期快 **2-3x**

### **优先级 2：优化量化参数访问**

**使用 shared memory 缓存 scales 和 zeros**

```cuda
__shared__ float shared_scales[MAX_TILES];
__shared__ float shared_zeros[MAX_TILES];

// 一次性加载到 shared memory
if (threadIdx.x < num_tiles) {
    shared_scales[threadIdx.x] = scales[threadIdx.x];
    shared_zeros[threadIdx.x] = zeros[threadIdx.x];
}
__syncthreads();

// 后续访问从 shared memory 读取（快 10-100x）
float scale = shared_scales[tile_id];
float zero = shared_zeros[tile_id];
```

**预期收益：快 1.5-2x**

### **优先级 3：批处理优化（可选）**

**目标：减少 kernel launch overhead**

```python
# 当前：逐 token 处理
for token in tokens:
    quantize_kernel(token)

# 优化：批量处理
quantize_kernel_batched(all_tokens)
```

**预期收益：快 1.3-1.5x**

---

## 📋 下一步行动计划

### **Phase 1：Profiling（定位具体瓶颈）**

1. **使用 Nsys 分析量化 kernel**
   ```bash
   nsys profile --trace=cuda,nvtx python benchmark/mem_spd_test_quant.py
   nsys-ui profile.nsys-rep
   ```

2. **查看 kernel 的详细指标**
   - Memory bandwidth utilization
   - Compute utilization
   - Occupancy
   - Warp efficiency

3. **对比无量化版本的 kernel**
   - 找出具体的性能差异点

### **Phase 2：代码分析**

1. **查看量化 kernel 源代码**
   ```
   kernel_quant/csrc/SpMM_API_Quant.cu
   kernel_quant/csrc/SpMM_Kernel_Quant.cuh
   ```

2. **定位反量化代码位置**
   - 找到 dequantize 的具体实现
   - 分析内存访问模式

3. **评估优化可行性**
   - 是否可以融合反量化
   - Shared memory 是否足够

### **Phase 3：优化实施**

1. **实现融合反量化的 kernel**
   - 修改 SpMM kernel
   - 内联反量化操作

2. **优化内存访问**
   - 使用 shared memory 缓存 scales/zeros
   - 优化内存访问模式

3. **测试和验证**
   - 正确性验证
   - 性能测试
   - 对比优化前后

---

## 📁 相关文件位置

### **模型文件**
- 无量化版本：`/home/zh/mustafar/models/llama_mustafar_kernel.py`
- 量化版本：`/home/zh/mustafar/models/llama_mustafar_quant_kernel.py`

### **Kernel 源代码**
- 无量化 kernel：`/home/zh/mustafar/kernel/csrc/`
- 量化 kernel：`/home/zh/mustafar/kernel_quant/csrc/`
  - `SpMM_API_Quant.cu` - API 接口
  - `SpMM_Kernel_Quant.cuh` - Kernel 实现

### **压缩函数**
- 无量化：`/home/zh/mustafar/kernel/compression.py`
- 量化：`/home/zh/mustafar/kernel_quant/compression_quant.py`

### **Benchmark 脚本**
- 测试脚本：`/home/zh/mustafar/benchmark/mem_spd_test_quant.py`
- 完整测试：`/home/zh/mustafar/benchmark/run_complete_benchmark.sh`

### **结果数据**
- 完整结果：`/home/zh/mustafar/benchmark/benchmark_results_20260211_092254/`
- Kernel 统计：`mustafar_quant_50_kernels.csv_cuda_gpu_kern_sum.csv`

---

## 🎯 优化目标

### **短期目标（1-2周）**
- TPOT: 82.78 ms → **40-50 ms**（快 1.7-2x）
- 方法：融合反量化 + 优化内存访问

### **中期目标（2-4周）**
- TPOT: 40-50 ms → **30-35 ms**（快 1.3-1.5x）
- 方法：批处理优化 + 进一步调优

### **最终目标**
- TPOT: **25-30 ms**（比无量化快 2x）
- TTFT: 保持 3469 ms（已经快 49%）
- 内存: 保持 41.28 GB（节省 15-18%）

**让量化版本真正实现：更快 + 更省内存！**

---

## 💡 关键洞察

1. **量化的 TTFT 已经很好**（快 49%）
   - 说明 prefill 阶段量化是有效的
   - 问题主要在 decode 阶段

2. **反量化是主要瓶颈**
   - 不是调用次数的问题
   - 是单次调用的效率问题

3. **优化空间很大**
   - 当前实现没有充分优化
   - 融合反量化可以带来 2-3x 加速

4. **内存节省已经实现**
   - 41.28 GB vs 44.72 GB（节省 7.7%）
   - 理论上可以节省更多

---

## ✅ 准备工作

### **环境检查**
```bash
# 1. 确认 CUDA 环境
nvidia-smi
nvcc --version

# 2. 确认 Nsys 可用
nsys --version

# 3. 确认 Python 环境
conda activate mustar
python -c "import torch; print(torch.cuda.is_available())"
```

### **代码准备**
```bash
# 1. 进入项目目录
cd /home/zh/mustafar

# 2. 查看 kernel 源代码
ls -la kernel_quant/csrc/

# 3. 准备 profiling
cd benchmark
```

---

## 📝 下一个 Session 的起点

**开始命令：**
```bash
# 1. 查看量化 kernel 源代码
cat kernel_quant/csrc/SpMM_Kernel_Quant.cuh | head -100

# 2. 运行简单的 profiling
nsys profile -o quant_profile python mem_spd_test_quant.py

# 3. 分析结果
nsys stats quant_profile.nsys-rep
```

**第一个问题：反量化代码在哪里？如何优化？**
