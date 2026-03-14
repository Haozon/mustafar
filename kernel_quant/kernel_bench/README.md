# Kernel Benchmark 测试套件

## 目标

系统性测试量化 kernel 在不同配置下的性能，对标论文 Figure 6a 的效率评估。

## 测试内容

### 1. 效率评估 (`benchmark_efficiency.py`)

对标论文 Figure 6a，测试各组件的延迟占比：

**测试组件**：
- **cuBLAS (Dense Inference)**: 密集矩阵乘法基准
- **SpMV**: 稀疏矩阵-向量乘法（量化稀疏 kernel）
- **Local MV**: Local window 密集矩阵乘法
- **Compress**: 压缩开销（量化 + 打包）

**测试配置**：
- Llama-2-7B: Input 2048, Output 1024
- Llama-3-8B: Input 4096, Output 1024
- 稀疏度: 50%, 70%

**论文参考值** (Llama-2-7B, 50% sparsity):
- Pruning: 1.84% of cuBLAS
- Compress: 6.25% of cuBLAS
- Local MV: 0.62% of cuBLAS
- SpMV: 81.07% of cuBLAS

## 快速开始

```bash
cd /home/zh/mustafar/kernel_quant/kernel_bench

# 运行效率评估
python benchmark_efficiency.py

# 或使用脚本
bash run_benchmark.sh
```

## 输出结果

### 终端输出
```
归一化延迟 (相对于 Dense = 100%)
======================================================================
  Dense Inference:  100.00%
  SpMV:              XX.XX%
  Local MV:           X.XX%
  Compress:           X.XX%
  Total (Mustafar):  XX.XX%

  加速比: X.XXx

组件占比 (相对于 cuBLAS 执行时间)
======================================================================
  SpMV:     XX.XX% of cuBLAS
  Local MV:  X.XX% of cuBLAS
  Compress:  X.XX% of cuBLAS
```

### JSON 文件
`efficiency_benchmark_YYYYMMDD_HHMMSS.json`

包含详细的测试结果和配置信息。

## 预期结果

### 理想情况（接近论文）
- SpMV: 60-90% of cuBLAS
- Local MV: < 1% of cuBLAS
- Compress: 5-10% of cuBLAS
- 总加速比: > 1.2x

### 当前问题（如果有）
如果 SpMV 占比过高（> 100%），说明：
1. Kernel 在大规模数据下性能下降
2. 需要优化内存访问模式
3. 需要优化反量化逻辑

## 分析方法

1. **对比论文值**：看各组件占比是否接近论文
2. **找出瓶颈**：哪个组件占比最高
3. **规模分析**：对比 Llama-2-7B 和 Llama-3-8B 的差异
4. **稀疏度分析**：对比 50% 和 70% 的差异

## 下一步

根据测试结果：
- 如果 SpMV 慢 → 优化反量化 kernel
- 如果 Compress 慢 → 优化量化逻辑
- 如果 Local MV 慢 → 检查 local window 实现
- 如果整体慢 → 检查数据传输和内存访问
