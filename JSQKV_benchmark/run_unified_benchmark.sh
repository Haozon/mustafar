#!/bin/bash

# 统一性能测试脚本
# 在相同负载条件下对比 Dense、Sparse-50%、Sparse-50%+Quant-2bit

echo "======================================================================"
echo "🚀 Unified Performance Benchmark"
echo "======================================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mustar

# 进入 JSQKV_benchmark 目录
cd "$(dirname "$0")"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "测试配置:"
echo "  - 模型: Meta-Llama-3-8B-Instruct"
echo "  - Batch Size: 8"
echo "  - Input Length: 4096 tokens"
echo "  - Output Length: 1024 tokens"
echo "  - 重复次数: 3"
echo ""
echo "测试配置:"
echo "  1. Dense (FP16)"
echo "  2. Sparse-50% (FP16)"
echo "  3. Sparse-50% + Quant-2bit"
echo ""

# 运行测试
python unified_benchmark.py

echo ""
echo "======================================================================"
echo "✅ Benchmark 完成!"
echo "======================================================================"
echo ""
echo "结果文件:"
echo "  - results/unified_benchmark_results.json"
echo ""
