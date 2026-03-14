#!/bin/bash

# Value 量化 Benchmark 测试脚本
# 测试不同配置下的性能

echo "======================================================================"
echo "🚀 Mustafar Value Quantization Benchmark"
echo "======================================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mustar

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "测试配置:"
echo "  - 模型: Meta-Llama-3-8B-Instruct"
echo "  - Batch Size: 8"
echo "  - Input Length: 4096 tokens"
echo "  - Output Length: 1024 tokens"
echo ""

# 测试 1: 量化模式 (2-bit, 50% 稀疏)
echo "======================================================================"
echo "📊 Test 1: Quantized Mode (2-bit, 50% sparse)"
echo "======================================================================"
python mem_spd_test_quant.py

echo ""
echo "======================================================================"
echo "✅ Benchmark 完成!"
echo "======================================================================"
echo ""
echo "结果文件:"
echo "  - mem_spd_test_quant_results_2bit.txt"
echo ""
