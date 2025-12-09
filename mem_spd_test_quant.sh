#!/bin/bash

# Mustafar Quantized Sparse Kernel Performance Test Script
# 测试配置：
# - 稀疏度: 50% (K_SPARSITY=0.5, V_SPARSITY=0.5)
# - 量化位宽: 2-bit
# - 预期内存压缩: ~16x

echo "========================================"
echo "Mustafar Quantized Sparse Kernel Test"
echo "========================================"
echo "Configuration:"
echo "  - Sparsity: 50%"
echo "  - Quantization: 2-bit per-token"
echo "  - Expected compression: ~16x"
echo "========================================"
echo ""

# 确保 CUDA 可见设备正确设置
export CUDA_VISIBLE_DEVICES=0

# 检查 CUDA 是否可用
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}');"

echo ""
echo "Starting quantized sparse kernel test..."
echo ""

# 运行量化稀疏测试
python3 mem_spd_test_quant.py

echo ""
echo "========================================"
echo "Test completed!"
echo "Check mem_spd_test_quant_results_2bit.txt for detailed results"
echo "========================================"