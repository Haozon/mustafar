#!/bin/bash

# Mustafar Quantized vs Non-Quantized Performance Comparison
# 对比测试：量化稀疏 vs 标准稀疏

echo "============================================================"
echo "Mustafar Performance Comparison Test"
echo "============================================================"
echo "This script will run two tests:"
echo "  1. Standard Mustafar (Sparse only)"
echo "  2. Mustafar with Quantization (Sparse + 2-bit Quant)"
echo "============================================================"
echo ""

# 确保 CUDA 可见设备正确设置
export CUDA_VISIBLE_DEVICES=0

# 检查环境
echo "Checking environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}');" || exit 1

echo ""
echo "============================================================"
echo "Test 1: Standard Mustafar (Sparse only, no quantization)"
echo "============================================================"
echo ""

# 修改 Python 脚本中的 QUANT_MODE 为 False
sed -i 's/QUANT_MODE = True/QUANT_MODE = False/' mem_spd_test_quant.py

# 运行标准稀疏测试
# python3 mem_spd_test_quant.py

# 保存结果
if [ -f "mem_spd_test_quant_results_2bit.txt" ]; then
    mv mem_spd_test_quant_results_2bit.txt mem_spd_test_sparse_only.txt
    echo "✅ Results saved to: mem_spd_test_sparse_only.txt"
fi

echo ""
echo "============================================================"
echo "Test 2: Mustafar with 2-bit Quantization"
echo "============================================================"
echo ""

# 修改 Python 脚本中的 QUANT_MODE 为 True
sed -i 's/QUANT_MODE = False/QUANT_MODE = True/' mem_spd_test_quant.py

# 运行量化稀疏测试
python3 mem_spd_test_quant.py

# 保存结果
if [ -f "mem_spd_test_quant_results_2bit.txt" ]; then
    echo "✅ Results saved to: mem_spd_test_quant_results_2bit.txt"
fi

echo ""
echo "============================================================"
echo "Comparison Summary"
echo "============================================================"

# 生成对比报告
python3 - << 'EOF'
import os

files = {
    'Sparse Only': 'mem_spd_test_sparse_only.txt',
    'Sparse + 2-bit Quant': 'mem_spd_test_quant_results_2bit.txt'
}

results = {}

for name, filename in files.items():
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            results[name] = {}
            for line in content.split('\n'):
                if 'TTFT:' in line:
                    results[name]['TTFT'] = float(line.split(':')[1].strip().split()[0])
                elif 'TPOT:' in line:
                    results[name]['TPOT'] = float(line.split(':')[1].strip().split()[0])
                elif 'Peak memory:' in line:
                    results[name]['Memory'] = float(line.split(':')[1].strip().split()[0])

if len(results) == 2:
    print("\n📊 Performance Comparison:")
    print("="*70)
    print(f"{'Metric':<25} {'Sparse Only':<20} {'Sparse+Quant':<20} {'Speedup':<10}")
    print("-"*70)
    
    sparse_ttft = results['Sparse Only'].get('TTFT', 0)
    quant_ttft = results['Sparse + 2-bit Quant'].get('TTFT', 0)
    if sparse_ttft and quant_ttft:
        speedup = sparse_ttft / quant_ttft
        print(f"{'TTFT (ms)':<25} {sparse_ttft:<20.2f} {quant_ttft:<20.2f} {speedup:<10.2f}x")
    
    sparse_tpot = results['Sparse Only'].get('TPOT', 0)
    quant_tpot = results['Sparse + 2-bit Quant'].get('TPOT', 0)
    if sparse_tpot and quant_tpot:
        speedup = sparse_tpot / quant_tpot
        print(f"{'TPOT (ms)':<25} {sparse_tpot:<20.2f} {quant_tpot:<20.2f} {speedup:<10.2f}x")
    
    sparse_mem = results['Sparse Only'].get('Memory', 0)
    quant_mem = results['Sparse + 2-bit Quant'].get('Memory', 0)
    if sparse_mem and quant_mem:
        reduction = (1 - quant_mem / sparse_mem) * 100
        print(f"{'Peak Memory (GB)':<25} {sparse_mem:<20.2f} {quant_mem:<20.2f} {reduction:<10.1f}%↓")
    
    print("="*70)
else:
    print("\n⚠️  Could not find both result files for comparison")
    print(f"Looking for:")
    for name, filename in files.items():
        status = "✅ Found" if os.path.exists(filename) else "❌ Missing"
        print(f"  {status}: {filename}")

EOF

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"
echo "Result files:"
echo "  - mem_spd_test_sparse_only.txt (Standard Mustafar)"
echo "  - mem_spd_test_quant_results_2bit.txt (Mustafar + Quantization)"
echo ""
