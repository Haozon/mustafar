#!/bin/bash
# 完整的 Benchmark 流程
# 参考论文方法：端到端吞吐量 + Nsys Profiling

set -e

echo "========================================================================"
echo "Mustafar Kernel Benchmark - 完整评测流程"
echo "========================================================================"

# 获取脚本所在目录（benchmark 目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 工作目录保持在 benchmark 目录
cd "$SCRIPT_DIR"
echo "工作目录: $(pwd)"

# 配置
CONDA_ENV="mustar"
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "📋 测试配置:"
echo "  - Conda 环境: $CONDA_ENV"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 模型: Llama-3-8B-Instruct"
echo "  - 输入长度: 4096 tokens"
echo "  - 输出长度: 1024 tokens"
echo "  - 稀疏度: 50%, 70%"
echo ""

# 激活环境
# 使用 conda run 来确保在正确的环境中运行
PYTHON_CMD="conda run -n $CONDA_ENV python"

echo "使用命令: $PYTHON_CMD"
echo ""

# ============================================================================
# 第零步：测试 Dense Baseline（无稀疏、无量化）
# ============================================================================

echo "========================================================================"
echo "第零步：测试 Dense Baseline（标准 LLaMA）"
echo "========================================================================"

echo ""
echo "配置 Baseline 模式（K=0, V=0, MUSTAFAR_MODE=False）..."
# 修改配置：关闭 Mustafar，设置稀疏度为 0
sed -i 's/MUSTAFAR_MODE = .*/MUSTAFAR_MODE = False/' mem_spd_test.py
sed -i 's/K_SPARSITY = .*/K_SPARSITY = 0.0/' mem_spd_test.py
sed -i 's/V_SPARSITY = .*/V_SPARSITY = 0.0/' mem_spd_test.py

# 运行测试
echo "运行 Dense Baseline 测试..."
$PYTHON_CMD mem_spd_test.py 2>&1 | tee "$OUTPUT_DIR/baseline_dense_output.txt"

# Nsys profile
echo "  运行 Nsys profiling..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTPUT_DIR/baseline_dense_profile" \
    --force-overwrite=true \
    $PYTHON_CMD mem_spd_test.py > /dev/null 2>&1

# 导出统计
if [ -f "$OUTPUT_DIR/baseline_dense_profile.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output "$OUTPUT_DIR/baseline_dense_kernels.csv" \
        "$OUTPUT_DIR/baseline_dense_profile.nsys-rep"
    echo "  ✓ 完成"
else
    echo "  ⚠️  Nsys profile 失败，跳过统计导出"
fi

echo ""
echo "恢复 Mustafar 模式..."
sed -i 's/MUSTAFAR_MODE = .*/MUSTAFAR_MODE = True/' mem_spd_test.py

# ============================================================================
# 第一步：测试 Mustafar（无量化）
# ============================================================================

echo ""
echo "========================================================================"
echo "第一步：测试 Mustafar（无量化）"
echo "========================================================================"

echo ""
echo "[1/2] 测试 50% 稀疏度..."
# 修改配置
sed -i 's/K_SPARSITY = .*/K_SPARSITY = 0.5/' mem_spd_test.py
sed -i 's/V_SPARSITY = .*/V_SPARSITY = 0.5/' mem_spd_test.py

# 运行测试
$PYTHON_CMD mem_spd_test.py 2>&1 | tee "$OUTPUT_DIR/mustafar_50_output.txt"

# Nsys profile
echo "  运行 Nsys profiling..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTPUT_DIR/mustafar_50_profile" \
    --force-overwrite=true \
    conda run -n $CONDA_ENV python mem_spd_test.py > /dev/null 2>&1

# 检查 profile 文件是否生成
if [ -f "$OUTPUT_DIR/mustafar_50_profile.nsys-rep" ]; then
    # 导出统计
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output "$OUTPUT_DIR/mustafar_50_kernels.csv" \
        "$OUTPUT_DIR/mustafar_50_profile.nsys-rep"
    echo "  ✓ 完成"
else
    echo "  ⚠️  Nsys profile 失败，跳过统计导出"
fi

echo ""
echo "[2/2] 测试 70% 稀疏度..."
# 修改配置
sed -i 's/K_SPARSITY = .*/K_SPARSITY = 0.7/' mem_spd_test.py
sed -i 's/V_SPARSITY = .*/V_SPARSITY = 0.7/' mem_spd_test.py

# 运行测试
$PYTHON_CMD mem_spd_test.py 2>&1 | tee "$OUTPUT_DIR/mustafar_70_output.txt"

# Nsys profile
echo "  运行 Nsys profiling..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTPUT_DIR/mustafar_70_profile" \
    --force-overwrite=true \
    $PYTHON_CMD mem_spd_test.py > /dev/null 2>&1

# 导出统计
nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTPUT_DIR/mustafar_70_kernels.csv" \
    "$OUTPUT_DIR/mustafar_70_profile.nsys-rep"

echo "  ✓ 完成"

# ============================================================================
# 第二步：测试 Mustafar_Quant（量化）
# ============================================================================

echo ""
echo "========================================================================"
echo "第二步：测试 Mustafar_Quant（量化）"
echo "========================================================================"

echo ""
echo "[1/2] 测试 50% 稀疏度 + 2-bit 量化..."
# 修改配置
sed -i 's/K_SPARSITY = .*/K_SPARSITY = 0.5/' mem_spd_test_quant.py
sed -i 's/V_SPARSITY = .*/V_SPARSITY = 0.5/' mem_spd_test_quant.py

# 运行测试
$PYTHON_CMD mem_spd_test_quant.py 2>&1 | tee "$OUTPUT_DIR/mustafar_quant_50_output.txt"

# Nsys profile
echo "  运行 Nsys profiling..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTPUT_DIR/mustafar_quant_50_profile" \
    --force-overwrite=true \
    $PYTHON_CMD mem_spd_test_quant.py > /dev/null 2>&1

# 导出统计
nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTPUT_DIR/mustafar_quant_50_kernels.csv" \
    "$OUTPUT_DIR/mustafar_quant_50_profile.nsys-rep"

echo "  ✓ 完成"

echo ""
echo "[2/2] 测试 70% 稀疏度 + 2-bit 量化..."
# 修改配置
sed -i 's/K_SPARSITY = .*/K_SPARSITY = 0.7/' mem_spd_test_quant.py
sed -i 's/V_SPARSITY = .*/V_SPARSITY = 0.7/' mem_spd_test_quant.py

# 运行测试
$PYTHON_CMD mem_spd_test_quant.py 2>&1 | tee "$OUTPUT_DIR/mustafar_quant_70_output.txt"

# Nsys profile
echo "  运行 Nsys profiling..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$OUTPUT_DIR/mustafar_quant_70_profile" \
    --force-overwrite=true \
    $PYTHON_CMD mem_spd_test_quant.py > /dev/null 2>&1

# 导出统计
if [ -f "$OUTPUT_DIR/mustafar_quant_70_profile.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output "$OUTPUT_DIR/mustafar_quant_70_kernels.csv" \
        "$OUTPUT_DIR/mustafar_quant_70_profile.nsys-rep"
    echo "  ✓ 完成"
else
    echo "  ⚠️  Nsys profile 失败，跳过统计导出"
fi

# ============================================================================
# 第三步：分析结果
# ============================================================================

echo ""
echo "========================================================================"
echo "第三步：分析结果"
echo "========================================================================"

echo ""
echo "分析 50% 稀疏度结果..."
$PYTHON_CMD analyze_nsys_results.py \
    "$OUTPUT_DIR/mustafar_50_kernels.csv" \
    "$OUTPUT_DIR/mustafar_quant_50_kernels.csv" \
    2>&1 | tee "$OUTPUT_DIR/analysis_50.txt"

echo ""
echo "分析 70% 稀疏度结果..."
$PYTHON_CMD analyze_nsys_results.py \
    "$OUTPUT_DIR/mustafar_70_kernels.csv" \
    "$OUTPUT_DIR/mustafar_quant_70_kernels.csv" \
    2>&1 | tee "$OUTPUT_DIR/analysis_70.txt"

# ============================================================================
# 第四步：与 Baseline 对比分析
# ============================================================================

echo ""
echo "========================================================================"
echo "第四步：与 Baseline 对比分析"
echo "========================================================================"

# 创建对比分析脚本
cat > "$OUTPUT_DIR/compare_with_baseline.py" << 'EOFPYTHON'
#!/usr/bin/env python3
import sys
import re

def parse_output(file_path):
    """从输出文件中提取性能指标"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # 提取 TTFT
            ttft_match = re.search(r'TTFT:\s*([\d.]+)\s*ms', content)
            if ttft_match:
                metrics['ttft'] = float(ttft_match.group(1))
            
            # 提取 TPOT
            tpot_match = re.search(r'TPOT:\s*([\d.]+)\s*ms', content)
            if tpot_match:
                metrics['tpot'] = float(tpot_match.group(1))
            
            # 提取内存
            mem_match = re.search(r'Peak memory:\s*([\d.]+)\s*GB', content)
            if mem_match:
                metrics['memory'] = float(mem_match.group(1))
                
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return metrics

# 读取所有结果
baseline = parse_output(sys.argv[1])
mustafar_50 = parse_output(sys.argv[2])
mustafar_70 = parse_output(sys.argv[3])
quant_50 = parse_output(sys.argv[4])
quant_70 = parse_output(sys.argv[5])

# 打印对比表格
print("\n" + "="*85)
print("性能对比分析（相对于 Dense Baseline）")
print("="*85)
print(f"\n{'配置':<30} {'TTFT (ms)':<15} {'TPOT (ms)':<15} {'内存 (GB)':<15}")
print("-"*85)

configs = [
    ("Dense Baseline (K=0, V=0)", baseline),
    ("Mustafar 50%", mustafar_50),
    ("Mustafar 70%", mustafar_70),
    ("Mustafar-Quant 50%", quant_50),
    ("Mustafar-Quant 70%", quant_70),
]

for name, metrics in configs:
    ttft = metrics.get('ttft', 0)
    tpot = metrics.get('tpot', 0)
    mem = metrics.get('memory', 0)
    print(f"{name:<30} {ttft:<15.2f} {tpot:<15.2f} {mem:<15.2f}")

print("\n" + "="*85)
print("加速比分析（相对于 Baseline）")
print("="*85)
print(f"\n{'配置':<30} {'TTFT 加速':<15} {'TPOT 加速':<15} {'内存节省':<15}")
print("-"*85)

baseline_ttft = baseline.get('ttft', 1)
baseline_tpot = baseline.get('tpot', 1)
baseline_mem = baseline.get('memory', 1)

for name, metrics in configs[1:]:  # 跳过 baseline 自己
    ttft = metrics.get('ttft', 1)
    tpot = metrics.get('tpot', 1)
    mem = metrics.get('memory', 1)
    
    ttft_speedup = baseline_ttft / ttft if ttft > 0 else 0
    tpot_speedup = baseline_tpot / tpot if tpot > 0 else 0
    mem_saving = (1 - mem / baseline_mem) * 100 if baseline_mem > 0 else 0
    
    speedup_str = f"{ttft_speedup:.2f}x" if ttft_speedup > 0 else "N/A"
    tpot_str = f"{tpot_speedup:.2f}x" if tpot_speedup > 0 else "N/A"
    mem_str = f"{mem_saving:.1f}%" if mem_saving != 0 else "0.0%"
    
    print(f"{name:<30} {speedup_str:<15} {tpot_str:<15} {mem_str:<15}")

print("\n" + "="*85)
print("\n关键发现:")
print("-"*85)

# 分析关键发现
if baseline_tpot > 0:
    for name, metrics in configs[1:]:
        tpot = metrics.get('tpot', 0)
        if tpot > 0:
            speedup = baseline_tpot / tpot
            if speedup > 1.1:
                print(f"✅ {name}: TPOT 加速 {speedup:.2f}x (更快)")
            elif speedup < 0.9:
                print(f"⚠️  {name}: TPOT 减速 {1/speedup:.2f}x (更慢)")
            else:
                print(f"➖ {name}: TPOT 性能相近")

print("\n" + "="*85)
EOFPYTHON

# 运行对比分析
echo ""
if [ -f "$OUTPUT_DIR/baseline_dense_output.txt" ]; then
    $PYTHON_CMD "$OUTPUT_DIR/compare_with_baseline.py" \
        "$OUTPUT_DIR/baseline_dense_output.txt" \
        "$OUTPUT_DIR/mustafar_50_output.txt" \
        "$OUTPUT_DIR/mustafar_70_output.txt" \
        "$OUTPUT_DIR/mustafar_quant_50_output.txt" \
        "$OUTPUT_DIR/mustafar_quant_70_output.txt" \
        2>&1 | tee "$OUTPUT_DIR/baseline_comparison.txt"
else
    echo "⚠️  Baseline 输出文件未找到，跳过对比分析"
fi

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ 测试完成！"
echo "========================================================================"
echo ""
echo "结果保存在: $OUTPUT_DIR/"
echo ""
echo "文件列表:"
echo "  - baseline_dense_output.txt       # Dense Baseline 输出"
echo "  - mustafar_50_output.txt          # Mustafar 50% 输出"
echo "  - mustafar_70_output.txt          # Mustafar 70% 输出"
echo "  - mustafar_quant_50_output.txt    # Mustafar_Quant 50% 输出"
echo "  - mustafar_quant_70_output.txt    # Mustafar_Quant 70% 输出"
echo "  - *_profile.nsys-rep              # Nsys profile 文件"
echo "  - *_kernels.csv                   # Kernel 统计"
echo "  - analysis_*.txt                  # Kernel 对比分析"
echo "  - baseline_comparison.txt         # Baseline 性能对比"
echo ""
echo "查看性能对比:"
echo "  cat $OUTPUT_DIR/baseline_comparison.txt"
echo ""
echo "查看 Nsys profile:"
echo "  nsys-ui $OUTPUT_DIR/baseline_dense_profile.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/mustafar_50_profile.nsys-rep"
echo ""
