#!/bin/bash

# JSQKV Benchmark - 一键运行脚本
# 运行吞吐量测试并生成对比图

echo "============================================================"
echo "JSQKV Benchmark - Throughput Testing"
echo "============================================================"
echo ""

# 设置工作目录
cd "$(dirname "$0")"

# 检查配置文件
if [ ! -f "benchmark_config.yaml" ]; then
    echo "❌ Error: benchmark_config.yaml not found!"
    exit 1
fi

# 检查Python环境
echo "Checking Python environment..."
python3 -c "import torch; import yaml; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required Python packages not found!"
    echo "Please install: torch, yaml, matplotlib"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# 解析命令行参数
RUN_BENCHMARK=true
RUN_PLOT=true
MODELS=""
CONFIGS=""
SCHEMES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-benchmark)
            RUN_BENCHMARK=false
            shift
            ;;
        --skip-plot)
            RUN_PLOT=false
            shift
            ;;
        --models)
            MODELS="--models $2"
            shift 2
            ;;
        --configs)
            CONFIGS="--configs $2"
            shift 2
            ;;
        --schemes)
            SCHEMES="--schemes $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-benchmark] [--skip-plot] [--models MODEL1 MODEL2] [--configs CONFIG1 CONFIG2] [--schemes SCHEME1 SCHEME2]"
            exit 1
            ;;
    esac
done

# 运行benchmark测试
if [ "$RUN_BENCHMARK" = true ]; then
    echo "============================================================"
    echo "Step 1: Running Benchmark Tests"
    echo "============================================================"
    echo ""
    
    python3 benchmark_throughput.py $MODELS $CONFIGS
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Benchmark failed!"
        exit 1
    fi
    
    echo ""
    echo "✅ Benchmark completed successfully!"
else
    echo "⏭️  Skipping benchmark tests (using existing results)"
fi

echo ""

# 生成图表
if [ "$RUN_PLOT" = true ]; then
    echo "============================================================"
    echo "Step 2: Generating Plots"
    echo "============================================================"
    echo ""
    
    python3 plot_throughput.py $SCHEMES
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Plot generation failed!"
        exit 1
    fi
    
    echo ""
    echo "✅ Plots generated successfully!"
else
    echo "⏭️  Skipping plot generation"
fi

echo ""
echo "============================================================"
echo "✅ All tasks completed!"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  - results/raw_data/    (JSON data)"
echo "  - results/plots/       (PDF plots)"
echo ""
