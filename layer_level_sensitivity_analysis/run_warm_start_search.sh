#!/bin/bash

# 贪心搜索层级稀疏度优化 - 可配置版本

# 默认参数
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
INITIAL_SPARSITY=0.4
STEP_SIZE=0.04
TARGET_SPARSITY=0.7
NUM_SAMPLES=30
MAX_LENGTH=512
RESIDUAL_LENGTH=128
OUTPUT_DIR="./sensitivity_results"
DEVICE="cuda"
CUDA_DEVICE=0
EVAL_METRIC="loss"  # 评估指标选项 ("ppl" 或 "loss")，默认使用loss

echo "========================================"
echo "🚀 贪心搜索层级稀疏度优化"
echo "========================================"
echo ""
echo "📋 配置参数:"
echo "   模型路径: $MODEL_PATH"
echo "   初始稀疏度: $INITIAL_SPARSITY"
echo "   步长: $STEP_SIZE"
echo "   目标稀疏度: $TARGET_SPARSITY"
echo "   验证样本数: $NUM_SAMPLES"
echo "   最大序列长度: $MAX_LENGTH"
echo "   Residual长度: $RESIDUAL_LENGTH"
echo "   评估指标: $EVAL_METRIC"
echo "   输出目录: $OUTPUT_DIR"
echo "   设备: $DEVICE"
echo "   CUDA设备: $CUDA_DEVICE"
echo ""
echo "========================================"
echo ""

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行分析
python sensitivity_analysis/greedy_search_simple.py \
    --model_path "$MODEL_PATH" \
    --initial_sparsity "$INITIAL_SPARSITY" \
    --step_size "$STEP_SIZE" \
    --target_sparsity "$TARGET_SPARSITY" \
    --num_samples "$NUM_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --residual_length "$RESIDUAL_LENGTH" \
    --eval_metric "$EVAL_METRIC" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "✅ Analysis completed!"
echo "========================================"
echo ""
echo "Visualizing results..."
python sensitivity_analysis/visualize_results.py

echo ""
echo "========================================"
echo "✅ All done!"
echo "========================================"
