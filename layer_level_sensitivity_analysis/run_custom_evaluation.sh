#!/bin/bash

# 基于自定义稀疏度配置的完整评估流程

set -e  # 遇到错误时退出

echo "=========================================="
echo "自定义稀疏度配置评估流程"
echo "=========================================="

# 配置参数
CUSTOM_RESULTS="sensitivity_custom/custom_results.json"
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
MAX_LENGTH=8192
RESIDUAL_LENGTH=128
OUTPUT_DIR="./pred/pred_same"

echo "使用自定义配置结果: $CUSTOM_RESULTS"
echo "模型路径: $MODEL_PATH"
echo "最大序列长度: $MAX_LENGTH"
echo "Residual长度: $RESIDUAL_LENGTH"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 步骤1: 运行自定义稀疏度评估
echo "步骤1: 运行自定义稀疏度评估..."
echo "----------------------------------------"

python eval_greedy_sparsity.py \
    --greedy_results "$CUSTOM_RESULTS" \
    --model_path "$MODEL_PATH" \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --residual_length $RESIDUAL_LENGTH \
    --datasets narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique \
              gov_report qmsum multi_news trec triviaqa samsum \
              passage_count passage_retrieval_en lcc repobench-p
if [ $? -ne 0 ]; then
    echo "错误: 自定义稀疏度评估失败"
    exit 1
fi

echo "步骤1完成: 自定义稀疏度评估"

# 步骤2: 查找生成的结果目录
echo ""
echo "步骤2: 查找生成的结果目录..."
echo "----------------------------------------"

# 查找最新生成的目录 (可能包含greedy或custom字样)
CUSTOM_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_*_avg_*" | sort | tail -1)

if [ -z "$CUSTOM_DIR" ]; then
    echo "错误: 未找到自定义评估结果目录"
    echo "在 $OUTPUT_DIR 中查找模式: *_*_avg_*"
    ls -la "$OUTPUT_DIR" || echo "输出目录不存在"
    exit 1
fi

echo "找到结果目录: $CUSTOM_DIR"

# 步骤3: 运行LongBench评估
echo ""
echo "步骤3: 运行LongBench评估..."
echo "----------------------------------------"

CUSTOM_SUBDIR=$(basename "$CUSTOM_DIR")
echo "评估目录名称: $CUSTOM_SUBDIR"

# 检查结果目录是否在正确的位置，并复制到pred目录
echo "将结果复制到 pred 目录..."
mkdir -p pred
cp -r "$CUSTOM_DIR" "pred/"

python eval_long_bench.py --model "$CUSTOM_SUBDIR"

if [ $? -ne 0 ]; then
    echo "错误: LongBench评估失败"
    exit 1
fi

echo "步骤3完成: LongBench评估"

# 步骤4: 显示结果
echo ""
echo "步骤4: 显示结果..."
echo "----------------------------------------"

RESULT_FILE="pred/$CUSTOM_SUBDIR/result.json"
if [ -f "$RESULT_FILE" ]; then
    echo "评估结果:"
    python -c "
import json
with open('$RESULT_FILE', 'r') as f:
    results = json.load(f)
    print('数据集评估结果:')
    for dataset, score in results.items():
        if dataset != 'average':
            print(f'  {dataset:20s}: {score:.4f}')
    if 'average' in results:
        print(f'  {'平均分数':20s}: {results[\"average\"]:.4f}')
"
else
    echo "警告: 未找到结果文件 $RESULT_FILE"
fi

echo ""
echo "=========================================="
echo "评估流程完成!"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  - 预测结果: pred/$CUSTOM_SUBDIR/"
echo "  - 评估分数: pred/$CUSTOM_SUBDIR/result.json"
echo "  - 稀疏度配置: pred/$CUSTOM_SUBDIR/sparsity_config.json"
echo ""
echo "自定义稀疏度配置特点:"
echo "  - 平均稀疏度: 70%"
echo "  - 浅层保守 (50-62%)"
echo "  - 中间层高稀疏度 (70-85%)"
echo "  - 深层适中 (55-70%)"