#!/bin/bash

# 基于Fisher敏感度分析结果的完整评估流程

set -e  # 遇到错误时退出

echo "=========================================="
echo "Fisher敏感度稀疏度配置评估流程"
echo "=========================================="

# 配置参数
FISHER_RESULTS="fisher_sensitivity_results/fisher_sparsity_allocation.json"
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
MAX_LENGTH=8192
RESIDUAL_LENGTH=128

# 检查Fisher敏感度分析结果文件是否存在
if [ ! -f "$FISHER_RESULTS" ]; then
    echo "错误: Fisher敏感度分析结果文件不存在: $FISHER_RESULTS"
    echo "请确保Fisher敏感度分析已完成并生成了结果文件"
    exit 1
fi

echo "使用Fisher敏感度结果: $FISHER_RESULTS"
echo "模型路径: $MODEL_PATH"
echo "最大序列长度: $MAX_LENGTH"
echo "Residual长度: $RESIDUAL_LENGTH"
echo ""

# 显示Fisher结果摘要
echo "Fisher敏感度分析结果摘要:"
echo "----------------------------------------"
python3 -c "
import json
with open('$FISHER_RESULTS', 'r') as f:
    data = json.load(f)
print(f'方法: {data[\"method\"]}')
print(f'求解器状态: {data[\"solver_status\"]}')
print(f'目标平均稀疏度: {data[\"target_avg_sparsity\"]}%')
print(f'实际平均稀疏度: {data[\"actual_avg_sparsity\"]}%')
print(f'层数: {data[\"n_layers\"]}')
print(f'稀疏度范围: {data[\"min_sparsity\"]}% - {data[\"max_sparsity\"]}%')
print(f'总敏感度: {data[\"total_sensitivity\"]:.2f}')
print('')
print('各层稀疏度配置:')
for i, sparsity in enumerate(data['optimal_sparsity']):
    print(f'  Layer {i:2d}: {sparsity:2d}%')
"

echo ""

# 步骤1: 运行Fisher稀疏度评估
echo "步骤1: 运行Fisher稀疏度评估..."
echo "----------------------------------------"

python eval_fisher_sparsity.py \
    --fisher_results "$FISHER_RESULTS" \
    --model_path "$MODEL_PATH" \
    --max_length $MAX_LENGTH \
    --residual_length $RESIDUAL_LENGTH \
    --datasets narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique \
              gov_report qmsum multi_news trec triviaqa samsum \
              passage_count passage_retrieval_en lcc repobench-p

if [ $? -ne 0 ]; then
    echo "错误: Fisher稀疏度评估失败"
    exit 1
fi

echo ""
echo "步骤1完成: Fisher稀疏度评估"

# 步骤2: 找到生成的结果目录
echo "步骤2: 查找生成的结果目录..."
echo "----------------------------------------"

FISHER_DIR=$(find pred_fisher -name "*fisher*" -type d 2>/dev/null | head -1)

if [ -z "$FISHER_DIR" ]; then
    echo "错误: 未找到Fisher评估结果目录"
    echo "检查 pred_fisher 目录是否存在..."
    if [ -d "pred_fisher" ]; then
        echo "pred_fisher 目录存在，但未找到匹配的子目录"
        ls -la pred_fisher/
    else
        echo "pred_fisher 目录不存在"
    fi
    exit 1
fi

echo "找到结果目录: $FISHER_DIR"

# 步骤3: 运行LongBench评估
echo ""
echo "步骤3: 运行LongBench评估..."
echo "----------------------------------------"

FISHER_SUBDIR=$(basename "$FISHER_DIR")
echo "评估目录名称: $FISHER_SUBDIR"

# 检查结果目录是否在正确的位置
if [ -d "pred_fisher/$FISHER_SUBDIR" ]; then
    # 将结果移动到 pred 目录以便 eval_long_bench.py 能找到
    echo "将结果复制到 pred 目录..."
    mkdir -p pred
    cp -r "pred_fisher/$FISHER_SUBDIR" "pred/"
fi

python eval_long_bench.py --model "$FISHER_SUBDIR"

if [ $? -ne 0 ]; then
    echo "错误: LongBench评估失败"
    exit 1
fi

echo ""
echo "步骤3完成: LongBench评估"

# 步骤4: 显示结果摘要
echo ""
echo "步骤4: 结果摘要..."
echo "----------------------------------------"

RESULT_FILE="pred/$FISHER_SUBDIR/result.json"
if [ -f "$RESULT_FILE" ]; then
    echo "LongBench评估结果:"
    python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    results = json.load(f)
    
print('数据集评估分数:')
total_score = 0
count = 0
for dataset, score in results.items():
    print(f'  {dataset:20s}: {score:6.2f}')
    total_score += score
    count += 1

if count > 0:
    avg_score = total_score / count
    print(f'')
    print(f'平均分数: {avg_score:.2f}')
"
else
    echo "警告: 未找到结果文件 $RESULT_FILE"
fi

echo ""
echo "=========================================="
echo "Fisher敏感度评估流程完成!"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  - 预测结果: pred/$FISHER_SUBDIR/"
echo "  - 评估分数: pred/$FISHER_SUBDIR/result.json"
echo "  - Fisher配置: pred_fisher/$FISHER_SUBDIR/fisher_sparsity_config.json"
echo ""
echo "主要发现:"
echo "  - 查看上方的评估分数了解Fisher-based稀疏度的性能"
echo "  - Fisher方法基于敏感度分析优化了per-layer稀疏度分配"
echo "  - 可以与均匀稀疏度和稠密模型进行对比分析"
echo ""
echo "后续分析建议:"
echo "  - 比较Fisher方法与贪心搜索方法的性能差异"
echo "  - 分析不同层稀疏度分配对各个任务的影响"
echo "  - 评估Fisher方法在不同目标稀疏度下的表现"