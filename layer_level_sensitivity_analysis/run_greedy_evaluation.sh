#!/bin/bash

# 基于贪心搜索结果的完整评估流程

set -e  # 遇到错误时退出

echo "=========================================="
echo "贪心搜索稀疏度配置评估流程"
echo "=========================================="

# 配置参数
GREEDY_RESULTS="sensitivity_results_step_0.2/greedy_search_results.json"
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
MAX_LENGTH=8192
RESIDUAL_LENGTH=128

# 检查贪心搜索结果文件是否存在
if [ ! -f "$GREEDY_RESULTS" ]; then
    echo "错误: 贪心搜索结果文件不存在: $GREEDY_RESULTS"
    echo "请先运行贪心搜索算法生成结果文件"
    exit 1
fi

echo "使用贪心搜索结果: $GREEDY_RESULTS"
echo "模型路径: $MODEL_PATH"
echo "最大序列长度: $MAX_LENGTH"
echo "Residual长度: $RESIDUAL_LENGTH"
echo ""

# 步骤1: 运行贪心稀疏度评估
echo "步骤1: 运行贪心稀疏度评估..."
echo "----------------------------------------"

python eval_greedy_sparsity.py \
    --greedy_results "$GREEDY_RESULTS" \
    --model_path "$MODEL_PATH" \
    --max_length $MAX_LENGTH \
    --residual_length $RESIDUAL_LENGTH \
    --datasets narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique \
              gov_report qmsum multi_news trec triviaqa samsum \
              passage_count passage_retrieval_en lcc repobench-p

if [ $? -ne 0 ]; then
    echo "错误: 贪心稀疏度评估失败"
    exit 1
fi

echo ""
echo "步骤1完成: 贪心稀疏度评估"

# 步骤2: 找到生成的结果目录
echo "步骤2: 查找生成的结果目录..."
echo "----------------------------------------"

GREEDY_DIR=$(find pred_greedy -name "*greedy*" -type d 2>/dev/null | head -1)

if [ -z "$GREEDY_DIR" ]; then
    echo "错误: 未找到贪心评估结果目录"
    echo "检查 pred_greedy 目录是否存在..."
    if [ -d "pred_greedy" ]; then
        echo "pred_greedy 目录存在，但未找到匹配的子目录"
        ls -la pred_greedy/
    else
        echo "pred_greedy 目录不存在"
    fi
    exit 1
fi

echo "找到结果目录: $GREEDY_DIR"

# 步骤3: 运行LongBench评估
echo ""
echo "步骤3: 运行LongBench评估..."
echo "----------------------------------------"

GREEDY_SUBDIR=$(basename "$GREEDY_DIR")
echo "评估目录名称: $GREEDY_SUBDIR"

# 检查结果目录是否在正确的位置
if [ -d "pred_greedy/$GREEDY_SUBDIR" ]; then
    # 将结果移动到 pred 目录以便 eval_long_bench.py 能找到
    echo "将结果复制到 pred 目录..."
    mkdir -p pred
    cp -r "pred_greedy/$GREEDY_SUBDIR" "pred/"
fi

python eval_long_bench.py --model "$GREEDY_SUBDIR"

if [ $? -ne 0 ]; then
    echo "错误: LongBench评估失败"
    exit 1
fi

echo ""
echo "步骤3完成: LongBench评估"

# 步骤4: 比较结果
echo ""
echo "步骤4: 比较结果..."
echo "----------------------------------------"

python compare_sparsity_results.py \
    --baseline_dir "pred/Meta-Llama-3-8B-Instruct_8192_K_0.7_V_0.7" \
    --greedy_dir "pred/$GREEDY_SUBDIR" \
    --dense_dir "pred/Meta-Llama-3-8B-Instruct_8192_K_0.0_V_0.0"

echo ""
echo "=========================================="
echo "评估流程完成!"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  - 预测结果: pred/$GREEDY_SUBDIR/"
echo "  - 评估分数: pred/$GREEDY_SUBDIR/result.json"
echo "  - 稀疏度配置: pred/$GREEDY_SUBDIR/sparsity_config.json"
echo ""
echo "主要发现:"
echo "  - 查看上方的比较表格了解性能差异"
echo "  - 贪心per-layer稀疏度 vs 均匀0.7稀疏度的对比"
echo "  - 与稠密模型(0.0稀疏度)的性能对比"