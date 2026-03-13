#!/bin/bash

# 运行 Baseline 的脚本
# 用法:
#   bash run_baseline.sh          # 运行完整 LongBench 测试

set -e

echo "=========================================="
echo "运行 Baseline (完整 LongBench 测试)"
echo "=========================================="
echo ""

# 获取脚本所在目录（DiffSparseKV）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 保持在 DiffSparseKV 目录
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo ""

# 激活环境
echo "激活环境..."
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mustar
    echo "✓ Conda 环境激活: mustar"
else
    echo "⚠️  Conda 未找到，使用当前 Python 环境"
fi
echo ""

# MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
MODEL_PATH="/home/zh/model/Llama-2-7b-hf"

# 提取模型名称（从路径中获取最后一部分）
MODEL_NAME=$(basename "$MODEL_PATH")
echo "模型名称: $MODEL_NAME"
echo ""

echo "配置:"
echo "  模型: $MODEL_PATH"
echo "  模式: mustafar (baseline)"
echo "  参数: k=0, v=0, group_size=32, residual_length=32"
echo "  数据集: 完整 LongBench (17个任务)"
echo ""

# 运行 Baseline
echo "=========================================="
echo "开始运行"
echo "=========================================="
echo ""

# 设置 PYTHONPATH 以便能够导入 models 模块（指向父目录）
export PYTHONPATH="$(dirname $PWD):$PYTHONPATH"

# 运行完整 LongBench 测试
echo "开始生成预测..."
python pred_long_bench_diff_sparse.py \
    --model_name_or_path "$MODEL_PATH" \
    --k_sparsity 0.0 \
    --v_sparsity 0.0 \
    --group_size 32 \
    --residual_length 32 \
    --mode mustafar \
    --e 0

echo ""
echo "✓ Baseline 预测完成"
echo ""

# 检查输出
BASELINE_DIR="pred/${MODEL_NAME}_8192_K_0.0_V_0.0"

if [ -d "$BASELINE_DIR" ]; then
    echo "✓ 输出目录: $BASELINE_DIR"
    echo ""
    
    # 统计生成的文件
    FILE_COUNT=$(ls -1 "$BASELINE_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "生成的数据集文件数: $FILE_COUNT"
    echo ""
    
    # 列出文件及大小
    echo "文件列表:"
    ls -lh "$BASELINE_DIR"/*.jsonl 2>/dev/null | awk '{printf "  %-40s %8s\n", $9, $5}'
    echo ""
fi

# 运行评估
echo "=========================================="
echo "运行评估"
echo "=========================================="
echo ""

python eval_long_bench.py --model ${MODEL_NAME}_8192_K_0.0_V_0.0

echo ""
echo "✓ 评估完成"
echo ""

    # 显示结果
    RESULT_FILE="pred/${MODEL_NAME}_8192_K_0.0_V_0.0/result.json"

    if [ -f "$RESULT_FILE" ]; then
        echo "=========================================="
        echo "Baseline 评估结果"
        echo "=========================================="
        echo ""
        
        # 格式化显示结果
        python -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
    
# 按类别分组
categories = {
    '单文档QA': ['narrativeqa', 'qasper', 'multifieldqa_en'],
    '多文档QA': ['hotpotqa', '2wikimqa', 'musique'],
    '摘要': ['gov_report', 'qmsum', 'multi_news'],
    '少样本学习': ['trec', 'triviaqa', 'samsum'],
    '合成任务': ['passage_count', 'passage_retrieval_en'],
    '代码': ['lcc', 'repobench-p']
}

print('各数据集得分:')
print('-' * 60)
for category, datasets in categories.items():
    print(f'\n{category}:')
    for dataset in datasets:
        if dataset in data:
            score = data[dataset]
            print(f'  {dataset:25s}: {score:6.2f}')

if 'average' in data:
    print('\n' + '=' * 60)
    print(f'平均分数: {data[\"average\"]:.2f}')
    print('=' * 60)
"
        echo ""
        
        # 提取平均分
        AVG_SCORE=$(python -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
    print(f'{data.get(\"average\", 0):.2f}')
" 2>/dev/null || echo "0")
        
        # 判断是否对齐
        if (( $(echo "$AVG_SCORE > 40" | bc -l 2>/dev/null || echo "0") )); then
            echo "✓ Baseline 分数正常 (>40)，对齐成功！"
        else
            echo "⚠️  Baseline 分数偏低 (<40)，可能存在问题"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
echo ""
echo "下一步: 运行 DiffSparseKV"
echo "  bash run_diffsparse.sh"
echo ""
