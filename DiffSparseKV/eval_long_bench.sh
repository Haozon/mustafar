#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数 - 使用绝对路径
PRED_BASE_DIR="$SCRIPT_DIR/pred"

# 查找所有结果目录
echo "查找结果目录..."
RESULT_DIRS=($(find "$PRED_BASE_DIR" -maxdepth 1 -type d -name "Meta-Llama*" | sort))

if [ ${#RESULT_DIRS[@]} -eq 0 ]; then
    echo "[ERROR] 未找到任何结果目录"
    echo "请先运行评估脚本生成预测结果"
    exit 1
fi

echo "找到 ${#RESULT_DIRS[@]} 个结果目录:"
for dir in "${RESULT_DIRS[@]}"; do
    echo "  • $(basename "$dir")"
done
echo ""

# 如果提供了参数，只评估指定的目录
if [ $# -gt 0 ]; then
    MODEL_INPUT=$1
    
    # 处理输入路径，移除可能的前缀
    MODEL_NAME=$(basename "$MODEL_INPUT")
    
    echo "评估指定模型: $MODEL_NAME"
    echo ""
    
    # 检查目录是否存在
    if [ ! -d "$PRED_BASE_DIR/$MODEL_NAME" ]; then
        echo "[ERROR] 目录不存在: $PRED_BASE_DIR/$MODEL_NAME"
        echo "可用的目录："
        for dir in "${RESULT_DIRS[@]}"; do
            echo "  • $(basename "$dir")"
        done
        exit 1
    fi
    
    echo "[$(date '+%H:%M:%S')] 开始评估: $MODEL_NAME"
    
    # 使用 eval_results.py 进行评估
    python eval_results.py --result_dir "$PRED_BASE_DIR/$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] ✓ 评估完成"
    else
        echo ""
        echo "[$(date '+%H:%M:%S')] ✗ 评估失败"
        exit 1
    fi
else
    # 评估所有目录
    echo "评估所有结果目录..."
    echo ""
    
    for result_dir in "${RESULT_DIRS[@]}"; do
        model_name=$(basename "$result_dir")
        
        echo "[$(date '+%H:%M:%S')] 开始评估: $model_name"
        
        # 使用 eval_results.py 进行评估
        python eval_results.py --result_dir "$result_dir"
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] ✓ 评估完成"
        else
            echo "[$(date '+%H:%M:%S')] ✗ 评估失败"
        fi
        echo ""
    done
    
    # 显示所有结果对比
    echo "=========================================="
    echo "所有结果对比"
    echo "=========================================="
    echo ""
    
    printf "%-55s | %-10s\n" "配置" "平均分数"
    echo "--------------------------------------------------------------------"
    
    for result_dir in "${RESULT_DIRS[@]}"; do
        model_name=$(basename "$result_dir")
        result_file="$result_dir/result.json"
        
        if [ -f "$result_file" ]; then
            avg_score=$(python -c "
import json
with open('$result_file', 'r') as f:
    results = json.load(f)
    avg = results.get('average', 0.0)
    print(f'{avg:.2f}')
" 2>/dev/null || echo "N/A")
            printf "%-55s | %-10s\n" "$model_name" "$avg_score"
        else
            printf "%-55s | %-10s\n" "$model_name" "未评估"
        fi
    done
    
    echo "--------------------------------------------------------------------"
fi

echo ""
echo "✓ 评估完成!"
echo ""
echo "使用方式:"
echo "  bash eval_long_bench.sh                                              # 评估所有结果"
echo "  bash eval_long_bench.sh [模型目录名]                                  # 评估指定结果"
echo ""
echo "示例:"
echo "  bash eval_long_bench.sh Meta-Llama-3-8B-Instruct_8192_K_0.0_V_0.0"
echo "  bash eval_long_bench.sh Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70"
echo ""
