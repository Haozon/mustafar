#!/bin/bash

# 运行 DiffSparseKV 的脚本
# 用法:
#   bash run_diffsparse.sh          # 运行完整 LongBench 测试
#   bash run_diffsparse.sh --window-size 64  # 自定义窗口大小

set -e

# 默认参数
WINDOW_SIZE=32
OBS_WINDOW_SIZE=128
USE_FLASH_ATTENTION="true"
# mustafar baseline:
TARGET_DIST="0.0,1.0,0.0"
SPARSITY_LEVELS="0.0,0.7,1.0"

# 保守型：保护重要 tokens，0.05×0 + 0.80×0.7 + 0.15×1.0 = 0.71 = 71%
# TARGET_DIST="0.05,0.80,0.15"
# SPARSITY_LEVELS="0.0,0.7,1.0"

# 渐进型：（更平滑的分布），0.15×0 + 0.50×0.7 + 0.35×1.0 = 0.70 = 70%
# TARGET_DIST="0.15,0.50,0.35"
# SPARSITY_LEVELS="0.0,0.7,1.0"

# 其他的稀疏度级别：0.0×0 + 0.60×0.5 + 0.40×1.0 = 0.70 = 70%
# TARGET_DIST="0.20,0.50,0.30"
# SPARSITY_LEVELS="0.0,0.6,1.0"

# TARGET_DIST="0.65,0.0,0.35"
# SPARSITY_LEVELS="0.0,0.7,1.0"

DEBUG_MODE="false"
ENABLE_STATS=1
IMPORTANCE_MODE="attention_only"
VALUE_SINK_KEEP=2
TARGET_BUDGET=""
BUDGET_TEMPLATE="default_3level"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --target-dist)
            TARGET_DIST="$2"
            shift 2
            ;;
        --sparsity-levels)
            SPARSITY_LEVELS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="True"
            shift
            ;;
        --importance-mode)
            IMPORTANCE_MODE="$2"
            shift 2
            ;;
        --value-sink-keep)
            VALUE_SINK_KEEP="$2"
            shift 2
            ;;
        --target-budget)
            TARGET_BUDGET="$2"
            shift 2
            ;;
        --budget-template)
            BUDGET_TEMPLATE="$2"
            shift 2
            ;;
        --no-stats)
            ENABLE_STATS=0
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 环境变量配置
export DIFFSPARSE_ENABLE_STATS=${ENABLE_STATS}

echo "=========================================="
echo "运行 DiffSparseKV (完整 LongBench 测试)"
echo "=========================================="
echo ""

# 获取脚本所在目录（DiffSparseKV）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 保持在 DiffSparseKV 目录
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo ""

# 检查 diffsparsekv 库是否已安装
echo "检查 diffsparsekv 库..."
if python -c "import diffsparsekv" 2>/dev/null; then
    echo "✓ diffsparsekv 库已安装"
else
    echo "⚠️  diffsparsekv 库未安装，将使用本地文件"
    echo "   建议运行: pip install -e ."
fi
echo ""

# 环境变量配置
export DIFFSPARSE_ENABLE_STATS=${ENABLE_STATS}

echo "环境变量配置:"
echo "  DIFFSPARSE_ENABLE_STATS: $DIFFSPARSE_ENABLE_STATS"
echo ""

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

# 计算预期稀疏度
IFS=',' read -ra DIST <<< "$TARGET_DIST"
IFS=',' read -ra LEVELS <<< "$SPARSITY_LEVELS"

# 调试：显示解析结果
echo "调试信息:"
echo "  TARGET_DIST: $TARGET_DIST"
echo "  SPARSITY_LEVELS: $SPARSITY_LEVELS"
echo "  解析后的 DIST 数量: ${#DIST[@]}"
echo "  解析后的 LEVELS 数量: ${#LEVELS[@]}"
if [ ${#DIST[@]} -ge 3 ]; then
    echo "  DIST: [${DIST[0]}, ${DIST[1]}, ${DIST[2]}]"
fi
if [ ${#LEVELS[@]} -ge 3 ]; then
    echo "  LEVELS: [${LEVELS[0]}, ${LEVELS[1]}, ${LEVELS[2]}]"
fi
echo ""

# 检查数组长度
if [ ${#DIST[@]} -ne 3 ] || [ ${#LEVELS[@]} -ne 3 ]; then
    echo "错误: 目标分布或稀疏度级别格式不正确"
    echo "  TARGET_DIST 应该是 3 个值，用逗号分隔，例如: 0.05,0.75,0.20"
    echo "  SPARSITY_LEVELS 应该是 3 个值，用逗号分隔，例如: 0.0,0.7,1.0"
    exit 1
fi

# 计算预期稀疏度（直接传递值给 Python）
EXPECTED_SPARSITY=$(python3 -c "
dist = [${DIST[0]}, ${DIST[1]}, ${DIST[2]}]
levels = [${LEVELS[0]}, ${LEVELS[1]}, ${LEVELS[2]}]
result = sum(d * l for d, l in zip(dist, levels))
print(f'{result:.3f}')
")

if [ $? -ne 0 ]; then
    echo "错误: 计算预期稀疏度失败"
    exit 1
fi

echo "配置:"
echo "  模型: $MODEL_PATH"
echo "  模式: diff_sparse_kv (观察窗口优化)"
echo "  窗口大小 (Window A/B): $WINDOW_SIZE"
echo "  观察窗口大小: $OBS_WINDOW_SIZE"
echo "  使用 Flash Attention: $USE_FLASH_ATTENTION"
echo "  目标分布: [$TARGET_DIST]"
echo "  稀疏度级别: [$SPARSITY_LEVELS]"
echo "  预期平均稀疏度: ${EXPECTED_SPARSITY}"
echo "  调试模式: $DEBUG_MODE"
echo "  统计收集: $([ $ENABLE_STATS -eq 1 ] && echo '启用' || echo '禁用')"
echo "  Importance 模式: $IMPORTANCE_MODE"
echo "  Value sink 保留: $VALUE_SINK_KEEP"
if [ -n "$TARGET_BUDGET" ]; then
    echo "  目标预算: $TARGET_BUDGET"
    echo "  预算模板: $BUDGET_TEMPLATE"
fi
echo "  数据集: 完整 LongBench (17个任务)"
echo ""

# 运行 DiffSparseKV
echo "=========================================="
echo "开始运行"
echo "=========================================="
echo ""

# 设置 PYTHONPATH 以便能够导入 models 模块（指向父目录）
export PYTHONPATH="$(dirname $PWD):$PYTHONPATH"

# 运行完整 LongBench 测试
echo "开始生成预测..."
START_TIME=$(date +%s)

CMD=(
    python
    pred_long_bench_diff_sparse.py
    --model_name_or_path "$MODEL_PATH"
    --group_size 32
    --residual_length 32
    --mode diff_sparse_kv
    --diff_sparse_target_distribution "$TARGET_DIST"
    --diff_sparse_sparsity_levels "$SPARSITY_LEVELS"
    --diff_sparse_window_size "$WINDOW_SIZE"
    --obs_window_size "$OBS_WINDOW_SIZE"
    --use_flash_attention "$USE_FLASH_ATTENTION"
    --diff_sparse_debug "$DEBUG_MODE"
    --diff_sparse_importance_mode "$IMPORTANCE_MODE"
    --diff_sparse_value_sink_keep "$VALUE_SINK_KEEP"
    --e 0
)

if [ -n "$TARGET_BUDGET" ]; then
    CMD+=(
        --diff_sparse_target_budget "$TARGET_BUDGET"
        --diff_sparse_budget_template "$BUDGET_TEMPLATE"
    )
fi

"${CMD[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "✓ DiffSparseKV 预测完成 (耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒)"
echo ""

# 检查输出
SPARSITY_LEVEL=$(python3 -c "
levels = [${LEVELS[0]}, ${LEVELS[1]}, ${LEVELS[2]}]
print(f'{levels[1]:.2f}')
")
DIFFSPARSE_DIR="pred/${MODEL_NAME}_8192_diff_sparse_kv_${SPARSITY_LEVEL}"

if [ -d "$DIFFSPARSE_DIR" ]; then
    echo "✓ 输出目录: $DIFFSPARSE_DIR"
    echo ""
    
    # 统计生成的文件
    FILE_COUNT=$(ls -1 "$DIFFSPARSE_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "生成的数据集文件数: $FILE_COUNT"
    echo ""
    
    # 列出文件及大小
    if [ $FILE_COUNT -gt 0 ]; then
        echo "文件列表:"
        ls -lh "$DIFFSPARSE_DIR"/*.jsonl 2>/dev/null | awk '{printf "  %-40s %8s\n", $9, $5}'
        echo ""
    fi
else
    echo "⚠️  输出目录不存在: $DIFFSPARSE_DIR"
    echo ""
fi

# 运行评估
echo "=========================================="
echo "运行评估"
echo "=========================================="
echo ""

EVAL_MODEL_NAME="${MODEL_NAME}_8192_diff_sparse_kv_${SPARSITY_LEVEL}"

if [ -d "$DIFFSPARSE_DIR" ] && [ $FILE_COUNT -gt 0 ]; then
    python eval_results.py --result_dir "$DIFFSPARSE_DIR"
    
    echo ""
    echo "✓ 评估完成"
    echo ""
else
    echo "⚠️  跳过评估（预测文件不存在）"
    echo ""
fi

# 显示结果
RESULT_FILE="$DIFFSPARSE_DIR/result.json"

if [ -f "$RESULT_FILE" ]; then
    echo "=========================================="
    echo "DiffSparseKV 评估结果"
    echo "=========================================="
    echo ""
    
    # 格式化显示结果
    python3 -c "
import json
import sys

try:
    with open('$RESULT_FILE') as f:
        data = json.load(f)
except Exception as e:
    print(f'错误: 无法读取结果文件 - {e}')
    sys.exit(1)
    
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
    has_data = any(dataset in data for dataset in datasets)
    if has_data:
        print(f'\n{category}:')
        for dataset in datasets:
            if dataset in data:
                score = data[dataset]
                print(f'  {dataset:25s}: {score:6.2f}')

# 计算平均分
scores = [v for k, v in data.items() if k != 'average' and isinstance(v, (int, float))]
if scores:
    avg_score = sum(scores) / len(scores)
    print('\n' + '=' * 60)
    print(f'平均分数: {avg_score:.2f} (基于 {len(scores)} 个数据集)')
    print('=' * 60)
"
    echo ""
fi

# 对比结果
echo ""
echo "=========================================="
echo "与 Baseline 对比"
echo "=========================================="
echo ""

BASELINE_FILE="./pred/${MODEL_NAME}_8192_K_0.0_V_0.0/result.json"

if [ -f "$BASELINE_FILE" ] && [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json
import sys

try:
    with open('$BASELINE_FILE') as f:
        baseline = json.load(f)
    with open('$RESULT_FILE') as f:
        diffsparse = json.load(f)
except Exception as e:
    print(f'错误: 无法读取结果文件 - {e}')
    sys.exit(1)

# 计算所有数据集的对比
print(f'{'数据集':<25s} {'Baseline':>10s} {'DiffSparse':>10s} {'差异':>10s}')
print('-' * 60)

all_datasets = set(baseline.keys()) | set(diffsparse.keys())
all_datasets.discard('average')

diffs = []
for dataset in sorted(all_datasets):
    if dataset in baseline and dataset in diffsparse:
        b_score = baseline[dataset]
        d_score = diffsparse[dataset]
        diff = d_score - b_score
        diffs.append(diff)
        print(f'{dataset:<25s} {b_score:>10.2f} {d_score:>10.2f} {diff:>+10.2f}')

if diffs:
    print('=' * 60)
    avg_diff = sum(diffs) / len(diffs)
    print(f'{'平均差异':<25s} {'':<10s} {'':<10s} {avg_diff:>+10.2f}')
    print('=' * 60)
    print()

    if avg_diff >= -0.5:
        print('✓ DiffSparseKV 性能保持良好 (差异 {:.2f})'.format(avg_diff))
    elif avg_diff >= -1.0:
        print('⚠️  DiffSparseKV 性能略有下降 (差异 {:.2f})'.format(avg_diff))
    else:
        print('⚠️  DiffSparseKV 性能下降较多 (差异 {:.2f})'.format(avg_diff))
else:
    print('⚠️  没有共同的数据集可以对比')
"
else
    if [ ! -f "$BASELINE_FILE" ]; then
        echo "⚠️  未找到 Baseline 结果文件"
        echo "   请先运行: bash run_baseline.sh"
    fi
    if [ ! -f "$RESULT_FILE" ]; then
        echo "⚠️  未找到 DiffSparseKV 结果文件"
    fi
fi

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
echo ""

# 显示稀疏度统计摘要
if [ "$DIFFSPARSE_ENABLE_STATS" = "1" ] && [ -f "$DIFFSPARSE_DIR/sparsity_statistics.json" ]; then
    echo ""
    echo "=========================================="
    echo "稀疏度统计摘要"
    echo "=========================================="
    echo ""
    
    python3 -c "
import json
import sys

try:
    with open('$DIFFSPARSE_DIR/sparsity_statistics.json') as f:
        stats = json.load(f)
except Exception as e:
    print(f'错误: 无法读取统计文件 - {e}')
    sys.exit(1)

config = stats.get('config', {})
expected = config.get('expected_avg_sparsity', 0.725)

print('预期值:')
print(f'  平均稀疏度: {expected:.1%}')
print(f'  Level分布: {config.get(\"target_distribution\", [0.05, 0.75, 0.20])}')
print()

per_dataset = stats.get('per_dataset', {})
if not per_dataset:
    print('⚠️  没有收集到统计数据')
    sys.exit(0)

print('实际统计:')
print('-' * 100)
print(f'{'数据集':<20s} {'K(最终)':>10s} {'V(最终)':>10s} {'Prefix K':>10s} {'Prefix V':>10s} {'Suffix K':>10s} {'Suffix V':>10s} {'样本数':>8s}')
print('-' * 100)

for dataset, data in sorted(per_dataset.items()):
    k = data.get('final_k', 0)
    v = data.get('final_v', 0)
    count = data.get('sample_count', 0)
    prefix_k = data.get('prefix_k', 0)
    prefix_v = data.get('prefix_v', 0)
    suffix_k = data.get('suffix_k', 0)
    suffix_v = data.get('suffix_v', 0)
    print(f'{dataset:<20s} {k:>9.1%} {v:>9.1%} {prefix_k:>9.1%} {prefix_v:>9.1%} {suffix_k:>9.1%} {suffix_v:>9.1%} {count:>8d}')

print('-' * 100)

# 计算全局平均
all_k = [d['final_k'] for d in per_dataset.values() if 'final_k' in d]
all_v = [d['final_v'] for d in per_dataset.values() if 'final_v' in d]
avg_k = sum(all_k) / len(all_k) if all_k else 0
avg_v = sum(all_v) / len(all_v) if all_v else 0

print(f'{'全局平均':<20s} {avg_k:>9.1%} {avg_v:>9.1%}')
print()

# 判断是否正常
if abs(avg_k - expected) < 0.05 and abs(avg_v - expected) < 0.05:
    print('✓ 稀疏度正常 (与预期值接近)')
else:
    print('⚠️  稀疏度异常 (与预期值差异较大)')
    print(f'   预期: {expected:.1%}, 实际: K={avg_k:.1%}, V={avg_v:.1%}')
    
    # 检查 prefix 是否被过度稀疏
    all_prefix_k = [d.get('prefix_k', 0) for d in per_dataset.values() if 'prefix_k' in d]
    if all_prefix_k:
        avg_prefix_k = sum(all_prefix_k) / len(all_prefix_k)
        if avg_prefix_k > 0.9:
            print(f'   ⚠️  Prefix 被过度稀疏 (K={avg_prefix_k:.1%})')
"
    echo ""
fi

# 显示使用提示
echo ""
echo "提示:"
echo "  - 查看详细结果: cat $RESULT_FILE"
if [ "$DIFFSPARSE_ENABLE_STATS" = "1" ]; then
    echo "  - 查看统计数据: cat $DIFFSPARSE_DIR/sparsity_statistics.json"
fi
echo "  - 重新运行评估: python eval_results.py --result_dir $DIFFSPARSE_DIR"
echo "  - 对比 Baseline: bash run_baseline.sh"
echo ""
