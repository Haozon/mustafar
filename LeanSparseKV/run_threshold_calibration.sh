#!/bin/bash

# LeanSparseKV 阈值校准脚本
# 自动运行完整的阈值校准和验证流程

set -e  # 遇到错误立即退出

echo "🎯 LeanSparseKV 阈值校准系统"
echo "=========================="

# 激活conda环境
echo "🔧 激活conda环境: mustar"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mustar

# 参数配置
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="calibration_results"
DATASET="wikitext"
NUM_SAMPLES=200
TARGET_SPARSITY="0.70"
SEQ_LENGTH=2048  # 统一序列长度，确保校准和验证一致

# 检查参数
if [ "$MODEL_PATH" = "/path/to/your/model" ]; then
    echo "❌ 错误: 请提供有效的模型路径"
    echo "用法: $0 <model_path> [output_dir] [dataset] [num_samples] [target_sparsity]"
    echo "示例: $0 /home/user/llama-3-8b-instruct results math 200 0.70"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "📋 配置信息:"
echo "  模型路径: $MODEL_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  数据集: $DATASET"
echo "  样本数量: $NUM_SAMPLES"
echo "  目标稀疏度: $TARGET_SPARSITY"
echo "  序列长度: $SEQ_LENGTH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查Python脚本是否存在
if [ ! -f "calibrate_sparsity_thresholds.py" ]; then
    echo "❌ 错误: calibrate_sparsity_thresholds.py 文件不存在"
    exit 1
fi

if [ ! -f "validate_thresholds.py" ]; then
    echo "❌ 错误: validate_thresholds.py 文件不存在"
    exit 1
fi

# 步骤1: 阈值校准
echo "🔍 步骤1: 运行阈值校准..."
echo "-----------------------------"
python calibrate_sparsity_thresholds.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --num_samples $NUM_SAMPLES \
    --target_sparsity $TARGET_SPARSITY \
    --granularity "per_layer" \
    --batch_size 4 \
    --seq_len $SEQ_LENGTH

if [ $? -ne 0 ]; then
    echo "❌ 阈值校准失败"
    exit 1
fi

echo "✅ 阈值校准完成"
echo ""

# 步骤2: 阈值验证
echo "🔍 步骤2: 运行阈值验证..."
echo "-----------------------------"
python validate_thresholds.py \
    --thresholds_file "$OUTPUT_DIR/thresholds.json" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR/validation" \
    --dataset "$DATASET" \
    --num_samples 100 \
    --batch_size 4 \
    --max_length $SEQ_LENGTH

if [ $? -ne 0 ]; then
    echo "❌ 阈值验证失败"
    exit 1
fi

echo "✅ 阈值验证完成"
echo ""

# 步骤3: 生成总结报告
echo "📊 步骤3: 生成总结报告..."
echo "-----------------------------"

python -c "
import json
import os
import numpy as np

output_dir = '$OUTPUT_DIR'
print('\\n🎯 LeanSparseKV 校准总结报告')
print('=' * 40)

# 加载校准结果
verification_file = os.path.join(output_dir, 'verification_results.json')
validation_file = os.path.join(output_dir, 'validation', 'validation_results.json')
thresholds_file = os.path.join(output_dir, 'thresholds.json')

success_count = 0
total_checks = 3

# 检查校准结果
if os.path.exists(verification_file):
    try:
        with open(verification_file, 'r') as f:
            results = json.load(f)
        
        sparsity_errors = []
        actual_sparsities = []
        
        for key, data in results.items():
            if isinstance(data, dict) and 'sparsity_error' in data:
                sparsity_errors.append(data['sparsity_error'])
                actual_sparsities.append(data['actual_sparsity'])
        
        if sparsity_errors:
            avg_sparsity_error = np.mean(sparsity_errors)
            avg_actual_sparsity = np.mean(actual_sparsities)
            layers_within_tolerance = sum(1 for e in sparsity_errors if e <= 0.02)
            
            print(f'\\n📈 校准结果:')
            print(f'  目标稀疏度: {float(\"$TARGET_SPARSITY\"):.1%}')
            print(f'  实际平均稀疏度: {avg_actual_sparsity:.1%}')
            print(f'  平均稀疏度误差: {avg_sparsity_error:.4f}')
            print(f'  误差≤2%的层数: {layers_within_tolerance}/{len(sparsity_errors)}')
            
            if avg_sparsity_error <= 0.02:
                print('  状态: ✅ 校准成功')
                success_count += 1
            else:
                print('  状态: ⚠️ 校准精度待改进')
        else:
            print('\\n❌ 无有效校准结果')
    except Exception as e:
        print(f'\\n❌ 读取校准结果失败: {e}')
else:
    print('\\n❌ 校准结果文件不存在')

# 检查验证结果
if os.path.exists(validation_file):
    try:
        with open(validation_file, 'r') as f:
            validation = json.load(f)
        
        overall = validation['overall_stats']
        print(f'\\n🔍 验证结果:')
        print(f'  验证稀疏度: {overall[\"avg_sparsity\"]:.1%}')
        print(f'  验证误差: {overall[\"sparsity_error\"]:.4f}')
        
        if overall['sparsity_error'] <= 0.02:
            print('  状态: ✅ 验证通过')
            success_count += 1
        else:
            print('  状态: ⚠️ 验证精度待改进')
    except Exception as e:
        print(f'\\n❌ 读取验证结果失败: {e}')
else:
    print('\\n⚠️ 验证结果文件不存在')

# 检查阈值文件
if os.path.exists(thresholds_file):
    try:
        with open(thresholds_file, 'r') as f:
            thresholds = json.load(f)
        
        print(f'\\n🎯 阈值信息:')
        print(f'  粒度: {thresholds[\"granularity\"]}')
        print(f'  阈值数量: {len(thresholds[\"thresholds\"])}')
        
        # 显示示例阈值
        if thresholds['thresholds']:
            sample_key = list(thresholds['thresholds'].keys())[0]
            sample_thresh = thresholds['thresholds'][sample_key]
            print(f'  示例阈值 (Layer {sample_key}):')
            print(f'    α_h: {sample_thresh[\"alpha_h\"]:.4f}')
            print(f'    α_mh: {sample_thresh[\"alpha_mh\"]:.4f}')
            print(f'    α_m: {sample_thresh[\"alpha_m\"]:.4f}')
            print(f'    α_ml: {sample_thresh[\"alpha_ml\"]:.4f}')
        
        print('  状态: ✅ 阈值文件生成成功')
        success_count += 1
    except Exception as e:
        print(f'\\n❌ 读取阈值文件失败: {e}')
else:
    print('\\n❌ 阈值文件不存在')

print(f'\\n📊 总体状态: {success_count}/{total_checks} 项检查通过')
if success_count == total_checks:
    print('🎉 校准流程完全成功!')
elif success_count >= 2:
    print('✅ 校准流程基本成功')
else:
    print('⚠️ 校准流程需要检查')
"

echo ""
echo "🎉 校准流程完成!"
echo ""
echo "📁 生成的文件:"
echo "  - $OUTPUT_DIR/thresholds.json: 校准得到的阈值参数"
echo "  - $OUTPUT_DIR/verification_results.json: 校准验证结果"
echo "  - $OUTPUT_DIR/threshold_analysis.png: 阈值分析图表"
echo "  - $OUTPUT_DIR/validation/validation_results.json: 验证结果"
echo "  - $OUTPUT_DIR/validation/validation_report.png: 验证报告图表"
echo ""
echo "🚀 下一步:"
echo "  1. 检查上述总结报告，确认校准质量"
echo "  2. 在你的应用中使用 $OUTPUT_DIR/thresholds.json"
echo "  3. 在目标任务上测试稀疏化效果"