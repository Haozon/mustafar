#!/bin/bash

# 完整对比分析：PPL vs Loss 评估方法

echo "========================================"
echo "🚀 完整对比分析：PPL vs Loss 评估方法"
echo "========================================"
echo ""

# 检查是否已有PPL结果
if [ -f "./sensitivity_results/greedy_search_results.json" ]; then
    echo "✅ PPL结果已存在，跳过PPL分析"
else
    echo "🔄 运行PPL-based分析..."
    bash sensitivity_analysis/run_warm_start_search.sh
    if [ $? -ne 0 ]; then
        echo "❌ PPL分析失败"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo ""

# 检查是否已有Loss结果
if [ -f "./sensitivity_results_loss/greedy_search_results.json" ]; then
    echo "✅ Loss结果已存在，跳过Loss分析"
else
    echo "🔄 运行Loss-based分析..."
    bash sensitivity_analysis/run_loss_based_search.sh
    if [ $? -ne 0 ]; then
        echo "❌ Loss分析失败"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo ""

# 对比分析
echo "🔄 进行对比分析..."
python sensitivity_analysis/compare_eval_methods.py

echo ""
echo "========================================"
echo "✅ 完整对比分析完成！"
echo "========================================"
echo ""
echo "📁 结果文件："
echo "   PPL结果:  ./sensitivity_results/greedy_search_results.json"
echo "   Loss结果: ./sensitivity_results_loss/greedy_search_results.json"
echo "   对比图表: ./sensitivity_results/eval_methods_comparison.png"
echo ""
echo "📊 可视化文件："
echo "   PPL可视化:  ./sensitivity_results/greedy_search_visualization.png"
echo "   Loss可视化: ./sensitivity_results_loss/greedy_search_visualization.png"
echo ""