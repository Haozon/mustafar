#!/usr/bin/env python3
"""
分析Fisher敏感度分析结果的脚本
可视化稀疏度分配模式并提供统计信息
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def load_fisher_results(fisher_path: str) -> Dict:
    """加载Fisher敏感度分析结果"""
    with open(fisher_path, 'r') as f:
        return json.load(f)

def analyze_sparsity_distribution(optimal_sparsity: List[int]) -> Dict:
    """分析稀疏度分布统计"""
    sparsity_array = np.array(optimal_sparsity)
    
    stats = {
        'mean': np.mean(sparsity_array),
        'std': np.std(sparsity_array),
        'min': np.min(sparsity_array),
        'max': np.max(sparsity_array),
        'median': np.median(sparsity_array),
        'q25': np.percentile(sparsity_array, 25),
        'q75': np.percentile(sparsity_array, 75),
    }
    
    return stats

def plot_sparsity_distribution(optimal_sparsity: List[int], save_path: str = None):
    """绘制稀疏度分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 各层稀疏度
    layers = list(range(len(optimal_sparsity)))
    ax1.bar(layers, optimal_sparsity, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title('Fisher-based Per-Layer Sparsity Allocation')
    ax1.grid(True, alpha=0.3)
    
    # 添加平均线
    mean_sparsity = np.mean(optimal_sparsity)
    ax1.axhline(y=mean_sparsity, color='red', linestyle='--', 
                label=f'Average: {mean_sparsity:.1f}%')
    ax1.legend()
    
    # 子图2: 稀疏度分布直方图
    ax2.hist(optimal_sparsity, bins=15, alpha=0.7, color='lightgreen', 
             edgecolor='darkgreen', density=True)
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Sparsity Distribution Histogram')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats = analyze_sparsity_distribution(optimal_sparsity)
    ax2.axvline(x=stats['mean'], color='red', linestyle='-', 
                label=f"Mean: {stats['mean']:.1f}%")
    ax2.axvline(x=stats['median'], color='orange', linestyle='--', 
                label=f"Median: {stats['median']:.1f}%")
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def compare_with_uniform_sparsity(optimal_sparsity: List[int], target_avg: float):
    """比较Fisher方法与均匀稀疏度的差异"""
    uniform_sparsity = [target_avg] * len(optimal_sparsity)
    
    print("Fisher vs Uniform Sparsity Comparison:")
    print("=" * 50)
    print(f"{'Layer':<6} {'Fisher':<8} {'Uniform':<8} {'Diff':<8}")
    print("-" * 50)
    
    total_diff = 0
    for i, (fisher, uniform) in enumerate(zip(optimal_sparsity, uniform_sparsity)):
        diff = fisher - uniform
        total_diff += abs(diff)
        print(f"{i:<6} {fisher:<8.1f} {uniform:<8.1f} {diff:<+8.1f}")
    
    print("-" * 50)
    print(f"Average absolute difference: {total_diff / len(optimal_sparsity):.2f}%")

def identify_sparsity_patterns(optimal_sparsity: List[int]) -> Dict:
    """识别稀疏度分配模式"""
    n_layers = len(optimal_sparsity)
    
    # 分析不同部分的稀疏度
    early_layers = optimal_sparsity[:n_layers//3]
    middle_layers = optimal_sparsity[n_layers//3:2*n_layers//3]
    late_layers = optimal_sparsity[2*n_layers//3:]
    
    patterns = {
        'early_layers_avg': np.mean(early_layers),
        'middle_layers_avg': np.mean(middle_layers),
        'late_layers_avg': np.mean(late_layers),
        'trend': 'decreasing' if optimal_sparsity[0] > optimal_sparsity[-1] else 'increasing'
    }
    
    # 检查是否有明显的趋势
    correlation = np.corrcoef(range(n_layers), optimal_sparsity)[0, 1]
    patterns['correlation_with_depth'] = correlation
    
    return patterns

def main():
    parser = argparse.ArgumentParser(description='分析Fisher敏感度分析结果')
    parser.add_argument('--fisher_results', type=str,
                       default='fisher_sensitivity_results/fisher_sparsity_allocation.json',
                       help='Fisher敏感度分析结果文件路径')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='保存图表的路径 (可选)')
    parser.add_argument('--no_plot', action='store_true',
                       help='不显示图表')
    
    args = parser.parse_args()
    
    # 加载Fisher结果
    print("Loading Fisher sensitivity analysis results...")
    fisher_results = load_fisher_results(args.fisher_results)
    
    optimal_sparsity = fisher_results['optimal_sparsity']
    target_avg = fisher_results['target_avg_sparsity']
    actual_avg = fisher_results['actual_avg_sparsity']
    
    print(f"\nFisher Sensitivity Analysis Summary:")
    print("=" * 60)
    print(f"Method: {fisher_results['method']}")
    print(f"Solver Status: {fisher_results['solver_status']}")
    print(f"Number of Layers: {fisher_results['n_layers']}")
    print(f"Target Average Sparsity: {target_avg}%")
    print(f"Actual Average Sparsity: {actual_avg}%")
    print(f"Sparsity Range: {fisher_results['min_sparsity']}% - {fisher_results['max_sparsity']}%")
    print(f"Total Sensitivity: {fisher_results['total_sensitivity']:.2f}")
    
    # 统计分析
    print(f"\nSparsity Distribution Statistics:")
    print("=" * 60)
    stats = analyze_sparsity_distribution(optimal_sparsity)
    print(f"Mean: {stats['mean']:.2f}%")
    print(f"Standard Deviation: {stats['std']:.2f}%")
    print(f"Median: {stats['median']:.2f}%")
    print(f"Min: {stats['min']:.2f}%")
    print(f"Max: {stats['max']:.2f}%")
    print(f"25th Percentile: {stats['q25']:.2f}%")
    print(f"75th Percentile: {stats['q75']:.2f}%")
    
    # 模式分析
    print(f"\nSparsity Allocation Patterns:")
    print("=" * 60)
    patterns = identify_sparsity_patterns(optimal_sparsity)
    print(f"Early Layers (0-{len(optimal_sparsity)//3-1}) Average: {patterns['early_layers_avg']:.2f}%")
    print(f"Middle Layers ({len(optimal_sparsity)//3}-{2*len(optimal_sparsity)//3-1}) Average: {patterns['middle_layers_avg']:.2f}%")
    print(f"Late Layers ({2*len(optimal_sparsity)//3}-{len(optimal_sparsity)-1}) Average: {patterns['late_layers_avg']:.2f}%")
    print(f"Overall Trend: {patterns['trend']}")
    print(f"Correlation with Layer Depth: {patterns['correlation_with_depth']:.3f}")
    
    # 详细的层级稀疏度
    print(f"\nPer-Layer Sparsity Configuration:")
    print("=" * 60)
    for i, sparsity in enumerate(optimal_sparsity):
        print(f"Layer {i:2d}: {sparsity:2d}%")
    
    # 与均匀稀疏度比较
    print(f"\n")
    compare_with_uniform_sparsity(optimal_sparsity, target_avg)
    
    # 生成可视化
    if not args.no_plot:
        print(f"\nGenerating visualization...")
        plot_sparsity_distribution(optimal_sparsity, args.save_plot)
    
    # 生成用于其他脚本的配置
    print(f"\nGenerated Configuration for eval_fisher_sparsity.py:")
    print("=" * 60)
    print("Sparsity config (as decimal):")
    sparsity_config = {i: s/100.0 for i, s in enumerate(optimal_sparsity)}
    print(json.dumps(sparsity_config, indent=2))

if __name__ == "__main__":
    main()