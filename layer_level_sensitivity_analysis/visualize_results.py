"""
可视化贪心搜索结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_results(result_file="./sensitivity_results/greedy_search_results.json"):
    """加载结果文件"""
    with open(result_file, 'r') as f:
        return json.load(f)

def visualize_results(results):
    """可视化结果"""
    
    sparsity_config = results['sparsity_config']
    sensitivity = results['sensitivity_scores']
    iteration_history = results['iteration_history']
    
    # 转换为列表
    layers = sorted([int(k) for k in sparsity_config.keys()])
    sparsities = [sparsity_config[str(l)] for l in layers]
    sensitivities = [sensitivity[str(l)] for l in layers]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================
    # 图1: 每层稀疏度配置
    # ========================================
    ax1 = axes[0, 0]
    colors = ['green' if s < 0.4 else 'orange' if s < 0.6 else 'red' for s in sparsities]
    ax1.bar(layers, sparsities, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Sparsity', fontsize=12)
    ax1.set_title('Layer-wise Sparsity Configuration', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(np.mean(sparsities), color='red', linestyle='--', 
                label=f'Mean = {np.mean(sparsities):.3f}')
    ax1.legend()
    
    # ========================================
    # 图2: 敏感度分数
    # ========================================
    ax2 = axes[0, 1]
    colors2 = ['red' if s > 0.7 else 'orange' if s > 0.4 else 'green' for s in sensitivities]
    ax2.bar(layers, sensitivities, color=colors2, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Normalized Sensitivity', fontsize=12)
    ax2.set_title('Layer Sensitivity Scores', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(np.mean(sensitivities), color='red', linestyle='--',
                label=f'Mean = {np.mean(sensitivities):.3f}')
    ax2.legend()
    
    # ========================================
    # 图3: 迭代过程 - 评估指标 vs Avg Sparsity
    # ========================================
    ax3 = axes[1, 0]
    iterations = [h['iteration'] for h in iteration_history]
    
    # 检查使用的评估指标
    eval_metric = results.get('config', {}).get('eval_metric', 'ppl')
    if eval_metric == 'ppl':
        metric_values = [h.get('perplexity', h.get('metric_value', 0)) for h in iteration_history]
        metric_label = 'Perplexity'
        title_suffix = 'Perplexity vs Sparsity'
    else:  # loss
        metric_values = [h.get('metric_value', h.get('loss', 0)) for h in iteration_history]
        metric_label = 'Loss'
        title_suffix = 'Loss vs Sparsity'
    
    avg_sparsities = [h.get('avg_sparsity', h.get('compression', 0)) for h in iteration_history]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(iterations, metric_values, 'b-o', label=metric_label, linewidth=2)
    line2 = ax3_twin.plot(iterations, avg_sparsities, 'r-s', label='Avg Sparsity', linewidth=2)
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel(metric_label, fontsize=12, color='b')
    ax3_twin.set_ylabel('Average Sparsity', fontsize=12, color='r')
    ax3.set_title(f'Search Progress: {title_suffix}', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # ========================================
    # 图4: 稀疏度 vs 敏感度散点图
    # ========================================
    ax4 = axes[1, 1]
    scatter = ax4.scatter(sparsities, sensitivities, s=100, alpha=0.6, 
                         c=layers, cmap='viridis', edgecolors='black')
    
    # 添加趋势线
    z = np.polyfit(sparsities, sensitivities, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(sparsities), p(sorted(sparsities)), "r--", alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('Sparsity', fontsize=12)
    ax4.set_ylabel('Sensitivity', fontsize=12)
    ax4.set_title('Sparsity vs Sensitivity Correlation', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Layer Index', fontsize=10)
    
    # 计算相关系数
    corr = np.corrcoef(sparsities, sensitivities)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "./sensitivity_results"
    if "loss" in str(results.get('config', {}).get('eval_metric', '')):
        output_dir = "./sensitivity_results_loss"
    
    output_file = f"{output_dir}/greedy_search_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("📊 GREEDY SEARCH RESULTS SUMMARY")
    print("="*70)
    
    config = results['config']
    sparsity_config = results['sparsity_config']
    sensitivity = results['sensitivity_scores']
    
    print(f"\n🔹 Configuration:")
    print(f"   Model: {config['model_path'].split('/')[-1]}")
    print(f"   Initial sparsity: {config.get('initial_sparsity', 'N/A')}")
    print(f"   Step size: {config.get('step_size', 'N/A')}")
    print(f"   Target sparsity: {config.get('target_sparsity', config.get('target_compression', 'N/A'))}")
    print(f"   Validation samples: {config['num_samples']}")
    print(f"   Number of layers: {config['num_layers']}")
    print(f"   Evaluation metric: {config.get('eval_metric', 'ppl')}")
    
    print(f"\n🔹 Final Results:")
    print(f"   Final avg sparsity: {results.get('final_avg_sparsity', results.get('final_compression', 'N/A')):.3f}")
    print(f"   Total iterations: {len(results['iteration_history'])}")
    
    # 转换为列表
    sparsities = [sparsity_config[str(l)] for l in range(config['num_layers'])]
    sensitivities = [sensitivity[str(l)] for l in range(config['num_layers'])]
    
    print(f"\n🔹 Sparsity Statistics:")
    print(f"   Mean: {np.mean(sparsities):.3f}")
    print(f"   Std:  {np.std(sparsities):.3f}")
    print(f"   Min:  {np.min(sparsities):.3f}")
    print(f"   Max:  {np.max(sparsities):.3f}")
    
    print(f"\n🔹 Top 10 Most Sparse Layers:")
    sorted_layers = sorted(enumerate(sparsities), key=lambda x: x[1], reverse=True)
    for i, (layer_idx, sparsity) in enumerate(sorted_layers[:10]):
        print(f"   {i+1}. Layer {layer_idx:2d}: {sparsity:.2f}")
    
    print(f"\n🔹 Top 10 Most Sensitive Layers:")
    sorted_sens = sorted(enumerate(sensitivities), key=lambda x: x[1], reverse=True)
    for i, (layer_idx, sens) in enumerate(sorted_sens[:10]):
        print(f"   {i+1}. Layer {layer_idx:2d}: {sens:.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """主函数"""
    import sys
    
    result_file = "./sensitivity_results/greedy_search_results.json"
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    
    print(f"Loading results from: {result_file}")
    results = load_results(result_file)
    
    # 打印摘要
    print_summary(results)
    
    # 可视化
    visualize_results(results)


if __name__ == "__main__":
    main()
