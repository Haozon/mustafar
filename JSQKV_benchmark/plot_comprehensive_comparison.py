#!/usr/bin/env python3
"""
综合性能对比图生成脚本
整合 JSQKV benchmark 和量化测试结果
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')

# 设置绘图样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 颜色方案
COLORS = {
    'dense': '#2E86AB',           # 深蓝
    'sparse_50': '#A23B72',       # 紫红
    'sparse_70': '#F18F01',       # 橙色
    'sparse_50_quant_2bit': '#C73E1D'  # 深红
}

MARKERS = {
    'dense': 'o',
    'sparse_50': 's',
    'sparse_70': '^',
    'sparse_50_quant_2bit': 'D'
}

LABELS = {
    'dense': 'Dense (FP16)',
    'sparse_50': 'Sparse-50% (FP16)',
    'sparse_70': 'Sparse-70% (FP16)',
    'sparse_50_quant_2bit': 'Sparse-50% + Quant-2bit'
}

def load_jsqkv_results():
    """加载 JSQKV benchmark 结果"""
    results_file = Path('JSQKV_benchmark/results/raw_data/llama3_8b_results_20260203_204148.json')
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded JSQKV results from: {results_file}")
    return data

def add_quant_results(data):
    """添加量化测试结果到数据中"""
    # 从 mem_spd_test_quant_results_2bit.txt 提取的数据
    quant_data = {
        'throughput': 34.50,  # tokens/sec (8192 tokens / 237.436 sec)
        'ttft': 10131.27,     # ms
        'tpot': 227.07,       # ms
        'peak_memory': 41.28, # GB
        'batch_size': 8,
        'input_length': 4098,
        'output_length': 1024,
        'total_tokens': 8192
    }
    
    # 添加到数据结构中
    if 'sparse_50_quant_2bit' not in data:
        data['sparse_50_quant_2bit'] = {}
    
    data['sparse_50_quant_2bit']['8'] = quant_data
    
    print(f"✅ Added quantization results (Batch=8)")
    return data

def plot_comprehensive_comparison(data, output_dir='results/plots'):
    """生成综合对比图"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mustafar Comprehensive Performance Comparison\n(Meta-Llama-3-8B-Instruct, Batch Size = 8)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    configs = ['dense', 'sparse_50', 'sparse_70', 'sparse_50_quant_2bit']
    
    # 提取 batch size = 8 的数据
    metrics = {
        'throughput': [],
        'ttft': [],
        'tpot': [],
        'peak_memory': []
    }
    
    labels_list = []
    colors_list = []
    
    for config in configs:
        if config in data and '8' in data[config]:
            config_data = data[config]['8']
            metrics['throughput'].append(config_data['throughput'])
            metrics['ttft'].append(config_data['ttft'])
            metrics['tpot'].append(config_data['tpot'])
            metrics['peak_memory'].append(config_data['peak_memory'])
            labels_list.append(LABELS[config])
            colors_list.append(COLORS[config])
    
    # 1. 吞吐量对比 (左上)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(labels_list)), metrics['throughput'], 
                    color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Throughput Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(labels_list)))
    ax1.set_xticklabels(labels_list, rotation=15, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, metrics['throughput'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加相对性能标注
    baseline = metrics['throughput'][0]  # Dense baseline
    for i, val in enumerate(metrics['throughput']):
        if i > 0:
            speedup = val / baseline
            color = 'green' if speedup > 1 else 'red'
            ax1.text(i, val * 0.5, f'{speedup:.2f}x',
                    ha='center', va='center', fontsize=9, 
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 2. TTFT 对比 (右上)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(labels_list)), metrics['ttft'], 
                    color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('TTFT (ms)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Time to First Token', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(labels_list)))
    ax2.set_xticklabels(labels_list, rotation=15, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, val in zip(bars2, metrics['ttft']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. TPOT 对比 (左下)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(labels_list)), metrics['tpot'], 
                    color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('TPOT (ms)', fontsize=13, fontweight='bold')
    ax3.set_title('(c) Time per Output Token', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(labels_list)))
    ax3.set_xticklabels(labels_list, rotation=15, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, val in zip(bars3, metrics['tpot']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 内存占用对比 (右下)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(labels_list)), metrics['peak_memory'], 
                    color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Peak Memory (GB)', fontsize=13, fontweight='bold')
    ax4.set_title('(d) Peak Memory Usage', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(labels_list)))
    ax4.set_xticklabels(labels_list, rotation=15, ha='right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, val in zip(bars4, metrics['peak_memory']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加内存节省标注
    baseline_mem = metrics['peak_memory'][0]
    for i, val in enumerate(metrics['peak_memory']):
        if i > 0:
            saving = (baseline_mem - val) / baseline_mem * 100
            if saving > 0:
                ax4.text(i, val * 0.5, f'-{saving:.1f}%',
                        ha='center', va='center', fontsize=9, 
                        color='green', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / 'comprehensive_comparison.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    
    png_path = Path(output_dir) / 'comprehensive_comparison.png'
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {png_path}")
    
    plt.close()

def plot_performance_tradeoff(data, output_dir='results/plots'):
    """绘制性能权衡图 (吞吐量 vs 内存)"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    configs = ['dense', 'sparse_50', 'sparse_70', 'sparse_50_quant_2bit']
    
    for config in configs:
        if config in data and '8' in data[config]:
            config_data = data[config]['8']
            throughput = config_data['throughput']
            memory = config_data['peak_memory']
            
            ax.scatter(memory, throughput, 
                      s=300, 
                      color=COLORS[config], 
                      marker=MARKERS[config],
                      alpha=0.7,
                      edgecolors='black',
                      linewidths=2,
                      label=LABELS[config],
                      zorder=3)
            
            # 添加标签
            ax.annotate(LABELS[config], 
                       xy=(memory, throughput),
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=COLORS[config], 
                                alpha=0.3,
                                edgecolor='black'),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0',
                                     color='black',
                                     lw=1.5))
    
    ax.set_xlabel('Peak Memory (GB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Memory Trade-off\n(Batch Size = 8)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 添加理想区域标注 (右上角 = 高吞吐量 + 低内存)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax.axvline(x=40, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax.text(39, 135, 'Ideal Region\n(High Throughput\nLow Memory)', 
           fontsize=10, color='green', fontweight='bold',
           ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'performance_memory_tradeoff.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    
    png_path = Path(output_dir) / 'performance_memory_tradeoff.png'
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {png_path}")
    
    plt.close()

def plot_normalized_comparison(data, output_dir='results/plots'):
    """绘制归一化对比图 (以 Dense 为基准)"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = ['dense', 'sparse_50', 'sparse_70', 'sparse_50_quant_2bit']
    metrics_names = ['Throughput', 'TTFT\n(inverse)', 'TPOT\n(inverse)', 'Memory\n(inverse)']
    
    # 提取数据并归一化
    baseline_data = data['dense']['8']
    
    normalized_data = []
    for config in configs:
        if config in data and '8' in data[config]:
            config_data = data[config]['8']
            
            # 归一化 (相对于 Dense baseline)
            # 吞吐量: 越高越好
            # TTFT/TPOT/Memory: 越低越好，所以取倒数
            norm_throughput = config_data['throughput'] / baseline_data['throughput']
            norm_ttft = baseline_data['ttft'] / config_data['ttft']
            norm_tpot = baseline_data['tpot'] / config_data['tpot']
            norm_memory = baseline_data['peak_memory'] / config_data['peak_memory']
            
            normalized_data.append([norm_throughput, norm_ttft, norm_tpot, norm_memory])
    
    # 绘制雷达图
    x = np.arange(len(metrics_names))
    width = 0.2
    
    for i, (config, norm_values) in enumerate(zip(configs, normalized_data)):
        offset = (i - len(configs)/2 + 0.5) * width
        bars = ax.bar(x + offset, norm_values, width, 
                     label=LABELS[config],
                     color=COLORS[config],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5)
        
        # 添加数值标签
        for bar, val in zip(bars, norm_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}x',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Normalized Performance\n(relative to Dense baseline)', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Normalized Performance Comparison\n(Higher is Better)', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Dense Baseline')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'normalized_comparison.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {output_path}")
    
    png_path = Path(output_dir) / 'normalized_comparison.png'
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {png_path}")
    
    plt.close()

def main():
    print("="*70)
    print("Mustafar Comprehensive Performance Comparison")
    print("="*70)
    
    # 加载 JSQKV 结果
    data = load_jsqkv_results()
    if data is None:
        return
    
    # 添加量化结果
    data = add_quant_results(data)
    
    # 生成图表
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70 + "\n")
    
    plot_comprehensive_comparison(data)
    plot_performance_tradeoff(data)
    plot_normalized_comparison(data)
    
    print("\n" + "="*70)
    print("✅ All plots generated successfully!")
    print("="*70)
    print("\nOutput files:")
    print("  - results/plots/comprehensive_comparison.pdf/png")
    print("  - results/plots/performance_memory_tradeoff.pdf/png")
    print("  - results/plots/normalized_comparison.pdf/png")
    print()

if __name__ == '__main__':
    main()
