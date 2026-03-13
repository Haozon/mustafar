"""
JSQKV Benchmark - 绘图脚本
根据配置文件生成多种对比图（PDF矢量图）
"""
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config

# 设置绘图样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体
plt.rcParams['ps.fonttype'] = 42

def load_results(results_dir='results/raw_data'):
    """加载测试结果"""
    results = {}
    
    for filename in os.listdir(results_dir):
        # 只加载格式为 xxx_results.json 的文件（不包含时间戳的）
        # 时间戳格式: xxx_results_20260203_204148.json
        if filename.endswith('_results.json'):
            # 检查是否包含时间戳（连续的日期数字）
            # 如果文件名中有类似 _20260203_ 这样的模式，说明是带时间戳的备份文件
            parts = filename.replace('.json', '').split('_')
            # 如果倒数第二个部分是8位数字（日期），说明是带时间戳的文件，跳过
            if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
                continue
            
            model_name = filename.replace('_results.json', '')
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
            
            print(f"✅ Loaded results for: {model_name}")
    
    return results

def plot_scheme(scheme_name, scheme_config, results, test_configs, model_configs, output_dir='results/plots'):
    """根据方案配置绘制对比图"""
    
    print(f"\n{'='*70}")
    print(f"Generating plot: {scheme_config['title']}")
    print(f"{'='*70}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs_to_plot = scheme_config['configs']
    styles = scheme_config['styles']
    
    # 获取模型列表
    model_names = list(results.keys())
    
    if len(model_names) < 2:
        print(f"⚠️  Warning: Need at least 2 models for comparison, found {len(model_names)}")
        return
    
    # 绘制每个模型的结果
    for idx, model_name in enumerate(sorted(model_names)):
        ax = axes[idx]
        model_data = results[model_name]
        model_display_name = model_configs.get(model_name, {}).get('display_name', model_name)
        
        # 绘制每个配置
        for config_name in configs_to_plot:
            if config_name not in model_data:
                print(f"  ⚠️  Config '{config_name}' not found in {model_name} results, skipping...")
                continue
            
            config_data = model_data[config_name]
            
            # 提取数据
            batch_sizes = sorted([int(bs) for bs in config_data.keys()])
            throughputs = [config_data[str(bs)]['throughput'] for bs in batch_sizes]
            
            # 获取样式
            style = styles.get(config_name, {})
            display_name = test_configs[config_name]['display_name']
            
            # 绘制曲线
            ax.plot(
                batch_sizes, 
                throughputs,
                color=style.get('color', 'blue'),
                marker=style.get('marker', 'o'),
                linestyle=style.get('linestyle', '-'),
                linewidth=2.5,
                markersize=9,
                label=display_name,
                markeredgewidth=1.5,
                markeredgecolor='white'
            )
            
            print(f"  ✅ Plotted: {config_name} for {model_name}")
        
        # 设置坐标轴
        ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
        ax.set_ylabel('Throughput (Tokens/Seconds)', fontsize=14, fontweight='bold')
        
        # 设置标题
        subplot_label = chr(97 + idx)  # 'a', 'b', 'c', ...
        ax.set_title(f'({subplot_label}) {model_display_name} Throughput', fontsize=14, fontweight='bold')
        
        # 设置图例
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 设置刻度
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 设置x轴刻度为整数
        ax.set_xticks(batch_sizes)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    output_path = os.path.join(output_dir, f"{scheme_config['name']}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n✅ Saved plot: {output_path}")
    
    # 同时保存PNG版本（便于预览）
    png_path = os.path.join(output_dir, f"{scheme_config['name']}.png")
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✅ Saved preview: {png_path}")
    
    plt.close()

def generate_all_plots(config, results, output_dir='results/plots', schemes_to_plot=None):
    """生成所有绘图方案"""
    
    plot_schemes = config['plot_schemes']
    test_configs = config['test_configs']
    model_configs = config['models']
    
    if schemes_to_plot is None:
        schemes_to_plot = list(plot_schemes.keys())
    
    print(f"\n{'='*70}")
    print(f"Generating plots for schemes: {schemes_to_plot}")
    print(f"{'='*70}")
    
    for scheme_name in schemes_to_plot:
        if scheme_name not in plot_schemes:
            print(f"⚠️  Warning: Scheme '{scheme_name}' not found in config, skipping...")
            continue
        
        scheme_config = plot_schemes[scheme_name]
        
        plot_scheme(
            scheme_name=scheme_name,
            scheme_config=scheme_config,
            results=results,
            test_configs=test_configs,
            model_configs=model_configs,
            output_dir=output_dir
        )

def main():
    parser = argparse.ArgumentParser(description='JSQKV Benchmark - Plot Generator')
    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--results-dir', type=str, default='results/raw_data',
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Output directory for plots')
    parser.add_argument('--schemes', type=str, nargs='+', default=None,
                        help='Specific schemes to plot (default: all)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载结果
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found! Please run benchmark_throughput.py first.")
        return
    
    # 生成图表
    generate_all_plots(
        config=config,
        results=results,
        output_dir=args.output_dir,
        schemes_to_plot=args.schemes
    )
    
    print(f"\n{'='*70}")
    print(f"✅ All plots generated successfully!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
