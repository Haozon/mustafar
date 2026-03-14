#!/usr/bin/env python3
"""
分析 Nsys Profile 结果
对比 Mustafar 和 Mustafar_Quant 的性能
"""
import pandas as pd
import json

def parse_nsys_csv(csv_file):
    """解析 nsys 导出的 CSV 文件"""
    df = pd.read_csv(csv_file)
    
    # 提取关键 kernel
    kernels = {
        'compression': [],
        'spmv': [],
        'local_mv': [],
        'pruning': [],
        'other': []
    }
    
    for _, row in df.iterrows():
        kernel_name = row['Kernel Name']
        time_pct = row['Time (%)']
        time_ms = row['Time (ms)']
        
        if 'compress' in kernel_name.lower():
            kernels['compression'].append((kernel_name, time_pct, time_ms))
        elif 'spmv' in kernel_name.lower() or 'formulation' in kernel_name.lower():
            kernels['spmv'].append((kernel_name, time_pct, time_ms))
        elif 'local' in kernel_name.lower() or 'window' in kernel_name.lower():
            kernels['local_mv'].append((kernel_name, time_pct, time_ms))
        elif 'prun' in kernel_name.lower():
            kernels['pruning'].append((kernel_name, time_pct, time_ms))
        else:
            kernels['other'].append((kernel_name, time_pct, time_ms))
    
    return kernels

def summarize_kernels(kernels):
    """汇总各类 kernel 的时间"""
    summary = {}
    
    for category, kernel_list in kernels.items():
        total_pct = sum(k[1] for k in kernel_list)
        total_ms = sum(k[2] for k in kernel_list)
        summary[category] = {
            'time_pct': total_pct,
            'time_ms': total_ms,
            'count': len(kernel_list)
        }
    
    return summary

def compare_results(mustafar_csv, mustafar_quant_csv):
    """对比两个版本的性能"""
    print("="*70)
    print("Mustafar vs Mustafar_Quant Performance Comparison")
    print("="*70)
    
    # 解析结果
    mustafar_kernels = parse_nsys_csv(mustafar_csv)
    mustafar_quant_kernels = parse_nsys_csv(mustafar_quant_csv)
    
    # 汇总
    mustafar_summary = summarize_kernels(mustafar_kernels)
    mustafar_quant_summary = summarize_kernels(mustafar_quant_kernels)
    
    # 打印对比
    print(f"\n{'Component':<15} {'Mustafar %':<12} {'Quant %':<12} {'Speedup':<10}")
    print("-"*70)
    
    for component in ['compression', 'spmv', 'local_mv', 'pruning']:
        mustafar_pct = mustafar_summary.get(component, {}).get('time_pct', 0)
        quant_pct = mustafar_quant_summary.get(component, {}).get('time_pct', 0)
        
        if mustafar_pct > 0 and quant_pct > 0:
            speedup = mustafar_pct / quant_pct
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{component:<15} {mustafar_pct:<12.2f} {quant_pct:<12.2f} {speedup_str:<10}")
    
    # 总体对比
    mustafar_total = sum(s['time_pct'] for s in mustafar_summary.values())
    quant_total = sum(s['time_pct'] for s in mustafar_quant_summary.values())
    
    print("-"*70)
    print(f"{'Total':<15} {mustafar_total:<12.2f} {quant_total:<12.2f} {mustafar_total/quant_total:.2f}x")
    
    # 与论文对比
    print(f"\n{'='*70}")
    print("Comparison with Paper (Llama-2-7B, 50% sparsity)")
    print(f"{'='*70}")
    
    paper_values = {
        'compression': 6.25,
        'spmv': 81.07,
        'local_mv': 0.62,
        'pruning': 1.84
    }
    
    print(f"\n{'Component':<15} {'Paper %':<12} {'Ours %':<12} {'Ratio':<10}")
    print("-"*70)
    
    for component, paper_pct in paper_values.items():
        our_pct = mustafar_summary.get(component, {}).get('time_pct', 0)
        
        if our_pct > 0:
            ratio = our_pct / paper_pct
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"
        
        print(f"{component:<15} {paper_pct:<12.2f} {our_pct:<12.2f} {ratio_str:<10}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python analyze_nsys_results.py mustafar_kernels.csv mustafar_quant_kernels.csv")
        sys.exit(1)
    
    mustafar_csv = sys.argv[1]
    mustafar_quant_csv = sys.argv[2]
    
    compare_results(mustafar_csv, mustafar_quant_csv)
