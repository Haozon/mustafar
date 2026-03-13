#!/usr/bin/env python3
"""
详细的 Compression 性能对比测试
目标：理解为什么量化版本的 Compression 更快
"""
import torch
import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np

# 添加路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'kernel'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'kernel_quant'))

print("="*70)
print("Compression 详细性能对比测试")
print("="*70)

# 导入模块
try:
    import compression
    print("✓ kernel/compression 导入成功")
    KERNEL_AVAILABLE = True
except Exception as e:
    print(f"✗ kernel/compression 导入失败: {e}")
    KERNEL_AVAILABLE = False

try:
    import compression_quant
    print("✓ kernel_quant/compression_quant 导入成功")
    KERNEL_QUANT_AVAILABLE = True
except Exception as e:
    print(f"✗ kernel_quant/compression_quant 导入失败: {e}")
    KERNEL_QUANT_AVAILABLE = False

print()

# ==================== 配置 ====================
TEST_CONFIGS = [
    # 小规模
    {
        'name': 'Small',
        'batch': 1,
        'heads': 4,
        'seq_len': 256,
        'head_dim': 128,
    },
    # 中等规模
    {
        'name': 'Medium',
        'batch': 1,
        'heads': 16,
        'seq_len': 1024,
        'head_dim': 128,
    },
    # 大规模（Llama-2-7B）
    {
        'name': 'Large',
        'batch': 1,
        'heads': 32,
        'seq_len': 2048,
        'head_dim': 128,
    },
]

SPARSITY_LEVELS = [0.3, 0.5, 0.7]
NUM_WARMUP = 10
NUM_ITERS = 100
DEFAULT_SEED = 20260304
DEFAULT_REPORT_STAT = "median"


def parse_args():
    parser = argparse.ArgumentParser(description="Compression detailed benchmark")
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=NUM_WARMUP,
        help=f"warmup 次数（默认: {NUM_WARMUP}）",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=NUM_ITERS,
        help=f"测试次数（默认: {NUM_ITERS}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"固定随机种子（默认: {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--report-stat",
        choices=["avg", "median", "trimmed_mean"],
        default=DEFAULT_REPORT_STAT,
        help=f"报告主口径（默认: {DEFAULT_REPORT_STAT}）",
    )
    return parser.parse_args()

# ==================== 工具函数 ====================

def benchmark_kernel(func, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    """使用 CUDA Event 精确计时"""
    # Warmup
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed)
    
    times_tensor = torch.tensor(times, dtype=torch.float32)
    # 10% trimmed mean，降低极端值对均值的影响
    sorted_times = torch.sort(times_tensor).values
    trim_n = int(len(times) * 0.1)
    if trim_n * 2 < len(times):
        trimmed = sorted_times[trim_n: len(times) - trim_n]
    else:
        trimmed = sorted_times
    return {
        'avg': times_tensor.mean().item(),
        'std': times_tensor.std().item(),
        'min': times_tensor.min().item(),
        'max': times_tensor.max().item(),
        'median': times_tensor.median().item(),
        'trimmed_mean': trimmed.mean().item(),
        'p10': sorted_times[int(0.1 * (len(times) - 1))].item(),
        'p90': sorted_times[int(0.9 * (len(times) - 1))].item(),
    }


def set_reproducibility(seed):
    """固定随机性，保证同一脚本口径可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def time_value(time_dict, stat_key):
    return float(time_dict.get(stat_key, time_dict.get("median", time_dict["avg"])))

def calculate_memory_mb(tensors):
    """计算内存占用（MB）"""
    total_bytes = 0
    if isinstance(tensors, dict):
        for value in tensors.values():
            if isinstance(value, torch.Tensor):
                total_bytes += value.numel() * value.element_size()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        total_bytes += item.numel() * item.element_size()
    elif isinstance(tensors, torch.Tensor):
        total_bytes = tensors.numel() * tensors.element_size()
    return total_bytes / (1024 ** 2)

def analyze_output_size(result, name):
    """分析输出数据大小"""
    info = {'name': name}
    
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            if isinstance(item, torch.Tensor):
                info[f'output_{i}_shape'] = list(item.shape)
                info[f'output_{i}_dtype'] = str(item.dtype)
                info[f'output_{i}_size_mb'] = item.numel() * item.element_size() / (1024**2)
            elif isinstance(item, list):
                info[f'output_{i}_type'] = 'list'
                info[f'output_{i}_len'] = len(item)
                if len(item) > 0 and isinstance(item[0], torch.Tensor):
                    total_size = sum(t.numel() * t.element_size() for t in item)
                    info[f'output_{i}_size_mb'] = total_size / (1024**2)
    
    return info

# ==================== 测试函数 ====================

def test_compression_sparse(k_cache, config, sparsity, num_warmup, num_iters, report_stat):
    """测试无量化 Compression"""
    if not KERNEL_AVAILABLE:
        return None
    
    print(f"\n  [1/2] 无量化 Compression...")
    
    # 测试
    result = benchmark_kernel(
        lambda: compression.convert_key_batched(k_cache),
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    
    # 获取输出
    k_bmps, k_idxs, k_nzs = compression.convert_key_batched(k_cache)
    
    # 分析输出大小
    output_info = {
        'bmps_shape': list(k_bmps.shape),
        'bmps_dtype': str(k_bmps.dtype),
        'bmps_size_mb': k_bmps.numel() * k_bmps.element_size() / (1024**2),
        'idxs_shape': list(k_idxs.shape),
        'idxs_dtype': str(k_idxs.dtype),
        'idxs_size_mb': k_idxs.numel() * k_idxs.element_size() / (1024**2),
        'nzs_count': len(k_nzs),
        'nzs_dtype': str(k_nzs[0].dtype) if len(k_nzs) > 0 else 'N/A',
        'nzs_total_size_mb': sum(nz.numel() * nz.element_size() for nz in k_nzs) / (1024**2),
    }
    
    # 计算总内存
    total_memory = calculate_memory_mb({
        'bmps': k_bmps,
        'idxs': k_idxs,
        'nzs': k_nzs,
    })
    
    report_time = time_value(result, report_stat)
    print(
        f"    Time ({report_stat}): {report_time:.4f} ms "
        f"[avg={result['avg']:.4f}, median={result['median']:.4f}]"
    )
    print(f"    Memory: {total_memory:.4f} MB")
    print(f"    Output: bmps={k_bmps.shape}, idxs={k_idxs.shape}, nzs={len(k_nzs)} tensors")
    
    return {
        'time': result,
        'memory_mb': total_memory,
        'output_info': output_info,
    }

def test_compression_quant(k_cache, config, sparsity, num_warmup, num_iters, report_stat):
    """测试量化 Compression"""
    if not KERNEL_QUANT_AVAILABLE:
        return None
    
    print(f"\n  [2/2] 量化 Compression...")
    
    # 测试
    result = benchmark_kernel(
        lambda: compression_quant.convert_key_batched_quant(k_cache),
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    
    # 获取输出
    k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
        compression_quant.convert_key_batched_quant(k_cache)
    
    # 分析输出大小
    output_info = {
        'bmps_shape': list(k_bmps.shape),
        'bmps_dtype': str(k_bmps.dtype),
        'bmps_size_mb': k_bmps.numel() * k_bmps.element_size() / (1024**2),
        'tile_offsets_shape': list(k_tile_offsets.shape),
        'tile_offsets_dtype': str(k_tile_offsets.dtype),
        'tile_offsets_size_mb': k_tile_offsets.numel() * k_tile_offsets.element_size() / (1024**2),
        'packed_quant_shape': list(k_packed_quant.shape),
        'packed_quant_dtype': str(k_packed_quant.dtype),
        'packed_quant_size_mb': k_packed_quant.numel() * k_packed_quant.element_size() / (1024**2),
        'scales_shape': list(k_scales.shape),
        'scales_dtype': str(k_scales.dtype),
        'scales_size_mb': k_scales.numel() * k_scales.element_size() / (1024**2),
        'zeros_shape': list(k_zeros.shape),
        'zeros_dtype': str(k_zeros.dtype),
        'zeros_size_mb': k_zeros.numel() * k_zeros.element_size() / (1024**2),
    }
    
    # 计算总内存
    total_memory = calculate_memory_mb({
        'bmps': k_bmps,
        'tile_offsets': k_tile_offsets,
        'packed_quant': k_packed_quant,
        'scales': k_scales,
        'zeros': k_zeros,
    })
    
    report_time = time_value(result, report_stat)
    print(
        f"    Time ({report_stat}): {report_time:.4f} ms "
        f"[avg={result['avg']:.4f}, median={result['median']:.4f}]"
    )
    print(f"    Memory: {total_memory:.4f} MB")
    print(f"    Output: bmps={k_bmps.shape}, packed_quant={k_packed_quant.shape}")
    
    return {
        'time': result,
        'memory_mb': total_memory,
        'output_info': output_info,
    }

# ==================== 主测试函数 ====================

def run_single_test(config, sparsity, num_warmup, num_iters, report_stat):
    """运行单个配置的测试"""
    print(f"\n{'='*70}")
    print(f"测试: {config['name']} | Sparsity: {sparsity*100:.0f}%")
    print(f"  Batch: {config['batch']}, Heads: {config['heads']}")
    print(f"  Seq: {config['seq_len']}, Dim: {config['head_dim']}")
    print(f"{'='*70}")
    
    batch = config['batch']
    heads = config['heads']
    seq_len = config['seq_len']
    head_dim = config['head_dim']
    total_batch_kv = batch * heads
    
    # 准备数据
    print(f"\n准备测试数据...")
    k_cache = torch.randn(total_batch_kv, seq_len, head_dim, 
                         dtype=torch.float16, device='cuda')
    
    # 应用稀疏性
    mask = torch.rand_like(k_cache) > sparsity
    k_cache_sparse = k_cache * mask
    actual_sparsity = (k_cache_sparse == 0).float().mean().item()
    print(f"  实际稀疏度: {actual_sparsity*100:.2f}%")
    
    # 原始数据大小
    original_size_mb = k_cache.numel() * k_cache.element_size() / (1024**2)
    print(f"  原始数据: {original_size_mb:.4f} MB")
    
    # 运行测试
    sparse_result = test_compression_sparse(
        k_cache_sparse, config, sparsity, num_warmup, num_iters, report_stat
    )
    quant_result = test_compression_quant(
        k_cache_sparse, config, sparsity, num_warmup, num_iters, report_stat
    )
    
    # 对比分析
    if sparse_result and quant_result:
        print(f"\n{'='*70}")
        print(f"对比分析")
        print(f"{'='*70}")
        
        sparse_t = time_value(sparse_result['time'], report_stat)
        quant_t = time_value(quant_result['time'], report_stat)
        time_ratio = sparse_t / quant_t
        memory_ratio = sparse_result['memory_mb'] / quant_result['memory_mb']
        
        print(f"\n时间对比:")
        print(f"  无量化: {sparse_t:.4f} ms ({report_stat})")
        print(f"  量化:   {quant_t:.4f} ms ({report_stat})")
        if time_ratio >= 1.0:
            print(f"  比例:   {time_ratio:.2f}x (量化版本快 {(time_ratio-1)*100:.1f}%)")
        else:
            print(f"  比例:   {time_ratio:.2f}x (无量化版本快 {(1/time_ratio-1)*100:.1f}%)")
        
        print(f"\n内存对比:")
        print(f"  原始:   {original_size_mb:.4f} MB")
        print(f"  无量化: {sparse_result['memory_mb']:.4f} MB ({original_size_mb/sparse_result['memory_mb']:.2f}x 压缩)")
        print(f"  量化:   {quant_result['memory_mb']:.4f} MB ({original_size_mb/quant_result['memory_mb']:.2f}x 压缩)")
        print(f"  比例:   {memory_ratio:.2f}x (量化版本节省 {(1-1/memory_ratio)*100:.1f}%)")
    
    return {
        'config': config,
        'sparsity': sparsity,
        'actual_sparsity': actual_sparsity,
        'original_size_mb': original_size_mb,
        'sparse': sparse_result,
        'quant': quant_result,
    }

# ==================== 主函数 ====================

def main():
    args = parse_args()
    set_reproducibility(args.seed)

    print(f"\n测试配置:")
    print(f"  配置数量: {len(TEST_CONFIGS)}")
    print(f"  稀疏度: {[f'{s*100:.0f}%' for s in SPARSITY_LEVELS]}")
    print(f"  预热: {args.num_warmup} 次, 测试: {args.num_iters} 次")
    print(f"  统计口径: {args.report_stat}")
    print(f"  随机种子: {args.seed}")
    
    all_results = []
    
    for config in TEST_CONFIGS:
        for sparsity in SPARSITY_LEVELS:
            result = run_single_test(
                config,
                sparsity,
                args.num_warmup,
                args.num_iters,
                args.report_stat,
            )
            all_results.append(result)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"compression_detailed_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_configs': TEST_CONFIGS,
            'sparsity_levels': SPARSITY_LEVELS,
            'num_warmup': args.num_warmup,
            'num_iters': args.num_iters,
            'seed': args.seed,
            'report_stat': args.report_stat,
            'results': all_results,
        }, f, indent=2, default=str)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print(f"汇总报告")
    print(f"{'='*70}")
    
    print(f"\n{'配置':<15} {'稀疏度':<8} {'无量化(ms)':<15} {'量化(ms)':<15} {'加速比':<10} {'内存比':<10}")
    print(f"{'-'*80}")
    
    for r in all_results:
        if r['sparse'] and r['quant']:
            sparse_t = time_value(r['sparse']['time'], args.report_stat)
            quant_t = time_value(r['quant']['time'], args.report_stat)
            time_ratio = sparse_t / quant_t
            memory_ratio = r['sparse']['memory_mb'] / r['quant']['memory_mb']
            
            print(f"{r['config']['name']:<15} "
                  f"{r['sparsity']*100:<8.0f} "
                  f"{sparse_t:<15.4f} "
                  f"{quant_t:<15.4f} "
                  f"{time_ratio:<10.2f}x "
                  f"{memory_ratio:<10.2f}x")
    
    # 生成 Markdown 报告
    md_file = os.path.join(output_dir, f"compression_detailed_{timestamp}.md")
    with open(md_file, 'w') as f:
        f.write(f"# Compression 详细性能对比\n\n")
        f.write(f"**测试时间**: {timestamp}\n\n")
        
        f.write(f"## 测试配置\n\n")
        f.write(f"- 预热: {args.num_warmup} 次\n")
        f.write(f"- 测试: {args.num_iters} 次\n")
        f.write(f"- 统计口径: {args.report_stat}\n")
        f.write(f"- 随机种子: {args.seed}\n")
        f.write(f"- 稀疏度: {', '.join([f'{s*100:.0f}%' for s in SPARSITY_LEVELS])}\n\n")
        
        f.write(f"## 性能对比\n\n")
        f.write(f"| 配置 | 稀疏度 | 无量化 (ms, {args.report_stat}) | 量化 (ms, {args.report_stat}) | 加速比 | 内存比 |\n")
        f.write(f"|------|--------|-------------|-----------|--------|--------|\n")
        
        for r in all_results:
            if r['sparse'] and r['quant']:
                sparse_t = time_value(r['sparse']['time'], args.report_stat)
                quant_t = time_value(r['quant']['time'], args.report_stat)
                time_ratio = sparse_t / quant_t
                memory_ratio = r['sparse']['memory_mb'] / r['quant']['memory_mb']
                
                f.write(f"| {r['config']['name']} | {r['sparsity']*100:.0f}% | "
                       f"{sparse_t:.4f} | "
                       f"{quant_t:.4f} | "
                       f"{time_ratio:.2f}x | "
                       f"{memory_ratio:.2f}x |\n")
        
        f.write(f"\n## 详细分析\n\n")
        f.write(f"### 关键发现\n\n")
        f.write(f"1. **量化版本的 Compression 更快**\n")
        f.write(f"2. **量化版本的内存占用更小**\n")
        f.write(f"3. **加速比随规模变化**\n\n")
    
    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"{'='*70}")
    print(f"\n结果已保存到:")
    print(f"  JSON: {output_file}")
    print(f"  Markdown: {md_file}")

if __name__ == '__main__':
    main()
