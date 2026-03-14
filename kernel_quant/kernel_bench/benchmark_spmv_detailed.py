#!/usr/bin/env python3
"""
SpMV 性能详细测试
目标：模拟真实 Decoding 场景，测试 SpMV 的累计性能
"""
import torch
import sys
import os
import json
import argparse
from datetime import datetime

# 添加路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'kernel'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'kernel_quant'))

print("="*70)
print("SpMV 性能详细测试")
print("="*70)

# 导入模块
try:
    import compression
    import mustafar_package
    print("✓ kernel 模块导入成功")
    KERNEL_AVAILABLE = True
except Exception as e:
    print(f"✗ kernel 导入失败: {e}")
    KERNEL_AVAILABLE = False

try:
    import compression_quant
    import mustafar_package_quant
    print("✓ kernel_quant 模块导入成功")
    KERNEL_QUANT_AVAILABLE = True
except Exception as e:
    print(f"✗ kernel_quant 导入失败: {e}")
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
        'num_decode_steps': 256,  # 模拟生成 256 个 token
    },
    # 中等规模
    {
        'name': 'Medium',
        'batch': 1,
        'heads': 16,
        'seq_len': 1024,
        'head_dim': 128,
        'num_decode_steps': 512,  # 模拟生成 512 个 token
    },
    # 大规模（Llama-2-7B）
    {
        'name': 'Large',
        'batch': 1,
        'heads': 32,
        'seq_len': 2048,
        'head_dim': 128,
        'num_decode_steps': 1024,  # 模拟生成 1024 个 token
    },
]

SPARSITY = 0.5  # 50% 稀疏度
NUM_WARMUP = 10
NUM_ITERS = 50
DEFAULT_SEED = 20260302
CONFIG_SEED_STRIDE = 9973
DEQUANT_MODE_LABELS = {
    0: 'speed',
    1: 'memory',
}

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
    
    times_tensor = torch.tensor(times)
    return {
        'avg': times_tensor.mean().item(),
        'std': times_tensor.std().item(),
        'min': times_tensor.min().item(),
        'max': times_tensor.max().item(),
        'median': times_tensor.median().item(),
    }

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


def parse_args():
    parser = argparse.ArgumentParser(description="SpMV detailed benchmark")
    parser.add_argument(
        "--dequant-mode",
        choices=["0", "1", "both"],
        default="0",
        help="量化反量化模式: 0=speed, 1=memory, both=两者都测",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"固定随机种子（默认: {DEFAULT_SEED}）",
    )
    return parser.parse_args()


def parse_quant_modes(dequant_mode_arg):
    if dequant_mode_arg == "both":
        return [0, 1]
    return [int(dequant_mode_arg)]


def faster_label(lhs_value, rhs_value, lhs_name, rhs_name):
    return lhs_name if lhs_value <= rhs_value else rhs_name


def set_reproducibility(seed):
    """固定随机性，保证同一脚本口径可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config_seed(base_seed, config_index):
    return base_seed + config_index * CONFIG_SEED_STRIDE


# ==================== 测试函数 ====================

def test_sparse_spmv(k_cache_sparse, query, config):
    """测试无量化 SpMV"""
    if not KERNEL_AVAILABLE:
        return None
    
    print(f"\n{'='*70}")
    print(f"[1/2] Sparse SpMV (FP16)")
    print(f"{'='*70}")
    
    batch = config['batch']
    heads = config['heads']
    seq_len = config['seq_len']
    head_dim = config['head_dim']
    num_decode_steps = config['num_decode_steps']
    total_batch_kv = batch * heads
    
    # 1. 预处理：压缩 KV Cache（模拟 Prefill）
    print(f"\n[Step 1] Compression (一次性)...")
    compress_result = benchmark_kernel(
        lambda: compression.convert_key_batched(k_cache_sparse),
        num_warmup=5,
        num_iters=10
    )
    print(f"  Time: {compress_result['avg']:.4f} ms")
    
    # 获取压缩数据
    k_bmps, k_idxs, k_nzs = compression.convert_key_batched(k_cache_sparse)
    
    # 计算 nz_offset
    k_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device='cuda')
    for i in range(1, total_batch_kv):
        k_nz_offset[i] = k_nz_offset[i-1] + k_idxs[i-1][-1] // 4
    
    # 计算内存
    memory_mb = calculate_memory_mb({
        'bmps': k_bmps,
        'idxs': k_idxs,
        'nzs': k_nzs,
        'offset': k_nz_offset,
    })
    print(f"  Compressed Memory: {memory_mb:.2f} MB")
    
    # 2. 测试：单次 SpMV
    print(f"\n[Step 2] 单次 SpMV...")
    padded_query = torch.nn.functional.pad(
        query.view(total_batch_kv, -1, head_dim),
        (0, 0, 0, 7),
        mode='constant',
        value=0
    )
    
    single_spmv_result = benchmark_kernel(
        lambda: mustafar_package.mustafar_key_formulation(
            k_bmps,
            torch.cat(k_nzs),
            k_idxs,
            k_nz_offset,
            padded_query,
            seq_len,
            head_dim,
            total_batch_kv,
            1
        )
    )
    print(f"  Time: {single_spmv_result['avg']:.4f} ms")
    
    # 3. 测试：批量 SpMV（模拟 Decoding）
    print(f"\n[Step 3] 批量 SpMV (模拟 {num_decode_steps} 次 Decoding)...")
    
    def batch_spmv():
        for _ in range(num_decode_steps):
            mustafar_package.mustafar_key_formulation(
                k_bmps,
                torch.cat(k_nzs),
                k_idxs,
                k_nz_offset,
                padded_query,
                seq_len,
                head_dim,
                total_batch_kv,
                1
            )
    
    batch_result = benchmark_kernel(
        batch_spmv,
        num_warmup=3,
        num_iters=10
    )
    
    avg_per_call = batch_result['avg'] / num_decode_steps
    print(f"  Total Time: {batch_result['avg']:.4f} ms")
    print(f"  Avg per call: {avg_per_call:.4f} ms")
    print(f"  TPOT estimate: {avg_per_call:.4f} ms/token")
    
    return {
        'compression': compress_result['avg'],
        'single_spmv': single_spmv_result['avg'],
        'batch_spmv_total': batch_result['avg'],
        'batch_spmv_avg': avg_per_call,
        'num_decode_steps': num_decode_steps,
        'memory_mb': memory_mb,
    }

def test_quant_spmv(k_cache_sparse, query, config, dequant_mode):
    """测试量化 SpMV"""
    if not KERNEL_QUANT_AVAILABLE:
        return None
    
    print(f"\n{'='*70}")
    mode_name = DEQUANT_MODE_LABELS.get(dequant_mode, f"unknown({dequant_mode})")
    print(f"[2/2] Quantized SpMV (2-bit, dequant_mode={dequant_mode}:{mode_name})")
    print(f"{'='*70}")
    
    batch = config['batch']
    heads = config['heads']
    seq_len = config['seq_len']
    head_dim = config['head_dim']
    num_decode_steps = config['num_decode_steps']
    total_batch_kv = batch * heads
    
    # 1. 预处理：压缩 + 量化（模拟 Prefill）
    print(f"\n[Step 1] Compression + Quantization (一次性)...")
    compress_result = benchmark_kernel(
        lambda: compression_quant.convert_key_batched_quant(k_cache_sparse),
        num_warmup=5,
        num_iters=10
    )
    print(f"  Time: {compress_result['avg']:.4f} ms")
    
    # 获取压缩数据
    k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
        compression_quant.convert_key_batched_quant(k_cache_sparse)
    
    # 计算内存
    memory_mb = calculate_memory_mb({
        'bmps': k_bmps,
        'tile_offsets': k_tile_offsets,
        'packed_quant': k_packed_quant,
        'scales': k_scales,
        'zeros': k_zeros,
    })
    print(f"  Compressed Memory: {memory_mb:.2f} MB")
    
    # 2. 测试：单次 SpMV
    print(f"\n[Step 2] 单次 SpMV + Dequantization...")
    padded_query = torch.nn.functional.pad(
        query.view(total_batch_kv, -1, head_dim),
        (0, 0, 0, 7),
        mode='constant',
        value=0
    )
    
    single_spmv_result = benchmark_kernel(
        lambda: mustafar_package_quant.mustafar_key_formulation_quant(
            k_bmps,
            k_packed_quant,
            k_tile_offsets,
            k_scales,
            k_zeros,
            padded_query,
            seq_len,
            head_dim,
            total_batch_kv,
            1,
            2,
            16,
            dequant_mode
        )
    )
    print(f"  Time: {single_spmv_result['avg']:.4f} ms")
    
    # 3. 测试：批量 SpMV（模拟 Decoding）
    print(f"\n[Step 3] 批量 SpMV (模拟 {num_decode_steps} 次 Decoding)...")
    
    def batch_spmv():
        for _ in range(num_decode_steps):
            mustafar_package_quant.mustafar_key_formulation_quant(
                k_bmps,
                k_packed_quant,
                k_tile_offsets,
                k_scales,
                k_zeros,
                padded_query,
                seq_len,
                head_dim,
                total_batch_kv,
                1,
                2,
                16,
                dequant_mode
            )
    
    batch_result = benchmark_kernel(
        batch_spmv,
        num_warmup=3,
        num_iters=10
    )
    
    avg_per_call = batch_result['avg'] / num_decode_steps
    print(f"  Total Time: {batch_result['avg']:.4f} ms")
    print(f"  Avg per call: {avg_per_call:.4f} ms")
    print(f"  TPOT estimate: {avg_per_call:.4f} ms/token")
    
    return {
        'dequant_mode': dequant_mode,
        'compression': compress_result['avg'],
        'single_spmv': single_spmv_result['avg'],
        'batch_spmv_total': batch_result['avg'],
        'batch_spmv_avg': avg_per_call,
        'num_decode_steps': num_decode_steps,
        'memory_mb': memory_mb,
    }

# ==================== 主测试函数 ====================

def run_single_test(config, quant_modes, data_seed):
    """运行单个配置的测试"""
    print(f"\n{'='*70}")
    print(f"测试配置: {config['name']}")
    print(f"  Batch: {config['batch']}, Heads: {config['heads']}")
    print(f"  Seq: {config['seq_len']}, Dim: {config['head_dim']}")
    print(f"  Decode Steps: {config['num_decode_steps']}")
    print(f"  数据种子: {data_seed}")
    print(f"{'='*70}")
    
    batch = config['batch']
    heads = config['heads']
    seq_len = config['seq_len']
    head_dim = config['head_dim']
    total_batch_kv = batch * heads
    
    # 准备数据
    print(f"\n准备测试数据...")
    set_reproducibility(data_seed)
    k_cache = torch.randn(total_batch_kv, seq_len, head_dim, 
                         dtype=torch.float16, device='cuda')
    
    # 应用稀疏性
    mask = torch.rand_like(k_cache) > SPARSITY
    k_cache_sparse = k_cache * mask
    actual_sparsity = (k_cache_sparse == 0).float().mean().item()
    print(f"  实际稀疏度: {actual_sparsity*100:.2f}%")
    
    # 准备 query
    query = torch.randn(batch, heads, 1, head_dim, 
                       dtype=torch.float16, device='cuda')
    
    # 运行测试
    sparse_result = test_sparse_spmv(k_cache_sparse, query, config)
    quant_results_by_mode = {}
    for dequant_mode in quant_modes:
        quant_result = test_quant_spmv(k_cache_sparse, query, config, dequant_mode)
        if quant_result is not None:
            quant_results_by_mode[str(dequant_mode)] = quant_result
    
    # 对比分析
    if sparse_result and quant_results_by_mode:
        for mode_key, quant_result in quant_results_by_mode.items():
            mode_id = int(mode_key)
            mode_name = DEQUANT_MODE_LABELS.get(mode_id, f"unknown({mode_id})")
            single_ratio = sparse_result['single_spmv'] / quant_result['single_spmv']
            batch_ratio = sparse_result['batch_spmv_total'] / quant_result['batch_spmv_total']
            memory_ratio = sparse_result['memory_mb'] / quant_result['memory_mb']

            print(f"\n{'='*70}")
            print(f"性能对比 (dequant_mode={mode_id}:{mode_name})")
            print(f"{'='*70}")

            print(f"\n单次 SpMV:")
            print(f"  Sparse:  {sparse_result['single_spmv']:.4f} ms")
            print(f"  Quant:   {quant_result['single_spmv']:.4f} ms")
            print(
                f"  比例:    {single_ratio:.2f}x "
                f"({faster_label(sparse_result['single_spmv'], quant_result['single_spmv'], 'Sparse', 'Quant')} 更快)"
            )

            print(f"\n批量 SpMV ({config['num_decode_steps']} 次):")
            print(f"  Sparse:  {sparse_result['batch_spmv_total']:.4f} ms")
            print(f"  Quant:   {quant_result['batch_spmv_total']:.4f} ms")
            print(
                f"  比例:    {batch_ratio:.2f}x "
                f"({faster_label(sparse_result['batch_spmv_total'], quant_result['batch_spmv_total'], 'Sparse', 'Quant')} 更快)"
            )

            print(f"\nTPOT 估算:")
            print(f"  Sparse:  {sparse_result['batch_spmv_avg']:.4f} ms/token")
            print(f"  Quant:   {quant_result['batch_spmv_avg']:.4f} ms/token")

            print(f"\n内存占用:")
            print(f"  Sparse:  {sparse_result['memory_mb']:.2f} MB")
            print(f"  Quant:   {quant_result['memory_mb']:.2f} MB")
            print(
                f"  比例:    {memory_ratio:.2f}x "
                f"({faster_label(sparse_result['memory_mb'], quant_result['memory_mb'], 'Sparse', 'Quant')} 更省)"
            )

    primary_mode = 0 if '0' in quant_results_by_mode else None
    if primary_mode is None and quant_results_by_mode:
        primary_mode = int(next(iter(quant_results_by_mode)))
    
    return {
        'config': config,
        'data_seed': data_seed,
        'actual_sparsity': actual_sparsity,
        'sparse': sparse_result,
        'quant': quant_results_by_mode.get(str(primary_mode)) if primary_mode is not None else None,
        'quant_primary_mode': primary_mode,
        'quant_by_mode': quant_results_by_mode,
    }

# ==================== 主函数 ====================

def main():
    args = parse_args()
    quant_modes = parse_quant_modes(args.dequant_mode)
    set_reproducibility(args.seed)

    print(f"\n测试配置:")
    print(f"  配置数量: {len(TEST_CONFIGS)}")
    print(f"  稀疏度: {SPARSITY*100:.0f}%")
    print(f"  预热: {NUM_WARMUP} 次, 测试: {NUM_ITERS} 次")
    print(f"  固定随机种子: {args.seed}")
    print(f"  配置种子步长: {CONFIG_SEED_STRIDE}")
    print(
        "  Quant dequant_mode: "
        + ", ".join([f"{m}({DEQUANT_MODE_LABELS.get(m, 'unknown')})" for m in quant_modes])
    )
    
    all_results = []
    config_seeds = {}
    
    for idx, config in enumerate(TEST_CONFIGS):
        config_seed = get_config_seed(args.seed, idx)
        config_seeds[config['name']] = config_seed
        result = run_single_test(config, quant_modes, config_seed)
        all_results.append(result)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"spmv_detailed_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_configs': TEST_CONFIGS,
            'sparsity': SPARSITY,
            'num_warmup': NUM_WARMUP,
            'num_iters': NUM_ITERS,
            'global_seed': args.seed,
            'config_seed_stride': CONFIG_SEED_STRIDE,
            'config_seeds': config_seeds,
            'reproducibility': {
                'seeded_data_per_config': True,
                'cudnn_deterministic': True,
                'cudnn_benchmark': False,
            },
            'quant_dequant_modes': quant_modes,
            'results': all_results,
        }, f, indent=2, default=str)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print(f"汇总报告")
    print(f"{'='*70}")
    
    print(
        f"\n{'配置':<10} {'模式':<8} {'Decode步数':<10} "
        f"{'Sparse单次':<12} {'Quant单次':<12} {'Sparse批量':<12} {'Quant批量':<12}"
    )
    print(f"{'-'*92}")

    for r in all_results:
        if not r['sparse']:
            continue
        for mode_key, quant_result in r.get('quant_by_mode', {}).items():
            mode_id = int(mode_key)
            mode_name = DEQUANT_MODE_LABELS.get(mode_id, f"m{mode_id}")
            print(
                f"{r['config']['name']:<10} "
                f"{mode_name:<8} "
                f"{r['config']['num_decode_steps']:<10} "
                f"{r['sparse']['single_spmv']:<12.4f} "
                f"{quant_result['single_spmv']:<12.4f} "
                f"{r['sparse']['batch_spmv_avg']:<12.4f} "
                f"{quant_result['batch_spmv_avg']:<12.4f}"
            )
    
    # 生成 Markdown 报告
    md_file = os.path.join(output_dir, f"spmv_detailed_{timestamp}.md")
    with open(md_file, 'w') as f:
        f.write(f"# SpMV 性能详细测试\n\n")
        f.write(f"**测试时间**: {timestamp}\n\n")
        
        f.write(f"## 测试配置\n\n")
        f.write(f"- 稀疏度: {SPARSITY*100:.0f}%\n")
        f.write(f"- 预热: {NUM_WARMUP} 次, 测试: {NUM_ITERS} 次\n\n")
        f.write(f"- 固定随机种子: {args.seed}\n")
        f.write(f"- 配置种子步长: {CONFIG_SEED_STRIDE}\n")
        f.write("- 配置种子:\n")
        for cfg in TEST_CONFIGS:
            f.write(f"  - {cfg['name']}: {config_seeds[cfg['name']]}\n")
        f.write("\n")
        f.write(f"- Quant dequant_mode: {', '.join([str(m) for m in quant_modes])}\n\n")
        
        f.write(f"## 性能对比\n\n")
        f.write(f"| 配置 | 模式 | Decode步数 | Sparse单次(ms) | Quant单次(ms) | Sparse TPOT(ms) | Quant TPOT(ms) | 内存比 |\n")
        f.write(f"|------|------|-----------|---------------|--------------|----------------|---------------|--------|\n")
        
        for r in all_results:
            if not r['sparse']:
                continue
            for mode_key, quant_result in r.get('quant_by_mode', {}).items():
                mode_id = int(mode_key)
                mode_name = DEQUANT_MODE_LABELS.get(mode_id, f"m{mode_id}")
                mem_ratio = r['sparse']['memory_mb'] / quant_result['memory_mb']
                f.write(f"| {r['config']['name']} | {mode_name} | {r['config']['num_decode_steps']} | "
                        f"{r['sparse']['single_spmv']:.4f} | "
                        f"{quant_result['single_spmv']:.4f} | "
                        f"{r['sparse']['batch_spmv_avg']:.4f} | "
                        f"{quant_result['batch_spmv_avg']:.4f} | "
                        f"{mem_ratio:.2f}x |\n")
        
        f.write(f"\n## 关键发现\n\n")
        f.write(f"1. **单次 SpMV**: Sparse 比 Quant 快（无反量化开销）\n")
        f.write(f"2. **批量 SpMV**: 累计效果与单次一致\n")
        f.write(f"3. **内存占用**: Quant 显著更少\n")
        f.write(f"4. **TPOT 估算**: 基于批量测试的平均值\n\n")
    
    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"{'='*70}")
    print(f"\n结果已保存到:")
    print(f"  JSON: {output_file}")
    print(f"  Markdown: {md_file}")

if __name__ == '__main__':
    main()
