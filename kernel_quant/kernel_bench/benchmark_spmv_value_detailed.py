#!/usr/bin/env python3
"""
Value SpMV 性能详细测试
目标：独立评估 Value kernel，不再使用 Key 路径结果替代。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kernel"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "kernel_quant"))

print("=" * 70)
print("Value SpMV 性能详细测试")
print("=" * 70)

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

TEST_CONFIGS = [
    {
        "name": "Small",
        "batch": 1,
        "heads": 4,
        "kv_heads": 4,
        "seq_len": 256,
        "head_dim": 128,
        "num_decode_steps": 256,
    },
    {
        "name": "Medium",
        "batch": 1,
        "heads": 16,
        "kv_heads": 4,
        "seq_len": 1024,
        "head_dim": 128,
        "num_decode_steps": 512,
    },
    {
        "name": "Large",
        "batch": 1,
        "heads": 32,
        "kv_heads": 8,
        "seq_len": 2048,
        "head_dim": 128,
        "num_decode_steps": 1024,
    },
]

SPARSITY = 0.5
NUM_WARMUP = 10
NUM_ITERS = 50
DEFAULT_SEED = 20260312
CONFIG_SEED_STRIDE = 9973
DEQUANT_MODE_LABELS = {
    0: "speed",
    1: "memory",
}


def benchmark_kernel(func, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times_tensor = torch.tensor(times)
    return {
        "avg": times_tensor.mean().item(),
        "std": times_tensor.std().item(),
        "min": times_tensor.min().item(),
        "max": times_tensor.max().item(),
        "median": times_tensor.median().item(),
    }


def calculate_memory_mb(tensors):
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
    parser = argparse.ArgumentParser(description="Value SpMV detailed benchmark")
    parser.add_argument(
        "--dequant-mode",
        choices=["0", "1", "both"],
        default="both",
        help="量化反量化模式: 0=speed, 1=memory, both=两者都测",
    )
    parser.add_argument(
        "--split-k-values",
        default="1,2,4,8",
        help="Value Split-K sweep，例如 1,2,4,8",
    )
    parser.add_argument(
        "--tile-config-values",
        default="0,1,2,3",
        help="Value tile config sweep: 0=auto, 1=tile64, 2=tile128, 3=fused",
    )
    parser.add_argument(
        "--config-name",
        choices=["all", "Small", "Medium", "Large"],
        default="all",
        help="只跑指定规模，默认 all",
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


def parse_split_k_values(split_k_values_arg):
    values = []
    for item in split_k_values_arg.split(","):
        item = item.strip()
        if not item:
            continue
        split_k = int(item)
        if split_k < 1:
            raise ValueError(f"split_k 必须 >= 1，收到: {split_k}")
        values.append(split_k)
    if not values:
        raise ValueError("split_k 列表不能为空")
    return values


def parse_tile_config_values(tile_config_values_arg):
    values = []
    for item in tile_config_values_arg.split(","):
        item = item.strip()
        if not item:
            continue
        tile_config = int(item)
        if tile_config not in (0, 1, 2, 3):
            raise ValueError(f"tile_config 必须是 0/1/2/3，收到: {tile_config}")
        values.append(tile_config)
    if not values:
        raise ValueError("tile_config 列表不能为空")
    return values


def set_reproducibility(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config_seed(base_seed, config_index):
    return base_seed + config_index * CONFIG_SEED_STRIDE


def filter_configs(config_name):
    if config_name == "all":
        return TEST_CONFIGS
    return [cfg for cfg in TEST_CONFIGS if cfg["name"] == config_name]


def test_sparse_value_spmv(v_cache_sparse, score, config):
    if not KERNEL_AVAILABLE:
        return None

    print(f"\n{'=' * 70}")
    print("[1/2] Sparse Value SpMV (FP16)")
    print(f"{'=' * 70}")

    batch = config["batch"]
    heads = config["heads"]
    seq_len = config["seq_len"]
    head_dim = config["head_dim"]
    num_decode_steps = config["num_decode_steps"]
    kv_heads = config.get("kv_heads", heads)
    total_batch_kv = batch * kv_heads
    total_batch_size = batch * heads
    num_key_value_groups = heads // kv_heads

    print("\n[Step 1] Compression (一次性)...")
    compress_result = benchmark_kernel(
        lambda: compression.convert_value_batched(v_cache_sparse),
        num_warmup=5,
        num_iters=10,
    )
    print(f"  Time: {compress_result['avg']:.4f} ms")

    v_bmps, v_idxs, v_nzs = compression.convert_value_batched(v_cache_sparse)
    v_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device="cuda")
    for i in range(1, total_batch_kv):
        v_nz_offset[i] = v_nz_offset[i - 1] + v_idxs[i - 1][-1] // 4

    memory_mb = calculate_memory_mb(
        {
            "bmps": v_bmps,
            "idxs": v_idxs,
            "nzs": v_nzs,
            "offset": v_nz_offset,
        }
    )
    print(f"  Compressed Memory: {memory_mb:.2f} MB")

    padded_score = torch.nn.functional.pad(
        score.view(total_batch_kv, -1, seq_len),
        (0, 0, 0, 7),
        mode="constant",
        value=0,
    ).contiguous()
    reduction_workspace = torch.empty((1,), dtype=torch.float16, device="cuda")

    print("\n[Step 2] 单次 Value SpMV...")
    single_spmv_result = benchmark_kernel(
        lambda: mustafar_package.mustafar_value_formulation(
            v_bmps,
            torch.cat(v_nzs),
            v_idxs,
            v_nz_offset,
            padded_score,
            reduction_workspace,
            head_dim,
            seq_len,
            total_batch_size,
            num_key_value_groups,
        )
    )
    print(f"  Time: {single_spmv_result['avg']:.4f} ms")

    print(f"\n[Step 3] 批量 Value SpMV (模拟 {num_decode_steps} 次 Decoding)...")

    def batch_spmv():
        for _ in range(num_decode_steps):
            mustafar_package.mustafar_value_formulation(
                v_bmps,
                torch.cat(v_nzs),
                v_idxs,
                v_nz_offset,
                padded_score,
                reduction_workspace,
                head_dim,
                seq_len,
                total_batch_size,
                num_key_value_groups,
            )

    batch_result = benchmark_kernel(batch_spmv, num_warmup=3, num_iters=10)
    avg_per_call = batch_result["avg"] / num_decode_steps
    print(f"  Total Time: {batch_result['avg']:.4f} ms")
    print(f"  Avg per call: {avg_per_call:.4f} ms")
    print(f"  TPOT estimate: {avg_per_call:.4f} ms/token")

    return {
        "compression": compress_result["avg"],
        "single_spmv": single_spmv_result["avg"],
        "batch_spmv_total": batch_result["avg"],
        "batch_spmv_avg": avg_per_call,
        "num_decode_steps": num_decode_steps,
        "memory_mb": memory_mb,
    }


def test_quant_value_spmv(v_cache_sparse, score, config, dequant_mode, split_k, tile_config):
    if not KERNEL_QUANT_AVAILABLE:
        return None

    print(f"\n{'=' * 70}")
    mode_name = DEQUANT_MODE_LABELS.get(dequant_mode, f"unknown({dequant_mode})")
    print(
        f"[2/2] Quantized Value SpMV (2-bit, dequant_mode={dequant_mode}:{mode_name}, split_k={split_k}, tile_config={tile_config})"
    )
    print(f"{'=' * 70}")

    batch = config["batch"]
    heads = config["heads"]
    seq_len = config["seq_len"]
    head_dim = config["head_dim"]
    num_decode_steps = config["num_decode_steps"]
    kv_heads = config.get("kv_heads", heads)
    total_batch_kv = batch * kv_heads
    total_batch_size = batch * heads
    num_key_value_groups = heads // kv_heads
    max_split_k = max(1, seq_len // 64)
    effective_split_k = min(split_k, max_split_k)
    if effective_split_k != split_k:
        print(f"  Requested split_k={split_k} 超出上限，自动裁剪为 {effective_split_k}")
    split_k = effective_split_k

    print("\n[Step 1] Compression + Quantization (一次性)...")
    compress_result = benchmark_kernel(
        lambda: compression_quant.convert_value_batched_quant(v_cache_sparse),
        num_warmup=5,
        num_iters=10,
    )
    print(f"  Time: {compress_result['avg']:.4f} ms")

    v_bmps, v_tile_offsets, v_packed_quant, v_counts, v_units, v_scales, v_zeros = compression_quant.convert_value_batched_quant(
        v_cache_sparse
    )
    memory_mb = calculate_memory_mb(
        {
            "bmps": v_bmps,
            "tile_offsets": v_tile_offsets,
            "packed_quant": v_packed_quant,
            "counts": v_counts,
            "units": v_units,
            "scales": v_scales,
            "zeros": v_zeros,
        }
    )
    print(f"  Compressed Memory: {memory_mb:.2f} MB")

    padded_score = torch.nn.functional.pad(
        score.view(total_batch_size, -1, seq_len),
        (0, 0, 0, 7),
        mode="constant",
        value=0,
    ).contiguous()
    workspace_numel = max(total_batch_size * head_dim * 8 * split_k, 1)
    reduction_workspace = torch.empty((workspace_numel,), dtype=torch.float16, device="cuda")

    print("\n[Step 2] 单次 Value SpMV + Dequantization...")
    single_spmv_result = benchmark_kernel(
        lambda: mustafar_package_quant.mustafar_value_formulation_quant(
            v_bmps,
            v_packed_quant,
            v_tile_offsets,
            v_counts,
            v_units,
            v_scales,
            v_zeros,
            padded_score,
            reduction_workspace,
            head_dim,
            seq_len,
            total_batch_size,
            num_key_value_groups,
            2,
            16,
            dequant_mode,
            split_k,
            tile_config,
        )
    )
    print(f"  Time: {single_spmv_result['avg']:.4f} ms")

    print(f"\n[Step 3] 批量 Value SpMV (模拟 {num_decode_steps} 次 Decoding)...")

    def batch_spmv():
        for _ in range(num_decode_steps):
            mustafar_package_quant.mustafar_value_formulation_quant(
                v_bmps,
                v_packed_quant,
                v_tile_offsets,
                v_counts,
                v_units,
                v_scales,
                v_zeros,
                padded_score,
                reduction_workspace,
                head_dim,
                seq_len,
                total_batch_size,
                num_key_value_groups,
                2,
                16,
                dequant_mode,
                split_k,
                tile_config,
            )

    batch_result = benchmark_kernel(batch_spmv, num_warmup=3, num_iters=10)
    avg_per_call = batch_result["avg"] / num_decode_steps
    print(f"  Total Time: {batch_result['avg']:.4f} ms")
    print(f"  Avg per call: {avg_per_call:.4f} ms")
    print(f"  TPOT estimate: {avg_per_call:.4f} ms/token")

    return {
        "dequant_mode": dequant_mode,
        "split_k": split_k,
        "tile_config": tile_config,
        "compression": compress_result["avg"],
        "single_spmv": single_spmv_result["avg"],
        "batch_spmv_total": batch_result["avg"],
        "batch_spmv_avg": avg_per_call,
        "num_decode_steps": num_decode_steps,
        "memory_mb": memory_mb,
    }


def make_score_tensor(config):
    batch = config["batch"]
    heads = config["heads"]
    kv_heads = config.get("kv_heads", heads)
    seq_len = config["seq_len"]
    total_batch_size = batch * heads
    return torch.randn((total_batch_size, 8, seq_len), dtype=torch.float16, device="cuda")


def main():
    args = parse_args()
    quant_modes = parse_quant_modes(args.dequant_mode)
    split_k_values = parse_split_k_values(args.split_k_values)
    tile_config_values = parse_tile_config_values(args.tile_config_values)
    configs = filter_configs(args.config_name)

    set_reproducibility(args.seed)
    config_seeds = {
        cfg["name"]: get_config_seed(args.seed, idx) for idx, cfg in enumerate(TEST_CONFIGS)
    }

    all_results = []
    print("测试配置:")
    print(f"  配置数量: {len(configs)}")
    print(f"  稀疏度: {SPARSITY * 100:.0f}%")
    print(f"  预热: {NUM_WARMUP} 次, 测试: {NUM_ITERS} 次")
    print(f"  固定随机种子: {args.seed}")
    print(f"  Split-K sweep: {split_k_values}")
    print(f"  Tile Config sweep: {tile_config_values}")

    for config in configs:
        cfg_seed = config_seeds[config["name"]]
        set_reproducibility(cfg_seed)

        print(f"\n{'=' * 70}")
        print(f"测试配置: {config['name']}")
        print(
            f"  Batch: {config['batch']}, Heads: {config['heads']}, KV Heads: {config.get('kv_heads', config['heads'])}"
        )
        print(f"  Seq: {config['seq_len']}, Dim: {config['head_dim']}")
        print(f"  Decode Steps: {config['num_decode_steps']}")
        print(f"  数据种子: {cfg_seed}")
        print(f"{'=' * 70}")

        total_batch_kv = config["batch"] * config.get("kv_heads", config["heads"])
        v_cache = torch.randn(
            (total_batch_kv, config["seq_len"], config["head_dim"]),
            dtype=torch.float16,
            device="cuda",
        )
        sparsity_mask = torch.rand_like(v_cache) > SPARSITY
        v_cache_sparse = v_cache * sparsity_mask
        score = make_score_tensor(config)

        actual_sparsity = (v_cache_sparse == 0).float().mean().item()
        print("\n准备测试数据...")
        print(f"  实际稀疏度: {actual_sparsity * 100:.2f}%")

        sparse_result = test_sparse_value_spmv(v_cache_sparse, score, config)
        quant_results = {}
        best_quant = None

        for dequant_mode in quant_modes:
            for split_k in split_k_values:
                for tile_config in tile_config_values:
                    quant_result = test_quant_value_spmv(
                        v_cache_sparse,
                        score,
                        config,
                        dequant_mode,
                        split_k,
                        tile_config,
                    )
                    if quant_result is None:
                        continue
                    key = f"mode{dequant_mode}_splitk{split_k}_tile{tile_config}"
                    quant_results[key] = quant_result
                    if best_quant is None or quant_result["batch_spmv_avg"] < best_quant["batch_spmv_avg"]:
                        best_quant = quant_result

        all_results.append(
            {
                "config": config,
                "data_seed": cfg_seed,
                "actual_sparsity": actual_sparsity,
                "sparse": sparse_result,
                "quant_results": quant_results,
                "best_quant": best_quant,
            }
        )

    output_dir = os.path.join(PROJECT_ROOT, "kernel_quant", "kernel_bench", "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"value_spmv_detailed_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "test_configs": configs,
                "sparsity": SPARSITY,
                "num_warmup": NUM_WARMUP,
                "num_iters": NUM_ITERS,
                "global_seed": args.seed,
                "config_seed_stride": CONFIG_SEED_STRIDE,
                "config_seeds": config_seeds,
                "quant_dequant_modes": quant_modes,
                "split_k_values": split_k_values,
                "tile_config_values": tile_config_values,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    md_file = os.path.join(output_dir, f"value_spmv_detailed_{timestamp}.md")
    with open(md_file, "w") as f:
        f.write("# Value SpMV 性能详细测试\n\n")
        f.write(f"**测试时间**: {timestamp}\n\n")
        f.write(f"- 稀疏度: {SPARSITY * 100:.0f}%\n")
        f.write(f"- 预热: {NUM_WARMUP} 次, 测试: {NUM_ITERS} 次\n")
        f.write(f"- 固定随机种子: {args.seed}\n")
        f.write(f"- Split-K sweep: {split_k_values}\n")
        f.write(f"- Tile Config sweep: {tile_config_values}\n\n")
        f.write("| 配置 | Sparse TPOT(ms) | Best Quant TPOT(ms) | Best Mode | Best Split-K | Best Tile |\n")
        f.write("|---|---:|---:|---|---:|---:|\n")
        for result in all_results:
            best = result["best_quant"]
            f.write(
                f"| {result['config']['name']} | {result['sparse']['batch_spmv_avg']:.4f} | "
                f"{best['batch_spmv_avg']:.4f} | {DEQUANT_MODE_LABELS.get(best['dequant_mode'], best['dequant_mode'])} | "
                f"{best['split_k']} | {best['tile_config']} |\n"
            )

    print(f"\n{'=' * 70}")
    print("Value SpMV 汇总报告")
    print(f"{'=' * 70}")
    for result in all_results:
        best = result["best_quant"]
        print(
            f"{result['config']['name']}: "
            f"Sparse={result['sparse']['batch_spmv_avg']:.4f} ms/token, "
            f"BestQuant={best['batch_spmv_avg']:.4f} ms/token "
            f"(mode={DEQUANT_MODE_LABELS.get(best['dequant_mode'], best['dequant_mode'])}, "
            f"split_k={best['split_k']}, tile={best['tile_config']})"
        )

    print("\n结果已保存到:")
    print(f"  JSON: {output_file}")
    print(f"  Markdown: {md_file}")


if __name__ == "__main__":
    main()
