#!/usr/bin/env python3
"""
DiffSparseKV 性能分析工具
精确测量每个操作的耗时
"""

import torch
import time
import numpy as np
from contextlib import contextmanager

# 导入模型
from llama_diff_sparse_kv import (
    LlamaDiffSparseKVAttention,
    create_diff_sparse_kv_config
)
from transformers import LlamaConfig

# 性能计时器
class PerformanceTimer:
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    @contextmanager
    def measure(self, name):
        """测量代码块的执行时间"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        
        self.timings[name].append(elapsed)
        self.counts[name] += 1
    
    def report(self):
        """生成性能报告"""
        print("\n" + "="*80)
        print("性能分析报告")
        print("="*80)
        
        # 按总时间排序
        total_times = {name: sum(times) for name, times in self.timings.items()}
        sorted_items = sorted(total_times.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(total_times.values())
        
        print(f"\n{'操作':<40s} {'调用次数':>8s} {'总时间(ms)':>12s} {'平均(ms)':>12s} {'占比':>8s}")
        print("-"*80)
        
        for name, total in sorted_items:
            count = self.counts[name]
            avg = total / count if count > 0 else 0
            percentage = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"{name:<40s} {count:>8d} {total*1000:>12.2f} {avg*1000:>12.2f} {percentage:>7.1f}%")
        
        print("-"*80)
        print(f"{'总计':<40s} {sum(self.counts.values()):>8d} {total_time*1000:>12.2f}")
        print("="*80)
        
        return total_times

# 全局计时器
timer = PerformanceTimer()

# Monkey patch 关键函数来测量性能
def patch_attention_for_profiling(attention_module):
    """给 attention 模块打补丁以测量性能"""
    
    # 保存原始方法
    original_prefill = attention_module._prefill_with_diff_sparse
    original_decode = attention_module._decode_with_diff_sparse
    original_decode_step = attention_module.decode_step_with_dual_window
    original_compression = attention_module._trigger_compression
    
    def profiled_prefill(self, query_states, key_states, value_states, attention_mask, kv_seq_len):
        with timer.measure("1. Prefill (总计)"):
            # 细分 prefill 的各个步骤
            
            # 1. Attention 计算
            with timer.measure("  1.1 Prefill: repeat_kv"):
                from transformers.models.llama.modeling_llama import repeat_kv
                key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
            
            with timer.measure("  1.2 Prefill: Q @ K^T"):
                attn_weights = torch.matmul(
                    query_states,
                    key_states_repeated.transpose(2, 3)
                ) / torch.sqrt(torch.tensor(self.head_dim, dtype=query_states.dtype))
            
            with timer.measure("  1.3 Prefill: Softmax"):
                import torch.nn.functional as F
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # 2. GQA 聚合
            with timer.measure("  1.4 Prefill: GQA Aggregation"):
                bsz = query_states.shape[0]
                q_len = query_states.shape[2]
                attn_weights_grouped = attn_weights.view(
                    bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, kv_seq_len
                )
                attn_weights_kv = attn_weights_grouped.mean(dim=2)
            
            # 3. Importance 计算
            with timer.measure("  1.5 Prefill: Importance Calculation"):
                importance_scores = self.importance_calculator.compute_diffkv_importance(
                    attn_weights_kv
                )
            
            # 4. Threshold 计算
            with timer.measure("  1.6 Prefill: Threshold Calculation"):
                threshold_high, threshold_low = self.threshold_manager.compute_per_layer_thresholds(
                    importance_scores
                )
            
            self.diff_sparse_thresholds = (threshold_high, threshold_low)
            self.thresholds_computed = True
            
            # 5. 稀疏化应用
            if kv_seq_len >= self.window_size:
                prefix_length = kv_seq_len - self.window_size
                prefix_keys = key_states[:, :, :prefix_length, :]
                prefix_values = value_states[:, :, :prefix_length, :]
                prefix_importance = importance_scores[:, :, :prefix_length]
                
                with timer.measure("  1.7 Prefill: Sparsity Application"):
                    compressed_keys, compressed_values = self.sparsity_applier.classify_and_apply_sparsity(
                        prefix_importance, prefix_keys, prefix_values,
                        thresholds=(threshold_high, threshold_low)
                    )
                
                window_keys = key_states[:, :, -self.window_size:, :]
                window_values = value_states[:, :, -self.window_size:, :]
                
                with timer.measure("  1.8 Prefill: Concatenation"):
                    key_states_full = torch.cat([compressed_keys, window_keys], dim=2)
                    value_states_full = torch.cat([compressed_values, window_values], dim=2)
            else:
                key_states_full = key_states
                value_states_full = value_states
            
            # 6. Attention output
            with timer.measure("  1.9 Prefill: Attention @ V"):
                from transformers.models.llama.modeling_llama import repeat_kv
                attn_output = torch.matmul(
                    attn_weights,
                    repeat_kv(value_states, self.num_key_value_groups)
                )
            
            # 7. 初始化双窗口
            with timer.measure("  1.10 Prefill: Initialize Windows"):
                self.initialize_dual_window_after_prefill(key_states_full, value_states_full)
            
            self.prefill_length = kv_seq_len
            self.current_sequence_length = kv_seq_len
            
            past_key_value = (key_states_full, value_states_full, kv_seq_len)
            return attn_output, None, past_key_value
    
    def profiled_decode_step(self, query_states, new_key, new_value, attention_mask):
        with timer.measure("2. Decode Step (总计)"):
            ws = self.window_state
            W = self.window_size
            
            # 1. 添加新 token
            with timer.measure("  2.1 Decode: Add Token (cat)"):
                if ws['window_a_size'] < W:
                    if ws['window_a_keys'] is None:
                        ws['window_a_keys'] = new_key
                        ws['window_a_values'] = new_value
                    else:
                        ws['window_a_keys'] = torch.cat([ws['window_a_keys'], new_key], dim=2)
                        ws['window_a_values'] = torch.cat([ws['window_a_values'], new_value], dim=2)
                    ws['window_a_size'] += 1
                else:
                    if ws['window_b_keys'] is None:
                        ws['window_b_keys'] = new_key
                        ws['window_b_values'] = new_value
                    else:
                        ws['window_b_keys'] = torch.cat([ws['window_b_keys'], new_key], dim=2)
                        ws['window_b_values'] = torch.cat([ws['window_b_values'], new_value], dim=2)
                    ws['window_b_size'] += 1
            
            # 2. 重建完整 KV cache
            with timer.measure("  2.2 Decode: Build Full Cache (cat)"):
                kv_list = []
                if ws['compressed_keys'] is not None:
                    kv_list.append((ws['compressed_keys'], ws['compressed_values']))
                if ws['window_a_keys'] is not None:
                    kv_list.append((ws['window_a_keys'], ws['window_a_values']))
                if ws['window_b_keys'] is not None:
                    kv_list.append((ws['window_b_keys'], ws['window_b_values']))
                
                full_keys = torch.cat([k for k, v in kv_list], dim=2)
                full_values = torch.cat([v for k, v in kv_list], dim=2)
            
            # 3. 计算 attention
            with timer.measure("  2.3 Decode: Q @ K^T"):
                from transformers.models.llama.modeling_llama import repeat_kv
                import math
                attn_weights = torch.matmul(
                    query_states,
                    repeat_kv(full_keys, self.num_key_value_groups).transpose(2, 3)
                ) / math.sqrt(self.head_dim)
            
            with timer.measure("  2.4 Decode: Softmax"):
                import torch.nn.functional as F
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # 4. GQA 聚合
            with timer.measure("  2.5 Decode: GQA Aggregation"):
                B, H_q = query_states.shape[:2]
                H_kv = self.num_key_value_heads
                attn_weights_grouped = attn_weights.view(
                    B, H_kv, self.num_key_value_groups, 1, -1
                )
                attn_weights_kv = attn_weights_grouped.mean(dim=2)
            
            # 5. 累积 attention
            with timer.measure("  2.6 Decode: Accumulate Attention"):
                if ws['window_a_size'] > 0:
                    compressed_len = 0 if ws['compressed_keys'] is None else ws['compressed_keys'].shape[2]
                    window_a_start = compressed_len
                    window_a_end = window_a_start + ws['window_a_size']
                    
                    window_a_attention = attn_weights_kv[:, :, 0, window_a_start:window_a_end]
                    ws['accumulator'][:, :, :ws['window_a_size']] += window_a_attention
                    ws['token_observation_count'][:, :, :ws['window_a_size']] += 1
            
            # 6. 计算 output
            with timer.measure("  2.7 Decode: Attention @ V"):
                from transformers.models.llama.modeling_llama import repeat_kv
                attn_output = torch.matmul(
                    attn_weights,
                    repeat_kv(full_values, self.num_key_value_groups)
                )
            
            # 7. 检查压缩
            if ws['window_a_size'] == W and ws['window_b_size'] >= W:
                with timer.measure("  2.8 Decode: Trigger Compression"):
                    self._trigger_compression()
            
            return attn_output
    
    # 替换方法
    attention_module._prefill_with_diff_sparse = lambda *args, **kwargs: profiled_prefill(attention_module, *args, **kwargs)
    attention_module.decode_step_with_dual_window = lambda *args, **kwargs: profiled_decode_step(attention_module, *args, **kwargs)


def run_performance_test():
    """运行性能测试"""
    print("="*80)
    print("DiffSparseKV 性能分析")
    print("="*80)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    # 创建配置
    config = LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=1,
    )
    
    config = create_diff_sparse_kv_config(
        base_config=config,
        enable_diff_sparse=True,
        target_distribution=[0.05, 0.75, 0.20],
        sparsity_levels=[0.0, 0.7, 1.0],
        diff_sparse_window_size=32,
        debug_diff_sparse=False
    )
    
    # 创建 attention 层
    attention = LlamaDiffSparseKVAttention(config).to(device).to(dtype)
    attention.eval()
    
    # 打补丁
    patch_attention_for_profiling(attention)
    
    # 测试参数
    batch_size = 1
    prefill_len = 2048  # 测试 2048 tokens
    decode_steps = 128  # 测试 128 步 decode
    
    print(f"\n测试配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_len}")
    print(f"  Decode steps: {decode_steps}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Num heads (Q): {config.num_attention_heads}")
    print(f"  Num heads (KV): {config.num_key_value_heads}")
    print(f"  Window size: {config.window_size}")
    
    # Warmup
    print("\n预热中...")
    with torch.no_grad():
        hidden_states = torch.randn(batch_size, 10, config.hidden_size, device=device, dtype=dtype)
        _ = attention(hidden_states, use_cache=True)
    
    # 测试 Prefill
    print("\n测试 Prefill 阶段...")
    with torch.no_grad():
        hidden_states = torch.randn(batch_size, prefill_len, config.hidden_size, device=device, dtype=dtype)
        
        output, _, past_kv = attention(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            use_cache=True
        )
    
    # 测试 Decode
    print(f"测试 Decode 阶段 ({decode_steps} 步)...")
    with torch.no_grad():
        for i in range(decode_steps):
            hidden_states_decode = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=dtype)
            
            output_decode, _, past_kv = attention(
                hidden_states=hidden_states_decode,
                attention_mask=None,
                position_ids=None,
                past_key_value=past_kv,
                use_cache=True
            )
    
    # 生成报告
    timer.report()
    
    # 额外分析
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    total_times = {name: sum(times) for name, times in timer.timings.items()}
    total_time = sum(total_times.values())
    
    # Prefill vs Decode
    prefill_time = total_times.get("1. Prefill (总计)", 0)
    decode_time = total_times.get("2. Decode Step (总计)", 0)
    
    print(f"\nPrefill vs Decode:")
    print(f"  Prefill: {prefill_time*1000:.2f} ms ({prefill_time/total_time*100:.1f}%)")
    print(f"  Decode:  {decode_time*1000:.2f} ms ({decode_time/total_time*100:.1f}%)")
    
    # Prefill 细分
    print(f"\nPrefill 细分 (前 5 项):")
    prefill_items = [(k, v) for k, v in total_times.items() if k.startswith("  1.")]
    prefill_items.sort(key=lambda x: x[1], reverse=True)
    for name, time in prefill_items[:5]:
        print(f"  {name}: {time*1000:.2f} ms ({time/prefill_time*100:.1f}%)")
    
    # Decode 细分
    print(f"\nDecode 细分 (前 5 项):")
    decode_items = [(k, v) for k, v in total_times.items() if k.startswith("  2.")]
    decode_items.sort(key=lambda x: x[1], reverse=True)
    for name, time in decode_items[:5]:
        print(f"  {name}: {time*1000:.2f} ms ({time/decode_time*100:.1f}%)")
    
    # Cat 操作统计
    cat_time = sum(v for k, v in total_times.items() if 'cat' in k.lower())
    print(f"\nCat 操作总耗时: {cat_time*1000:.2f} ms ({cat_time/total_time*100:.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    run_performance_test()
