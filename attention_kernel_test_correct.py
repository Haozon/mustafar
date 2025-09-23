#!/usr/bin/env python3
"""
Mustafar Attention Kernel Isolated Testing Script (Corrected Version)
使用官方压缩函数的正确版本

功能：
1. 使用官方compression.py中的convert_key_batched/convert_value_batched
2. 正确调用mustafar_key_formulation/mustafar_value_formulation
3. 模拟prefill和decoding阶段测试
4. 性能对比和正确性验证
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import json
import os
import sys
import psutil
import gc
from typing import Dict, List, Tuple, Optional

# 导入官方压缩函数
try:
    from compression import convert_key_batched, convert_value_batched
    COMPRESSION_AVAILABLE = True
    print("✓ Official compression functions loaded successfully")
except ImportError as e:
    COMPRESSION_AVAILABLE = False
    print(f"✗ Failed to load compression functions: {e}")

# 导入Mustafar kernel
try:
    import mustafar_package
    MUSTAFAR_AVAILABLE = True
    print("✓ Mustafar CUDA kernel loaded successfully")
except ImportError:
    MUSTAFAR_AVAILABLE = False
    print("✗ Mustafar CUDA kernel not available")

class AttentionKernelTester:
    """Attention核心测试器 - 使用官方接口"""
    
    def __init__(self, device='cuda:1'):
        self.device = device
        torch.cuda.set_device(device)
        self.results = {
            'mustafar_prefill': {},
            'mustafar_decoding': {},
            'original_prefill': {},
            'original_decoding': {},
            'performance_comparison': {},
            'memory_comparison': {},
            'correctness': {}
        }
        
    def create_sparse_attention_matrix(self, batch_size: int, seq_len: int, 
                                     head_dim: int, sparsity: float = 0.7) -> torch.Tensor:
        """创建稀疏attention矩阵"""
        print(f"Creating sparse attention matrix: {batch_size}x{seq_len}x{seq_len}, sparsity={sparsity}")
        
        # 生成标准attention分数
        attention_scores = torch.randn(batch_size, seq_len, seq_len, 
                                     dtype=torch.float16, device=self.device)
        
        # 创建稀疏mask (保留top-k)
        num_keep = int(seq_len * (1 - sparsity))
        
        # 对每行保留top-k个最大值，其余置0
        for b in range(batch_size):
            for i in range(seq_len):
                row = attention_scores[b, i, :]
                _, top_indices = torch.topk(row, num_keep)
                mask = torch.zeros_like(row, dtype=torch.bool)
                mask[top_indices] = True
                attention_scores[b, i, :] = torch.where(mask, row, torch.tensor(0.0, device=self.device))
        
        return attention_scores
    
    def measure_memory_usage(self) -> Dict:
        """测量当前GPU显存使用"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)    # GB
            return {
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved
            }
        return {'allocated_gb': 0, 'reserved_gb': 0}
    
    def standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """标准Attention计算"""
        # Q, K, V shape: [batch, seq_len, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -float('inf'))
        
        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output, attn_probs
    
    def test_original_attention_prefill(self, batch_size: int, seq_len: int, head_dim: int) -> Dict:
        """测试原始Attention的Prefill阶段"""
        print(f"\n--- Testing Original Attention (Prefill) ---")
        print(f"Config: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
        
        results = {'config': {'batch_size': batch_size, 'seq_len': seq_len, 'head_dim': head_dim}}
        
        # 记录初始显存
        initial_memory = self.measure_memory_usage()
        
        # 生成输入数据
        Q = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=self.device)
        K = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=self.device)
        V = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=self.device)
        
        peak_memory = self.measure_memory_usage()
        
        # 预热
        for _ in range(3):
            output, _ = self.standard_attention(Q, K, V)
            torch.cuda.synchronize()
        
        # 正式测试 - 多次运行取平均
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            
            output, attn_probs = self.standard_attention(Q, K, V)
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.update({
            'execution_time_ms': avg_time,
            'time_std_ms': std_time,
            'output_shape': list(output.shape),
            'memory_usage': {
                'initial': initial_memory,
                'peak': peak_memory
            }
        })
        
        print(f"Original Attention (Prefill) - Time: {avg_time:.2f}±{std_time:.2f} ms")
        print(f"Memory usage: {peak_memory['allocated_gb']:.2f} GB")
        
        return results
    
    def test_original_attention_decoding(self, batch_size: int, seq_len: int, head_dim: int) -> Dict:
        """测试原始Attention的Decoding阶段 (KV Cache)"""
        print(f"\n--- Testing Original Attention (Decoding) ---")
        print(f"Config: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
        
        results = {'config': {'batch_size': batch_size, 'seq_len': seq_len, 'head_dim': head_dim}}
        
        # 记录初始显存
        initial_memory = self.measure_memory_usage()
        
        # 模拟KV Cache (已缓存的seq_len-1个token)
        kv_cache_len = seq_len - 1
        K_cache = torch.randn(batch_size, kv_cache_len, head_dim, dtype=torch.float16, device=self.device)
        V_cache = torch.randn(batch_size, kv_cache_len, head_dim, dtype=torch.float16, device=self.device)
        
        # 新token的QKV
        Q_new = torch.randn(batch_size, 1, head_dim, dtype=torch.float16, device=self.device)
        K_new = torch.randn(batch_size, 1, head_dim, dtype=torch.float16, device=self.device)
        V_new = torch.randn(batch_size, 1, head_dim, dtype=torch.float16, device=self.device)
        
        # 拼接得到完整的K, V
        K_full = torch.cat([K_cache, K_new], dim=1)
        V_full = torch.cat([V_cache, V_new], dim=1)
        
        peak_memory = self.measure_memory_usage()
        
        # 预热
        for _ in range(3):
            output, _ = self.standard_attention(Q_new, K_full, V_full)
            torch.cuda.synchronize()
        
        # 正式测试
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            
            output, attn_probs = self.standard_attention(Q_new, K_full, V_full)
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.update({
            'execution_time_ms': avg_time,
            'time_std_ms': std_time,
            'output_shape': list(output.shape),
            'kv_cache_length': kv_cache_len,
            'memory_usage': {
                'initial': initial_memory,
                'peak': peak_memory
            }
        })
        
        print(f"Original Attention (Decoding) - Time: {avg_time:.2f}±{std_time:.2f} ms")
        print(f"KV Cache length: {kv_cache_len}, Memory: {peak_memory['allocated_gb']:.2f} GB")
        
        return results
    
    def test_mustafar_attention_prefill(self, batch_size: int, seq_len: int, head_dim: int, 
                                      sparsity: float = 0.7) -> Dict:
        """测试Mustafar Attention的Prefill阶段"""
        print(f"\n--- Testing Mustafar Attention (Prefill) ---")
        print(f"Config: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
        
        results = {'config': {'batch_size': batch_size, 'seq_len': seq_len, 'head_dim': head_dim}}
        
        if not (COMPRESSION_AVAILABLE and MUSTAFAR_AVAILABLE):
            print("Skipping test - required modules not available")
            return results
        
        # 记录初始显存
        initial_memory = self.measure_memory_usage()
        
        # 1. 创建稀疏attention矩阵 (相当于Q*K^T的结果)
        sparse_attention = self.create_sparse_attention_matrix(batch_size, seq_len, seq_len, sparsity)
        
        # 2. 使用官方压缩函数
        print("Compressing sparse matrix using official convert_key_batched...")
        torch.cuda.synchronize()
        compress_start = time.time()
        
        try:
            bitmaps, accum_counts, packed_values = convert_key_batched(sparse_attention)
            torch.cuda.synchronize()
            compress_time = (time.time() - compress_start) * 1000
            
            print(f"Compression completed in {compress_time:.2f} ms")
            print(f"Bitmap shape: {bitmaps.shape}")
            print(f"Counts shape: {accum_counts.shape}")
            print(f"Packed values: {len(packed_values)} batches")
            
            results['compression'] = {
                'time_ms': compress_time,
                'bitmap_shape': list(bitmaps.shape),
                'counts_shape': list(accum_counts.shape),
                'num_packed_batches': len(packed_values)
            }
            
        except Exception as e:
            print(f"Compression failed: {e}")
            results['compression'] = {'error': str(e)}
            return results
        
        # 3. 准备Key矩阵 (随机生成)
        K_matrix = torch.randn(batch_size, seq_len, head_dim, 
                              dtype=torch.float16, device=self.device)
        
        # 4. 调用Mustafar Key内核
        print("Calling mustafar_key_formulation...")
        try:
            # 准备输入参数
            # 注意：需要根据实际接口调整参数格式
            
            # 合并所有batch的packed_values
            total_packed_size = sum(len(pv) for pv in packed_values)
            nz_combined = torch.zeros(total_packed_size, dtype=torch.float16, device=self.device)
            nz_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            
            offset = 0
            for i, pv in enumerate(packed_values):
                
                nz_offsets[i] = offset
                nz_combined[offset:offset+len(pv)] = pv
                offset += len(pv)
            nz_offsets[-1] = offset
            
            torch.cuda.synchronize()
            kernel_start = time.time()
            
            # 调用Mustafar内核 - 修复数据类型
            output = mustafar_package.mustafar_key_formulation(
                bitmaps.to(torch.int64),      # bmp
                nz_combined.to(torch.float16),  # NZ (必须是float16)
                accum_counts.to(torch.int32), # idx
                nz_offsets.to(torch.int32),   # NZ_offset
                K_matrix,                     # B matrix
                seq_len,                      # M_Global  
                head_dim,                     # K_Global
                batch_size,                   # Batch_Size
                1                             # num_key_value_groups (假设为1)
            )
            
            torch.cuda.synchronize()
            kernel_time = (time.time() - kernel_start) * 1000
            
            print(f"Kernel execution completed in {kernel_time:.2f} ms")
            print(f"Output shape: {output.shape}")
            
            peak_memory = self.measure_memory_usage()
            
            results['kernel'] = {
                'time_ms': kernel_time,
                'output_shape': list(output.shape),
                'output_dtype': str(output.dtype)
            }
            
            results['memory_usage'] = {
                'initial': initial_memory,
                'peak': peak_memory
            }
            
        except Exception as e:
            print(f"Kernel execution failed: {e}")
            results['kernel'] = {'error': str(e)}
            import traceback
            traceback.print_exc()
        
        return results
    
    def test_mustafar_attention_decoding(self, batch_size: int, seq_len: int, head_dim: int,
                                       sparsity: float = 0.7) -> Dict:
        """测试Mustafar Attention的Decoding阶段"""
        print(f"\n--- Testing Mustafar Attention (Decoding) ---") 
        print(f"Config: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
        
        results = {'config': {'batch_size': batch_size, 'seq_len': seq_len, 'head_dim': head_dim}}
        
        if not (COMPRESSION_AVAILABLE and MUSTAFAR_AVAILABLE):
            print("Skipping test - required modules not available")
            return results
        
        # 记录初始显存
        initial_memory = self.measure_memory_usage()
        
        # 1. 创建稀疏attention概率矩阵
        sparse_attention = self.create_sparse_attention_matrix(batch_size, seq_len, seq_len, sparsity)
        
        # 应用softmax得到attention概率
        sparse_attention_probs = torch.softmax(torch.where(
            sparse_attention != 0, sparse_attention, torch.tensor(-float('inf'), device=self.device)
        ), dim=-1)
        
        # 2. 使用官方压缩函数
        print("Compressing attention probabilities using convert_value_batched...")
        torch.cuda.synchronize()
        compress_start = time.time()
        
        try:
            bitmaps, accum_counts, packed_values = convert_value_batched(sparse_attention_probs)
            torch.cuda.synchronize()
            compress_time = (time.time() - compress_start) * 1000
            
            print(f"Compression completed in {compress_time:.2f} ms")
            results['compression'] = {
                'time_ms': compress_time,
                'bitmap_shape': list(bitmaps.shape),
                'counts_shape': list(accum_counts.shape),
                'num_packed_batches': len(packed_values)
            }
            
        except Exception as e:
            print(f"Compression failed: {e}")
            results['compression'] = {'error': str(e)}
            return results
        
        # 3. 准备Value矩阵
        V_matrix = torch.randn(batch_size, seq_len, head_dim,
                              dtype=torch.float16, device=self.device)
        
        # 4. 调用Mustafar Value内核
        print("Calling mustafar_value_formulation...")
        try:
            # 准备workspace
            workspace_size = batch_size * seq_len * head_dim * 4  # 预留足够空间
            workspace = torch.zeros(workspace_size, dtype=torch.float16, device=self.device)
            
            # 合并packed values
            total_packed_size = sum(len(pv) for pv in packed_values)
            nz_combined = torch.zeros(total_packed_size, dtype=torch.float16, device=self.device)
            nz_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            
            offset = 0
            for i, pv in enumerate(packed_values):
                nz_offsets[i] = offset
                nz_combined[offset:offset+len(pv)] = pv  
                offset += len(pv)
            nz_offsets[-1] = offset
            
            torch.cuda.synchronize()
            kernel_start = time.time()
            
            # 调用Value内核 - 修复数据类型
            output = mustafar_package.mustafar_value_formulation(
                bitmaps.to(torch.int64),      # bmp
                nz_combined.to(torch.float16),  # NZ (必须是float16)
                accum_counts.to(torch.int32), # idx
                nz_offsets.to(torch.int32),   # NZ_offset
                V_matrix,                     # B matrix
                workspace,                    # Reduction_Workspace
                seq_len,                      # M_Global
                head_dim,                     # K_Global  
                batch_size,                   # Batch_Size
                1                             # num_key_value_groups
            )
            
            torch.cuda.synchronize()
            kernel_time = (time.time() - kernel_start) * 1000
            
            print(f"Kernel execution completed in {kernel_time:.2f} ms")
            print(f"Output shape: {output.shape}")
            
            results['kernel'] = {
                'time_ms': kernel_time,
                'output_shape': list(output.shape),
                'output_dtype': str(output.dtype)
            }
            
            # 5. 正确性验证 - 对比标准计算
            expected_output = torch.matmul(sparse_attention_probs, V_matrix)
            mse_error = torch.mean((expected_output - output) ** 2).item()
            max_error = torch.max(torch.abs(expected_output - output)).item()
            
            results['correctness'] = {
                'mse_error': mse_error,
                'max_error': max_error
            }
            
            print(f"Correctness - MSE: {mse_error:.6f}, Max Error: {max_error:.6f}")
            
            peak_memory = self.measure_memory_usage()
            results['memory_usage'] = {
                'initial': initial_memory,
                'peak': peak_memory
            }
            
        except Exception as e:
            print(f"Kernel execution failed: {e}")
            results['kernel'] = {'error': str(e)}
            import traceback
            traceback.print_exc()
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """运行综合测试"""
        print("="*60)
        print("COMPREHENSIVE ATTENTION KERNEL TEST")
        print("Mustafar vs Original Attention Comparison")
        print("="*60)
        
        test_configs = [
            # 小规模测试
            {'batch_size': 1, 'seq_len': 512, 'head_dim': 64, 'sparsity': 0.5},
            {'batch_size': 2, 'seq_len': 1024, 'head_dim': 128, 'sparsity': 0.7},
            
            # 中等规模测试
            {'batch_size': 4, 'seq_len': 2048, 'head_dim': 128, 'sparsity': 0.8},
        ]
        
        all_results = {
            'test_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'compression_available': COMPRESSION_AVAILABLE,
                'mustafar_available': MUSTAFAR_AVAILABLE,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            },
            'original_attention_prefill': {},
            'original_attention_decoding': {},
            'mustafar_attention_prefill': {},
            'mustafar_attention_decoding': {},
            'performance_comparison': {},
            'memory_comparison': {}
        }
        
        # 测试原始Attention - Prefill阶段
        print("\n" + "="*50)
        print("TESTING ORIGINAL ATTENTION - PREFILL PHASE")
        print("="*50)
        
        for i, config in enumerate(test_configs):
            config_name = f"config_{i}_{config['batch_size']}x{config['seq_len']}x{config['head_dim']}"
            print(f"\nTesting {config_name}...")
            all_results['original_attention_prefill'][config_name] = self.test_original_attention_prefill(
                config['batch_size'], config['seq_len'], config['head_dim']
            )
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
        
        # 测试原始Attention - Decoding阶段
        print("\n" + "="*50)
        print("TESTING ORIGINAL ATTENTION - DECODING PHASE")
        print("="*50)
        
        for i, config in enumerate(test_configs):
            config_name = f"config_{i}_{config['batch_size']}x{config['seq_len']}x{config['head_dim']}"
            print(f"\nTesting {config_name}...")
            all_results['original_attention_decoding'][config_name] = self.test_original_attention_decoding(
                config['batch_size'], config['seq_len'], config['head_dim']
            )
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
        
        # 测试Mustafar Attention - Prefill阶段  
        print("\n" + "="*50)
        print("TESTING MUSTAFAR ATTENTION - PREFILL PHASE")
        print("="*50)
        
        for i, config in enumerate(test_configs):
            config_name = f"config_{i}_{config['batch_size']}x{config['seq_len']}x{config['head_dim']}"
            print(f"\nTesting {config_name}...")
            all_results['mustafar_attention_prefill'][config_name] = self.test_mustafar_attention_prefill(**config)
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
        
        # 测试Mustafar Attention - Decoding阶段
        print("\n" + "="*50)
        print("TESTING MUSTAFAR ATTENTION - DECODING PHASE")
        print("="*50)
        
        for i, config in enumerate(test_configs):
            config_name = f"config_{i}_{config['batch_size']}x{config['seq_len']}x{config['head_dim']}"
            print(f"\nTesting {config_name}...")
            all_results['mustafar_attention_decoding'][config_name] = self.test_mustafar_attention_decoding(**config)
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
        
        # 计算性能对比
        all_results['performance_comparison'] = self._calculate_performance_comparison(all_results)
        all_results['memory_comparison'] = self._calculate_memory_comparison(all_results)
        
        return all_results
    
    def generate_report(self, results: Dict, json_file: str = "mustafar_attention_test_report.json", 
                       md_file: str = "ATTENTION_KERNEL_TEST_REPORT.md"):
        """生成测试报告"""
        print("\n" + "="*60)
        print("GENERATING TEST REPORTS")
        print("="*60)
        
        # 保存详细JSON报告
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed JSON report saved to: {json_file}")
        
        # 生成Markdown报告
        self._generate_markdown_report(results, md_file)
        print(f"Markdown report saved to: {md_file}")
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _calculate_performance_comparison(self, results: Dict) -> Dict:
        """计算性能对比"""
        comparison = {
            'prefill_phase': {},
            'decoding_phase': {}
        }
        
        # Prefill阶段对比
        original_prefill = results.get('original_attention_prefill', {})
        mustafar_prefill = results.get('mustafar_attention_prefill', {})
        
        for config_name in original_prefill.keys():
            if config_name in mustafar_prefill:
                orig_time = original_prefill[config_name].get('execution_time_ms', 0)
                mustafar_compression_time = mustafar_prefill[config_name].get('compression', {}).get('time_ms', 0)
                mustafar_kernel_time = mustafar_prefill[config_name].get('kernel', {}).get('time_ms', 0)
                mustafar_total_time = mustafar_compression_time + mustafar_kernel_time
                
                speedup = orig_time / mustafar_total_time if mustafar_total_time > 0 else 0
                
                comparison['prefill_phase'][config_name] = {
                    'original_time_ms': orig_time,
                    'mustafar_compression_time_ms': mustafar_compression_time,
                    'mustafar_kernel_time_ms': mustafar_kernel_time,
                    'mustafar_total_time_ms': mustafar_total_time,
                    'speedup_ratio': speedup,
                    'performance_gain_percent': (speedup - 1) * 100 if speedup > 0 else 0
                }
        
        # Decoding阶段对比
        original_decoding = results.get('original_attention_decoding', {})
        mustafar_decoding = results.get('mustafar_attention_decoding', {})
        
        for config_name in original_decoding.keys():
            if config_name in mustafar_decoding:
                orig_time = original_decoding[config_name].get('execution_time_ms', 0)
                mustafar_compression_time = mustafar_decoding[config_name].get('compression', {}).get('time_ms', 0)
                mustafar_kernel_time = mustafar_decoding[config_name].get('kernel', {}).get('time_ms', 0)
                mustafar_total_time = mustafar_compression_time + mustafar_kernel_time
                
                speedup = orig_time / mustafar_total_time if mustafar_total_time > 0 else 0
                
                comparison['decoding_phase'][config_name] = {
                    'original_time_ms': orig_time,
                    'mustafar_compression_time_ms': mustafar_compression_time,
                    'mustafar_kernel_time_ms': mustafar_kernel_time,
                    'mustafar_total_time_ms': mustafar_total_time,
                    'speedup_ratio': speedup,
                    'performance_gain_percent': (speedup - 1) * 100 if speedup > 0 else 0
                }
        
        return comparison
    
    def _calculate_memory_comparison(self, results: Dict) -> Dict:
        """计算显存使用对比"""
        comparison = {
            'prefill_phase': {},
            'decoding_phase': {}
        }
        
        # Prefill阶段显存对比
        original_prefill = results.get('original_attention_prefill', {})
        mustafar_prefill = results.get('mustafar_attention_prefill', {})
        
        for config_name in original_prefill.keys():
            if config_name in mustafar_prefill:
                orig_memory = original_prefill[config_name].get('memory_usage', {}).get('peak', {})
                mustafar_memory = mustafar_prefill[config_name].get('memory_usage', {}).get('peak', {})
                
                orig_allocated = orig_memory.get('allocated_gb', 0)
                mustafar_allocated = mustafar_memory.get('allocated_gb', 0)
                
                memory_reduction = (orig_allocated - mustafar_allocated) / orig_allocated * 100 if orig_allocated > 0 else 0
                
                comparison['prefill_phase'][config_name] = {
                    'original_memory_gb': orig_allocated,
                    'mustafar_memory_gb': mustafar_allocated,
                    'memory_reduction_percent': memory_reduction
                }
        
        # Decoding阶段显存对比
        original_decoding = results.get('original_attention_decoding', {})
        mustafar_decoding = results.get('mustafar_attention_decoding', {})
        
        for config_name in original_decoding.keys():
            if config_name in mustafar_decoding:
                orig_memory = original_decoding[config_name].get('memory_usage', {}).get('peak', {})
                mustafar_memory = mustafar_decoding[config_name].get('memory_usage', {}).get('peak', {})
                
                orig_allocated = orig_memory.get('allocated_gb', 0)
                mustafar_allocated = mustafar_memory.get('allocated_gb', 0)
                
                memory_reduction = (orig_allocated - mustafar_allocated) / orig_allocated * 100 if orig_allocated > 0 else 0
                
                comparison['decoding_phase'][config_name] = {
                    'original_memory_gb': orig_allocated,
                    'mustafar_memory_gb': mustafar_allocated,
                    'memory_reduction_percent': memory_reduction
                }
        
        return comparison
    
    def _generate_markdown_report(self, results: Dict, output_file: str):
        """生成Markdown格式测试报告"""
        markdown_content = f"""# Attention Kernel Performance Test Report

## Test Information
- **Timestamp**: {results['test_info']['timestamp']}
- **Device**: {results['test_info']['device']}
- **PyTorch Version**: {results['test_info']['pytorch_version']}
- **CUDA Version**: {results['test_info']['cuda_version']}
- **Compression Available**: {'✓' if results['test_info']['compression_available'] else '✗'}
- **Mustafar Kernel Available**: {'✓' if results['test_info']['mustafar_available'] else '✗'}

## Performance Comparison

### Prefill Phase Performance

| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |
|---------------|---------------|---------------------|------------------|-------------|---------|------------------|
"""
        
        # 添加Prefill阶段数据
        prefill_comparison = results.get('performance_comparison', {}).get('prefill_phase', {})
        for config_name, data in prefill_comparison.items():
            config_display = config_name.replace('config_', '').replace('_', ' ')
            markdown_content += f"| {config_display} | {data['original_time_ms']:.2f} | {data['mustafar_total_time_ms']:.2f} | {data['mustafar_compression_time_ms']:.2f} | {data['mustafar_kernel_time_ms']:.2f} | {data['speedup_ratio']:.2f}x | {data['performance_gain_percent']:.1f}% |\n"
        
        markdown_content += "\n### Decoding Phase Performance\n\n| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |\n|---------------|---------------|---------------------|------------------|-------------|---------|------------------|\n"
        
        # 添加Decoding阶段数据
        decoding_comparison = results.get('performance_comparison', {}).get('decoding_phase', {})
        for config_name, data in decoding_comparison.items():
            config_display = config_name.replace('config_', '').replace('_', ' ')
            markdown_content += f"| {config_display} | {data['original_time_ms']:.2f} | {data['mustafar_total_time_ms']:.2f} | {data['mustafar_compression_time_ms']:.2f} | {data['mustafar_kernel_time_ms']:.2f} | {data['speedup_ratio']:.2f}x | {data['performance_gain_percent']:.1f}% |\n"
        
        markdown_content += "\n## Memory Usage Comparison\n\n### Prefill Phase Memory\n\n| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |\n|---------------|---------------|---------------|------------------|\n"
        
        # 添加Prefill阶段显存数据
        prefill_memory = results.get('memory_comparison', {}).get('prefill_phase', {})
        for config_name, data in prefill_memory.items():
            config_display = config_name.replace('config_', '').replace('_', ' ')
            markdown_content += f"| {config_display} | {data['original_memory_gb']:.2f} | {data['mustafar_memory_gb']:.2f} | {data['memory_reduction_percent']:.1f}% |\n"
        
        markdown_content += "\n### Decoding Phase Memory\n\n| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |\n|---------------|---------------|---------------|------------------|\n"
        
        # 添加Decoding阶段显存数据
        decoding_memory = results.get('memory_comparison', {}).get('decoding_phase', {})
        for config_name, data in decoding_memory.items():
            config_display = config_name.replace('config_', '').replace('_', ' ')
            markdown_content += f"| {config_display} | {data['original_memory_gb']:.2f} | {data['mustafar_memory_gb']:.2f} | {data['memory_reduction_percent']:.1f}% |\n"
        
        # 添加正确性信息
        markdown_content += "\n## Correctness Verification\n\n"
        
        # 检查Mustafar decoding结果中的正确性信息
        mustafar_decoding = results.get('mustafar_attention_decoding', {})
        has_correctness_data = False
        for config_name, data in mustafar_decoding.items():
            if 'correctness' in data:
                has_correctness_data = True
                config_display = config_name.replace('config_', '').replace('_', ' ')
                mse = data['correctness'].get('mse_error', 0)
                max_error = data['correctness'].get('max_error', 0)
                markdown_content += f"- **{config_display}**: MSE Error = {mse:.6f}, Max Error = {max_error:.6f}\n"
        
        if not has_correctness_data:
            markdown_content += "No correctness verification data available.\n"
        
        markdown_content += "\n## Summary\n\n"
        
        # 计算平均性能提升
        prefill_gains = [data['performance_gain_percent'] for data in prefill_comparison.values() if data['performance_gain_percent'] != 0]
        decoding_gains = [data['performance_gain_percent'] for data in decoding_comparison.values() if data['performance_gain_percent'] != 0]
        
        avg_prefill_gain = np.mean(prefill_gains) if prefill_gains else 0
        avg_decoding_gain = np.mean(decoding_gains) if decoding_gains else 0
        
        markdown_content += f"- **Average Prefill Phase Performance Gain**: {avg_prefill_gain:.1f}%\n"
        markdown_content += f"- **Average Decoding Phase Performance Gain**: {avg_decoding_gain:.1f}%\n"
        
        # 计算平均显存减少
        prefill_memory_reductions = [data['memory_reduction_percent'] for data in prefill_memory.values()]
        decoding_memory_reductions = [data['memory_reduction_percent'] for data in decoding_memory.values()]
        
        avg_prefill_memory_reduction = np.mean(prefill_memory_reductions) if prefill_memory_reductions else 0
        avg_decoding_memory_reduction = np.mean(decoding_memory_reductions) if decoding_memory_reductions else 0
        
        markdown_content += f"- **Average Prefill Phase Memory Reduction**: {avg_prefill_memory_reduction:.1f}%\n"
        markdown_content += f"- **Average Decoding Phase Memory Reduction**: {avg_decoding_memory_reduction:.1f}%\n"
        
        markdown_content += f"\n---\n*Report generated on {results['test_info']['timestamp']}*\n"
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _print_summary(self, results: Dict):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        
        print(f"Timestamp: {results['test_info']['timestamp']}")
        print(f"Device: {results['test_info']['device']}")
        print(f"Compression Available: {'✓' if results['test_info']['compression_available'] else '✗'}")
        print(f"Mustafar Kernel Available: {'✓' if results['test_info']['mustafar_available'] else '✗'}")
        
        # 原始Attention测试摘要
        print(f"\nORIGINAL ATTENTION TESTS:")
        orig_prefill_success = 0
        orig_prefill_total = len(results.get('original_attention_prefill', {}))
        for name, result in results.get('original_attention_prefill', {}).items():
            if 'execution_time_ms' in result:
                orig_prefill_success += 1
                print(f"  ✓ Prefill {name}: {result['execution_time_ms']:.2f}ms, Memory: {result.get('memory_usage', {}).get('peak', {}).get('allocated_gb', 0):.2f}GB")
            else:
                print(f"  ✗ Prefill {name}: Failed")
        
        orig_decoding_success = 0
        orig_decoding_total = len(results.get('original_attention_decoding', {}))
        for name, result in results.get('original_attention_decoding', {}).items():
            if 'execution_time_ms' in result:
                orig_decoding_success += 1
                print(f"  ✓ Decoding {name}: {result['execution_time_ms']:.2f}ms, Memory: {result.get('memory_usage', {}).get('peak', {}).get('allocated_gb', 0):.2f}GB")
            else:
                print(f"  ✗ Decoding {name}: Failed")
        
        # Mustafar Attention测试摘要
        print(f"\nMUSTAFAR ATTENTION TESTS:")
        mustafar_prefill_success = 0
        mustafar_prefill_total = len(results.get('mustafar_attention_prefill', {}))
        for name, result in results.get('mustafar_attention_prefill', {}).items():
            if 'kernel' in result and 'error' not in result['kernel']:
                mustafar_prefill_success += 1
                total_time = result.get('compression', {}).get('time_ms', 0) + result['kernel'].get('time_ms', 0)
                print(f"  ✓ Prefill {name}: {total_time:.2f}ms (compression: {result.get('compression', {}).get('time_ms', 0):.2f}ms + kernel: {result['kernel'].get('time_ms', 0):.2f}ms)")
            else:
                print(f"  ✗ Prefill {name}: Failed")
        
        mustafar_decoding_success = 0
        mustafar_decoding_total = len(results.get('mustafar_attention_decoding', {}))
        for name, result in results.get('mustafar_attention_decoding', {}).items():
            if 'kernel' in result and 'error' not in result['kernel']:
                mustafar_decoding_success += 1
                total_time = result.get('compression', {}).get('time_ms', 0) + result['kernel'].get('time_ms', 0)
                mse = result.get('correctness', {}).get('mse_error', 0)
                print(f"  ✓ Decoding {name}: {total_time:.2f}ms, MSE: {mse:.6f}")
            else:
                print(f"  ✗ Decoding {name}: Failed")
        
        # 性能对比摘要
        print(f"\nPERFORMANCE COMPARISON:")
        prefill_comparison = results.get('performance_comparison', {}).get('prefill_phase', {})
        for name, data in prefill_comparison.items():
            if data['speedup_ratio'] > 0:
                print(f"  Prefill {name}: {data['speedup_ratio']:.2f}x speedup ({data['performance_gain_percent']:.1f}% gain)")
        
        decoding_comparison = results.get('performance_comparison', {}).get('decoding_phase', {})
        for name, data in decoding_comparison.items():
            if data['speedup_ratio'] > 0:
                print(f"  Decoding {name}: {data['speedup_ratio']:.2f}x speedup ({data['performance_gain_percent']:.1f}% gain)")
        
        print(f"\nOVERALL RESULTS:")
        total_tests = orig_prefill_total + orig_decoding_total + mustafar_prefill_total + mustafar_decoding_total
        total_success = orig_prefill_success + orig_decoding_success + mustafar_prefill_success + mustafar_decoding_success
        print(f"  Original Attention tests passed: {orig_prefill_success + orig_decoding_success}/{orig_prefill_total + orig_decoding_total}")
        print(f"  Mustafar Attention tests passed: {mustafar_prefill_success + mustafar_decoding_success}/{mustafar_prefill_total + mustafar_decoding_total}")
        print(f"  Overall success rate: {(total_success / total_tests * 100):.1f}%" if total_tests > 0 else "  No tests completed")


def main():
    """主测试函数"""
    print("Comprehensive Attention Kernel Test (Mustafar vs Original)")
    print("Testing on GPU device: cuda:1")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    if torch.cuda.device_count() < 2:
        print("WARNING: Less than 2 GPUs available. Using cuda:0 instead of cuda:1")
        device = 'cuda:0'
    else:
        device = 'cuda:1'
    
    print(f"Using device: {device}")
    
    # 运行测试
    tester = AttentionKernelTester(device=device)
    
    try:
        results = tester.run_comprehensive_test()
        tester.generate_report(results)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()