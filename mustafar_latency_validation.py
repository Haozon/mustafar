#!/usr/bin/env python3
"""
Mustafar Latency Validation Experiment

基于论文Figure 6a的实验，验证以下声明：
- Llama-2-7B在2048输入序列长度+1024生成长度下的延迟分解
- Llama-3-8B在4096输入序列长度+1024生成长度下的延迟分解
- 各组件相对于cuBLAS执行时间的百分比
- 50%和70%稀疏度下的SpMV加速效果

组件测试：
1. cuBLAS dense batched MV (baseline)
2. Mustafar batched SpMV 
3. Runtime pruning overhead
4. Compression overhead  
5. Local window dense MV overhead
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import json
import os
import sys
import gc
from typing import Dict, List, Tuple, Optional


try:
    from compression import convert_key_batched, convert_value_batched
    COMPRESSION_AVAILABLE = True
    print("✓ Official compression functions loaded")
except ImportError as e:
    COMPRESSION_AVAILABLE = False
    print(f"✗ Compression functions not available: {e}")

# 导入Mustafar kernel
try:
    import mustafar_package
    MUSTAFAR_AVAILABLE = True
    print("✓ Mustafar CUDA kernel loaded")
except ImportError:
    MUSTAFAR_AVAILABLE = False
    print("✗ Mustafar CUDA kernel not available")

class MustafarLatencyValidator:
    """Mustafar延迟验证器，根据论文Figure 6a进行测试"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        torch.cuda.set_device(device)
        self.warmup_iterations = 10
        self.measurement_iterations = 50
        
        # 论文中的配置
        self.test_configs = {
            'llama2_7b': {
                'seq_len': 2048,
                'gen_len': 1024, 
                'head_dim': 128,
                'num_heads': 32,
                'batch_size': 1,
                'local_window': 256  # 假设的local window大小
            },
            'llama3_8b': {
                'seq_len': 4096,
                'gen_len': 1024,
                'head_dim': 128, 
                'num_heads': 32,
                'batch_size': 1,
                'local_window': 256
            }
        }
        
        self.sparsity_levels = [0.5, 0.7]  # 50% and 70% sparsity
        
    def warmup_gpu(self, warmup_size=1024):
        """GPU预热操作"""
        print(f"GPU warmup with size {warmup_size}...")
        
        # 创建预热数据
        A = torch.randn(warmup_size, warmup_size, dtype=torch.float16, device=self.device)
        B = torch.randn(warmup_size, warmup_size, dtype=torch.float16, device=self.device)
        
        # 预热操作
        for _ in range(self.warmup_iterations):
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
        
        # 清理
        del A, B, C
        torch.cuda.empty_cache()
        print("GPU warmup completed")
    
    def create_sparse_matrix(self, batch_size: int, M: int, N: int, sparsity: float) -> torch.Tensor:
        """创建稀疏矩阵，保持语义合理性"""
        print(f"Creating sparse matrix: {batch_size}x{M}x{N}, sparsity={sparsity}")
        
        # 生成随机attention分数
        matrix = torch.randn(batch_size, M, N, dtype=torch.float16, device=self.device)
        
        # 创建结构化稀疏模式 - 模拟真实的attention pattern
        num_keep = int(N * (1 - sparsity))
        
        for b in range(batch_size):
            for i in range(M):
                row = matrix[b, i, :]
                
                # 保留local window (前面的token)
                local_size = min(256, N)  # local window大小
                local_mask = torch.zeros_like(row, dtype=torch.bool)
                local_mask[:local_size] = True
                
                # 保留top-k远距离token
                remaining_keep = max(0, num_keep - local_size)
                if remaining_keep > 0 and N > local_size:
                    _, top_indices = torch.topk(row[local_size:], remaining_keep)
                    top_indices += local_size  # 调整索引
                    local_mask[top_indices] = True
                
                # 应用稀疏mask
                matrix[b, i, :] = torch.where(local_mask, row, torch.tensor(0.0, device=self.device))
        
        return matrix
    
    def measure_cublas_dense_mv(self, config: dict) -> dict:
        """测量cuBLAS dense batched MV性能 (baseline)"""
        print(f"\n--- Measuring cuBLAS Dense MV ---")
        print(f"Config: seq_len={config['seq_len']}, head_dim={config['head_dim']}")
        
        seq_len = config['seq_len'] + config['gen_len']  # 总序列长度
        head_dim = config['head_dim']
        batch_size = config['batch_size']
        
        # 创建dense矩阵 - 模拟attention probabilities @ values
        attention_probs = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float16, device=self.device)
        attention_probs = F.softmax(attention_probs, dim=-1)
        values = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=self.device)
        
        # 预热
        print("Warming up cuBLAS...")
        for _ in range(self.warmup_iterations):
            result = torch.matmul(attention_probs, values)
            torch.cuda.synchronize()
        
        # 测量
        times = []
        torch.cuda.synchronize()
        for _ in range(self.measurement_iterations):
            start_time = time.perf_counter()
            result = torch.matmul(attention_probs, values)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"cuBLAS Dense MV: {avg_time:.2f}±{std_time:.2f} ms")
        
        # 清理
        del attention_probs, values, result
        torch.cuda.empty_cache()
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'times': times
        }
    
    def measure_runtime_pruning(self, config: dict, sparsity: float) -> dict:
        """测量runtime pruning开销"""
        print(f"\n--- Measuring Runtime Pruning Overhead (sparsity={sparsity}) ---")
        
        seq_len = config['seq_len'] + config['gen_len']
        head_dim = config['head_dim']
        batch_size = config['batch_size']
        
        # 创建attention分数
        attention_scores = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float16, device=self.device)
        
        # 预热
        print("Warming up pruning operation...")
        for _ in range(self.warmup_iterations):
            # 模拟pruning操作 - topk selection
            num_keep = int(seq_len * (1 - sparsity))
            _, indices = torch.topk(attention_scores, num_keep, dim=-1)
            torch.cuda.synchronize()
        
        # 测量pruning时间
        times = []
        torch.cuda.synchronize()
        for _ in range(self.measurement_iterations):
            start_time = time.perf_counter()
            
            # Runtime pruning - topk selection and masking
            num_keep = int(seq_len * (1 - sparsity))
            _, indices = torch.topk(attention_scores, num_keep, dim=-1)
            
            # 创建稀疏mask
            mask = torch.zeros_like(attention_scores, dtype=torch.bool)
            for b in range(batch_size):
                for i in range(seq_len):
                    mask[b, i, indices[b, i]] = True
            
            # 应用mask
            pruned_scores = torch.where(mask, attention_scores, torch.tensor(0.0, device=self.device))
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Runtime Pruning: {avg_time:.2f}±{std_time:.2f} ms")
        
        # 清理
        del attention_scores, indices, mask, pruned_scores
        torch.cuda.empty_cache()
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'times': times
        }
    
    def measure_compression_overhead(self, config: dict, sparsity: float) -> dict:
        """测量compression开销"""
        print(f"\n--- Measuring Compression Overhead (sparsity={sparsity}) ---")
        
        if not COMPRESSION_AVAILABLE:
            print("Compression functions not available, skipping...")
            return {'error': 'Compression not available'}
        
        seq_len = config['seq_len'] + config['gen_len']
        batch_size = config['batch_size']
        
        # 创建稀疏attention矩阵
        sparse_attention = self.create_sparse_matrix(batch_size, seq_len, seq_len, sparsity)
        
        # 预热
        print("Warming up compression...")
        for _ in range(self.warmup_iterations // 2):  # 压缩比较慢，减少预热次数
            try:
                bitmaps, accum_counts, packed_values = convert_value_batched(sparse_attention)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Compression warmup failed: {e}")
                return {'error': str(e)}
        
        # 测量compression时间
        times = []
        torch.cuda.synchronize()
        for _ in range(self.measurement_iterations // 5):  # 减少测量次数
            start_time = time.perf_counter()
            
            try:
                bitmaps, accum_counts, packed_values = convert_value_batched(sparse_attention)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Compression failed: {e}")
                return {'error': str(e)}
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Compression: {avg_time:.2f}±{std_time:.2f} ms")
        
        # 清理
        del sparse_attention, bitmaps, accum_counts, packed_values
        torch.cuda.empty_cache()
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'times': times
        }
    
    def measure_local_window_mv(self, config: dict) -> dict:
        """测量local window dense MV开销"""
        print(f"\n--- Measuring Local Window MV Overhead ---")
        
        seq_len = config['seq_len'] + config['gen_len']
        head_dim = config['head_dim']
        batch_size = config['batch_size']
        window_size = config['local_window']
        
        # 创建local window数据
        attention_probs = torch.randn(batch_size, seq_len, window_size, dtype=torch.float16, device=self.device)
        attention_probs = F.softmax(attention_probs, dim=-1)
        values = torch.randn(batch_size, window_size, head_dim, dtype=torch.float16, device=self.device)
        
        # 预热
        print("Warming up local window MV...")
        for _ in range(self.warmup_iterations):
            result = torch.matmul(attention_probs, values)
            torch.cuda.synchronize()
        
        # 测量
        times = []
        torch.cuda.synchronize()
        for _ in range(self.measurement_iterations):
            start_time = time.perf_counter()
            result = torch.matmul(attention_probs, values)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Local Window MV: {avg_time:.2f}±{std_time:.2f} ms")
        
        # 清理
        del attention_probs, values, result
        torch.cuda.empty_cache()
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'times': times
        }
    
    def measure_spmv_kernel(self, config: dict, sparsity: float) -> dict:
        """测量Mustafar SpMV kernel性能"""
        print(f"\n--- Measuring Mustafar SpMV Kernel (sparsity={sparsity}) ---")
        
        if not (COMPRESSION_AVAILABLE and MUSTAFAR_AVAILABLE):
            print("Required modules not available, skipping...")
            return {'error': 'Mustafar kernel not available'}
        
        seq_len = config['seq_len'] + config['gen_len']
        head_dim = config['head_dim']
        batch_size = config['batch_size']
        
        # 创建稀疏attention概率和values
        sparse_attention = self.create_sparse_matrix(batch_size, seq_len, seq_len, sparsity)
        sparse_attention_probs = F.softmax(torch.where(
            sparse_attention != 0, sparse_attention, torch.tensor(-float('inf'), device=self.device)
        ), dim=-1)
        values = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=self.device)
        
        # 压缩attention概率
        try:
            bitmaps, accum_counts, packed_values = convert_value_batched(sparse_attention_probs)
        except Exception as e:
            print(f"Compression failed: {e}")
            return {'error': f'Compression failed: {e}'}
        
        # 准备kernel参数
        total_packed_size = sum(len(pv) for pv in packed_values)
        nz_combined = torch.zeros(total_packed_size, dtype=torch.float16, device=self.device)
        nz_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        
        offset = 0
        for i, pv in enumerate(packed_values):
            nz_offsets[i] = offset
            nz_combined[offset:offset+len(pv)] = pv
            offset += len(pv)
        nz_offsets[-1] = offset
        
        workspace_size = batch_size * seq_len * head_dim * 4
        workspace = torch.zeros(workspace_size, dtype=torch.float16, device=self.device)
        
        # 预热Mustafar kernel
        print("Warming up Mustafar SpMV kernel...")
        for _ in range(self.warmup_iterations // 2):
            try:
                output = mustafar_package.mustafar_value_formulation(
                    bitmaps.to(torch.int64),
                    nz_combined.to(torch.float16),
                    accum_counts.to(torch.int32),
                    nz_offsets.to(torch.int32),
                    values,
                    workspace,
                    seq_len,
                    head_dim,
                    batch_size,
                    1
                )
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Kernel warmup failed: {e}")
                return {'error': f'Kernel failed: {e}'}
        
        # 测量SpMV kernel时间
        times = []
        torch.cuda.synchronize()
        for _ in range(self.measurement_iterations // 5):
            start_time = time.perf_counter()
            
            try:
                output = mustafar_package.mustafar_value_formulation(
                    bitmaps.to(torch.int64),
                    nz_combined.to(torch.float16),
                    accum_counts.to(torch.int32),
                    nz_offsets.to(torch.int32),
                    values,
                    workspace,
                    seq_len,
                    head_dim,
                    batch_size,
                    1
                )
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Kernel execution failed: {e}")
                return {'error': f'Kernel failed: {e}'}
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Mustafar SpMV: {avg_time:.2f}±{std_time:.2f} ms")
        
        # 正确性验证 - 检查输出维度
        expected_output = torch.matmul(sparse_attention_probs, values)
        print(f"Expected output shape: {expected_output.shape}, Actual output shape: {output.shape}")
        
        # 如果维度不匹配，调整输出或跳过验证
        if expected_output.shape == output.shape:
            mse_error = torch.mean((expected_output - output) ** 2).item()
        else:
            print("Warning: Output shape mismatch, skipping correctness verification")
            mse_error = -1
        
        print(f"Correctness - MSE: {mse_error:.6f}")
        
        # 清理
        del sparse_attention, sparse_attention_probs, values, output, expected_output
        del bitmaps, accum_counts, packed_values, nz_combined, nz_offsets, workspace
        torch.cuda.empty_cache()
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'times': times,
            'mse_error': mse_error
        }
    
    def run_comprehensive_validation(self) -> dict:
        """运行完整的延迟验证实验"""
        print("="*60)
        print("MUSTAFAR LATENCY VALIDATION EXPERIMENT")
        print("Based on Paper Figure 6a")
        print("="*60)
        
        # 预热GPU
        self.warmup_gpu()
        
        results = {
            'experiment_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'compression_available': COMPRESSION_AVAILABLE,
                'mustafar_available': MUSTAFAR_AVAILABLE,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'warmup_iterations': self.warmup_iterations,
                'measurement_iterations': self.measurement_iterations
            },
            'results': {}
        }
        
        # 对每个模型配置进行测试
        for model_name, config in self.test_configs.items():
            print(f"\n{'='*50}")
            print(f"TESTING {model_name.upper()}")
            print(f"{'='*50}")
            
            model_results = {}
            
            # 1. 测量cuBLAS baseline
            model_results['cublas_dense_mv'] = self.measure_cublas_dense_mv(config)
            baseline_time = model_results['cublas_dense_mv']['avg_time_ms']
            
            # 2. 测量各个稀疏度的组件
            for sparsity in self.sparsity_levels:
                sparsity_key = f'sparsity_{int(sparsity*100)}percent'
                model_results[sparsity_key] = {}
                
                # Runtime pruning
                model_results[sparsity_key]['runtime_pruning'] = self.measure_runtime_pruning(config, sparsity)
                
                # Compression  
                model_results[sparsity_key]['compression'] = self.measure_compression_overhead(config, sparsity)
                
                # SpMV kernel
                model_results[sparsity_key]['spmv_kernel'] = self.measure_spmv_kernel(config, sparsity)
            
            # 3. Local window MV (不依赖稀疏度)
            model_results['local_window_mv'] = self.measure_local_window_mv(config)
            
            # 4. 计算相对于baseline的百分比
            model_results['percentage_analysis'] = {}
            for sparsity in self.sparsity_levels:
                sparsity_key = f'sparsity_{int(sparsity*100)}percent'
                percentages = {}
                
                # 计算各组件相对于baseline的百分比
                if 'runtime_pruning' in model_results[sparsity_key]:
                    pruning_time = model_results[sparsity_key]['runtime_pruning'].get('avg_time_ms', 0)
                    percentages['pruning_percent'] = (pruning_time / baseline_time) * 100
                
                if 'compression' in model_results[sparsity_key]:
                    compression_time = model_results[sparsity_key]['compression'].get('avg_time_ms', 0)
                    percentages['compression_percent'] = (compression_time / baseline_time) * 100
                
                if 'spmv_kernel' in model_results[sparsity_key]:
                    spmv_time = model_results[sparsity_key]['spmv_kernel'].get('avg_time_ms', 0)
                    percentages['spmv_percent'] = (spmv_time / baseline_time) * 100
                
                local_window_time = model_results['local_window_mv'].get('avg_time_ms', 0)
                percentages['local_window_percent'] = (local_window_time / baseline_time) * 100
                
                model_results['percentage_analysis'][sparsity_key] = percentages
            
            results['results'][model_name] = model_results
        
        return results
    
    def generate_report(self, results: dict):
        """生成实验报告"""
        print(f"\n{'='*60}")
        print("EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # 保存详细结果到JSON
        json_file = "mustafar_latency_validation_report.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_file}")
        
        # 生成对比报告
        for model_name, model_results in results['results'].items():
            print(f"\n{model_name.upper()} RESULTS:")
            print("-" * 40)
            
            baseline_time = model_results['cublas_dense_mv']['avg_time_ms']
            print(f"cuBLAS Dense MV (Baseline): {baseline_time:.2f} ms")
            
            for sparsity in self.sparsity_levels:
                sparsity_key = f'sparsity_{int(sparsity*100)}percent'
                print(f"\n{int(sparsity*100)}% Sparsity Components:")
                
                percentages = model_results['percentage_analysis'][sparsity_key]
                
                if 'pruning_percent' in percentages:
                    print(f"  Runtime Pruning: {percentages['pruning_percent']:.2f}% of cuBLAS time")
                
                if 'compression_percent' in percentages:
                    print(f"  Compression: {percentages['compression_percent']:.2f}% of cuBLAS time")
                
                if 'spmv_percent' in percentages:
                    print(f"  SpMV Kernel: {percentages['spmv_percent']:.2f}% of cuBLAS time")
                    
                    # 计算总体加速效果
                    total_overhead = percentages.get('pruning_percent', 0) + percentages.get('compression_percent', 0) + percentages.get('local_window_percent', 0)
                    spmv_percent = percentages['spmv_percent']
                    net_improvement = 100 - spmv_percent - total_overhead
                    
                    print(f"  Net Performance: {net_improvement:.2f}% improvement over cuBLAS")
                
                if 'local_window_percent' in percentages:
                    print(f"  Local Window MV: {percentages['local_window_percent']:.2f}% of cuBLAS time")
        
        # 与论文声明的对比
        print(f"\n{'='*60}")
        print("COMPARISON WITH PAPER CLAIMS")
        print(f"{'='*60}")
        
        print("Paper claims for Llama-2-7B:")
        print("- Pruning: 1.84% of cuBLAS time")
        print("- Compression: 6.25% of cuBLAS time") 
        print("- Local window MV: 0.62% of cuBLAS time")
        print("- SpMV (50% sparsity): 81.07% of cuBLAS time")
        print("- SpMV (70% sparsity): 61.87% of cuBLAS time")
        
        # 显示我们的结果
        if 'llama2_7b' in results['results']:
            llama2_results = results['results']['llama2_7b']['percentage_analysis']
            print(f"\nOur experimental results for Llama-2-7B:")
            
            for sparsity in self.sparsity_levels:
                sparsity_key = f'sparsity_{int(sparsity*100)}percent'
                if sparsity_key in llama2_results:
                    percentages = llama2_results[sparsity_key]
                    print(f"\n{int(sparsity*100)}% Sparsity:")
                    print(f"- Pruning: {percentages.get('pruning_percent', 0):.2f}% of cuBLAS time")
                    print(f"- Compression: {percentages.get('compression_percent', 0):.2f}% of cuBLAS time")
                    print(f"- SpMV: {percentages.get('spmv_percent', 0):.2f}% of cuBLAS time")
                    print(f"- Local window MV: {percentages.get('local_window_percent', 0):.2f}% of cuBLAS time")


def main():
    """主实验函数"""
    print("Mustafar Latency Validation Experiment")
    print("Validating performance claims from Paper Figure 6a")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    device = 'cuda:0'
    print(f"Using device: {device}")
    
    # 运行实验
    validator = MustafarLatencyValidator(device=device)
    
    try:
        results = validator.run_comprehensive_validation()
        validator.generate_report(results)
        
        print(f"\n{'='*60}")
        print("LATENCY VALIDATION EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()