#!/usr/bin/env python3
"""
统一性能测试脚本
在相同负载条件下对比 Dense、Sparse-50%、Sparse-50%+Quant-2bit
"""
import torch
import os
import sys
import time
import json
from datetime import datetime
from transformers import AutoTokenizer
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 8
INPUT_LENGTH = 4096
OUTPUT_LENGTH = 1024
NUM_REPEATS = 3

MODEL_PATH = '/home/zh/model/Meta-Llama-3-8B-Instruct'

# 测试配置
CONFIGS = {
    'dense': {
        'name': 'Dense (FP16)',
        'k_sparsity': 0.0,
        'v_sparsity': 0.0,
        'use_quant': False,
        'model_class': 'llama_mustafar_kernel'
    },
    'sparse_50': {
        'name': 'Sparse-50% (FP16)',
        'k_sparsity': 0.5,
        'v_sparsity': 0.5,
        'use_quant': False,
        'model_class': 'llama_mustafar_kernel'
    },
    'sparse_50_quant_2bit': {
        'name': 'Sparse-50% + Quant-2bit',
        'k_sparsity': 0.5,
        'v_sparsity': 0.5,
        'use_quant': True,
        'model_class': 'llama_mustafar_quant_kernel'
    }
}

def print_separator(char='=', length=70):
    print(char * length)

def print_header(text):
    print_separator()
    print(f"🔥 {text}")
    print_separator()

def get_gpu_memory():
    """获取当前 GPU 内存使用"""
    return torch.cuda.max_memory_allocated() / 1024**3  # GB

def reset_gpu_memory():
    """重置 GPU 内存统计"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

def load_model(config_name, config_params):
    """加载模型"""
    from transformers import LlamaConfig
    
    print(f"\n📦 Loading model: {config_params['name']}")
    
    # 创建配置
    config = LlamaConfig.from_pretrained(MODEL_PATH)
    config.k_sparsity = config_params['k_sparsity']
    config.v_sparsity = config_params['v_sparsity']
    config.group_size = 32
    config.residual_length = 256
    config.use_flash = True
    
    if config_params['use_quant']:
        config.quant_bits = 2
        from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT
        model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
            MODEL_PATH,
            config=config,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR
        model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
            MODEL_PATH,
            config=config,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    
    model.eval()
    print(f"✅ Model loaded: {config_params['name']}")
    return model

def run_benchmark(config_name, config_params):
    """运行单个配置的 benchmark"""
    print_header(f"Testing: {config_params['name']}")
    
    results = {
        'config_name': config_name,
        'display_name': config_params['name'],
        'k_sparsity': config_params['k_sparsity'],
        'v_sparsity': config_params['v_sparsity'],
        'use_quant': config_params['use_quant'],
        'batch_size': BATCH_SIZE,
        'input_length': INPUT_LENGTH,
        'output_length': OUTPUT_LENGTH,
        'repeats': []
    }
    
    # 加载模型
    reset_gpu_memory()
    model = load_model(config_name, config_params)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备输入 - 使用与 benchmark_throughput.py 相同的方法
    context = []
    for _ in range(BATCH_SIZE):
        string = 'apple bear' * (INPUT_LENGTH // 2)
        context.append(string[:-1])
    
    inputs = tokenizer(context, return_tensors="pt").to('cuda')
    
    print(f"\n📝 Test Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Input length: {inputs['input_ids'].shape[1]}")
    print(f"   Output length: {OUTPUT_LENGTH}")
    print(f"   K Sparsity: {config_params['k_sparsity']*100:.0f}%")
    print(f"   V Sparsity: {config_params['v_sparsity']*100:.0f}%")
    print(f"   Quantization: {'2-bit' if config_params['use_quant'] else 'None'}")
    
    # Warmup
    print(f"\n⏳ Running warmup...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None
        )
    torch.cuda.synchronize()
    print("✅ Warmup completed")
    
    # 运行测试
    print(f"\n🔥 Running {NUM_REPEATS} test iterations...")
    
    for i in range(NUM_REPEATS):
        print(f"\n   Iteration {i+1}/{NUM_REPEATS}...")
        
        reset_gpu_memory()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=OUTPUT_LENGTH,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # ms
        peak_memory = get_gpu_memory()
        
        # 计算指标
        total_tokens = BATCH_SIZE * OUTPUT_LENGTH
        throughput = total_tokens / (total_time / 1000)  # tokens/sec
        avg_tpot = total_time / OUTPUT_LENGTH  # ms per token
        
        repeat_result = {
            'iteration': i + 1,
            'total_time_ms': total_time,
            'throughput': throughput,
            'tpot': avg_tpot,
            'peak_memory_gb': peak_memory
        }
        
        results['repeats'].append(repeat_result)
        
        print(f"      Total time: {total_time:.2f} ms")
        print(f"      Throughput: {throughput:.2f} tokens/sec")
        print(f"      TPOT: {avg_tpot:.2f} ms")
        print(f"      Peak memory: {peak_memory:.2f} GB")
    
    # 计算平均值
    avg_total_time = sum(r['total_time_ms'] for r in results['repeats']) / NUM_REPEATS
    avg_throughput = sum(r['throughput'] for r in results['repeats']) / NUM_REPEATS
    avg_tpot = sum(r['tpot'] for r in results['repeats']) / NUM_REPEATS
    avg_memory = sum(r['peak_memory_gb'] for r in results['repeats']) / NUM_REPEATS
    
    results['averages'] = {
        'total_time_ms': avg_total_time,
        'throughput': avg_throughput,
        'tpot': avg_tpot,
        'peak_memory_gb': avg_memory
    }
    
    print(f"\n📊 Average Results:")
    print(f"   Total time: {avg_total_time:.2f} ms")
    print(f"   Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"   TPOT: {avg_tpot:.2f} ms")
    print(f"   Peak memory: {avg_memory:.2f} GB")
    
    # 清理
    del model
    del tokenizer
    reset_gpu_memory()
    
    return results

def print_comparison(all_results):
    """打印对比结果"""
    print_header("📊 Performance Comparison")
    
    # 提取数据
    configs = []
    throughputs = []
    tpots = []
    memories = []
    
    for result in all_results:
        configs.append(result['display_name'])
        throughputs.append(result['averages']['throughput'])
        tpots.append(result['averages']['tpot'])
        memories.append(result['averages']['peak_memory_gb'])
    
    # 打印表格
    print("\n| Configuration | Throughput | TPOT | Memory |")
    print("|---------------|------------|------|--------|")
    for i, config in enumerate(configs):
        print(f"| {config:30s} | {throughputs[i]:8.2f} tok/s | {tpots[i]:6.2f} ms | {memories[i]:6.2f} GB |")
    
    # 计算相对性能（以 Dense 为基准）
    if len(all_results) > 0:
        baseline_throughput = throughputs[0]
        baseline_tpot = tpots[0]
        baseline_memory = memories[0]
        
        print("\n| Configuration | Throughput vs Dense | TPOT vs Dense | Memory vs Dense |")
        print("|---------------|---------------------|---------------|-----------------|")
        for i, config in enumerate(configs):
            throughput_ratio = throughputs[i] / baseline_throughput
            tpot_ratio = tpots[i] / baseline_tpot
            memory_ratio = memories[i] / baseline_memory
            
            throughput_str = f"{throughput_ratio:.2f}x"
            if throughput_ratio > 1:
                throughput_str += " ↑"
            elif throughput_ratio < 1:
                throughput_str += " ↓"
            
            tpot_str = f"{tpot_ratio:.2f}x"
            if tpot_ratio < 1:
                tpot_str += " ↑"
            elif tpot_ratio > 1:
                tpot_str += " ↓"
            
            memory_str = f"{memory_ratio:.2f}x"
            if memory_ratio < 1:
                memory_str += " ↑"
            
            print(f"| {config:30s} | {throughput_str:19s} | {tpot_str:13s} | {memory_str:15s} |")

def save_results(all_results, filename='results/unified_benchmark_results.json'):
    """保存结果到文件"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'test_config': {
            'batch_size': BATCH_SIZE,
            'input_length': INPUT_LENGTH,
            'output_length': OUTPUT_LENGTH,
            'num_repeats': NUM_REPEATS,
            'model_path': MODEL_PATH
        },
        'results': all_results
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

def check_gpu_status():
    """检查 GPU 状态"""
    print_header("🖥️  GPU Status Check")
    
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Failed to run nvidia-smi: {e}")
    
    print("\n⚠️  Please verify:")
    print("   1. GPU utilization is low (< 10%) before starting")
    print("   2. No other processes are using the GPU")
    print("   3. Memory usage is minimal")
    
    response = input("\n▶️  Continue with benchmark? (y/n): ")
    return response.lower() == 'y'

def main():
    print_separator('=', 70)
    print("🚀 Unified Performance Benchmark")
    print("   Testing: Dense, Sparse-50%, Sparse-50%+Quant-2bit")
    print("   Under SAME load conditions")
    print_separator('=', 70)
    
    # 检查 GPU 状态
    if not check_gpu_status():
        print("\n❌ Benchmark cancelled by user")
        return
    
    # 运行所有配置的测试
    all_results = []
    
    for config_name, config_params in CONFIGS.items():
        try:
            result = run_benchmark(config_name, config_params)
            all_results.append(result)
            
            # 等待一下，让 GPU 冷却
            print("\n⏸️  Waiting 10 seconds before next test...")
            time.sleep(10)
            
        except Exception as e:
            print(f"\n❌ Error testing {config_params['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印对比结果
    if all_results:
        print("\n")
        print_comparison(all_results)
        
        # 保存结果
        save_results(all_results)
    
    print_separator('=', 70)
    print("✅ Benchmark completed!")
    print_separator('=', 70)

if __name__ == '__main__':
    main()
