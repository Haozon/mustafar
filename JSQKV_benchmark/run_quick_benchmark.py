#!/usr/bin/env python3
"""
快速 Benchmark 测试 - 测试量化模型的基本性能
"""
import torch
import os
from transformers import LlamaConfig, AutoTokenizer
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 配置
K_SPARSITY = 0.5
V_SPARSITY = 0.5
QUANT_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 256
BATCH_SIZE = 1  # 使用小 batch 以加快测试
PROMPT_LENGTH = 1024  # 使用较短的 prompt
OUTPUT_LENGTH = 128  # 使用较短的输出

model_name_or_path = '/home/zh/model/Meta-Llama-3-8B-Instruct'

print("="*70)
print("🚀 快速 Benchmark 测试 - 量化模型")
print("="*70)
print(f"\n配置:")
print(f"  K Sparsity: {K_SPARSITY*100}%")
print(f"  V Sparsity: {V_SPARSITY*100}%")
print(f"  Quantization: {QUANT_BITS}-bit")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Prompt Length: ~{PROMPT_LENGTH} tokens")
print(f"  Output Length: {OUTPUT_LENGTH} tokens")

# 加载模型
print(f"\n加载模型...")
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_sparsity = K_SPARSITY
config.v_sparsity = V_SPARSITY
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
config.use_flash = True
config.quant_bits = QUANT_BITS

from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT

model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
    trust_remote_code=True,
)

model.eval()

# 准备输入
torch.manual_seed(42)
context = []
for _ in range(BATCH_SIZE):
    string = 'apple bear ' * (PROMPT_LENGTH // 2)
    context.append(string[:-1])

inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']

print(f"\n实际输入长度: {input_ids.shape[1]} tokens")

# Warmup
print(f"\n⏳ Warmup...")
with torch.no_grad():
    torch.cuda.synchronize()
    _ = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
    torch.cuda.synchronize()

# Benchmark
print(f"\n🔥 运行 Benchmark...")
num_repeats = 3
times = []

torch.cuda.reset_peak_memory_stats()

for i in range(num_repeats):
    print(f"  Repeat {i+1}/{num_repeats}...")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=OUTPUT_LENGTH, eos_token_id=None)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        print(f"    Time: {elapsed:.2f} ms")

avg_time = sum(times) / len(times)
peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

# 计算指标
total_tokens = BATCH_SIZE * (input_ids.shape[1] + OUTPUT_LENGTH)
throughput = total_tokens / (avg_time / 1000)  # tokens/sec
ttft_estimate = avg_time / OUTPUT_LENGTH  # 粗略估计

print(f"\n{'='*70}")
print(f"📊 Benchmark 结果")
print(f"{'='*70}")
print(f"\n性能指标:")
print(f"  平均生成时间: {avg_time:.2f} ms")
print(f"  峰值内存: {peak_mem:.2f} GB")
print(f"  吞吐量: {throughput:.2f} tokens/sec")
print(f"  估计 TPOT: {ttft_estimate:.2f} ms/token")
print(f"\n工作负载:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Input length: {input_ids.shape[1]} tokens")
print(f"  Output length: {OUTPUT_LENGTH} tokens")
print(f"  Total tokens: {total_tokens}")
print(f"\n压缩配置:")
print(f"  K Sparsity: {K_SPARSITY*100}%")
print(f"  V Sparsity: {V_SPARSITY*100}%")
print(f"  Quantization: {QUANT_BITS}-bit")
theoretical_compression = 16 / (2 ** (QUANT_BITS - 1))
print(f"  理论压缩比: ~{theoretical_compression:.1f}x")
print(f"{'='*70}\n")

# 保存结果
result_file = f"quick_benchmark_results_{QUANT_BITS}bit.txt"
with open(result_file, 'w') as f:
    f.write(f"Quick Benchmark Results\n")
    f.write(f"{'='*70}\n")
    f.write(f"Configuration:\n")
    f.write(f"  K Sparsity: {K_SPARSITY*100}%\n")
    f.write(f"  V Sparsity: {V_SPARSITY*100}%\n")
    f.write(f"  Quantization: {QUANT_BITS}-bit\n")
    f.write(f"  Batch size: {BATCH_SIZE}\n")
    f.write(f"  Input length: {input_ids.shape[1]}\n")
    f.write(f"  Output length: {OUTPUT_LENGTH}\n")
    f.write(f"\nResults:\n")
    f.write(f"  Average time: {avg_time:.2f} ms\n")
    f.write(f"  Peak memory: {peak_mem:.2f} GB\n")
    f.write(f"  Throughput: {throughput:.2f} tokens/sec\n")
    f.write(f"  Estimated TPOT: {ttft_estimate:.2f} ms/token\n")

print(f"✅ 结果已保存到: {result_file}")
