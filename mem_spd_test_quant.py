# LLaMA model with Mustafar Quantized Sparse Kernel
import torch
import os
from transformers import LlamaConfig, AutoTokenizer
import time

'''
量化稀疏 KV Cache 性能测评
- 稀疏度: 50% (K_SPARSITY = 0.5, V_SPARSITY = 0.5)
- 量化位宽: 2-bit
- 测试指标: TTFT, TPOT, 内存占用
'''

# ==================== 配置参数 ====================
K_SPARSITY = 0.5      # Key 稀疏度 50%
V_SPARSITY = 0.5      # Value 稀疏度 50%
QUANT_BITS = 2        # 2-bit 量化
GROUP_SIZE = 32
RESIDUAL_LENGTH = 256
BATCH_SIZE = 8

# 测试模式选择
QUANT_MODE = True     # True: 使用量化稀疏, False: 使用标准 Mustafar
# QUANT_MODE = True

PROMPT_LENGTH = 4096
OUTPUT_LENGTH = 1024
NUM_REPEATS = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== 模型配置 ====================
model_name_or_path = '/home/zh/model/Meta-Llama-3-8B-Instruct'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_sparsity = K_SPARSITY
config.v_sparsity = V_SPARSITY
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
config.use_flash = True

if QUANT_MODE:
    # 使用量化稀疏内核
    from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT
    print("="*60)
    print("🚀 Using Mustafar Quantized Sparse Kernel")
    print(f"   - K Sparsity: {K_SPARSITY*100}%")
    print(f"   - V Sparsity: {V_SPARSITY*100}%")
    print(f"   - Quantization: {QUANT_BITS}-bit")
    print(f"   - Expected Memory Compression: ~{16/(2**(QUANT_BITS-1))}x")
    print("="*60)
    
    config.quant_bits = QUANT_BITS
    model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()
else:
    # 使用标准 Mustafar（仅稀疏，无量化）
    from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR
    print("="*60)
    print("📊 Using Standard Mustafar (Sparse only, no quantization)")
    print(f"   - K Sparsity: {K_SPARSITY*100}%")
    print(f"   - V Sparsity: {V_SPARSITY*100}%")
    print(f"   - Expected Memory Compression: ~{1/K_SPARSITY}x")
    print("="*60)
    
    model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
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

# ==================== 测评函数 ====================

def benchmark_with_token_timing(model, inputs, model_name, max_tokens=600, num_repeats=3):
    """
    测量TTFT和TPOT的函数
    返回: (ttft_ms, tpot_ms, total_time_ms, peak_memory_gb)
    """
    print(f"\n{'='*60}")
    print(f"🔍 Token-level timing for {model_name}")
    print(f"{'='*60}")
    
    # Warmup
    print("⏳ Running token-level warmup...")
    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
        torch.cuda.synchronize()
    
    all_ttft_times = []
    all_tpot_times = []
    
    for repeat in range(num_repeats):
        print(f"\n📌 Repeat {repeat + 1}/{num_repeats}...")
        token_times = []
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            input_ids = inputs['input_ids'].clone()
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.clone()
            past_key_values = None
            
            for token_idx in range(max_tokens):
                torch.cuda.synchronize()
                token_start = time.time()
                
                # Generate next token
                if past_key_values is None:
                    # First token - full forward pass with all input tokens
                    current_input = {'input_ids': input_ids}
                    if attention_mask is not None:
                        current_input['attention_mask'] = attention_mask
                else:
                    # Subsequent tokens - only the last generated token
                    current_input = {'input_ids': input_ids[:, -1:]}
                
                current_input['past_key_values'] = past_key_values
                current_input['use_cache'] = True
                
                outputs = model(**current_input)
                
                # Get next token
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                
                torch.cuda.synchronize()
                token_time = (time.time() - token_start) * 1000  # ms
                token_times.append(token_time)
                
                # Print progress every 100 tokens
                if (token_idx + 1) % 100 == 0:
                    print(f"   Generated {token_idx + 1}/{max_tokens} tokens...")
                
                # Update for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                past_key_values = outputs.past_key_values
        
        if len(token_times) > 0:
            ttft = token_times[0]
            tpot = sum(token_times[1:]) / len(token_times[1:]) if len(token_times) > 1 else token_times[0]
            all_ttft_times.append(ttft)
            all_tpot_times.append(tpot)
        
            print(f"   ✅ Repeat {repeat + 1}: TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms")
    
    # Calculate averages
    avg_ttft = sum(all_ttft_times) / len(all_ttft_times) if all_ttft_times else 0
    avg_tpot = sum(all_tpot_times) / len(all_tpot_times) if all_tpot_times else 0
    total_avg_time = avg_ttft + avg_tpot * (max_tokens - 1)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    print(f"\n{'='*60}")
    print(f"📊 Token-level Results:")
    print(f"   Average TTFT: {avg_ttft:.2f} ms")
    print(f"   Average TPOT: {avg_tpot:.2f} ms")
    print(f"   Total generation time: {total_avg_time:.2f} ms")
    print(f"   Peak memory: {peak_memory:.2f} GB")
    print(f"{'='*60}")
    
    return avg_ttft, avg_tpot, total_avg_time, peak_memory

# ==================== 主测试流程 ====================

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

context = []
batch_size = BATCH_SIZE
prompt_length = PROMPT_LENGTH
output_length = OUTPUT_LENGTH
num_repeats = NUM_REPEATS

# 构造测试输入
for _ in range(batch_size):
    string = 'apple bear' * (prompt_length // 2)
    context.append(string[:-1])
    
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']

print(f"\n{'='*60}")
print(f"📝 Test Configuration:")
print(f"   Batch size: {batch_size}")
print(f"   Input length: {input_ids.shape[1]}")
print(f"   Output length: {output_length}")
print(f"   Model: {model_name_or_path}")
print(f"{'='*60}")

# Warmup
print("\n⏳ Running warmup iteration...")
with torch.no_grad():
    torch.cuda.synchronize()
    outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
    torch.cuda.synchronize()

# ==================== Batch Generation Test ====================
print(f"\n{'='*60}")
print("🔥 Running Batch Generation Test")
print(f"{'='*60}")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        print(f"   Batch repeat {i+1}/{num_repeats}...")
        outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
    torch.cuda.synchronize()
    batch_time = (time.time() - st) / num_repeats * 1000
    used_mem = torch.cuda.max_memory_allocated()
    
    print(f"\n📊 Batch Generation Results:")
    print(f"   Average time: {batch_time:.2f} ms")
    print(f"   Peak memory: {used_mem / 1024 ** 3:.2f} GB")
    print(f"   Throughput: {batch_size * output_length / (batch_time / 1000):.2f} tokens/sec")

# ==================== Token-level Timing Test ====================
model_name = f"Mustafar-Quant-{QUANT_BITS}bit" if QUANT_MODE else f"Mustafar-Sparse"
ttft, tpot, total_time, peak_mem = benchmark_with_token_timing(
    model, inputs, model_name, max_tokens=output_length, num_repeats=num_repeats
)

# ==================== Final Summary ====================
print(f"\n{'='*60}")
print("🎯 FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Model Configuration:")
print(f"   Mode: {model_name}")
if QUANT_MODE:
    print(f"   Quantization: {QUANT_BITS}-bit per-token")
print(f"   K Sparsity: {K_SPARSITY*100}%")
print(f"   V Sparsity: {V_SPARSITY*100}%")
print(f"\nPerformance Metrics:")
print(f"   TTFT (Time to First Token): {ttft:.2f} ms")
print(f"   TPOT (Time per Output Token): {tpot:.2f} ms")
print(f"   Total generation time: {total_time:.2f} ms")
print(f"   Peak memory: {peak_mem:.2f} GB")
print(f"\nWorkload:")
print(f"   Batch size: {batch_size}")
print(f"   Input length: {input_ids.shape[1]} tokens")
print(f"   Output length: {output_length} tokens")
print(f"   Total tokens: {batch_size * (input_ids.shape[1] + output_length)}")

if QUANT_MODE:
    theoretical_compression = 16 / (2 ** (QUANT_BITS - 1))  # FP16 to 2-bit with 50% sparsity
    print(f"\nMemory Compression:")
    print(f"   Theoretical: ~{theoretical_compression:.1f}x")
    print(f"   (FP16 baseline → {QUANT_BITS}-bit + {K_SPARSITY*100}% sparse)")

print(f"{'='*60}\n")

# Save results to file
result_file = f"mem_spd_test_quant_results_{QUANT_BITS}bit.txt"
with open(result_file, 'w') as f:
    f.write(f"Mustafar Quantized Sparse Kernel Test Results\n")
    f.write(f"{'='*60}\n")
    f.write(f"Configuration:\n")
    f.write(f"  Model: {model_name}\n")
    if QUANT_MODE:
        f.write(f"  Quantization: {QUANT_BITS}-bit per-token\n")
    f.write(f"  K Sparsity: {K_SPARSITY*100}%\n")
    f.write(f"  V Sparsity: {V_SPARSITY*100}%\n")
    f.write(f"  Batch size: {batch_size}\n")
    f.write(f"  Input length: {input_ids.shape[1]}\n")
    f.write(f"  Output length: {output_length}\n")
    f.write(f"\nResults:\n")
    f.write(f"  TTFT: {ttft:.2f} ms\n")
    f.write(f"  TPOT: {tpot:.2f} ms\n")
    f.write(f"  Total time: {total_time:.2f} ms\n")
    f.write(f"  Peak memory: {peak_mem:.2f} GB\n")
    f.write(f"  Batch time: {batch_time:.2f} ms\n")

print(f"✅ Results saved to: {result_file}")