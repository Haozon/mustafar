# LLaMA model with Mustafar Quantized Sparse Kernel
import torch
import os
import sys
from transformers import LlamaConfig, AutoTokenizer
import time
import gc

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from benchmark.low_memory_generation import forward_last_logits, greedy_generate

'''
量化稀疏 KV Cache 性能测评
- 稀疏度: 50% (K_SPARSITY = 0.7
- 量化位宽: 2-bit
- 测试指标: TTFT, TPOT, 内存占用
'''

# ==================== 配置参数 ====================
K_SPARSITY = 0.7
V_SPARSITY = 0.7
QUANT_BITS = 2        # 2-bit 量化
QUANT_K_DEQUANT_MODE = 0
QUANT_K_USE_META = False
QUANT_V_DEQUANT_MODE = 0
QUANT_V_SPLIT_K = 4
QUANT_V_TILE_CONFIG = 0
QUANT_V_DECODE_N1 = False
GROUP_SIZE = 32
RESIDUAL_LENGTH = 256
BATCH_SIZE = 8

# 测试模式选择
QUANT_MODE = True     # True: 使用量化稀疏, False: 使用标准 Mustafar
# QUANT_MODE = True

PROMPT_LENGTH = 4096
OUTPUT_LENGTH = 1024
NUM_REPEATS = 3
TOKEN_TIMING_STEPS = 64

def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value != "" else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value != "" else default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


K_SPARSITY = _get_env_float("K_SPARSITY", K_SPARSITY)
V_SPARSITY = _get_env_float("V_SPARSITY", V_SPARSITY)
QUANT_BITS = _get_env_int("QUANT_BITS", QUANT_BITS)
QUANT_K_DEQUANT_MODE = _get_env_int("QUANT_K_DEQUANT_MODE", QUANT_K_DEQUANT_MODE)
QUANT_K_USE_META = _get_env_bool("QUANT_K_USE_META", QUANT_K_USE_META)
QUANT_V_DEQUANT_MODE = _get_env_int("QUANT_V_DEQUANT_MODE", QUANT_V_DEQUANT_MODE)
QUANT_V_SPLIT_K = _get_env_int("QUANT_V_SPLIT_K", QUANT_V_SPLIT_K)
QUANT_V_TILE_CONFIG = _get_env_int("QUANT_V_TILE_CONFIG", QUANT_V_TILE_CONFIG)
QUANT_V_DECODE_N1 = _get_env_bool("QUANT_V_DECODE_N1", QUANT_V_DECODE_N1)
GROUP_SIZE = _get_env_int("GROUP_SIZE", GROUP_SIZE)
RESIDUAL_LENGTH = _get_env_int("RESIDUAL_LENGTH", RESIDUAL_LENGTH)
BATCH_SIZE = _get_env_int("BATCH_SIZE", BATCH_SIZE)
PROMPT_LENGTH = _get_env_int("PROMPT_LENGTH", PROMPT_LENGTH)
OUTPUT_LENGTH = _get_env_int("OUTPUT_LENGTH", OUTPUT_LENGTH)
NUM_REPEATS = _get_env_int("NUM_REPEATS", NUM_REPEATS)
TOKEN_TIMING_STEPS = _get_env_int("TOKEN_TIMING_STEPS", TOKEN_TIMING_STEPS)
QUANT_MODE = _get_env_bool("QUANT_MODE", QUANT_MODE)

auto_tuned_quant_v = False
if os.getenv("QUANT_V_SPLIT_K") is None and os.getenv("QUANT_V_TILE_CONFIG") is None:
    if PROMPT_LENGTH >= 4096 and V_SPARSITY >= 0.65 and BATCH_SIZE >= 3:
        QUANT_V_SPLIT_K = 8
        QUANT_V_TILE_CONFIG = 1
        auto_tuned_quant_v = True
    elif PROMPT_LENGTH >= 4096 and V_SPARSITY <= 0.55 and 3 <= BATCH_SIZE <= 6:
        QUANT_V_SPLIT_K = 8
        QUANT_V_TILE_CONFIG = 1
        auto_tuned_quant_v = True

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ==================== 模型配置 ====================
model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", '/home/zh/model/Meta-Llama-3-8B-Instruct')
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
    print(f"   - K Dequant Mode: {QUANT_K_DEQUANT_MODE}")
    print(f"   - K Use Meta: {QUANT_K_USE_META}")
    print(f"   - V Dequant Mode: {QUANT_V_DEQUANT_MODE}")
    print(f"   - V Split-K: {QUANT_V_SPLIT_K}")
    print(f"   - V Tile Config: {QUANT_V_TILE_CONFIG}")
    print(f"   - V Decode N1: {QUANT_V_DECODE_N1}")
    if auto_tuned_quant_v:
        print("   - V Tuning: auto-selected for long-context decode")
    print(f"   - Expected Memory Compression: ~{16/(2**(QUANT_BITS-1))}x")
    print("="*60)
    
    config.quant_bits = QUANT_BITS
    config.quant_k_dequant_mode = QUANT_K_DEQUANT_MODE
    config.quant_k_use_meta = QUANT_K_USE_META
    config.quant_v_dequant_mode = QUANT_V_DEQUANT_MODE
    config.quant_v_split_k = QUANT_V_SPLIT_K
    config.quant_v_tile_config = QUANT_V_TILE_CONFIG
    config.quant_v_decode_n1 = QUANT_V_DECODE_N1
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
        warmup_outputs = greedy_generate(model, inputs, max_new_tokens=min(10, max_tokens))
        torch.cuda.synchronize()
    del warmup_outputs
    gc.collect()
    torch.cuda.empty_cache()
    
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
                    current_input_ids = input_ids
                else:
                    current_input_ids = input_ids[:, -1:]

                logits, past_key_values = forward_last_logits(
                    model,
                    current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
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
    outputs = greedy_generate(model, inputs, max_new_tokens=output_length)
    torch.cuda.synchronize()
del outputs
gc.collect()
torch.cuda.empty_cache()

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
        outputs = greedy_generate(model, inputs, max_new_tokens=output_length)
    torch.cuda.synchronize()
    batch_time = (time.time() - st) / num_repeats * 1000
    used_mem = torch.cuda.max_memory_allocated()
    
    print(f"\n📊 Batch Generation Results:")
    print(f"   Average time: {batch_time:.2f} ms")
    print(f"   Peak memory: {used_mem / 1024 ** 3:.2f} GB")
    print(f"   Throughput: {batch_size * output_length / (batch_time / 1000):.2f} tokens/sec")
del outputs
gc.collect()
torch.cuda.empty_cache()

# ==================== Token-level Timing Test ====================
model_name = f"Mustafar-Quant-{QUANT_BITS}bit" if QUANT_MODE else f"Mustafar-Sparse"
ttft, tpot, total_time, peak_mem = benchmark_with_token_timing(
    model, inputs, model_name, max_tokens=min(output_length, TOKEN_TIMING_STEPS), num_repeats=num_repeats
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
