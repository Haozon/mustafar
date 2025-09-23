# LLaMA model with KIVI
import torch
import os
from transformers import LlamaConfig, AutoTokenizer
import time

'''
一些信息：
当 sequence_length 为2048 ，bs 为32时，测试 dense 模型 KV cache 无法放下因此有如下可用信息：
1. 强制使用前两张GPU卡,因为只测1张卡 sequence length 太长放不下；
2. 注释了 .cuda(), 因为该操作会强制使用一张卡。
由于 mustafar_kernel 只能使用 一张卡，否则会出现RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!， 
'''
K_SPARSITY = 0.7
V_SPARSITY = 0.7
GROUP_SIZE = 32
BATCH_SIZE = 8

MUSTAFAR_MODE = True
# MUSTAFAR_MODE = False

PROMPT_LENGTH = 4096
OUTPUT_LENGTH = 1024
NUM_REPEATS = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name_or_path = '/home/zh/model/Meta-Llama-3-8B-Instruct'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_sparsity = K_SPARSITY
config.v_sparsity = V_SPARSITY
config.group_size = GROUP_SIZE
config.residual_length = GROUP_SIZE

if MUSTAFAR_MODE:
    # 设置导入路径
    from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR
    print("@@@@@@@@@@@Using Mustafar")
    config.use_flash = True
    model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()
else:
    from transformers import LlamaForCausalLM
    print("@@@@@@@@@@@Using Naive")
    config.use_flash = True
    config.use_cache = True
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
    trust_remote_code=True,
)

model.eval()

def benchmark_with_token_timing(model, inputs, model_name, max_tokens=600, num_repeats=3):
    """
    测量TTFT和TPOT的函数
    返回: (ttft_ms, tpot_ms, total_time_ms, peak_memory_gb)
    """
    print(f"\n=== Token-level timing for {model_name} ===")
    
    # Warmup
    print("Running token-level warmup...")
    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
        torch.cuda.synchronize()
    
    all_ttft_times = []
    all_tpot_times = []
    
    for repeat in range(num_repeats):
        print(f"Running repeat {repeat + 1}/{num_repeats}...")
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
        
            print(f"  Repeat {repeat + 1}: TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms")
    
    # Calculate averages
    avg_ttft = sum(all_ttft_times) / len(all_ttft_times) if all_ttft_times else 0
    avg_tpot = sum(all_tpot_times) / len(all_tpot_times) if all_tpot_times else 0
    total_avg_time = avg_ttft + avg_tpot * (max_tokens - 1)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    print(f"Average TTFT: {avg_ttft:.2f} ms")
    print(f"Average TPOT: {avg_tpot:.2f} ms")
    print(f"Total generation time: {total_avg_time:.2f} ms")
    print(f"Peak memory: {peak_memory:.2f} GB")
    
    return avg_ttft, avg_tpot, total_avg_time, peak_memory

# Main execution
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

context = []
batch_size = BATCH_SIZE
prompt_lenth = PROMPT_LENGTH
output_length = OUTPUT_LENGTH
num_repeats = NUM_REPEATS

for _ in range(batch_size):
    string = 'apple bear' * (prompt_lenth // 2)
    context.append(string[:-1])
    
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']

print("Running warmup iteration...")
with torch.no_grad():
    torch.cuda.synchronize()
    outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
    torch.cuda.synchronize()
    
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")

# Original batch generation test
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
    torch.cuda.synchronize()
    batch_time = (time.time() - st) / num_repeats * 1000
    used_mem = torch.cuda.max_memory_allocated()
    print(f'\n=== Original Batch Generation Results ===')
    print(f'Batch generation time: {batch_time:.2f} ms')
    print(f'Peak mem: {used_mem / 1024 ** 3:.2f} GB')

# Token-level timing test
model_name = "Mustafar" if MUSTAFAR_MODE else "Naive"
ttft, tpot, total_time, peak_mem = benchmark_with_token_timing(
    model, inputs, model_name, max_tokens=output_length, num_repeats=num_repeats
)

print(f'\n=== Final Results Summary ===')
print(f'Model: {model_name}')
print(f'TTFT: {ttft:.2f} ms')
print(f'TPOT: {tpot:.2f} ms')
print(f'Total generation time: {total_time:.2f} ms')
print(f'Peak memory: {peak_mem:.2f} GB')
print(f'Batch size: {batch_size}, Input length: {input_ids.shape[1]}, Output length: {output_length}')