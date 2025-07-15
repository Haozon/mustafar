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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name_or_path = 'meta-llama/Llama-2-7b-hf'
# model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name_or_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b'
# model_name_or_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/Meta-Llama-3-8B'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_sparsity = K_SPARSITY
config.v_sparsity = V_SPARSITY
config.group_size = GROUP_SIZE
config.residual_length = GROUP_SIZE



if MUSTAFAR_MODE:
    from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR

    print("@@@@@@@@@@@Using Mustafar")
    config.use_flash = True
    model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        #cache_dir=cache_dir,
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
        #cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
    trust_remote_code=True,
    # cache_dir=cache_dir,
    )


model.eval()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

context = []
batch_size = BATCH_SIZE
# prompt_lenth = 4096
# output_length = 4096
# prompt_lenth = 2048
# output_length = 2048
prompt_lenth = 300
output_length = 600
num_repeats = 3
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
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
    torch.cuda.synchronize()
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')
    # total_mem = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
    # print(f'total peak mem: {total_mem / 1024 ** 3:.2f} GB')

