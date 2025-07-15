# LLaMA model performance comparison with NVTX profiling
import torch
import os
from transformers import LlamaConfig, AutoTokenizer
import time

'''
性能对比测试：
对比 MUSTAFAR 和 Naive 模式的 generate 性能
添加 NVTX 标签用于 nsys 分析
'''
K_SPARSITY = 0.7
V_SPARSITY = 0.7
GROUP_SIZE = 32
BATCH_SIZE = 8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name_or_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b'

def create_model(use_mustafar=True):
    """创建模型"""
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.k_sparsity = K_SPARSITY
    config.v_sparsity = V_SPARSITY
    config.group_size = GROUP_SIZE
    config.residual_length = GROUP_SIZE
    config.use_flash = True
    
    if use_mustafar:
        from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR
        print("Creating Mustafar model...")
        model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).cuda()
    else:
        from transformers import LlamaForCausalLM
        print("Creating Naive model...")
        config.use_cache = True
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).cuda()
    
    return model

def benchmark_model(model, inputs, model_name, output_length=600, num_repeats=3):
    """性能测试函数"""
    print(f"\n=== Testing {model_name} ===")
    
    # Warmup
    torch.cuda.nvtx.range_push(f"{model_name}_warmup")
    print("Running warmup iteration...")
    with torch.no_grad():
        torch.cuda.synchronize()
        outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # Benchmark
    torch.cuda.nvtx.range_push(f"{model_name}_benchmark")
    with torch.no_grad():
        torch.cuda.synchronize()
        st = time.time()
        
        for i in range(num_repeats):
            torch.cuda.nvtx.range_push(f"{model_name}_generate_iter_{i}")
            outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
            torch.cuda.nvtx.range_pop()
        
        torch.cuda.synchronize()
        avg_time = (time.time() - st) / num_repeats * 1000
    torch.cuda.nvtx.range_pop()
    
    print(f'{model_name} - Average time: {avg_time:.2f} ms')
    
    return avg_time

def main():
    # 准备数据
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    batch_size = BATCH_SIZE
    prompt_length = 300
    output_length = 600
    num_repeats = 3
    
    # 创建输入数据
    context = []
    for _ in range(batch_size):
        string = 'apple bear' * (prompt_length // 2)
        context.append(string[:-1])
    inputs = tokenizer(context, return_tensors="pt").to('cuda')
    input_ids = inputs['input_ids']
    
    print(f"Batch size: {batch_size}, Input length: {input_ids.shape[1]}, Output length: {output_length}")
    print(f"Model: {model_name_or_path}")
    
    results = {}
    
    # 测试 Mustafar 模型
    torch.cuda.nvtx.range_push("mustafar_model_creation")
    mustafar_model = create_model(use_mustafar=True)
    torch.cuda.nvtx.range_pop()
    mustafar_model.eval()
    
    mustafar_time = benchmark_model(
        mustafar_model, inputs, "Mustafar", output_length, num_repeats
    )
    results['mustafar'] = mustafar_time
    

    
    # # 测试 Naive 模型
    # torch.cuda.nvtx.range_push("naive_model_creation")
    # naive_model = create_model(use_mustafar=False)
    # torch.cuda.nvtx.range_pop()
    # naive_model.eval()
    
    # naive_time = benchmark_model(
    #     naive_model, inputs, "Naive", output_length, num_repeats
    # )
    # results['naive'] = naive_time
    

if __name__ == "__main__":
    main()