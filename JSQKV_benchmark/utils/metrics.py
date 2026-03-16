"""性能指标计算"""
import torch
import time

def build_fixed_length_inputs(tokenizer, batch_size, input_length, device="cuda"):
    """
    构造严格等长的输入，避免 tokenizer 后 token 数超出预期。
    """
    seed_text = ("apple bear " * max(input_length, 16)).strip()
    encoded = tokenizer(seed_text, return_tensors="pt")
    input_ids = encoded["input_ids"][:, :input_length].contiguous()
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[:, :input_length].contiguous()

    input_ids = input_ids.repeat(batch_size, 1).to(device)
    attention_mask = attention_mask.repeat(batch_size, 1).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def measure_throughput(
    model,
    tokenizer,
    batch_size,
    input_length,
    output_length,
    num_repeats=3,
    warmup_tokens=10,
    measure_token_metrics=True,
):
    """
    测量模型吞吐量
    
    Args:
        model: 待测试的模型
        tokenizer: tokenizer
        batch_size: 批次大小
        input_length: 输入序列长度
        output_length: 输出序列长度
        num_repeats: 重复测试次数
        warmup_tokens: 预热token数量
    
    Returns:
        dict: 包含吞吐量、TTFT、TPOT、内存等指标
    """
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 构造严格等长输入
    inputs = build_fixed_length_inputs(tokenizer, batch_size, input_length, device='cuda')
    input_ids = inputs['input_ids']
    
    # Warmup
    print(f"    Running warmup...")
    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model.generate(**inputs, max_new_tokens=warmup_tokens, eos_token_id=None)
        torch.cuda.synchronize()
    
    # 测量批次生成时间
    torch.cuda.reset_peak_memory_stats()
    batch_times = []
    
    with torch.no_grad():
        for i in range(num_repeats):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model.generate(**inputs, max_new_tokens=output_length, eos_token_id=None)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            batch_times.append(elapsed)
            print(f"    Repeat {i+1}/{num_repeats}: {elapsed:.3f}s")
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    # 计算吞吐量 (tokens/second)
    total_tokens = batch_size * output_length
    throughput = total_tokens / avg_batch_time
    
    # 可选：测量 TTFT 和 TPOT。大规模 batch-size sweep 默认可关闭以节省时间。
    if measure_token_metrics:
        ttft, tpot = measure_token_timing(model, inputs, output_length, num_repeats=1)
    else:
        ttft, tpot = None, None
    
    results = {
        'throughput': throughput,  # tokens/second
        'avg_batch_time': avg_batch_time,  # seconds
        'ttft': ttft,  # ms
        'tpot': tpot,  # ms
        'peak_memory': peak_memory,  # GB
        'batch_size': batch_size,
        'input_length': input_ids.shape[1],
        'output_length': output_length,
        'total_tokens': total_tokens
    }
    
    print(f"    ✅ Throughput: {throughput:.2f} tokens/sec")
    print(f"    ✅ Peak Memory: {peak_memory:.2f} GB")
    if ttft is not None and tpot is not None:
        print(f"    ✅ TTFT: {ttft:.2f} ms, TPOT: {tpot:.2f} ms")
    
    return results

def measure_token_timing(model, inputs, max_tokens, num_repeats=1):
    """
    测量 TTFT 和 TPOT
    
    Returns:
        ttft: Time to First Token (ms)
        tpot: Time per Output Token (ms)
    """
    all_ttft_times = []
    all_tpot_times = []
    
    for repeat in range(num_repeats):
        token_times = []
        
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
                    current_input = {'input_ids': input_ids}
                    if attention_mask is not None:
                        current_input['attention_mask'] = attention_mask
                else:
                    current_input = {'input_ids': input_ids[:, -1:]}
                
                current_input['past_key_values'] = past_key_values
                current_input['use_cache'] = True
                
                outputs = model(**current_input)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                
                torch.cuda.synchronize()
                token_time = (time.time() - token_start) * 1000  # ms
                token_times.append(token_time)
                
                # Update for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                past_key_values = outputs.past_key_values
                
                # 只测量前几个token即可
                if token_idx >= 50:
                    break
        
        if len(token_times) > 0:
            ttft = token_times[0]
            tpot = sum(token_times[1:]) / len(token_times[1:]) if len(token_times) > 1 else token_times[0]
            all_ttft_times.append(ttft)
            all_tpot_times.append(tpot)
    
    avg_ttft = sum(all_ttft_times) / len(all_ttft_times) if all_ttft_times else 0
    avg_tpot = sum(all_tpot_times) / len(all_tpot_times) if all_tpot_times else 0
    
    return avg_ttft, avg_tpot
