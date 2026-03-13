#!/usr/bin/env python3
"""
测试不同量化位数的性能
验证是否是 2-bit 位操作导致的性能问题
"""
import torch
import os
import sys
import time
from transformers import AutoTokenizer, LlamaConfig

# 添加项目根目录到路径（向上两级到 mustafar/）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = '/home/zh/model/Meta-Llama-3-8B-Instruct'
BATCH_SIZE = 8
INPUT_LENGTH = 4096
OUTPUT_LENGTH = 100  # 快速测试

def test_config(quant_bits, use_quant):
    """测试单个配置"""
    print(f"\n{'='*70}")
    if use_quant:
        print(f"🔥 Testing: Sparse-50% + Quant-{quant_bits}bit")
    else:
        print(f"🔥 Testing: Sparse-50% (FP16)")
    print(f"{'='*70}")
    
    # 加载模型
    config = LlamaConfig.from_pretrained(MODEL_PATH)
    config.k_sparsity = 0.5
    config.v_sparsity = 0.5
    config.group_size = 32
    config.residual_length = 256
    config.use_flash = True
    
    if use_quant:
        config.quant_bits = quant_bits
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
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备输入
    context = []
    for _ in range(BATCH_SIZE):
        string = 'apple bear' * (INPUT_LENGTH // 2)
        context.append(string[:-1])
    
    inputs = tokenizer(context, return_tensors="pt").to('cuda')
    
    # Warmup
    print("⏳ Warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
    torch.cuda.synchronize()
    
    # 测试
    print("🔥 Running test...")
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=OUTPUT_LENGTH, eos_token_id=None)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = sum(times) / len(times)
    total_tokens = BATCH_SIZE * OUTPUT_LENGTH
    throughput = total_tokens / avg_time
    tpot = avg_time * 1000 / OUTPUT_LENGTH
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    result = {
        'config': f"Quant-{quant_bits}bit" if use_quant else "FP16",
        'avg_time': avg_time,
        'throughput': throughput,
        'tpot': tpot,
        'peak_memory': peak_memory
    }
    
    print(f"\n📊 Results:")
    print(f"   Throughput: {throughput:.2f} tokens/sec")
    print(f"   TPOT: {tpot:.2f} ms")
    print(f"   Peak memory: {peak_memory:.2f} GB")
    
    # 清理
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return result

def main():
    print("="*70)
    print("🚀 量化位数性能对比测试")
    print("="*70)
    print(f"\n测试配置:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Input length: {INPUT_LENGTH}")
    print(f"   Output length: {OUTPUT_LENGTH}")
    
    results = []
    
    # 测试 FP16 (无量化)
    try:
        result = test_config(quant_bits=None, use_quant=False)
        results.append(result)
    except Exception as e:
        print(f"❌ FP16 test failed: {e}")
    
    time.sleep(5)  # 等待 GPU 冷却
    
    # 测试 2-bit 量化
    try:
        result = test_config(quant_bits=2, use_quant=True)
        results.append(result)
    except Exception as e:
        print(f"❌ 2-bit test failed: {e}")
    
    # 打印对比
    if len(results) >= 2:
        print("\n" + "="*70)
        print("📊 Performance Comparison")
        print("="*70)
        
        print(f"\n| Configuration | Throughput | TPOT | Memory |")
        print(f"|---------------|------------|------|--------|")
        for r in results:
            print(f"| {r['config']:13s} | {r['throughput']:8.2f} tok/s | {r['tpot']:6.2f} ms | {r['peak_memory']:6.2f} GB |")
        
        if len(results) == 2:
            baseline = results[0]
            quant = results[1]
            
            throughput_ratio = quant['throughput'] / baseline['throughput']
            tpot_ratio = quant['tpot'] / baseline['tpot']
            memory_ratio = quant['peak_memory'] / baseline['peak_memory']
            
            print(f"\n| Metric | Quant vs FP16 |")
            print(f"|--------|---------------|")
            print(f"| Throughput | {throughput_ratio:.2f}x {'↑' if throughput_ratio > 1 else '↓'} |")
            print(f"| TPOT | {tpot_ratio:.2f}x {'↑' if tpot_ratio < 1 else '↓'} |")
            print(f"| Memory | {memory_ratio:.2f}x {'↑' if memory_ratio < 1 else '↓'} |")
            
            print(f"\n💡 Analysis:")
            if throughput_ratio < 0.8:
                print(f"   ⚠️  Quantization causes {(1-throughput_ratio)*100:.1f}% performance loss")
                print(f"   🔍 Main bottleneck is likely in:")
                print(f"      - Dequantization in CUDA kernel")
                print(f"      - Bit unpacking operations (2-bit)")
                print(f"      - Memory access patterns")
            elif throughput_ratio < 1.0:
                print(f"   ⚠️  Quantization causes {(1-throughput_ratio)*100:.1f}% performance loss")
                print(f"   🔍 Minor overhead, acceptable for memory savings")
            else:
                print(f"   ✅ Quantization improves performance!")
    
    print("\n" + "="*70)
    print("✅ Test completed!")
    print("="*70)

if __name__ == '__main__':
    main()
