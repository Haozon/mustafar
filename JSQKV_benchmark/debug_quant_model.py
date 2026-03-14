#!/usr/bin/env python3
"""
调试量化模型 - 使用 CUDA_LAUNCH_BLOCKING 获取详细错误信息
"""
import torch
import os
from transformers import LlamaConfig, AutoTokenizer

# 启用同步模式以获取准确的错误位置
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("="*70)
print("🔍 调试量化模型")
print("="*70)

# 配置
K_SPARSITY = 0.5
V_SPARSITY = 0.5
QUANT_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 256

model_name_or_path = '/home/zh/model/Meta-Llama-3-8B-Instruct'

print(f"\n加载配置...")
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_sparsity = K_SPARSITY
config.v_sparsity = V_SPARSITY
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
config.use_flash = True
config.quant_bits = QUANT_BITS

print(f"配置:")
print(f"  K Sparsity: {K_SPARSITY*100}%")
print(f"  V Sparsity: {V_SPARSITY*100}%")
print(f"  Quantization: {QUANT_BITS}-bit")

try:
    print(f"\n加载模型...")
    from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT
    
    model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()
    
    print(f"✅ 模型加载成功!")
    
    print(f"\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    print(f"✅ Tokenizer 加载成功!")
    
    model.eval()
    
    # 使用较长的输入以触发压缩
    print(f"\n准备输入...")
    test_text = "Hello, how are you? " * 100  # 更长的输入
    inputs = tokenizer(test_text, return_tensors="pt").to('cuda')
    
    print(f"输入:")
    print(f"  Input IDs shape: {inputs['input_ids'].shape}")
    print(f"  Input length: {inputs['input_ids'].shape[1]} tokens")
    
    # 逐步测试
    print(f"\n步骤 1: 测试 forward pass...")
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            print(f"✅ Forward pass 成功!")
            print(f"  Logits shape: {outputs.logits.shape}")
            
            # 检查 logits 是否有异常值
            if torch.isnan(outputs.logits).any():
                print(f"⚠️  Logits 包含 NaN!")
            if torch.isinf(outputs.logits).any():
                print(f"⚠️  Logits 包含 Inf!")
            
            print(f"  Logits min: {outputs.logits.min().item():.4f}")
            print(f"  Logits max: {outputs.logits.max().item():.4f}")
            print(f"  Logits mean: {outputs.logits.mean().item():.4f}")
            
        except Exception as e:
            print(f"❌ Forward pass 失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"\n步骤 2: 测试生成 (1 token)...")
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, max_new_tokens=1, eos_token_id=None)
            print(f"✅ 生成 1 token 成功!")
            print(f"  Output shape: {outputs.shape}")
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"\n步骤 3: 测试生成 (10 tokens)...")
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
            print(f"✅ 生成 10 tokens 成功!")
            print(f"  Output shape: {outputs.shape}")
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Generated (first 200 chars): {generated_text[:200]}...")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"\n{'='*70}")
    print(f"🎉 所有测试通过!")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"\n{'='*70}")
    print(f"❌ 测试失败: {e}")
    print(f"{'='*70}")
    import traceback
    traceback.print_exc()
