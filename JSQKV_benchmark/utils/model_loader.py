"""模型加载器"""
import torch
import sys
import os
from transformers import LlamaConfig, AutoTokenizer

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_model(model_config, test_config):
    """
    根据配置加载相应的模型
    
    Args:
        model_config: 模型配置（路径、序列长度等）
        test_config: 测试配置（稀疏度、量化等）
    
    Returns:
        model: 加载的模型
        tokenizer: 对应的tokenizer
    """
    model_path = model_config['path']
    
    # 创建配置
    config = LlamaConfig.from_pretrained(model_path)
    config.k_sparsity = test_config['k_sparsity']
    config.v_sparsity = test_config['v_sparsity']
    config.use_flash = True
    config.use_cache = True
    
    # 根据测试配置选择模型类型
    if test_config['use_quant']:
        # 量化稀疏模型
        from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT
        
        config.quant_bits = test_config.get('quant_bits', 2)
        config.group_size = test_config.get('group_size', 32)
        config.residual_length = test_config.get('residual_length', 256)
        config.quant_k_dequant_mode = test_config.get('quant_k_dequant_mode', 0)
        config.quant_v_dequant_mode = test_config.get('quant_v_dequant_mode', 0)
        config.quant_v_split_k = test_config.get('quant_v_split_k', 4)
        config.quant_v_tile_config = test_config.get('quant_v_tile_config', 0)
        
        print(f"  Loading Mustafar Quantized Model:")
        print(f"    - K Sparsity: {config.k_sparsity*100}%")
        print(f"    - V Sparsity: {config.v_sparsity*100}%")
        print(f"    - Quantization: {config.quant_bits}-bit")
        print(f"    - K Dequant Mode: {config.quant_k_dequant_mode}")
        print(f"    - V Dequant Mode: {config.quant_v_dequant_mode}")
        print(f"    - V Split-K: {config.quant_v_split_k}")
        print(f"    - V Tile Config: {config.quant_v_tile_config}")
        
        model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).cuda()
        
    elif test_config['k_sparsity'] < 1.0 or test_config['v_sparsity'] < 1.0:
        # 稀疏模型（无量化）
        from models.llama_mustafar_kernel import LlamaForCausalLM_MUSTAFAR
        
        config.group_size = test_config.get('group_size', 32)
        config.residual_length = test_config.get('residual_length', 256)
        
        print(f"  Loading Mustafar Sparse Model:")
        print(f"    - K Sparsity: {config.k_sparsity*100}%")
        print(f"    - V Sparsity: {config.v_sparsity*100}%")
        
        model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).cuda()
        
    else:
        # Dense模型
        from transformers import LlamaForCausalLM
        
        print(f"  Loading Dense Model (baseline)")
        
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).cuda()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    
    model.eval()
    
    return model, tokenizer
