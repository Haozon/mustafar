"""模型加载器"""
import os
import sys

import torch
from transformers import AutoTokenizer, LlamaConfig

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def resolve_quant_runtime_tuning(model_config, test_config, batch_size):
    """
    为量化路径选择运行时参数。

    说明:
    - benchmark/ 下的 mem_spd_test_quant.py 已有一版按 BS 自动调 split_k/tile 的逻辑。
    - JSQKV benchmark 之前一直固定使用 YAML 中的量化参数，导致小中 batch 场景下的
      量化版本没有吃到这部分 tuning。
    - 这里把调优逻辑前移到 runner，允许按 batch size 动态切换，不需要为每个 batch 重载模型。
    """
    split_k = test_config.get("quant_v_split_k", 4)
    tile_config = test_config.get("quant_v_tile_config", 0)
    decode_n1 = test_config.get("quant_v_decode_n1", False)
    tuned = False
    reason = "yaml-default"

    if not test_config.get("use_quant", False):
        return {
            "quant_v_split_k": split_k,
            "quant_v_tile_config": tile_config,
            "quant_v_decode_n1": decode_n1,
            "auto_tuned": False,
            "reason": reason,
        }

    if not test_config.get("auto_quant_v_tune", False):
        tuning = {
            "quant_v_split_k": split_k,
            "quant_v_tile_config": tile_config,
            "quant_v_decode_n1": decode_n1,
            "auto_tuned": False,
            "reason": reason,
        }
        overrides = test_config.get("quant_runtime_overrides", {})
        override = overrides.get(batch_size, overrides.get(str(batch_size)))
        if override:
            tuning["quant_v_split_k"] = override.get("quant_v_split_k", tuning["quant_v_split_k"])
            tuning["quant_v_tile_config"] = override.get("quant_v_tile_config", tuning["quant_v_tile_config"])
            tuning["quant_v_decode_n1"] = override.get("quant_v_decode_n1", tuning["quant_v_decode_n1"])
            tuning["auto_tuned"] = True
            tuning["reason"] = f"explicit-override@bs={batch_size}"
        return tuning

    prompt_length = model_config.get("input_length", 0)
    v_sparsity = test_config.get("v_sparsity", 1.0)
    profile = test_config.get("auto_quant_v_tune_profile", "long_context_small_bs")

    if profile == "long_context_small_bs" and prompt_length >= 4096:
        # 复用 mem_spd_test_quant.py 中验证过的长上下文小中 batch 经验:
        # - Sparse70: BS>=3 时优先 split_k=8 + tile64
        # - Sparse50: BS=3~6 时优先 split_k=8 + tile64
        if v_sparsity >= 0.65 and batch_size >= 3:
            split_k = 8
            tile_config = 1
            tuned = True
            reason = "profile=long_context_small_bs,v>=0.65,bs>=3"
        elif v_sparsity <= 0.55 and 3 <= batch_size <= 6:
            split_k = 8
            tile_config = 1
            tuned = True
            reason = "profile=long_context_small_bs,v<=0.55,3<=bs<=6"

    tuning = {
        "quant_v_split_k": split_k,
        "quant_v_tile_config": tile_config,
        "quant_v_decode_n1": decode_n1,
        "auto_tuned": tuned,
        "reason": reason,
    }
    overrides = test_config.get("quant_runtime_overrides", {})
    override = overrides.get(batch_size, overrides.get(str(batch_size)))
    if override:
        tuning["quant_v_split_k"] = override.get("quant_v_split_k", tuning["quant_v_split_k"])
        tuning["quant_v_tile_config"] = override.get("quant_v_tile_config", tuning["quant_v_tile_config"])
        tuning["quant_v_decode_n1"] = override.get("quant_v_decode_n1", tuning["quant_v_decode_n1"])
        tuning["auto_tuned"] = True
        tuning["reason"] = f"explicit-override@bs={batch_size}"

    return tuning


def apply_quant_runtime_tuning(model, model_config, test_config, batch_size):
    """
    将量化参数动态写回所有 attention 模块。

    量化 kernel 在 forward 中会读取模块上的 `self.quant_v_*` 属性，因此无需重载模型。
    """
    tuning = resolve_quant_runtime_tuning(model_config, test_config, batch_size)
    if not test_config.get("use_quant", False):
        return tuning

    if hasattr(model, "config"):
        model.config.quant_v_split_k = tuning["quant_v_split_k"]
        model.config.quant_v_tile_config = tuning["quant_v_tile_config"]
        model.config.quant_v_decode_n1 = tuning["quant_v_decode_n1"]

    updated = 0
    for module in model.modules():
        if hasattr(module, "quant_v_split_k"):
            module.quant_v_split_k = tuning["quant_v_split_k"]
            module.quant_v_tile_config = tuning["quant_v_tile_config"]
            module.quant_v_decode_n1 = tuning["quant_v_decode_n1"]
            updated += 1

    tuning["updated_modules"] = updated
    return tuning

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
        config.quant_k_use_meta = test_config.get('quant_k_use_meta', False)
        config.quant_v_dequant_mode = test_config.get('quant_v_dequant_mode', 0)
        config.quant_v_split_k = test_config.get('quant_v_split_k', 4)
        config.quant_v_tile_config = test_config.get('quant_v_tile_config', 0)
        config.quant_v_decode_n1 = test_config.get('quant_v_decode_n1', False)

        print(f"  Loading Mustafar Quantized Model:")
        print(f"    - K Sparsity: {config.k_sparsity*100}%")
        print(f"    - V Sparsity: {config.v_sparsity*100}%")
        print(f"    - Quantization: {config.quant_bits}-bit")
        print(f"    - K Dequant Mode: {config.quant_k_dequant_mode}")
        print(f"    - K Use Meta: {config.quant_k_use_meta}")
        print(f"    - V Dequant Mode: {config.quant_v_dequant_mode}")
        print(f"    - V Split-K: {config.quant_v_split_k}")
        print(f"    - V Tile Config: {config.quant_v_tile_config}")
        print(f"    - V Decode N1: {config.quant_v_decode_n1}")
        if test_config.get("auto_quant_v_tune", False):
            print(
                f"    - Runtime V Tuning: enabled ({test_config.get('auto_quant_v_tune_profile', 'long_context_small_bs')})"
            )
        
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
