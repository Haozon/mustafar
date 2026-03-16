#!/usr/bin/env python3
"""
DiffSparseKV LongBench Evaluation Script

This script performs full LongBench evaluation with DiffSparseKV,
following the pattern of eval_greedy_sparsity.py
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DiffSparseKV components (for future integration)
# from DiffSparseKV.config_models import DiffSparseKVConfig, create_default_config
# from DiffSparseKV.importance_calculator import DiffKVImportanceCalculator
# from DiffSparseKV.threshold_manager import GlobalThresholdManager
# from DiffSparseKV.sparsity_applier import SparsityClassifierApplier


def build_chat(tokenizer, prompt, model_name):
    """构建聊天格式的prompt"""
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response, model_name):
    """后处理响应"""
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


class KVCacheSparsityWrapper:
    """KV Cache稀疏度应用包装器"""
    
    def __init__(self, sparsity_level: float = 0.0, sparsity_type: str = 'uniform'):
        """
        初始化KV Cache稀疏度包装器
        
        Args:
            sparsity_level: 稀疏度级别 (0.0-1.0)
            sparsity_type: 稀疏度类型 ('uniform' 或 'diff_sparse_kv')
        """
        self.sparsity_level = sparsity_level
        self.sparsity_type = sparsity_type
        self.enabled = sparsity_level > 0.0
        self.stats = {
            'total_tokens': 0,
            'pruned_tokens': 0,
            'compression_ratio': 0.0
        }
    
    def apply_sparsity_to_kv(self, key_cache, value_cache):
        """
        对KV Cache应用稀疏度
        
        Args:
            key_cache: 键缓存 (list of tensors for each layer)
            value_cache: 值缓存 (list of tensors for each layer)
            
        Returns:
            稀疏化后的 (key_cache, value_cache)
        """
        if not self.enabled or self.sparsity_level == 0.0:
            return key_cache, value_cache
        
        sparse_key_cache = []
        sparse_value_cache = []
        
        for layer_idx, (keys, values) in enumerate(zip(key_cache, value_cache)):
            # keys/values shape: [batch, num_heads, seq_len, head_dim]
            if keys is None or values is None:
                sparse_key_cache.append(keys)
                sparse_value_cache.append(values)
                continue
            
            B, H, T, D = keys.shape
            
            # 计算要保留的token数量
            num_keep = max(1, int(T * (1 - self.sparsity_level)))
            
            if self.sparsity_type == 'uniform':
                # 统一稀疏度：保留最后的token和一些重要的token
                # 策略：保留最后 num_keep 个token
                sparse_keys = keys[:, :, -num_keep:, :]
                sparse_values = values[:, :, -num_keep:, :]
            
            elif self.sparsity_type == 'diff_sparse_kv':
                # 差分稀疏度：基于token重要性
                # 简单实现：保留最后的token和基于norm的重要token
                key_norms = torch.norm(keys, dim=-1)  # [B, H, T]
                
                # 获取最重要的token索引
                _, top_indices = torch.topk(key_norms.view(B, H, -1), num_keep, dim=-1)
                
                # 创建稀疏张量
                sparse_keys = torch.zeros_like(keys)
                sparse_values = torch.zeros_like(values)
                
                for b in range(B):
                    for h in range(H):
                        indices = top_indices[b, h]
                        sparse_keys[b, h, indices] = keys[b, h, indices]
                        sparse_values[b, h, indices] = values[b, h, indices]
            
            else:
                sparse_keys = keys
                sparse_values = values
            
            sparse_key_cache.append(sparse_keys)
            sparse_value_cache.append(sparse_values)
            
            # 更新统计信息
            self.stats['total_tokens'] += T
            self.stats['pruned_tokens'] += (T - num_keep)
        
        # 计算压缩比
        if self.stats['total_tokens'] > 0:
            self.stats['compression_ratio'] = self.stats['pruned_tokens'] / self.stats['total_tokens']
        
        return sparse_key_cache, sparse_value_cache
    
    def get_stats(self):
        """获取稀疏度统计信息"""
        return self.stats.copy()


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name,
             kv_sparsity_wrapper=None):
    """获取模型预测结果"""
    preds = []
    
    for json_obj in tqdm(data, desc=f"Processing {dataset}"):
        prompt = prompt_format.format(**json_obj)
        
        # 截断到最大长度
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # 对于某些任务不使用chat格式
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_tokens.input_ids.shape[-1]
        
        # 生成文本
        with torch.no_grad():
            if dataset == "samsum":
                output = model.generate(
                    **input_tokens,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input_tokens,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({
            "pred": pred, 
            "answers": json_obj["answers"], 
            "all_classes": json_obj["all_classes"], 
            "length": json_obj["length"]
        })
    
    return preds


def main():
    parser = argparse.ArgumentParser(description='DiffSparseKV LongBench Evaluation')
    parser.add_argument('--model_path', type=str,
                       default='/home/zh/model/Meta-Llama-3-8B-Instruct',
                       help='模型路径')
    parser.add_argument('--max_length', type=int, default=8192,
                       help='最大序列长度')
    parser.add_argument('--datasets', nargs='+', 
                       default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", 
                               "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", 
                               "trec", "triviaqa", "samsum", "passage_count", 
                               "passage_retrieval_en", "lcc", "repobench-p"],
                       help='要评估的数据集列表')
    parser.add_argument('--output_dir', type=str, default='pred/pred_diff_sparse_kv',
                       help='输出目录')
    parser.add_argument('--kv_sparsity', type=float, default=0.0,
                       help='KV Cache稀疏度 (0.0-1.0, 0.0=无稀疏, 1.0=完全稀疏)')
    parser.add_argument('--sparsity_levels', type=str, default='',
                       help='完整三级稀疏度，例如 0.0,0.7,1.0；为空时使用 [0.0, kv_sparsity, 1.0]')
    parser.add_argument('--sparsity_type', type=str, default='none',
                       choices=['none', 'uniform', 'diff_sparse_kv'],
                       help='稀疏度类型: none(无), uniform(统一), diff_sparse_kv(差分稀疏)')
    parser.add_argument('--target_distribution', type=str, default='0.05,0.75,0.20',
                       help='DiffSparseKV 的三级目标分布，例如 0.05,0.75,0.20')
    parser.add_argument('--window_size', type=int, default=128,
                       help='DiffSparseKV Window A/B 大小')
    parser.add_argument('--obs_window_size', type=int, default=128,
                       help='DiffSparseKV 观察窗口大小')
    parser.add_argument('--limit', type=int, default=0,
                       help='每个数据集只评估前 N 条样本，0 表示全量')
<<<<<<< HEAD
    parser.add_argument('--sample_seed', type=int, default=-1,
                       help='当 limit > 0 时，>=0 表示随机抽样种子；-1 表示取前 N 条')
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
    parser.add_argument('--output_tag', type=str, default='',
                       help='附加到输出目录名的标签，便于保存多次实验')
    parser.add_argument('--debug_diff_sparse', action='store_true',
                       help='开启 DiffSparseKV 调试输出')
    parser.add_argument('--level_2_mode', type=str, default='evict',
                       choices=['evict', 'zero'],
                       help='Level 2 在 100%% 稀疏时的处理方式')
    parser.add_argument('--importance_mode', type=str, default='attention_only',
                       choices=['attention_only', 'value_aware'],
                       help='importance 指标类型')
    parser.add_argument('--value_sink_keep', type=int, default=2,
                       help='value-aware 时强制保留的前缀 sink token 数量')
    parser.add_argument('--head_aggregation_mode', type=str, default='mean',
                       choices=['mean', 'max', 'hybrid', 'top2_mean'],
                       help='跨头聚合 importance 的方式')
    parser.add_argument('--head_aggregation_alpha', type=float, default=0.5,
                       help='hybrid 聚合时 mean 的权重')
    parser.add_argument('--head_disagreement_ratio', type=float, default=-1.0,
                       help='若 max/mean 超过该阈值，则使用 max 保护 token；-1 禁用')
<<<<<<< HEAD
    parser.add_argument('--selector_mode', type=str, default='diffsparse',
                       choices=['diffsparse', 'snapkv'],
                       help='token 选择模式')
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
    parser.add_argument('--target_budget', type=float, default=-1.0,
                       help='目标平均稀疏度预算；>=0 时将通过 budget generator 生成配置')
    parser.add_argument('--budget_template', type=str, default='default_3level',
                       help='budget generator 使用的模板名称')
    
    args = parser.parse_args()
    if args.target_budget >= 0.0:
        from diffsparsekv import resolve_budget_config
        resolved_budget = resolve_budget_config(args.target_budget, args.budget_template)
        target_distribution = resolved_budget.target_distribution
        sparsity_levels = resolved_budget.sparsity_levels
    else:
        resolved_budget = None
        target_distribution = [float(x) for x in args.target_distribution.split(',')]
        if args.sparsity_levels:
            sparsity_levels = [float(x) for x in args.sparsity_levels.split(',')]
        else:
            sparsity_levels = [0.0, args.kv_sparsity, 1.0]
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from: {args.model_path}")
    
    if args.sparsity_type == 'diff_sparse_kv':
        # 加载 DiffSparseKV 模型
        try:
            try:
                from diffsparsekv import LlamaForCausalLMDiffSparseKV, create_diff_sparse_kv_config
            except ImportError:
                from llama_diff_sparse_kv import LlamaForCausalLMDiffSparseKV, create_diff_sparse_kv_config

            config = LlamaConfig.from_pretrained(args.model_path)
            
            # 配置 MUSTAFAR 基础参数
            config.k_sparsity = 0.0
            config.v_sparsity = 0.0
            config.group_size = 32
            config.residual_length = 32  # 与原始代码一致
            config.use_flash = True
            
            # 配置 DiffSparseKV 参数
            config = create_diff_sparse_kv_config(
                base_config=config,
                enable_diff_sparse=True,
                target_distribution=target_distribution,
                sparsity_levels=sparsity_levels,
                diff_sparse_window_size=args.window_size,
                obs_window_size=args.obs_window_size,
                debug_diff_sparse=args.debug_diff_sparse,
                level_2_mode=args.level_2_mode,
                importance_mode=args.importance_mode,
                value_sink_keep=args.value_sink_keep,
                head_aggregation_mode=args.head_aggregation_mode,
                head_aggregation_alpha=args.head_aggregation_alpha,
                head_disagreement_ratio=args.head_disagreement_ratio,
<<<<<<< HEAD
                selector_mode=args.selector_mode,
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
            )
            
            model = LlamaForCausalLMDiffSparseKV.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print("✓ DiffSparseKV model loaded successfully")
            print(f"  - Target distribution: {target_distribution}")
            print(f"  - Sparsity levels: {sparsity_levels}")
            if resolved_budget is not None:
                print(f"  - Target budget: {resolved_budget.target_budget:.3f}")
                print(f"  - Budget template: {resolved_budget.template_name}")
            print(f"  - Window size: {args.window_size}")
            print(f"  - Observation window: {args.obs_window_size}")
            print(f"  - Level 2 mode: {args.level_2_mode}")
            print(f"  - Importance mode: {args.importance_mode}")
            print(f"  - Value sink keep: {args.value_sink_keep}")
            print(f"  - Head aggregation mode: {args.head_aggregation_mode}")
            print(f"  - Head aggregation alpha: {args.head_aggregation_alpha}")
            print(f"  - Head disagreement ratio: {args.head_disagreement_ratio}")
<<<<<<< HEAD
            print(f"  - Selector mode: {args.selector_mode}")
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
            
        except Exception as e:
            print(f"✗ Failed to load DiffSparseKV model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # 加载 MUSTAFAR baseline 模型
        try:
            from models.llama_mustafar_Kt_Mag_Vt_Mag import LlamaForCausalLM_MUSTAFAR
            
            config = LlamaConfig.from_pretrained(args.model_path)
            config.k_sparsity = 0.0 if args.sparsity_type == 'none' else args.kv_sparsity
            config.v_sparsity = 0.0 if args.sparsity_type == 'none' else args.kv_sparsity
            config.group_size = 32
            config.residual_length = 32  # 与原始代码一致
            config.use_flash = True
            
            model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print(f"✓ MUSTAFAR model loaded successfully (sparsity_type: {args.sparsity_type})")
        except Exception as e:
            print(f"Warning: Failed to load MUSTAFAR model: {e}")
            print("Falling back to standard model")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✓ Standard model loaded successfully")
    
    model.eval()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_name = args.model_path.split("/")[-1]
    
    def build_output_subdir(base_name: str) -> str:
        if args.output_tag:
            return f"{base_name}_{args.output_tag}"
        return base_name
    
    # 定义数据集的 prompt 格式和最大生成长度（硬编码默认值）
    dataset2prompt = {
        "narrativeqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "qasper": "{context}\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_en": "{context}\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "{context}\n\n问题: {input}\n答案:",
        "hotpotqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "musique": "{context}\n\nQuestion: {input}\nAnswer:",
        "dureader": "{context}\n\n问题: {input}\n答案:",
        "gov_report": "{context}\n\nQuestion: {input}\nAnswer:",
        "qmsum": "{context}\n\nQuestion: {input}\nAnswer:",
        "multi_news": "{context}\n\nQuestion: {input}\nAnswer:",
        "vcsum": "{context}\n\n问题: {input}\n答案:",
        "trec": "{input}",
        "triviaqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "samsum": "{context}\n\nSummarize the above conversation.",
        "lsht": "{input}",
        "passage_retrieval_en": "{input}",
        "passage_count": "{context}\n\nHow many passages are there?",
        "passage_retrieval_zh": "{input}",
        "lcc": "{input}",
        "repobench-p": "{input}",
    }
    
    # 最大输出 token 的长度，不同数据集有不同的值，不同的任务需要不同长度的答案
    dataset2maxlen = {
        "narrativeqa": 100,
        "qasper": 100,
        "multifieldqa_en": 100,
        "multifieldqa_zh": 100,
        "hotpotqa": 100,
        "2wikimqa": 100,
        "musique": 100,
        "dureader": 100,
        "gov_report": 100,
        "qmsum": 100,
        "multi_news": 100,
        "vcsum": 100,
        "trec": 50,
        "triviaqa": 100,
        "samsum": 100,
        "lsht": 50,
        "passage_retrieval_en": 100,
        "passage_count": 50,
        "passage_retrieval_zh": 100,
        "lcc": 100,
        "repobench-p": 100,
    }
    
    # Try to load from config files if they exist, otherwise use defaults
    try:
        with open("config/dataset2prompt.json", "r") as f:
            dataset2prompt.update(json.load(f))
        with open("config/dataset2maxlen.json", "r") as f:
            dataset2maxlen.update(json.load(f))
        print("✓ Loaded dataset configurations from config files")
    except FileNotFoundError:
        print("✓ Using default dataset configurations (config files not found)")
    
    # 创建输出目录
    output_subdir = build_output_subdir(f"{model_name}_{args.max_length}_baseline")
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # 初始化KV Cache稀疏度包装器
    kv_sparsity_wrapper = None
    if args.sparsity_type != 'none':
        kv_sparsity_wrapper = KVCacheSparsityWrapper(
            sparsity_level=args.kv_sparsity,
            sparsity_type=args.sparsity_type
        )
        print(f"KV Cache Sparsity: {args.sparsity_type} ({args.kv_sparsity*100:.1f}%)")
        
        # 更新输出目录名称
        output_subdir = build_output_subdir(
            (
                f"{model_name}_{args.max_length}_{args.sparsity_type}_budget_{args.target_budget:.2f}"
                if args.target_budget >= 0.0
                else f"{model_name}_{args.max_length}_{args.sparsity_type}_{args.kv_sparsity:.2f}"
            )
        )
        output_path = os.path.join(args.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory: {output_path}")
    
    # 对每个数据集进行预测
    for dataset in args.datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*50}")
        
        try:
            # 加载数据集
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            if args.limit > 0:
<<<<<<< HEAD
                sample_count = min(args.limit, len(data))
                if args.sample_seed >= 0:
                    rng = np.random.default_rng(args.sample_seed)
                    sampled_indices = sorted(rng.choice(len(data), size=sample_count, replace=False).tolist())
                    data = data.select(sampled_indices)
                    print(
                        f"Using {len(data)} random samples for quick evaluation "
                        f"(seed={args.sample_seed})"
                    )
                else:
                    data = data.select(range(sample_count))
                    print(f"Using first {len(data)} samples for quick evaluation")
=======
                data = data.select(range(min(args.limit, len(data))))
                print(f"Using first {len(data)} samples for quick evaluation")
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
            
            # 获取prompt格式和最大生成长度
            prompt_format = dataset2prompt.get(dataset, "{input}")
            max_gen = dataset2maxlen.get(dataset, 100)
            
            # 获取预测
            preds = get_pred(
                model, tokenizer, data, args.max_length, max_gen, 
                prompt_format, dataset, device, model_name,
                kv_sparsity_wrapper=kv_sparsity_wrapper
            )
            
            # 保存预测结果
            output_file = os.path.join(output_path, f"{dataset}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for pred in preds:
                    f.write(json.dumps(pred, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved {len(preds)} predictions to {output_file}")
        
        except Exception as e:
            print(f"✗ Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存配置信息
    config_file = os.path.join(output_path, "sparsity_config.json")
    with open(config_file, 'w') as f:
        config_info = {
            "model": model_name,
            "max_length": args.max_length,
            "sparsity_type": args.sparsity_type,
            "kv_sparsity": args.kv_sparsity,
            "sparsity_levels": sparsity_levels,
            "target_distribution": target_distribution,
            "window_size": args.window_size,
            "obs_window_size": args.obs_window_size,
            "level_2_mode": args.level_2_mode,
            "importance_mode": args.importance_mode,
            "value_sink_keep": args.value_sink_keep,
            "head_aggregation_mode": args.head_aggregation_mode,
            "head_aggregation_alpha": args.head_aggregation_alpha,
            "head_disagreement_ratio": args.head_disagreement_ratio,
<<<<<<< HEAD
            "selector_mode": args.selector_mode,
            "target_budget": args.target_budget,
            "budget_template": args.budget_template,
            "limit": args.limit,
            "sample_seed": args.sample_seed,
=======
            "target_budget": args.target_budget,
            "budget_template": args.budget_template,
            "limit": args.limit,
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
            "output_tag": args.output_tag,
        }
        
        # 添加稀疏度统计信息
        if kv_sparsity_wrapper:
            config_info["sparsity_stats"] = kv_sparsity_wrapper.get_stats()
        
        json.dump(config_info, f, indent=2)
    
    print(f"\n{'='*50}")
    print("✓ Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
