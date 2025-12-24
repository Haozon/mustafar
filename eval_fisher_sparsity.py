#!/usr/bin/env python3
"""
基于Fisher敏感度分析结果的per-layer稀疏度配置评估脚本
读取fisher_sparsity_allocation.json文件并进行LongBench测试
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from typing import Dict

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

def load_fisher_sparsity_config(fisher_results_path: str) -> Dict[int, float]:
    """
    从Fisher敏感度分析结果文件中加载稀疏度配置
    
    Args:
        fisher_results_path: Fisher结果JSON文件路径
        
    Returns:
        Dict[int, float]: 层索引到稀疏度的映射 (稀疏度以小数形式，如0.7表示70%)
    """
    print(f"Loading Fisher sensitivity results from: {fisher_results_path}")
    
    with open(fisher_results_path, 'r') as f:
        fisher_results = json.load(f)
    
    optimal_sparsity = fisher_results['optimal_sparsity']
    n_layers = fisher_results['n_layers']
    
    # 验证层数匹配
    if len(optimal_sparsity) != n_layers:
        raise ValueError(f"Sparsity list length ({len(optimal_sparsity)}) doesn't match n_layers ({n_layers})")
    
    # 转换为字典格式，稀疏度从百分比转换为小数
    sparsity_config = {}
    for layer_idx, sparsity_percent in enumerate(optimal_sparsity):
        sparsity_config[layer_idx] = sparsity_percent / 100.0  # 转换为小数形式
    
    return sparsity_config, fisher_results

def apply_per_layer_sparsity_to_model(model, sparsity_config: Dict[int, float]):
    """
    将per-layer稀疏度配置应用到现有的MUSTAFAR模型
    通过修改每层的k_sparsity和v_sparsity参数
    """
    print("Applying Fisher-based per-layer sparsity configuration...")
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'k_sparsity') and hasattr(layer.self_attn, 'v_sparsity'):
            sparsity = sparsity_config.get(layer_idx, 0.0)
            layer.self_attn.k_sparsity = sparsity
            layer.self_attn.v_sparsity = sparsity
            print(f"  Layer {layer_idx:2d}: sparsity = {sparsity:.3f} ({sparsity*100:.1f}%)")
        else:
            print(f"  Layer {layer_idx:2d}: No sparsity attributes found, skipping")

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
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
    parser = argparse.ArgumentParser(description='基于Fisher敏感度分析结果评估per-layer稀疏度')
    parser.add_argument('--fisher_results', type=str, 
                       default='fisher_sensitivity_results/fisher_sparsity_allocation.json',
                       help='Fisher敏感度分析结果文件路径')
    parser.add_argument('--model_path', type=str,
                       default='/home/zh/model/Meta-Llama-3-8B-Instruct',
                       help='模型路径')
    parser.add_argument('--max_length', type=int, default=8192,
                       help='最大序列长度')
    parser.add_argument('--residual_length', type=int, default=128,
                       help='residual window大小')
    parser.add_argument('--datasets', nargs='+', 
                       default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", 
                               "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", 
                               "trec", "triviaqa", "samsum", "passage_count", 
                               "passage_retrieval_en", "lcc", "repobench-p"],
                       help='要评估的数据集列表')
    parser.add_argument('--output_dir', type=str, default='pred_fisher',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Fisher敏感度分析结果
    sparsity_config, fisher_results = load_fisher_sparsity_config(args.fisher_results)
    
    print("\nFisher-based per-layer sparsity configuration:")
    for layer_idx, sparsity in sparsity_config.items():
        print(f"  Layer {layer_idx:2d}: {sparsity:.3f} ({sparsity*100:.1f}%)")
    
    avg_sparsity = np.mean(list(sparsity_config.values()))
    target_avg_sparsity = fisher_results['target_avg_sparsity'] / 100.0
    actual_avg_sparsity = fisher_results['actual_avg_sparsity'] / 100.0
    
    print(f"\nSparsity Statistics:")
    print(f"  Target average sparsity: {target_avg_sparsity:.3f} ({target_avg_sparsity*100:.1f}%)")
    print(f"  Actual average sparsity: {actual_avg_sparsity:.3f} ({actual_avg_sparsity*100:.1f}%)")
    print(f"  Calculated average sparsity: {avg_sparsity:.3f} ({avg_sparsity*100:.1f}%)")
    print(f"  Method: {fisher_results['method']}")
    print(f"  Solver status: {fisher_results['solver_status']}")
    
    # 加载模型 - 使用现有的MUSTAFAR模型
    print(f"\nLoading MUSTAFAR model from: {args.model_path}")
    
    # 导入MUSTAFAR模型
    from models.llama_mustafar_Kt_Mag_Vt_Mag import LlamaForCausalLM_MUSTAFAR
    
    # 配置模型
    config = LlamaConfig.from_pretrained(args.model_path)
    config.k_sparsity = 0.0  # 初始设置为0，后面会per-layer设置
    config.v_sparsity = 0.0
    config.group_size = 32
    config.residual_length = args.residual_length
    config.use_flash = True
    
    model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    model.eval()
    
    # 应用Fisher-based per-layer稀疏度配置
    apply_per_layer_sparsity_to_model(model, sparsity_config)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_name = args.model_path.split("/")[-1]
    
    # 加载配置文件
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # 创建输出目录
    output_subdir = f"{model_name}_{args.max_length}_fisher_avg_{avg_sparsity:.3f}"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nOutput directory: {output_path}")
    
    # 对每个数据集进行预测
    for dataset in args.datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*50}")
        
        try:
            # 加载数据集
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            
            # 获取prompt格式和最大生成长度
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            
            # 生成预测
            preds = get_pred(
                model, tokenizer, data, args.max_length, 
                max_gen, prompt_format, dataset, device, model_name
            )
            
            # 保存结果
            out_file = os.path.join(output_path, f"{dataset}.jsonl")
            with open(out_file, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Saved {len(preds)} predictions to {out_file}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存完整的配置信息
    config_info = {
        "fisher_results_file": args.fisher_results,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "residual_length": args.residual_length,
        "sparsity_config": sparsity_config,
        "target_avg_sparsity": target_avg_sparsity,
        "actual_avg_sparsity": actual_avg_sparsity,
        "calculated_avg_sparsity": avg_sparsity,
        "method": fisher_results['method'],
        "solver_status": fisher_results['solver_status'],
        "total_sensitivity": fisher_results['total_sensitivity'],
        "n_layers": fisher_results['n_layers'],
        "min_sparsity": fisher_results['min_sparsity'] / 100.0,
        "max_sparsity": fisher_results['max_sparsity'] / 100.0,
        "original_fisher_results": fisher_results
    }
    
    config_file = os.path.join(output_path, "fisher_sparsity_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nFisher sparsity configuration saved to: {config_file}")
    print(f"\nEvaluation completed! Results saved in: {output_path}")
    print(f"\nTo evaluate the results, run:")
    print(f"python eval_long_bench.py --model {output_subdir}")
    
    print(f"\nSummary:")
    print(f"  - Fisher method: {fisher_results['method']}")
    print(f"  - Solver status: {fisher_results['solver_status']}")
    print(f"  - Target sparsity: {target_avg_sparsity*100:.1f}%")
    print(f"  - Actual sparsity: {actual_avg_sparsity*100:.1f}%")
    print(f"  - Sparsity range: {fisher_results['min_sparsity']:.1f}% - {fisher_results['max_sparsity']:.1f}%")

if __name__ == "__main__":
    main()