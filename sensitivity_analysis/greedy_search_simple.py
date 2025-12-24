"""
贪心搜索层级敏感度分析 - 快速验证版本
使用Perplexity作为评估指标，K和V使用相同的per-layer稀疏度
"""

import torch
import json
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn

@dataclass
class GreedySearchConfig:
    """贪心搜索配置 - 改进版（Warm Start）"""
    model_path: str = "/home/zh/model/Meta-Llama-3-8B-Instruct"
    initial_sparsity: float = 0.4  # 初始稀疏度（warm start）
    step_size: float = 0.05  # 统一步长
    target_sparsity: float = 0.7  # 目标稀疏度
    num_samples: int = 30  # 快速验证用30个样本
    max_length: int = 512  # 序列长度
    device: str = "cuda"
    output_dir: str = "./sensitivity_results"
    residual_length: int = 128  # residual window 大小
    eval_metric: str = "ppl"  # 评估指标: "ppl" (perplexity) 或 "loss" (cross entropy loss)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value states for Grouped Query Attention.
    
    Args:
        hidden_states: [batch, num_key_value_heads, seq_len, head_dim]
        n_rep: number of repetitions
        
    Returns:
        repeated tensor: [batch, num_key_value_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_kv_sparsity(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    对 key/value states 应用 magnitude-based pruning
    
    Args:
        tensor: [batch, num_heads, seq_len, head_dim]
        sparsity: 稀疏度 (0.0 - 1.0)
        
    Returns:
        pruned_tensor
    """
    if sparsity == 0.0:
        return tensor
    
    B, H, T, D = tensor.shape
    keep_ratio = 1.0 - sparsity
    num_to_keep = max(1, int(keep_ratio * D))
    
    # Reshape to [B*H*T, D]
    tensor_flat = tensor.reshape(-1, D)
    
    # Compute pruning threshold per vector
    threshold_values, _ = torch.kthvalue(torch.abs(tensor_flat), num_to_keep, dim=-1, keepdim=True)
    
    # Create a mask: Keep only values larger than or equal to the threshold
    mask = torch.abs(tensor_flat) >= threshold_values
    
    # Apply the mask (zero out pruned elements)
    pruned_tensor = tensor_flat * mask
    
    return pruned_tensor.view(B, H, T, D)


def apply_kv_sparsity_with_residual(tensor: torch.Tensor, sparsity: float, residual_length: int) -> torch.Tensor:
    """
    对 key/value states 应用 magnitude-based pruning，保留最后 residual_length 个 tokens 为稠密
    
    Args:
        tensor: [batch, num_heads, seq_len, head_dim]
        sparsity: 稀疏度 (0.0 - 1.0)
        residual_length: 保留为稠密的 token 数量
        
    Returns:
        pruned_tensor
    """
    if sparsity == 0.0:
        return tensor
    
    seq_len = tensor.shape[-2]
    
    # 如果序列长度小于 residual_length，对整个序列稀疏化
    if seq_len < residual_length:
        return apply_kv_sparsity(tensor, sparsity)
    
    # 否则，只对前面的部分稀疏化，保留最后 residual_length 个 tokens
    prefix = tensor[:, :, :-(residual_length), :]
    suffix = tensor[:, :, -(residual_length):, :]
    
    prefix_pruned = apply_kv_sparsity(prefix, sparsity)
    
    return torch.cat([prefix_pruned, suffix], dim=2)


class LlamaAttention_Sparse(LlamaAttention):
    """支持稀疏化的 Llama Attention 层"""
    
    def __init__(self, config: LlamaConfig, layer_idx: int, sparsity_config: Dict[int, float], 
                 residual_length: int = 128):
        super().__init__(config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.sparsity_config = sparsity_config
        self.residual_length = residual_length
    
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value states for GQA"""
        return repeat_kv(hidden_states, n_rep)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播，在计算注意力前应用稀疏化
        """
        bsz, q_len, _ = hidden_states.size()
        
        # 计算 Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(value_states, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # 处理 past_key_value
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # 更新 kv_seq_len
        kv_seq_len = key_states.shape[-2]
        
        # ⭐ 关键：在计算注意力前应用稀疏化
        sparsity = self.sparsity_config.get(self.layer_idx, 0.0)
        if sparsity > 0.0:
            key_states = apply_kv_sparsity_with_residual(key_states, sparsity, self.residual_length)
            value_states = apply_kv_sparsity_with_residual(value_states, sparsity, self.residual_length)
        
        # Repeat key/value states for GQA (Grouped Query Attention)
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)
        
        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
        
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # 准备 past_key_value
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None
        
        return attn_output, attn_weights if output_attentions else None, past_key_value


class PerLayerSparsityModel:
    """支持per-layer稀疏度的模型包装器"""
    
    def __init__(self, model_path: str, device: str = "cuda", residual_length: int = 128):
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 获取层数
        self.num_layers = len(self.model.model.layers)
        print(f"Model loaded with {self.num_layers} layers")
        
        # 初始化per-layer稀疏度配置
        self.sparsity_config = {i: 0.0 for i in range(self.num_layers)}
        self.residual_length = residual_length
        self.original_attention_layers = {}
        self.layers_replaced = False  # 标记是否已替换层
        
    def set_sparsity_config(self, config: Dict[int, float]):
        """设置每层的稀疏度"""
        self.sparsity_config = config.copy()
    
    def replace_attention_layers(self):
        """
        用 LlamaAttention_Sparse 替换所有 attention 层
        """
        if self.layers_replaced:
            # 如果已经替换过，只更新稀疏度配置
            for layer_idx, layer in enumerate(self.model.model.layers):
                if isinstance(layer.self_attn, LlamaAttention_Sparse):
                    layer.self_attn.sparsity_config = self.sparsity_config
            return
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # 保存原始 attention 层
            self.original_attention_layers[layer_idx] = layer.self_attn
            
            # 创建新的 sparse attention 层
            sparse_attn = LlamaAttention_Sparse(
                self.model.config,
                layer_idx,
                self.sparsity_config,
                self.residual_length
            )
            
            # 复制权重
            sparse_attn.load_state_dict(layer.self_attn.state_dict())
            
            # 移动到与原始层相同的设备和dtype
            sparse_attn = sparse_attn.to(
                device=layer.self_attn.q_proj.weight.device,
                dtype=layer.self_attn.q_proj.weight.dtype
            )
            
            # 替换
            layer.self_attn = sparse_attn
        
        self.layers_replaced = True
    
    def restore_attention_layers(self):
        """
        恢复原始 attention 层
        """
        if not self.layers_replaced:
            return
        
        for layer_idx, original_attn in self.original_attention_layers.items():
            self.model.model.layers[layer_idx].self_attn = original_attn
        
        self.layers_replaced = False
    
    def forward_with_sparsity(self, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播，应用per-layer稀疏度到 key/value states
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # 前向传播（attention 层已经被替换）
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False  # Prefill 阶段不需要 cache
            )
        
        return outputs.logits



class GreedyLayerSensitivityAnalyzer:
    """贪心搜索层级敏感度分析器"""
    
    def __init__(self, config: GreedySearchConfig):
        self.config = config
        
        # 加载模型
        self.model_wrapper = PerLayerSparsityModel(
            config.model_path, 
            config.device,
            config.residual_length
        )
        self.num_layers = self.model_wrapper.num_layers
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 准备验证数据
        self.val_loader = self._prepare_validation_data()
        
        # 初始化记录
        self.iteration_history = []
        
    def _prepare_validation_data(self):
        """准备验证数据集"""
        print(f"Loading validation data ({self.config.num_samples} samples)...")
        
        # 使用wikitext-2作为验证集
        dataset = load_dataset(
            "wikitext", 
            "wikitext-2-raw-v1", 
            split=f"validation[:{self.config.num_samples}]"
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config.max_length,
                padding='max_length',
                return_tensors='pt'
            )
        
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # 使用更大的 batch size 加速评估
        return DataLoader(tokenized, batch_size=4, shuffle=False)
    
    def evaluate_perplexity(self, sparsity_config: Dict[int, float]) -> float:
        """
        评估给定稀疏度配置的perplexity
        
        Args:
            sparsity_config: {layer_idx: sparsity}
            
        Returns:
            perplexity (float)
        """
        # 应用稀疏度配置并替换 attention 层（只做一次）
        self.model_wrapper.set_sparsity_config(sparsity_config)
        self.model_wrapper.replace_attention_layers()
        
        try:
            total_loss = 0.0
            total_tokens = 0
            
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                # 前向传播
                logits = self.model_wrapper.forward_with_sparsity(
                    input_ids, attention_mask
                )
                
                # 计算loss (shift for causal LM)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # 计算cross entropy
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # 只计算非padding token的loss
                mask = (shift_labels != self.tokenizer.pad_token_id).view(-1)
                loss = loss[mask].sum()
                
                total_loss += loss.item()
                total_tokens += mask.sum().item()
            
            # 计算perplexity
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            
            return perplexity
        finally:
            # 恢复原始 attention 层
            self.model_wrapper.restore_attention_layers()
    
    def evaluate_loss(self, sparsity_config: Dict[int, float]) -> float:
        """
        评估给定稀疏度配置的平均 cross entropy loss
        
        Args:
            sparsity_config: {layer_idx: sparsity}
            
        Returns:
            average_loss (float)
        """
        # 应用稀疏度配置并替换 attention 层（只做一次）
        self.model_wrapper.set_sparsity_config(sparsity_config)
        self.model_wrapper.replace_attention_layers()
        
        try:
            total_loss = 0.0
            total_tokens = 0
            
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                # 前向传播
                logits = self.model_wrapper.forward_with_sparsity(
                    input_ids, attention_mask
                )
                
                # 计算loss (shift for causal LM)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # 计算cross entropy
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # 只计算非padding token的loss
                mask = (shift_labels != self.tokenizer.pad_token_id).view(-1)
                loss = loss[mask].sum()
                
                total_loss += loss.item()
                total_tokens += mask.sum().item()
            
            # 返回平均loss
            avg_loss = total_loss / total_tokens
            return avg_loss
        finally:
            # 恢复原始 attention 层
            self.model_wrapper.restore_attention_layers()
    
    def evaluate_metric(self, sparsity_config: Dict[int, float]) -> float:
        """
        根据配置的评估指标评估给定稀疏度配置
        
        Args:
            sparsity_config: {layer_idx: sparsity}
            
        Returns:
            metric_value (float): perplexity 或 loss
        """
        if self.config.eval_metric == "ppl":
            return self.evaluate_perplexity(sparsity_config)
        elif self.config.eval_metric == "loss":
            return self.evaluate_loss(sparsity_config)
        else:
            raise ValueError(f"Unsupported eval_metric: {self.config.eval_metric}. Use 'ppl' or 'loss'.")
    
    def compute_average_sparsity(self, sparsity_config: Dict[int, float]) -> float:
        """计算平均稀疏度"""
        return np.mean(list(sparsity_config.values()))
    
    def search(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        执行贪心搜索（改进版 - Warm Start）
        
        Returns:
            (sparsity_config, sensitivity_scores)
        """
        print("\n" + "="*70)
        print("🚀 Starting Warm-Start Greedy Search")
        print("="*70)
        print(f"Initial sparsity: {self.config.initial_sparsity}")
        print(f"Step size: {self.config.step_size}")
        print(f"Target sparsity: {self.config.target_sparsity}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Validation samples: {self.config.num_samples}")
        print("="*70 + "\n")
        
        # 初始化：所有层从 initial_sparsity 开始
        sparsity_config = {l: self.config.initial_sparsity for l in range(self.num_layers)}
        sensitivity = {l: 0.0 for l in range(self.num_layers)}
        selection_count = {l: 0 for l in range(self.num_layers)}
        
        # 评估baseline
        print(f"📊 Evaluating baseline (all layers at {self.config.initial_sparsity:.1%} sparsity)...")
        baseline_metric = self.evaluate_metric(sparsity_config)
        current_avg_sparsity = self.compute_average_sparsity(sparsity_config)
        
        metric_name = "perplexity" if self.config.eval_metric == "ppl" else "loss"
        print(f"✅ Baseline {metric_name}: {baseline_metric:.4f}")
        print(f"   Current avg sparsity: {current_avg_sparsity:.3f}\n")
        
        iteration = 0
        
        # 贪心搜索主循环
        while current_avg_sparsity < self.config.target_sparsity:
            iteration += 1
            print(f"\n{'─'*70}")
            print(f"🔄 Iteration {iteration}")
            print(f"{'─'*70}")
            
            best_layer = None
            min_relative_error = float('inf')
            best_metric = None
            
            # 尝试所有层
            print("Testing all layers...")
            for layer_idx in tqdm(range(self.num_layers), desc="Layers"):
                current_sparsity = sparsity_config[layer_idx]
                
                # 计算下一个稀疏度
                next_sparsity = current_sparsity + self.config.step_size
                if next_sparsity > 1.0:
                    continue  # 该层已达到最大稀疏度
                
                # 临时配置
                temp_config = sparsity_config.copy()
                temp_config[layer_idx] = next_sparsity
                
                # 评估指标
                temp_metric = self.evaluate_metric(temp_config)
                
                # 计算相对指标增量
                relative_error = (temp_metric - baseline_metric) / max(baseline_metric, 1e-6)
                
                # 更新最佳选择（最小化相对误差）
                if relative_error < min_relative_error:
                    min_relative_error = relative_error
                    best_layer = layer_idx
                    best_metric = temp_metric
            
            # 检查终止条件
            if best_layer is None:
                print("\n⚠️  Warning: Cannot reach target sparsity")
                print("   All layers have reached maximum sparsity (1.0)")
                break
            
            # 应用最佳选择
            old_sparsity = sparsity_config[best_layer]
            sparsity_config[best_layer] += self.config.step_size
            current_avg_sparsity = self.compute_average_sparsity(sparsity_config)
            
            # 更新 baseline 指标
            baseline_metric = best_metric
            
            # 更新敏感度（累积相对误差）
            sensitivity[best_layer] += min_relative_error
            selection_count[best_layer] += 1
            
            # 记录迭代信息
            iter_info = {
                'iteration': iteration,
                'layer': best_layer,
                'old_sparsity': old_sparsity,
                'new_sparsity': sparsity_config[best_layer],
                'metric_value': best_metric,
                'metric_name': metric_name,
                'avg_sparsity': current_avg_sparsity,
                'relative_metric_increase': min_relative_error
            }
            self.iteration_history.append(iter_info)
            
            # 打印结果
            print(f"\n✅ Best choice:")
            print(f"   Layer {best_layer}: {old_sparsity:.3f} → {sparsity_config[best_layer]:.3f}")
            print(f"   {metric_name.capitalize()}: {best_metric:.4f}")
            print(f"   Relative {metric_name} increase: {min_relative_error:.6f} ({min_relative_error*100:.4f}%)")
            print(f"   Avg sparsity: {current_avg_sparsity:.3f}")
            
            # 检查是否达到目标
            if current_avg_sparsity >= self.config.target_sparsity:
                print(f"\n🎯 Target sparsity reached!")
                break
        
        # 计算平均敏感度
        for l in range(self.num_layers):
            if selection_count[l] > 0:
                sensitivity[l] = sensitivity[l] / selection_count[l]
            else:
                sensitivity[l] = 0.0  # 从未被选中 → 不敏感
        
        # 归一化敏感度到 [0, 1]
        sens_values = list(sensitivity.values())
        if len(sens_values) > 0:
            sens_min, sens_max = min(sens_values), max(sens_values)
            
            if sens_max - sens_min > 1e-6:
                sensitivity = {
                    l: (s - sens_min) / (sens_max - sens_min)
                    for l, s in sensitivity.items()
                }
        
        print("\n" + "="*70)
        print("✅ Greedy search completed!")
        print("="*70)
        print(f"Final avg sparsity: {current_avg_sparsity:.3f}")
        print(f"Final {metric_name}: {baseline_metric:.4f}")
        print(f"Total iterations: {iteration}")
        print("="*70 + "\n")
        
        return sparsity_config, sensitivity
    
    def save_results(self, sparsity_config: Dict[int, float], 
                    sensitivity: Dict[int, float]):
        """保存结果到文件"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        results = {
            'config': {
                'model_path': self.config.model_path,
                'initial_sparsity': self.config.initial_sparsity,
                'step_size': self.config.step_size,
                'target_sparsity': self.config.target_sparsity,
                'num_samples': self.config.num_samples,
                'num_layers': self.num_layers,
                'eval_metric': self.config.eval_metric
            },
            'sparsity_config': sparsity_config,
            'sensitivity_scores': sensitivity,
            'iteration_history': self.iteration_history,
            'final_avg_sparsity': self.compute_average_sparsity(sparsity_config)
        }
        
        # 保存JSON
        output_file = f"{self.config.output_dir}/greedy_search_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: {output_file}")
        
        return output_file


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='贪心搜索层级稀疏度分析')
    parser.add_argument('--model_path', type=str, 
                       default="/home/zh/model/Meta-Llama-3-8B-Instruct",
                       help='模型路径')
    parser.add_argument('--initial_sparsity', type=float, default=0.4,
                       help='初始稀疏度 (默认: 0.4)')
    parser.add_argument('--step_size', type=float, default=0.05,
                       help='稀疏度增加步长 (默认: 0.05)')
    parser.add_argument('--target_sparsity', type=float, default=0.7,
                       help='目标稀疏度 (默认: 0.7)')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='验证样本数量 (默认: 30)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度 (默认: 512)')
    parser.add_argument('--residual_length', type=int, default=128,
                       help='residual window 大小 (默认: 128)')
    parser.add_argument('--output_dir', type=str, default="./sensitivity_results",
                       help='输出目录 (默认: ./sensitivity_results)')
    parser.add_argument('--eval_metric', type=str, default="ppl",
                       choices=["ppl", "loss"],
                       help='评估指标: ppl (perplexity) 或 loss (cross entropy loss) (默认: ppl)')
    parser.add_argument('--device', type=str, default="cuda",
                       help='设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("🔧 配置参数:")
    print(f"   模型路径: {args.model_path}")
    print(f"   初始稀疏度: {args.initial_sparsity}")
    print(f"   步长: {args.step_size}")
    print(f"   目标稀疏度: {args.target_sparsity}")
    print(f"   验证样本数: {args.num_samples}")
    print(f"   最大序列长度: {args.max_length}")
    print(f"   Residual长度: {args.residual_length}")
    print(f"   评估指标: {args.eval_metric}")
    print(f"   输出目录: {args.output_dir}")
    print()
    
    # 创建配置
    config = GreedySearchConfig(
        model_path=args.model_path,
        initial_sparsity=args.initial_sparsity,
        step_size=args.step_size,
        target_sparsity=args.target_sparsity,
        num_samples=args.num_samples,
        max_length=args.max_length,
        residual_length=args.residual_length,
        output_dir=args.output_dir,
        device=args.device,
        eval_metric=args.eval_metric
    )
    
    # 创建分析器
    analyzer = GreedyLayerSensitivityAnalyzer(config)
    
    # 执行搜索
    start_time = time.time()
    sparsity_config, sensitivity = analyzer.search()
    elapsed_time = time.time() - start_time
    
    # 打印结果摘要
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    print("\n🔹 Sparsity Configuration (top 10 most sparse):")
    sorted_layers = sorted(sparsity_config.items(), key=lambda x: x[1], reverse=True)
    for layer_idx, sparsity in sorted_layers[:10]:
        print(f"   Layer {layer_idx:2d}: {sparsity:.3f}")
    
    print("\n🔹 Sparsity Configuration (top 10 least sparse):")
    sorted_layers_asc = sorted(sparsity_config.items(), key=lambda x: x[1])
    for layer_idx, sparsity in sorted_layers_asc[:10]:
        print(f"   Layer {layer_idx:2d}: {sparsity:.3f}")
    
    print("\n🔹 Sensitivity Scores (top 10 most sensitive):")
    sorted_sens = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    for layer_idx, sens in sorted_sens[:10]:
        print(f"   Layer {layer_idx:2d}: {sens:.4f}")
    
    print(f"\n⏱️  Total time: {elapsed_time/60:.2f} minutes")
    
    # 保存结果
    analyzer.save_results(sparsity_config, sensitivity)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
