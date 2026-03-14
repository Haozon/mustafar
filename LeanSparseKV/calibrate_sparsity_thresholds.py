#!/usr/bin/env python3
"""
LeanSparseKV Threshold Calibration using TEAL-style Statistical Method

This script implements TEAL's proven statistical approach for threshold calibration:
1. Load real datasets with concatenated text + overflowing tokens
2. Collect attention scores from multiple batches
3. Global flatten all scores into single 1D array
4. Direct percentile calculation with outlier removal
5. Target percentiles: [95, 80, 50, 20] for 5-level sparsity

This approach fixes the overfitting problem by using stable statistical methods
instead of optimization algorithms.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# For dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Only built-in datasets will work.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TEALStyleThresholdCalibrator:
    """
    TEAL-Style Threshold Calibration for LeanSparseKV
    
    Uses TEAL's proven statistical approach:
    1. Real dataset loading with concatenated text + overflowing tokens
    2. Global flatten of all attention scores across batches
    3. Direct percentile calculation with outlier removal
    4. Target percentiles: [95, 80, 50, 20] for 5-level sparsity
    """
    
    def __init__(self, 
                 model_path: str,
                 target_sparsity: float = 0.70,
                 granularity: str = "per_layer",
                 device: str = "auto",
                 outlier_threshold: float = 0.01):
        """
        Initialize the calibrator using TEAL's approach
        
        Args:
            model_path: Path to the model
            target_sparsity: Target average sparsity (default: 0.70)
            granularity: "per_layer" or "per_head"
            device: Device to use ("auto", "cuda", "cpu")
            outlier_threshold: Outlier removal threshold (default: 0.01, keep 98% data)
        """
        self.model_path = model_path
        self.target_sparsity = target_sparsity
        self.granularity = granularity
        self.outlier_threshold = outlier_threshold
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # 修正的目标分布：更合理的5级分配
        # 目标：[5%, 15%, 30%, 30%, 20%] -> 平均约70%稀疏度
        self.target_distribution = np.array([0.05, 0.15, 0.30, 0.30, 0.20])
        # 不再使用错误的分位数，改为直接索引计算
        self.target_percentiles = None  # 废弃，使用新的计算方法
        
        logger.info(f"Target distribution: {self.target_distribution}")
        logger.info(f"Target percentiles: {self.target_percentiles}")
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            use_cache=False,  # Disable KV cache to save memory
        )
        self.model.eval()
        
        # Additional memory optimizations
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")
        
        # Set model to not store intermediate activations
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Model info - handle different architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA, Mistral, etc.
            self.num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2, GPT-Neo, etc.
            self.num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert.encoder, 'layer'):
            # BERT-like models
            self.num_layers = len(self.model.bert.encoder.layer)
        else:
            # Fallback: try to infer from config
            if hasattr(self.model.config, 'num_hidden_layers'):
                self.num_layers = self.model.config.num_hidden_layers
            elif hasattr(self.model.config, 'n_layer'):
                self.num_layers = self.model.config.n_layer
            else:
                raise ValueError(f"Cannot determine number of layers for model type: {type(self.model)}")
        
        self.num_heads = self.model.config.num_attention_heads
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads")
        
        # Storage for collected scores - TEAL style global flatten
        self.all_scores = {}  # {layer_id: [all_flattened_scores]}
        self.thresholds = {}  # {layer_id: (α_h, α_mh, α_m, α_ml)}
        
    def compute_diffkv_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute DiffKV importance scores with correct causal attention handling
        
        Args:
            attention_weights: [B, H, T_q, T_k] attention weights (already softmax normalized)
            
        Returns:
            importance_scores: [B, H, T_k] probability-based importance scores
        """
        B, H, T_q, T_k = attention_weights.shape
        
        # Create causal mask to identify valid query-key pairs
        # In causal attention, query i can only attend to keys 0, 1, ..., i
        causal_mask = torch.tril(torch.ones(T_q, T_k, device=attention_weights.device, dtype=torch.bool))
        
        # Count how many queries can attend to each key
        # Key k can be attended by queries k, k+1, ..., T_q-1
        valid_queries_per_key = causal_mask.sum(dim=0).float()  # [T_k]
        
        # Ensure we don't have zero counts (handle edge cases)
        valid_queries_per_key = torch.clamp(valid_queries_per_key, min=1)
        
        # Mask out invalid attention weights (should be 0 due to causal mask anyway)
        masked_attention = attention_weights * causal_mask[None, None, :, :]
        
        # Sum attention for each key across all valid queries
        importance_sum = masked_attention.sum(dim=2)  # [B, H, T_k]
        
        # Divide by the actual number of queries that can attend to each key
        # This gives the average attention each key receives from valid queries
        importance = importance_sum / valid_queries_per_key[None, None, :]  # [B, H, T_k]
        
        # Apply T_k scaling for proper normalization
        # Physical meaning: average probability that this key token is attended to
        # by the queries that can actually see it (respecting causal constraint), scaled by T_k
        importance = importance * T_k
        return importance

    def load_teal_style_dataset(self, 
                               dataset_name: str = "wikitext",
                               num_samples: int = 300,
                               seq_len: int = 2048) -> str:
        """
        Load dataset using TEAL's approach: concatenated text + overflowing tokens
        
        Args:
            dataset_name: Dataset to use
            num_samples: Number of samples to concatenate
            seq_len: Sequence length for tokenization
            
        Returns:
            Concatenated text string
        """
        logger.info(f"Loading {dataset_name} dataset with TEAL-style concatenation...")
        
        if dataset_name == "wikitext" and DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                logger.info(f"WikiText dataset loaded, total items: {len(dataset)}")
                
                # TEAL-style: concatenate all text into one long string
                text = ""
                valid_samples = 0
                for item in dataset:
                    article_text = item["text"].strip()
                    if article_text and len(article_text) > 100:  # Skip very short articles
                        text += article_text + "\n\n"
                        valid_samples += 1
                        if valid_samples >= num_samples:
                            break
                
                logger.info(f"Concatenated {valid_samples} articles, total length: {len(text)} chars")
                return text
                
            except Exception as e:
                logger.error(f"Error loading WikiText: {e}")
                # Fallback to simple text
                
        # Fallback: create simple repeated text for testing
        logger.warning(f"Using fallback text for dataset {dataset_name}")
        simple_text = "The quick brown fox jumps over the lazy dog. " * 1000
        return simple_text * (num_samples // 10 + 1)

    def collect_teal_style_scores(self, 
                                 dataset_name: str = "wikitext",
                                 num_samples: int = 300,
                                 seq_len: int = 2048,
                                 batch_size: int = 10) -> None:
        """
        Collect attention scores using TEAL's approach with memory optimization:
        1. Concatenated text + overflowing tokens
        2. Multiple batches processing
        3. Global flatten all scores per layer
        4. Memory-efficient processing with gradient checkpointing
        
        Args:
            dataset_name: Dataset to use
            num_samples: Number of samples to concatenate
            seq_len: Sequence length
            batch_size: Batch size for processing
        """
        logger.info("🔄 Collecting scores using TEAL-style approach (memory optimized)...")
        
        # Step 1: Load concatenated text (TEAL approach)
        text = self.load_teal_style_dataset(dataset_name, num_samples, seq_len)
        
        # Step 2: Tokenize with overflowing tokens (TEAL approach)
        logger.info("Tokenizing with overflowing tokens...")
        encodings = self.tokenizer(
            text, 
            truncation=True, 
            return_tensors="pt", 
            max_length=seq_len,
            return_overflowing_tokens=True,
            padding="max_length"
        )
        
        # Memory optimization: Process smaller batches
        total_batches = min(batch_size, encodings.input_ids.size(0))
        logger.info(f"Processing {total_batches} batches for memory efficiency")
        
        # Initialize storage for global flatten (TEAL approach)
        if self.granularity == "per_layer":
            for layer_id in range(self.num_layers):
                self.all_scores[layer_id] = []
        else:  # per_head (not recommended, but supported)
            for layer_id in range(self.num_layers):
                for head_id in range(self.num_heads):
                    self.all_scores[(layer_id, head_id)] = []
        
        # Step 3: Memory-efficient batch processing
        logger.info("Forward pass and score collection (memory optimized)...")
        
        # Process batches one by one to reduce memory usage
        for batch_idx in range(0, total_batches, 2):  # Process 2 samples at a time
            end_idx = min(batch_idx + 2, total_batches)
            current_batch_size = end_idx - batch_idx
            
            input_ids = encodings.input_ids[batch_idx:end_idx, :].to(self.device)
            logger.info(f"Processing batch {batch_idx//2 + 1}/{(total_batches + 1)//2}, shape: {input_ids.shape}")
            
            with torch.no_grad():
                # Memory optimization: Enable gradient checkpointing if available
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                
                # Forward pass with attention output
                outputs = self.model(input_ids, output_attentions=True)
                attentions = outputs.attentions  # List of [B, H, T, T] tensors
                
                # Process each layer immediately and free memory
                for layer_id, attention_weights in enumerate(attentions):
                    # Compute importance scores
                    importance_scores = self.compute_diffkv_importance(attention_weights)  # [B, H, T]
                    
                    if self.granularity == "per_layer":
                        # TEAL approach: global flatten across heads and batches
                        layer_scores = importance_scores.mean(dim=1)  # [B, T] - average across heads
                        
                        flat_scores = layer_scores.flatten().cpu().numpy()  # Move to CPU immediately
                        self.all_scores[layer_id].extend(flat_scores)
                    else:  # per_head
                        # Store per-head scores (also globally flattened)
                        for head_id in range(self.num_heads):
                            head_scores = importance_scores[:, head_id, :]  # [B, T]
                            flat_scores = head_scores.flatten().cpu().numpy()  # Move to CPU immediately
                            self.all_scores[(layer_id, head_id)].extend(flat_scores)
                    
                    # Free GPU memory immediately after processing each layer
                    del attention_weights, importance_scores
                
                # Free memory after each batch
                del outputs, attentions
                torch.cuda.empty_cache()  # Force GPU memory cleanup
        
        # Log collection results
        for key in self.all_scores:
            logger.info(f"Collected {len(self.all_scores[key])} scores for layer/head {key}")

    def compute_teal_style_thresholds(self) -> None:
        """
        Compute thresholds using TEAL's statistical approach:
        1. Global flatten of all scores
        2. Outlier removal (keep 98% data)
        3. Direct percentile calculation
        4. No optimization algorithms - pure statistics
        """
        logger.info("🎯 Computing thresholds using TEAL-style statistics...")
        for key, scores in self.all_scores.items():
            if len(scores) == 0:
                logger.warning(f"No scores collected for {key}, skipping...")
                continue
            
            # Convert to numpy array
            scores = np.array(scores)
            logger.info(f"Processing {key}: {len(scores)} scores")
            logger.info(f"  Score range: [{np.min(scores):.6f}, {np.max(scores):.6f}]")
            logger.info(f"  Score mean: {np.mean(scores):.6f}")
            logger.info(f"  Score std: {np.std(scores):.6f}")
            
            # TEAL-style outlier removal
            scores_sorted = np.sort(scores)
            lower_idx = int(self.outlier_threshold * len(scores_sorted))
            upper_idx = len(scores_sorted) - int(self.outlier_threshold * len(scores_sorted))
            
            if upper_idx <= lower_idx:
                logger.error(f"Too few scores after outlier removal for {key}")
                self.thresholds[key] = (1e-3, 1e-4, 1e-5, 1e-6)
                continue
                
            clean_scores = scores_sorted[lower_idx:upper_idx]
            logger.info(f"  After outlier removal: {len(clean_scores)} scores")
            logger.info(f"  Cleaned range: [{np.min(clean_scores):.6f}, {np.max(clean_scores):.6f}]")
            
            try:
                # 修正的阈值计算：直接根据目标分布计算索引
                # 目标分布：[0.05, 0.15, 0.30, 0.30, 0.20]
                # 累积分布：[0.05, 0.20, 0.50, 0.80, 1.00]
                
                logger.info(f"  Using corrected threshold calculation based on target distribution")
                
                # 降序排列scores（重要性从高到低）
                clean_scores_desc = np.sort(clean_scores)[::-1]
                total_tokens = len(clean_scores_desc)
                
                # 根据目标分布计算阈值索引
                idx_5_percent = max(0, int(0.05 * total_tokens) - 1)      # 前5%的边界
                idx_20_percent = max(0, int(0.20 * total_tokens) - 1)     # 前20%的边界  
                idx_50_percent = max(0, int(0.50 * total_tokens) - 1)     # 前50%的边界
                idx_80_percent = max(0, int(0.80 * total_tokens) - 1)     # 前80%的边界
                
                # 从降序排列的scores中取阈值
                α_h = clean_scores_desc[idx_5_percent]    # 0%稀疏度阈值（前5%）
                α_mh = clean_scores_desc[idx_20_percent]  # 50%稀疏度阈值（前20%）
                α_m = clean_scores_desc[idx_50_percent]   # 70%稀疏度阈值（前50%）
                α_ml = clean_scores_desc[idx_80_percent]  # 90%稀疏度阈值（前80%）
                
                logger.info(f"  Threshold indices: 5%={idx_5_percent}, 20%={idx_20_percent}, 50%={idx_50_percent}, 80%={idx_80_percent}")
                logger.info(f"  Raw thresholds: α_h={α_h:.8f}, α_mh={α_mh:.8f}, α_m={α_m:.8f}, α_ml={α_ml:.8f}")
                
                # Ensure thresholds are positive and in descending order
                α_h = max(α_h, 1e-8)
                α_mh = max(α_mh, 1e-9)
                α_m = max(α_m, 1e-10)
                α_ml = max(α_ml, 1e-11)
                
                # Ensure strict descending order
                if α_mh >= α_h:
                    α_mh = α_h * 0.9
                if α_m >= α_mh:
                    α_m = α_mh * 0.9
                if α_ml >= α_m:
                    α_ml = α_m * 0.9
                
                self.thresholds[key] = (float(α_h), float(α_mh), float(α_m), float(α_ml))
                
                logger.info(f"{key}: α_h={α_h:.8f}, α_mh={α_mh:.8f}, α_m={α_m:.8f}, α_ml={α_ml:.8f}")
                
                # Verify achieved distribution
                sparsity_counts = [
                    np.sum(clean_scores >= α_h),
                    np.sum((clean_scores >= α_mh) & (clean_scores < α_h)),
                    np.sum((clean_scores >= α_m) & (clean_scores < α_mh)),
                    np.sum((clean_scores >= α_ml) & (clean_scores < α_m)),
                    np.sum(clean_scores < α_ml)
                ]
                achieved_distribution = np.array(sparsity_counts) / len(clean_scores)
                
                logger.info(f"  Achieved distribution: {achieved_distribution}")
                logger.info(f"  Target distribution: {self.target_distribution}")
                
                # Calculate achieved sparsity
                sparsity_levels = [0.0, 0.5, 0.7, 0.9, 1.0]
                achieved_sparsity = np.sum(np.array(sparsity_levels) * achieved_distribution)
                logger.info(f"  Achieved sparsity: {achieved_sparsity:.3f} (target: {self.target_sparsity:.3f})")
                
            except Exception as e:
                logger.error(f"Error computing thresholds for {key}: {e}")
                self.thresholds[key] = (1e-3, 1e-4, 1e-5, 1e-6)

    def save_results(self, output_dir: str) -> None:
        """
        Save calibration results to files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save thresholds
        thresholds_data = {
            'model_path': self.model_path,
            'target_sparsity': self.target_sparsity,
            'granularity': self.granularity,
            'sparsity_levels': [0.0, 0.5, 0.7, 0.9, 1.0],  # Fixed sparsity levels
            'target_distribution': self.target_distribution.tolist(),
            'target_percentiles': self.target_percentiles,
            'outlier_threshold': self.outlier_threshold,
            'method': 'TEAL-style statistical approach',
            'thresholds': {}
        }
        
        for key, (α_h, α_mh, α_m, α_ml) in self.thresholds.items():
            if self.granularity == "per_layer":
                layer_id = key
                thresholds_data['thresholds'][f'layer_{layer_id}'] = {
                    'alpha_h': α_h,
                    'alpha_mh': α_mh,
                    'alpha_m': α_m,
                    'alpha_ml': α_ml
                }
            else:  # per_head
                layer_id, head_id = key
                thresholds_data['thresholds'][f'layer_{layer_id}_head_{head_id}'] = {
                    'alpha_h': α_h,
                    'alpha_mh': α_mh,
                    'alpha_m': α_m,
                    'alpha_ml': α_ml
                }
        
        thresholds_file = os.path.join(output_dir, 'thresholds.json')
        with open(thresholds_file, 'w') as f:
            json.dump(thresholds_data, f, indent=2)
        logger.info(f"Thresholds saved to {thresholds_file}")


def load_calibration_data(dataset_name: str, 
                         num_samples: int = 200, 
                         min_length: int = 512,
                         max_length: int = 2048) -> List[str]:
    """
    Load calibration dataset (kept for compatibility, but TEAL-style approach is preferred)
    
    Args:
        dataset_name: Name of the dataset ("math", "gsm8k", "alpaca", "wikitext")
        num_samples: Number of samples to load
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        List of text samples
    """
    logger.info(f"Loading {dataset_name} dataset...")
    
    texts = []
    
    if dataset_name == "wikitext" and DATASETS_AVAILABLE:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            logger.info(f"WikiText dataset loaded, total items: {len(dataset)}")
            
            valid_texts = 0
            for item in dataset:
                text = item["text"].strip()
                # Skip empty lines and very short articles
                if text and len(text) >= min_length:
                    # Only keep articles that are within our length range
                    if len(text) <= max_length:
                        texts.append(text)
                        valid_texts += 1
                    else:
                        # For very long articles, take the first part
                        truncated_text = text[:max_length]
                        # Try to end at a sentence boundary
                        last_period = truncated_text.rfind('.')
                        if last_period > min_length:
                            truncated_text = truncated_text[:last_period + 1]
                        texts.append(truncated_text)
                        valid_texts += 1
                    
                    if len(texts) >= num_samples:
                        break
                        
            logger.info(f"Selected {valid_texts} valid WikiText articles")
            
        except Exception as e:
            logger.error(f"Error loading WikiText: {e}")
            # Fallback to simple text
            
    if len(texts) == 0:
        # Fallback: create simple repeated text for testing
        logger.warning(f"Using fallback text for dataset {dataset_name}")
        simple_text = "The quick brown fox jumps over the lazy dog. This is a sample text for calibration. " * 100
        texts = [simple_text] * num_samples
    
    if len(texts) < num_samples:
        logger.warning(f"Only found {len(texts)} samples, requested {num_samples}")
    
    logger.info(f"Loaded {len(texts)} calibration samples")
    return texts[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="LeanSparseKV TEAL-Style Threshold Calibration")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="LeanSparseKV/calibration_results",
                       help="Output directory for results")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "alpaca"],
                       help="Calibration dataset")
    parser.add_argument("--num_samples", type=int, default=300,
                       help="Number of samples to concatenate (TEAL-style)")
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--granularity", type=str, default="per_layer",
                       choices=["per_layer", "per_head"],
                       help="Threshold granularity")
    parser.add_argument("--target_sparsity", type=float, default=0.70,
                       help="Target average sparsity")
    parser.add_argument("--outlier_threshold", type=float, default=0.01,
                       help="Outlier removal threshold (TEAL-style)")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = vars(args)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Initialize TEAL-style calibrator
    calibrator = TEALStyleThresholdCalibrator(
        model_path=config['model_path'],
        target_sparsity=config['target_sparsity'],
        granularity=config['granularity'],
        outlier_threshold=config['outlier_threshold']
    )
    
    # Run TEAL-style calibration
    logger.info("🚀 Starting TEAL-style threshold calibration...")
    logger.info("📊 Method: Global flatten + direct percentile calculation")
    logger.info("🎯 Target: Fix overfitting problem with stable statistics")
    
    # Step 1: Collect scores using TEAL approach
    calibrator.collect_teal_style_scores(
        dataset_name=config['dataset'],
        num_samples=config['num_samples'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size']
    )
    
    # Step 2: Compute thresholds using TEAL statistics
    calibrator.compute_teal_style_thresholds()
    
    # Step 3: Save results
    calibrator.save_results(config['output_dir'])
    
    logger.info("✅ TEAL-style threshold calibration completed successfully!")
    logger.info("🔍 Next step: Run validation to check if overfitting is fixed")


if __name__ == "__main__":
    main()