#!/usr/bin/env python3
"""
DiffKV Threshold Validation System

This script validates the computed thresholds by testing them on evaluation datasets
and measuring the actual sparsity distribution and model performance.

Usage:
    python validate_thresholds.py --thresholds_file thresholds.json --model_path /path/to/model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt

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

class DiffKVThresholdValidator:
    """
    Validates DiffKV thresholds on evaluation datasets
    """
    
    def __init__(self, 
                 thresholds_file: str,
                 model_path: str,
                 device: str = "auto"):
        """
        Initialize validator
        
        Args:
            thresholds_file: Path to thresholds JSON file
            model_path: Path to the model
            device: Device for inference
        """
        self.thresholds_file = thresholds_file
        self.model_path = model_path
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load thresholds
        with open(thresholds_file, 'r') as f:
            threshold_data = json.load(f)
        
        self.target_sparsity = threshold_data['target_sparsity']
        
        # Handle missing sparsity_levels field (use default 5-level sparsity)
        if 'sparsity_levels' in threshold_data:
            self.sparsity_levels = np.array(threshold_data['sparsity_levels'])
        else:
            # Default 5-level sparsity: [0%, 50%, 70%, 90%, 100%]
            self.sparsity_levels = np.array([0.0, 0.5, 0.7, 0.9, 1.0])
            logger.info("Using default sparsity levels: [0%, 50%, 70%, 90%, 100%]")
        
        # Handle missing target_distribution field
        if 'target_distribution' in threshold_data:
            self.target_distribution = np.array(threshold_data['target_distribution'])
        else:
            # Default distribution
            self.target_distribution = np.array([0.05, 0.15, 0.30, 0.30, 0.20])
            logger.info("Using default target distribution: [0.05, 0.15, 0.30, 0.30, 0.20]")
        
        self.granularity = threshold_data['granularity']
        
        # Parse thresholds
        self.thresholds = {}
        for key_str, threshold_dict in threshold_data['thresholds'].items():
            # Handle different key formats from calibration script
            if self.granularity == "per_layer":
                # Key format: "layer_X" -> convert to int
                if key_str.startswith('layer_'):
                    layer_id = int(key_str.split('_')[1])
                    key = layer_id
                else:
                    # Fallback: try to convert directly
                    key = int(key_str)
            else:  # per_head
                # Key format: "layer_X_head_Y" -> convert to tuple
                if 'layer_' in key_str and 'head_' in key_str:
                    parts = key_str.split('_')
                    layer_id = int(parts[1])
                    head_id = int(parts[3])
                    key = (layer_id, head_id)
                else:
                    # Fallback: try to eval as tuple
                    key = eval(key_str)
            
            self.thresholds[key] = (
                threshold_dict['alpha_h'],
                threshold_dict['alpha_mh'],
                threshold_dict['alpha_m'],
                threshold_dict['alpha_ml']
            )
        
        logger.info(f"Loaded thresholds for {len(self.thresholds)} {'layers' if self.granularity == 'per_layer' else 'heads'}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Model info
        self.num_layers = len(self.model.model.layers)
        self.num_heads = self.model.config.num_attention_heads
        
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
    
    def assign_sparsity_levels(self, importance_scores: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        Assign sparsity levels based on importance scores and thresholds
        
        Args:
            importance_scores: [B, H, T] importance scores
            layer_id: Layer index
            
        Returns:
            sparsity_map: [B, H, T] sparsity level assignments
        """
        B, H, T = importance_scores.shape
        
        if self.granularity == "per_layer":

            α_h, α_mh, α_m, α_ml = self.thresholds[layer_id]
            
            # Apply same thresholds to all heads in this layer
            sparsity_map = torch.zeros_like(importance_scores)
            
            # 5-level classification
            mask_0 = importance_scores >= α_h      # Level 0: 0% sparsity
            mask_1 = (importance_scores >= α_mh) & (importance_scores < α_h)   # Level 1: 50% sparsity
            mask_2 = (importance_scores >= α_m) & (importance_scores < α_mh)   # Level 2: 70% sparsity
            mask_3 = (importance_scores >= α_ml) & (importance_scores < α_m)   # Level 3: 90% sparsity
            mask_4 = importance_scores < α_ml      # Level 4: 100% sparsity (pruned)
            
            sparsity_map[mask_0] = self.sparsity_levels[0]  # 0%
            sparsity_map[mask_1] = self.sparsity_levels[1]  # 50%
            sparsity_map[mask_2] = self.sparsity_levels[2]  # 70%
            sparsity_map[mask_3] = self.sparsity_levels[3]  # 90%
            sparsity_map[mask_4] = self.sparsity_levels[4]  # 100%
            
        else:  # per_head
            sparsity_map = torch.zeros_like(importance_scores)
            
            for head_id in range(H):
                α_h, α_mh, α_m, α_ml = self.thresholds[(layer_id, head_id)]
                head_scores = importance_scores[:, head_id, :]  # [B, T]
                
                # 5-level classification for this head
                mask_0 = head_scores >= α_h
                mask_1 = (head_scores >= α_mh) & (head_scores < α_h)
                mask_2 = (head_scores >= α_m) & (head_scores < α_mh)
                mask_3 = (head_scores >= α_ml) & (head_scores < α_m)
                mask_4 = head_scores < α_ml
                
                sparsity_map[:, head_id, :][mask_0] = self.sparsity_levels[0]
                sparsity_map[:, head_id, :][mask_1] = self.sparsity_levels[1]
                sparsity_map[:, head_id, :][mask_2] = self.sparsity_levels[2]
                sparsity_map[:, head_id, :][mask_3] = self.sparsity_levels[3]
                sparsity_map[:, head_id, :][mask_4] = self.sparsity_levels[4]
        
        return sparsity_map
    
    def validate_on_dataset(self, 
                           validation_texts: List[str],
                           max_length: int = 2048,
                           batch_size: int = 1) -> Dict:
        """
        Validate thresholds on evaluation dataset
        
        Args:
            validation_texts: List of validation text samples
            max_length: Maximum sequence length
            batch_size: Batch size for processing (default 1 for no padding)
            
        Returns:
            validation_results: Dictionary with validation statistics
        """
        logger.info(f"Validating on {len(validation_texts)} samples...")
        
        # Statistics collection
        all_sparsity_maps = []
        layer_stats = {i: {'distributions': [], 'avg_sparsities': []} for i in range(self.num_layers)}
        
        # Process samples individually or in small batches without padding
        for i in tqdm(range(0, len(validation_texts), batch_size), desc="Validating"):
            batch_texts = validation_texts[i:i+batch_size]
            
            # Process each text individually to avoid padding
            for text in batch_texts:
                # Tokenize single text without padding
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False  # No padding
                ).to(self.device)
                
                # Skip if sequence is too short
                seq_len = inputs['input_ids'].shape[1]
                if seq_len < 10:  # Skip very short sequences
                    continue
                
                # Forward pass with attention output
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                    attentions = outputs.attentions  # List of [1, H, T, T]
                
                # Process each layer
                sample_sparsity_maps = []
                for layer_id, attention_weights in enumerate(attentions):
                    # Compute importance scores
                    importance_scores = self.compute_diffkv_importance(attention_weights)  # [1, H, T]
                    
                    # Apply cross-head averaging (same as calibrate)
                    if self.granularity == "per_layer":
                        # Average across heads to match calibrate behavior
                        layer_scores = importance_scores.mean(dim=1)  # [1, T]
                        
                        # Debug: Log score statistics to verify consistency with calibrate
                        if layer_id == 0 and i == 0 and text == batch_texts[0]:  # Only log once
                            logger.info(f"Validate - Layer {layer_id} score stats:")
                            logger.info(f"  Sequence length: {seq_len}")
                            logger.info(f"  Original importance_scores shape: {importance_scores.shape}")
                            logger.info(f"  After head averaging shape: {layer_scores.shape}")
                            logger.info(f"  Score range: [{layer_scores.min():.6f}, {layer_scores.max():.6f}]")
                            logger.info(f"  Score mean: {layer_scores.mean():.6f}")
                        
                        # Expand back to [1, H, T] for assign_sparsity_levels compatibility
                        layer_scores_expanded = layer_scores.unsqueeze(1).expand(-1, importance_scores.shape[1], -1)
                        sparsity_map = self.assign_sparsity_levels(layer_scores_expanded, layer_id)
                    else:  # per_head
                        sparsity_map = self.assign_sparsity_levels(importance_scores, layer_id)
                    
                    sample_sparsity_maps.append(sparsity_map.cpu().numpy())
                    
                    # Collect statistics (only from actual tokens, no padding)
                    sparsity_flat = sparsity_map.view(-1).cpu().numpy()
                    avg_sparsity = np.mean(sparsity_flat)
                    
                    # Count distribution
                    distribution = []
                    for level in self.sparsity_levels:
                        count = np.sum(sparsity_flat == level)
                        distribution.append(count / len(sparsity_flat))
                    
                    layer_stats[layer_id]['distributions'].append(distribution)
                    layer_stats[layer_id]['avg_sparsities'].append(avg_sparsity)
                
                all_sparsity_maps.append(sample_sparsity_maps)
                
                # Clear GPU memory
                del outputs, attentions
                torch.cuda.empty_cache()
        
        # Aggregate statistics
        validation_results = {
            'overall_stats': {},
            'layer_stats': {},
            'distribution_analysis': {}
        }
        
        # Overall statistics
        all_avg_sparsities = []
        all_distributions = []
        
        for layer_id in range(self.num_layers):
            layer_avg_sparsity = np.mean(layer_stats[layer_id]['avg_sparsities'])
            layer_avg_distribution = np.mean(layer_stats[layer_id]['distributions'], axis=0)
            
            all_avg_sparsities.append(layer_avg_sparsity)
            all_distributions.append(layer_avg_distribution)
            
            validation_results['layer_stats'][layer_id] = {
                'avg_sparsity': layer_avg_sparsity,
                'avg_distribution': layer_avg_distribution.tolist(),
                'sparsity_std': np.std(layer_stats[layer_id]['avg_sparsities']),
                'target_sparsity': self.target_sparsity,
                'sparsity_error': abs(layer_avg_sparsity - self.target_sparsity)
            }
        
        # Overall results
        overall_avg_sparsity = np.mean(all_avg_sparsities)
        overall_avg_distribution = np.mean(all_distributions, axis=0)
        
        validation_results['overall_stats'] = {
            'avg_sparsity': overall_avg_sparsity,
            'avg_distribution': overall_avg_distribution.tolist(),
            'target_sparsity': self.target_sparsity,
            'target_distribution': self.target_distribution.tolist(),
            'sparsity_error': abs(overall_avg_sparsity - self.target_sparsity),
            'distribution_error': np.mean(np.abs(overall_avg_distribution - self.target_distribution)),
            'sparsity_std_across_layers': np.std(all_avg_sparsities)
        }
        
        # Distribution analysis
        validation_results['distribution_analysis'] = {
            'layer_consistency': {
                'sparsity_cv': np.std(all_avg_sparsities) / np.mean(all_avg_sparsities),
                'distribution_variance': np.var(all_distributions, axis=0).tolist()
            }
        }
        
        return validation_results
    
    def create_validation_report(self, validation_results: Dict, output_dir: str) -> None:
        """
        Create validation report with visualizations
        
        Args:
            validation_results: Results from validation
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert validation results to JSON serializable format
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.ndarray, np.generic)):
                if obj.ndim == 0:  # scalar
                    return float(obj)
                else:  # array
                    return obj.astype(float).tolist()
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        validation_results_serializable = convert_to_serializable(validation_results)
        
        # Save validation results
        results_file = os.path.join(output_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(validation_results_serializable, f, indent=2)
        
        logger.info(f"Validation results saved to {results_file}")
        
        # Create main validation report
        self._create_main_validation_plots(validation_results, output_dir)
        
        # Create per-layer distribution analysis
        self._create_per_layer_distribution_plots(validation_results, output_dir)
        
        # Print summary
        overall_stats = validation_results['overall_stats']
        layers = list(range(self.num_layers))
        sparsity_errors = [validation_results['layer_stats'][i]['sparsity_error'] for i in layers]
        
        logger.info("🎯 Validation Summary:")
        logger.info(f"  Overall sparsity: {overall_stats['avg_sparsity']:.3f} (target: {overall_stats['target_sparsity']:.3f})")
        logger.info(f"  Sparsity error: {overall_stats['sparsity_error']:.3f}")
        logger.info(f"  Distribution error: {overall_stats['distribution_error']:.3f}")
        logger.info(f"  Layers within ±2%: {sum(1 for e in sparsity_errors if e <= 0.02)}/{len(sparsity_errors)}")
    
    def _create_main_validation_plots(self, validation_results: Dict, output_dir: str) -> None:
        """Create main validation plots"""
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DiffKV Threshold Validation Results', fontsize=16)
        
        # Plot 1: Sparsity by layer
        layers = list(range(self.num_layers))
        layer_sparsities = [validation_results['layer_stats'][i]['avg_sparsity'] for i in layers]
        target_line = [self.target_sparsity] * len(layers)
        
        axes[0, 0].plot(layers, layer_sparsities, 'o-', label='Actual Sparsity', color='blue')
        axes[0, 0].plot(layers, target_line, '--', label=f'Target ({self.target_sparsity:.1%})', color='red')
        axes[0, 0].fill_between(layers, 
                               [self.target_sparsity - 0.02] * len(layers),
                               [self.target_sparsity + 0.02] * len(layers),
                               alpha=0.2, color='red', label='±2% tolerance')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Sparsity')
        axes[0, 0].set_title('Sparsity by Layer')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution comparison
        overall_stats = validation_results['overall_stats']
        x = np.arange(len(self.sparsity_levels))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, overall_stats['target_distribution'], width, 
                      label='Target', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, overall_stats['avg_distribution'], width,
                      label='Actual', alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('Sparsity Level')
        axes[0, 1].set_ylabel('Token Proportion')
        axes[0, 1].set_title('Overall Distribution Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'{int(s*100)}%' for s in self.sparsity_levels])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sparsity error by layer
        sparsity_errors = [validation_results['layer_stats'][i]['sparsity_error'] for i in layers]
        axes[1, 0].bar(layers, sparsity_errors, alpha=0.7, color='purple')
        axes[1, 0].axhline(y=0.02, color='red', linestyle='--', label='2% threshold')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Sparsity Error')
        axes[1, 0].set_title('Sparsity Error by Layer')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        summary_text = f"""Validation Summary:
        
Overall Average Sparsity: {overall_stats['avg_sparsity']:.3f}
Target Sparsity: {overall_stats['target_sparsity']:.3f}
Sparsity Error: {overall_stats['sparsity_error']:.3f}

Distribution Error: {overall_stats['distribution_error']:.3f}
Layer Consistency (CV): {validation_results['distribution_analysis']['layer_consistency']['sparsity_cv']:.3f}

Layers within ±2%: {sum(1 for e in sparsity_errors if e <= 0.02)}/{len(sparsity_errors)}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "validation_report.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Main validation report saved to {plot_file}")
    
    def _create_per_layer_distribution_plots(self, validation_results: Dict, output_dir: str) -> None:
        """Create per-layer distribution analysis plots"""
        layers = list(range(self.num_layers))
        
        # Determine grid size for subplots
        n_layers = len(layers)
        cols = min(4, n_layers)  # Maximum 4 columns
        rows = (n_layers + cols - 1) // cols  # Ceiling division
        
        # Create figure for per-layer distributions
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Per-Layer Token Distribution Analysis', fontsize=16)
        
        # Handle single row/column cases
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot distribution for each layer
        x = np.arange(len(self.sparsity_levels))
        width = 0.35
        
        for i, layer_id in enumerate(layers):
            if i >= len(axes):
                break
                
            layer_stats = validation_results['layer_stats'][layer_id]
            layer_distribution = layer_stats['avg_distribution']
            
            # Plot target vs actual distribution for this layer
            axes[i].bar(x - width/2, self.target_distribution, width, 
                       label='Target', alpha=0.8, color='skyblue')
            axes[i].bar(x + width/2, layer_distribution, width,
                       label='Actual', alpha=0.8, color='lightcoral')
            
            axes[i].set_xlabel('Sparsity Level')
            axes[i].set_ylabel('Token Proportion')
            axes[i].set_title(f'Layer {layer_id}\n(Sparsity: {layer_stats["avg_sparsity"]:.3f})')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([f'{int(s*100)}%' for s in self.sparsity_levels])
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            
            # Add distribution error as text
            dist_error = np.mean(np.abs(np.array(layer_distribution) - self.target_distribution))
            axes[i].text(0.02, 0.98, f'Dist Error: {dist_error:.3f}', 
                        transform=axes[i].transAxes, fontsize=8,
                        verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        per_layer_plot_file = os.path.join(output_dir, "per_layer_distributions.png")
        plt.savefig(per_layer_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-layer distribution analysis saved to {per_layer_plot_file}")
        
        # Create heatmap for distribution comparison across layers
        self._create_distribution_heatmap(validation_results, output_dir)
    
    def _create_distribution_heatmap(self, validation_results: Dict, output_dir: str) -> None:
        """Create heatmap showing distribution patterns across layers"""
        layers = list(range(self.num_layers))
        
        # Prepare data for heatmap
        layer_distributions = []
        layer_labels = []
        
        for layer_id in layers:
            layer_stats = validation_results['layer_stats'][layer_id]
            layer_distributions.append(layer_stats['avg_distribution'])
            layer_labels.append(f'Layer {layer_id}')
        
        # Add target distribution for comparison
        layer_distributions.append(self.target_distribution.tolist())
        layer_labels.append('Target')
        
        # Convert to numpy array
        distribution_matrix = np.array(layer_distributions)
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(layers) * 0.3)))
        fig.suptitle('Token Distribution Analysis Across Layers', fontsize=16)
        
        # Heatmap 1: Actual distributions
        im1 = ax1.imshow(distribution_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Token Distribution by Layer')
        ax1.set_xlabel('Sparsity Level')
        ax1.set_ylabel('Layer')
        ax1.set_xticks(range(len(self.sparsity_levels)))
        ax1.set_xticklabels([f'{int(s*100)}%' for s in self.sparsity_levels])
        ax1.set_yticks(range(len(layer_labels)))
        ax1.set_yticklabels(layer_labels)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Token Proportion')
        
        # Add text annotations
        for i in range(len(layer_labels)):
            for j in range(len(self.sparsity_levels)):
                text = ax1.text(j, i, f'{distribution_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Heatmap 2: Difference from target
        target_matrix = np.tile(self.target_distribution, (len(layers), 1))
        diff_matrix = distribution_matrix[:-1] - target_matrix  # Exclude target row
        
        im2 = ax2.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
        ax2.set_title('Difference from Target Distribution')
        ax2.set_xlabel('Sparsity Level')
        ax2.set_ylabel('Layer')
        ax2.set_xticks(range(len(self.sparsity_levels)))
        ax2.set_xticklabels([f'{int(s*100)}%' for s in self.sparsity_levels])
        ax2.set_yticks(range(len(layers)))
        ax2.set_yticklabels([f'Layer {i}' for i in layers])
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Difference from Target')
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(self.sparsity_levels)):
                color = "white" if abs(diff_matrix[i, j]) > 0.05 else "black"
                text = ax2.text(j, i, f'{diff_matrix[i, j]:+.2f}',
                               ha="center", va="center", color=color, fontsize=8)
        
        plt.tight_layout()
        heatmap_file = os.path.join(output_dir, "distribution_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution heatmap saved to {heatmap_file}")
        
        # Save detailed per-layer statistics
        self._save_per_layer_statistics(validation_results, output_dir)
    
    def _save_per_layer_statistics(self, validation_results: Dict, output_dir: str) -> None:
        """Save detailed per-layer statistics to CSV and text files"""
        layers = list(range(self.num_layers))
        
        # Prepare data for CSV
        csv_data = []
        for layer_id in layers:
            layer_stats = validation_results['layer_stats'][layer_id]
            row = {
                'Layer': layer_id,
                'Avg_Sparsity': layer_stats['avg_sparsity'],
                'Target_Sparsity': layer_stats['target_sparsity'],
                'Sparsity_Error': layer_stats['sparsity_error'],
                'Sparsity_Std': layer_stats['sparsity_std']
            }
            
            # Add distribution data
            for i, level in enumerate(self.sparsity_levels):
                row[f'Dist_{int(level*100)}%'] = layer_stats['avg_distribution'][i]
                row[f'Target_Dist_{int(level*100)}%'] = self.target_distribution[i]
                row[f'Dist_Error_{int(level*100)}%'] = abs(layer_stats['avg_distribution'][i] - self.target_distribution[i])
            
            csv_data.append(row)
        
        # Save to CSV
        import pandas as pd
        try:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, "per_layer_statistics.csv")
            df.to_csv(csv_file, index=False)
            logger.info(f"Per-layer statistics saved to {csv_file}")
        except ImportError:
            # Fallback: save as JSON if pandas not available
            json_file = os.path.join(output_dir, "per_layer_statistics.json")
            with open(json_file, 'w') as f:
                json.dump(csv_data, f, indent=2)
            logger.info(f"Per-layer statistics saved to {json_file}")
        
        # Create detailed text report
        text_report_file = os.path.join(output_dir, "per_layer_analysis.txt")
        with open(text_report_file, 'w') as f:
            f.write("DiffKV Per-Layer Distribution Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            overall_stats = validation_results['overall_stats']
            f.write(f"Overall Statistics:\n")
            f.write(f"  Target Sparsity: {overall_stats['target_sparsity']:.3f}\n")
            f.write(f"  Actual Sparsity: {overall_stats['avg_sparsity']:.3f}\n")
            f.write(f"  Sparsity Error: {overall_stats['sparsity_error']:.3f}\n")
            f.write(f"  Distribution Error: {overall_stats['distribution_error']:.3f}\n")
            f.write(f"  Layer Consistency (CV): {validation_results['distribution_analysis']['layer_consistency']['sparsity_cv']:.3f}\n\n")
            
            # Target distribution
            f.write("Target Distribution:\n")
            for i, level in enumerate(self.sparsity_levels):
                f.write(f"  {int(level*100)}% sparsity: {self.target_distribution[i]:.3f}\n")
            f.write("\n")
            
            # Per-layer details
            f.write("Per-Layer Analysis:\n")
            f.write("-" * 30 + "\n")
            
            for layer_id in layers:
                layer_stats = validation_results['layer_stats'][layer_id]
                f.write(f"\nLayer {layer_id}:\n")
                f.write(f"  Sparsity: {layer_stats['avg_sparsity']:.3f} (error: {layer_stats['sparsity_error']:.3f})\n")
                f.write(f"  Sparsity Std: {layer_stats['sparsity_std']:.3f}\n")
                f.write(f"  Distribution:\n")
                
                total_dist_error = 0
                for i, level in enumerate(self.sparsity_levels):
                    actual = layer_stats['avg_distribution'][i]
                    target = self.target_distribution[i]
                    error = abs(actual - target)
                    total_dist_error += error
                    f.write(f"    {int(level*100)}% sparsity: {actual:.3f} (target: {target:.3f}, error: {error:.3f})\n")
                
                f.write(f"  Total Distribution Error: {total_dist_error:.3f}\n")
            
            # Layer ranking by performance
            f.write("\n" + "=" * 50 + "\n")
            f.write("Layer Performance Ranking:\n")
            f.write("-" * 30 + "\n")
            
            # Sort layers by sparsity error
            layer_errors = [(i, validation_results['layer_stats'][i]['sparsity_error']) for i in layers]
            layer_errors.sort(key=lambda x: x[1])
            
            f.write("\nBy Sparsity Error (best to worst):\n")
            for rank, (layer_id, error) in enumerate(layer_errors, 1):
                f.write(f"  {rank:2d}. Layer {layer_id:2d}: {error:.3f}\n")
            
            # Sort layers by distribution error
            layer_dist_errors = []
            for i in layers:
                layer_stats = validation_results['layer_stats'][i]
                dist_error = np.mean(np.abs(np.array(layer_stats['avg_distribution']) - self.target_distribution))
                layer_dist_errors.append((i, dist_error))
            layer_dist_errors.sort(key=lambda x: x[1])
            
            f.write("\nBy Distribution Error (best to worst):\n")
            for rank, (layer_id, error) in enumerate(layer_dist_errors, 1):
                f.write(f"  {rank:2d}. Layer {layer_id:2d}: {error:.3f}\n")
        
        logger.info(f"Detailed per-layer analysis saved to {text_report_file}")

def load_validation_data(dataset_name: str, num_samples: int = 100) -> List[str]:
    """
    Load validation dataset
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples to load
        
    Returns:
        List of text samples
    """
    logger.info(f"Loading {num_samples} validation samples from {dataset_name}...")
    
    if dataset_name.lower() == "wikitext" and DATASETS_AVAILABLE:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            logger.info(f"WikiText validation dataset loaded, total items: {len(dataset)}")
            
            texts = []
            valid_texts = 0
            min_length = 200  # Minimum text length in characters
            max_length = 2000  # Maximum text length in characters
            
            for item in dataset:
                text = item["text"].strip()
                # Skip empty lines and very short articles
                if text and len(text) >= min_length:
                    # For long texts, take a reasonable chunk
                    if len(text) > max_length:
                        # Find a good breaking point (sentence end)
                        truncated_text = text[:max_length]
                        last_period = truncated_text.rfind('.')
                        if last_period > min_length:
                            text = truncated_text[:last_period + 1]
                        else:
                            text = truncated_text
                    
                    texts.append(text)
                    valid_texts += 1
                    if len(texts) >= num_samples:
                        break
                        
            logger.info(f"Selected {valid_texts} valid WikiText validation articles")
            logger.info(f"Text length range: {min([len(t) for t in texts])}-{max([len(t) for t in texts])} characters")
            logger.info(f"Average text length: {np.mean([len(t) for t in texts]):.1f} characters")
            
            if len(texts) > 0:
                return texts[:num_samples]
                
        except Exception as e:
            logger.error(f"Error loading WikiText: {e}")
            # Fall through to fallback
    
    elif dataset_name.lower() == "math":
        texts = [
            "Find the value of x in the equation 3x - 7 = 14. To solve this linear equation, we need to isolate x on one side. First, add 7 to both sides: 3x - 7 + 7 = 14 + 7, which gives us 3x = 21. Then divide both sides by 3: x = 21/3 = 7. Therefore, x = 7.",
            "Calculate the integral of x^2 from 0 to 3. The integral of x^2 is (x^3)/3 + C. To evaluate the definite integral, we use the fundamental theorem of calculus: ∫[0 to 3] x^2 dx = [(x^3)/3] evaluated from 0 to 3 = (3^3)/3 - (0^3)/3 = 27/3 - 0 = 9.",
            "What is the probability of rolling a sum of 7 with two dice? When rolling two standard six-sided dice, there are 6 × 6 = 36 possible outcomes. The ways to get a sum of 7 are: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). That's 6 favorable outcomes. So the probability is 6/36 = 1/6 ≈ 0.167 or about 16.7%.",
            "Solve the quadratic equation x^2 - 5x + 6 = 0. We can factor this equation by finding two numbers that multiply to 6 and add to -5. Those numbers are -2 and -3. So we can write: (x - 2)(x - 3) = 0. This gives us x - 2 = 0 or x - 3 = 0, so x = 2 or x = 3.",
            "Find the slope of the line passing through (2, 3) and (5, 9). The slope formula is m = (y2 - y1)/(x2 - x1). Using the points (2, 3) and (5, 9): m = (9 - 3)/(5 - 2) = 6/3 = 2. Therefore, the slope is 2."
        ] * (num_samples // 5 + 1)
        return texts[:num_samples]
    
    elif dataset_name.lower() == "gsm8k":
        texts = [
            "A store sells pencils for $0.25 each and erasers for $0.75 each. If Sarah buys 8 pencils and 3 erasers, how much does she spend in total? First, let's calculate the cost of pencils: 8 × $0.25 = $2.00. Next, the cost of erasers: 3 × $0.75 = $2.25. The total cost is $2.00 + $2.25 = $4.25.",
            "Tom has 24 marbles. He gives away 1/3 of them to his sister and 1/4 of the remaining marbles to his brother. How many marbles does Tom have left? First, Tom gives 1/3 of 24 = 8 marbles to his sister. He has 24 - 8 = 16 marbles left. Then he gives 1/4 of 16 = 4 marbles to his brother. Finally, Tom has 16 - 4 = 12 marbles left.",
            "A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies? We can set up a proportion: 2.5 cups / 12 cookies = x cups / 30 cookies. Cross multiply: 2.5 × 30 = 12 × x, so 75 = 12x. Divide both sides by 12: x = 75/12 = 6.25 cups of flour."
        ] * (num_samples // 3 + 1)
        return texts[:num_samples]
    
    # Fallback: general text samples with more substantial content
    logger.warning(f"Using fallback text for dataset {dataset_name}")
    texts = [
        "The capital of France is Paris, which is known for its beautiful architecture, rich history, and cultural significance. The city sits on the Seine River and has been a major European city for over two millennia. Famous landmarks include the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which houses thousands of works of art including the Mona Lisa.",
        "Artificial intelligence is transforming many industries including healthcare, finance, transportation, and education. Machine learning algorithms can now diagnose diseases, detect fraud, optimize supply chains, and personalize learning experiences. However, these advances also raise important questions about privacy, job displacement, and the ethical use of AI systems.",
        "Climate change is one of the most pressing issues of our time, affecting weather patterns, sea levels, and ecosystems worldwide. Rising global temperatures are caused primarily by greenhouse gas emissions from human activities such as burning fossil fuels, deforestation, and industrial processes. Addressing this challenge requires coordinated international action and significant changes in how we produce and consume energy.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This complex network enables consciousness, memory, learning, and all cognitive functions. Neuroscientists continue to study how neural circuits process information and how brain plasticity allows us to adapt and learn throughout our lives.",
        "Renewable energy sources like solar and wind are becoming more cost-effective and efficient, making them increasingly competitive with traditional fossil fuels. Solar panel efficiency has improved dramatically while costs have fallen, and wind turbines are now capable of generating electricity even in low-wind conditions. These technologies are essential for reducing carbon emissions and achieving sustainable energy systems."
    ] * (num_samples // 5 + 1)
    return texts[:num_samples]

def main():
    parser = argparse.ArgumentParser(description="DiffKV Threshold Validation")
    parser.add_argument("--thresholds_file", type=str, required=True,
                       help="Path to thresholds JSON file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="validation_results",
                       help="Output directory for validation results")
    parser.add_argument("--dataset", type=str, default="math",
                       choices=["math", "gsm8k", "general", "wikitext"],
                       help="Validation dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of validation samples")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing (default 1 for no padding)")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DiffKVThresholdValidator(
        thresholds_file=args.thresholds_file,
        model_path=args.model_path
    )
    
    # Load validation data
    validation_texts = load_validation_data(args.dataset, args.num_samples)
    
    # Run validation
    logger.info("🔍 Starting threshold validation...")
    
    validation_results = validator.validate_on_dataset(
        validation_texts=validation_texts,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # Create validation report
    validator.create_validation_report(validation_results, args.output_dir)
    
    logger.info("✅ Threshold validation completed successfully!")
    logger.info(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()