"""
DiffKV Importance Calculator

This module implements the core importance calculation for DiffSparseKV system.
It computes token importance scores using causal attention processing with
proper numerical stability.

Requirements addressed: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings


class DiffKVImportanceCalculator:
    """
    Calculates token importance scores using DiffKV method with causal attention processing.
    
    The importance calculation assumes attention weights already have causal masking applied,
    computes effective query counts, and applies proper T_k scaling for normalization.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize the importance calculator.
        
        Args:
            eps: Small epsilon value for numerical stability
        """
        self.eps = eps
    
    def compute_diffkv_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute DiffKV importance scores from pre-masked attention weights.
        
        Args:
            attention_weights: [B, H, T_q, T_k] - Pre-masked softmax attention weights
                             
        Returns:
            importance_scores: [B, H, T_k] - Average N-scaled attention per key token
        """
        B, H, T_q, T_k = attention_weights.shape
        device = attention_weights.device
        
        # Scale each query's attention by its sequence length N
        N_per_query = torch.arange(T_k - T_q + 1, T_k + 1, device=device, dtype=torch.float32)
        N_per_query = N_per_query.view(1, 1, T_q, 1)
        
        # Sum N-scaled attention across queries
        importance_sum = (attention_weights * N_per_query).sum(dim=2)
        
        # Count valid queries per key (causal constraint)
        valid_queries = torch.full((T_k,), float(T_q), device=device, dtype=torch.float32)
        if T_k > T_q:
            for j in range(T_q):
                valid_queries[T_k - T_q + j] = T_q - j
        
        # Average
        return importance_sum / valid_queries[None, None, :]
    
    def compute_importance_batch(self, attention_weights_list: list) -> list:
        """
        Compute importance scores for multiple layers in batch.
        
        Args:
            attention_weights_list: List of attention weight tensors, each of shape [B, H, T_q, T_k]
                                  Represents attention weights from different layers
                                  Note: Tensors can have different sequence lengths
                                  
        Returns:
            importance_scores: List of importance score tensors, each of shape [B, H, T_k]
        """
        if not attention_weights_list:
            raise ValueError("Empty attention weights list provided")
        
        importance_list = []
        for layer_idx, attention_weights in enumerate(attention_weights_list):
            try:
                layer_importance = self.compute_diffkv_importance(attention_weights)
                importance_list.append(layer_importance)
            except Exception as e:
                raise RuntimeError(f"Failed to compute importance for layer {layer_idx}: {str(e)}")
        
        return importance_list
    
    def validate_attention_weights(self, attention_weights: torch.Tensor) -> bool:
        """
        Validate that attention weights are properly formatted.
        
        Args:
            attention_weights: Tensor to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check basic properties
            if attention_weights.dim() != 4:
                return False
            
            # Check for invalid values
            if torch.any(torch.isnan(attention_weights)) or torch.any(torch.isinf(attention_weights)):
                return False
            
            # Check if values are in reasonable range for softmax output
            if torch.any(attention_weights < -self.eps) or torch.any(attention_weights > 1 + self.eps):
                return False
            
            return True
        except Exception:
            return False

    def compute_importance_batch_same_length(self, attention_weights_list: list) -> torch.Tensor:
        """
        Compute importance scores for multiple layers with same sequence length.
        
        Args:
            attention_weights_list: List of attention weight tensors, each of shape [B, H, T_q, T_k]
                                  All tensors must have the same sequence length
                                  
        Returns:
            importance_scores: Stacked importance scores of shape [num_layers, B, H, T_k]
        """
        if not attention_weights_list:
            raise ValueError("Empty attention weights list provided")
        
        # Verify all tensors have the same shape
        first_shape = attention_weights_list[0].shape
        for i, tensor in enumerate(attention_weights_list):
            if tensor.shape != first_shape:
                raise ValueError(f"All tensors must have the same shape. "
                               f"Tensor 0: {first_shape}, Tensor {i}: {tensor.shape}")
        
        importance_list = []
        for layer_idx, attention_weights in enumerate(attention_weights_list):
            try:
                layer_importance = self.compute_diffkv_importance(attention_weights)
                importance_list.append(layer_importance)
            except Exception as e:
                raise RuntimeError(f"Failed to compute importance for layer {layer_idx}: {str(e)}")
        
        # Stack along new dimension for layers
        return torch.stack(importance_list, dim=0)  # [num_layers, B, H, T_k]


def create_importance_calculator(eps: float = 1e-8) -> DiffKVImportanceCalculator:
    """
    Factory function to create a DiffKV importance calculator.
    
    Args:
        eps: Epsilon value for numerical stability
        
    Returns:
        DiffKVImportanceCalculator: Configured importance calculator instance
    """
    return DiffKVImportanceCalculator(eps=eps)