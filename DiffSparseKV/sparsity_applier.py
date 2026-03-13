"""
Sparsity Classifier and Applier for DiffSparseKV

This module implements the sparsity classification and application system that
assigns tokens to 3 sparsity levels and applies magnitude-based pruning.

Requirements addressed: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import warnings


class SparsityClassifierApplier:
    """
    Classifies tokens into 3 sparsity levels and applies magnitude-based pruning.
    
    This class handles the complete sparsity application pipeline:
    1. Classifies tokens based on importance scores and thresholds
    2. Applies magnitude-based top-k selection for K and V tensors
    3. Ensures consistent sparsity application across K and V
    4. Handles complete removal for 100% sparsity level
    """
    
    def __init__(self, 
                 sparsity_levels: List[float] = [0.0, 0.7, 1.0],
                 preserve_shapes: bool = True):
        """
        Initialize the sparsity classifier and applier.
        
        Args:
            sparsity_levels: Sparsity rates for each level [Level 0, Level 1, Level 2]
                           Default: [0%, 70%, 100%]
            preserve_shapes: Whether to preserve tensor shapes (pad with zeros) or 
                           return sparse tensors with reduced dimensions
        """
        if len(sparsity_levels) != 3:
            raise ValueError("sparsity_levels must have exactly 3 elements")
        
        if not all(0.0 <= s <= 1.0 for s in sparsity_levels):
            raise ValueError("All sparsity levels must be between 0.0 and 1.0")
        
        if not all(sparsity_levels[i] <= sparsity_levels[i+1] for i in range(len(sparsity_levels)-1)):
            raise ValueError("Sparsity levels must be in non-decreasing order")
        
        self.sparsity_levels = sparsity_levels
        self.preserve_shapes = preserve_shapes
        
        # Statistics tracking
        self.last_classification_stats = None
    
    def classify_and_apply_sparsity(self, 
                                   importance_scores: torch.Tensor,
                                   key_states: torch.Tensor,
                                   value_states: torch.Tensor,
                                   thresholds: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify tokens and apply magnitude-based pruning to K and V tensors.
        
        This method implements the complete sparsity application pipeline:
        1. Classifies tokens into 3 levels based on importance and thresholds
        2. Applies magnitude-based top-k selection for each sparsity level
        3. Ensures K and V tensors receive identical sparsity patterns
        4. Handles complete removal for 100% sparsity tokens
        
        Args:
            importance_scores: Token importance scores of shape [B, H, T]
            key_states: Key tensor of shape [B, H, T, D]
            value_states: Value tensor of shape [B, H, T, D]
            thresholds: (threshold_high, threshold_low) for level boundaries
                       threshold_high: Level 0 vs Level 1 boundary
                       threshold_low: Level 1 vs Level 2 boundary
                       
        Returns:
            (pruned_keys, pruned_values): Sparsified K and V tensors with same shapes as input
                                        
        Raises:
            ValueError: If tensor shapes are incompatible or thresholds are invalid
        """
        # Input validation
        if importance_scores.dim() != 3:
            raise ValueError(f"importance_scores must be 3D [B, H, T], got {importance_scores.dim()}D")
        
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("key_states and value_states must be 4D [B, H, T, D]")
        B, H, T = importance_scores.shape
        B_k, H_k, T_k, D_k = key_states.shape
        B_v, H_v, T_v, D_v = value_states.shape
        
        if (B, H, T) != (B_k, H_k, T_k) or (B, H, T) != (B_v, H_v, T_v):
            raise ValueError(f"Shape mismatch: importance {(B, H, T)}, keys {(B_k, H_k, T_k)}, values {(B_v, H_v, T_v)}")
        
        if D_k != D_v:
            raise ValueError(f"Key and value dimensions must match: D_k={D_k}, D_v={D_v}")
        
        threshold_high, threshold_low = thresholds
        if threshold_high < threshold_low:
            raise ValueError(f"threshold_high ({threshold_high}) must be >= threshold_low ({threshold_low})")
        
        # Classify tokens into sparsity levels
        sparsity_levels = self._classify_tokens(importance_scores, thresholds)
        
        # Check if Level 2 has 100% sparsity - if so, use token eviction
        level_2_count = (sparsity_levels == 2).sum().item()
        
        if self.sparsity_levels[2] >= 1.0 and level_2_count > 0:
            # Use token eviction instead of magnitude pruning
            pruned_keys, pruned_values = self._apply_token_eviction(
                key_states, value_states, sparsity_levels
            )
        else:
            # Use original magnitude pruning for partial sparsity
            pruned_keys, pruned_values = self._apply_magnitude_pruning(
                key_states, value_states, sparsity_levels
            )
        
        # Update statistics
        self._update_classification_stats(sparsity_levels)
        
        return pruned_keys, pruned_values
    
    def _classify_tokens(self, importance_scores: torch.Tensor, 
                        thresholds: Tuple[float, float]) -> torch.Tensor:
        """
        Classify tokens into 3 sparsity levels based on importance and thresholds.
        
        Args:
            importance_scores: [B, H, T] importance scores
            thresholds: (threshold_high, threshold_low)
            
        Returns:
            sparsity_levels: [B, H, T] tensor with sparsity level indices (0, 1, 2)
        """
        threshold_high, threshold_low = thresholds
        
        # Initialize all tokens to Level 2 (highest sparsity)
        levels = torch.full_like(importance_scores, 2, dtype=torch.long)
        
        # Assign Level 1 (medium sparsity) 
        levels[importance_scores >= threshold_low] = 1
        
        # Assign Level 0 (no sparsity)
        levels[importance_scores >= threshold_high] = 0
        
        return levels
    
    def _apply_magnitude_pruning(self, 
                                key_states: torch.Tensor,
                                value_states: torch.Tensor,
                                sparsity_levels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply magnitude-based pruning to K and V tensors based on sparsity levels.
        
        Args:
            key_states: [B, H, T, D] key tensor
            value_states: [B, H, T, D] value tensor  
            sparsity_levels: [B, H, T] sparsity level indices
            
        Returns:
            (pruned_keys, pruned_values): Pruned tensors with same shapes
        """
        B, H, T, D = key_states.shape
        
        # Clone tensors to avoid modifying originals
        pruned_keys = key_states.clone()
        pruned_values = value_states.clone()
        
        # Process each sparsity level
        for level_idx, sparsity_rate in enumerate(self.sparsity_levels):
            # Find tokens assigned to this sparsity level
            level_mask = (sparsity_levels == level_idx)  # [B, H, T]
            
            if sparsity_rate == 0.0:
                # Level 0: No sparsity, keep all elements
                continue
            
            if not torch.any(level_mask):
                # No tokens at this level
                continue
            
            if sparsity_rate >= 1.0:
                # Level 2: 100% sparsity - complete removal
                pruned_keys[level_mask] = 0.0
                pruned_values[level_mask] = 0.0
            else:
                # Level 1: Partial sparsity - magnitude-based top-k selection
                self._apply_topk_pruning(pruned_keys, pruned_values, level_mask, sparsity_rate)
        
        return pruned_keys, pruned_values
    
    def _apply_token_eviction(self,
                             key_states: torch.Tensor,
                             value_states: torch.Tensor,
                             sparsity_levels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token-level eviction for 100% sparsity tokens.
        
        This method removes (evicts) tokens classified as Level 2 instead of zeroing them out.
        This avoids the softmax normalization issue where zero-valued tokens still consume
        attention weight.
        
        Args:
            key_states: [B, H, T, D] key tensor
            value_states: [B, H, T, D] value tensor
            sparsity_levels: [B, H, T] sparsity level indices
            
        Returns:
            (evicted_keys, evicted_values): Tensors with Level 2 tokens removed
                                           Shape: [B, H, T_kept, D] where T_kept < T
        """
        B, H, T, D = key_states.shape
        
        # Since all heads use the same classification (after aggregation),
        # we can use the first head's classification
        level_mask_per_token = sparsity_levels[0, 0, :]  # [T]
        
        # Find tokens to keep (Level 0 and Level 1)
        keep_mask = (level_mask_per_token != 2)  # [T]
        keep_indices = torch.where(keep_mask)[0]
        
        # Evict tokens by gathering only the kept ones
        evicted_keys = key_states[:, :, keep_indices, :]  # [B, H, T_kept, D]
        evicted_values = value_states[:, :, keep_indices, :]  # [B, H, T_kept, D]
        
        # Apply dimension-level sparsity to Level 1 tokens if any
        level_1_mask_kept = (level_mask_per_token[keep_indices] == 1)
        num_level_1 = level_1_mask_kept.sum().item()
        
        if num_level_1 > 0 and self.sparsity_levels[1] > 0.0:
            # Expand mask to [B, H, T_kept]
            level_1_mask_expanded = level_1_mask_kept.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
            
            # Apply top-k pruning to Level 1 tokens
            self._apply_topk_pruning(
                evicted_keys, evicted_values, 
                level_1_mask_expanded, 
                self.sparsity_levels[1]
            )
        
        return evicted_keys, evicted_values
    
    def _apply_topk_pruning(self, 
                           key_tensor: torch.Tensor,
                           value_tensor: torch.Tensor,
                           token_mask: torch.Tensor,
                           sparsity_rate: float) -> None:
        """
        Apply top-k magnitude pruning to selected tokens (in-place operation).
        K and V are pruned INDEPENDENTLY based on their own magnitudes.
        
        Vectorized implementation for better performance.
        
        Args:
            key_tensor: [B, H, T, D] key tensor (modified in-place)
            value_tensor: [B, H, T, D] value tensor (modified in-place)
            token_mask: [B, H, T] boolean mask for tokens to prune
            sparsity_rate: Fraction of elements to prune (0.0 to 1.0)
        """
        B, H, T, D = key_tensor.shape
        
        # Calculate number of elements to keep
        keep_ratio = 1.0 - sparsity_rate
        num_to_keep = max(1, int(keep_ratio * D))
        
        if num_to_keep >= D:
            return  # No pruning needed
        
        # Check if there are any tokens to process
        if not torch.any(token_mask):
            return
        
        # Expand mask to [B, H, T, D]
        token_mask_expanded = token_mask.unsqueeze(-1).expand(B, H, T, D)
        
        # === Vectorized K pruning ===
        # Compute magnitude for all tokens
        k_magnitude = torch.abs(key_tensor)  # [B, H, T, D]
        
        # For tokens not in mask, set magnitude to -inf so they won't be selected
        k_magnitude_masked = k_magnitude.clone()
        k_magnitude_masked[~token_mask_expanded] = float('-inf')
        
        # Find top-k indices for each token
        _, k_top_indices = torch.topk(k_magnitude_masked, num_to_keep, dim=-1, largest=True)
        # k_top_indices: [B, H, T, num_to_keep]
        
        # Create keep mask: [B, H, T, D]
        k_keep_mask = torch.zeros_like(key_tensor, dtype=torch.bool)
        k_keep_mask.scatter_(-1, k_top_indices, True)
        
        # Apply pruning: zero out elements not in top-k AND in token_mask
        prune_mask = token_mask_expanded & (~k_keep_mask)
        key_tensor[prune_mask] = 0.0
        
        # === Vectorized V pruning ===
        v_magnitude = torch.abs(value_tensor)  # [B, H, T, D]
        
        v_magnitude_masked = v_magnitude.clone()
        v_magnitude_masked[~token_mask_expanded] = float('-inf')
        
        _, v_top_indices = torch.topk(v_magnitude_masked, num_to_keep, dim=-1, largest=True)
        
        v_keep_mask = torch.zeros_like(value_tensor, dtype=torch.bool)
        v_keep_mask.scatter_(-1, v_top_indices, True)
        
        prune_mask = token_mask_expanded & (~v_keep_mask)
        value_tensor[prune_mask] = 0.0
    
    def _update_classification_stats(self, sparsity_levels: torch.Tensor) -> None:
        """Update classification statistics for monitoring."""
        total_tokens = sparsity_levels.numel()
        
        if total_tokens == 0:
            self.last_classification_stats = None
            return
        
        level_counts = []
        for level in range(3):
            count = torch.sum(sparsity_levels == level).item()
            level_counts.append(count)
        
        self.last_classification_stats = {
            "total_tokens": total_tokens,
            "level_counts": level_counts,
            "level_proportions": [count / total_tokens for count in level_counts],
            "average_sparsity": sum(self.sparsity_levels[i] * level_counts[i] for i in range(3)) / total_tokens
        }
    
    def get_classification_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics from the last classification operation.
        
        Returns:
            Dictionary with classification statistics or None if no classification performed
        """
        return self.last_classification_stats
    
    def apply_sparsity_to_tensor(self, 
                                tensor: torch.Tensor,
                                sparsity_level: int) -> torch.Tensor:
        """
        Apply sparsity to a single tensor based on sparsity level.
        
        Args:
            tensor: Input tensor of any shape ending with dimension D
            sparsity_level: Sparsity level index (0, 1, or 2)
            
        Returns:
            Sparsified tensor with same shape as input
        """
        if sparsity_level < 0 or sparsity_level >= len(self.sparsity_levels):
            raise ValueError(f"Invalid sparsity level {sparsity_level}. Must be 0, 1, or 2.")
        
        sparsity_rate = self.sparsity_levels[sparsity_level]
        
        if sparsity_rate == 0.0:
            # No sparsity
            return tensor.clone()
        elif sparsity_rate >= 1.0:
            # Complete sparsity
            return torch.zeros_like(tensor)
        else:
            # Partial sparsity - magnitude-based top-k
            return self._apply_topk_to_tensor(tensor, sparsity_rate)
    
    def _apply_topk_to_tensor(self, tensor: torch.Tensor, sparsity_rate: float) -> torch.Tensor:
        """Apply top-k magnitude pruning to a tensor."""
        original_shape = tensor.shape
        D = original_shape[-1]  # Last dimension
        
        # Reshape to [..., D] for easier processing
        reshaped = tensor.view(-1, D)
        N = reshaped.shape[0]
        
        # Calculate number of elements to keep
        keep_ratio = 1.0 - sparsity_rate
        num_to_keep = max(1, int(keep_ratio * D))
        
        if num_to_keep >= D:
            return tensor.clone()
        
        # Apply top-k to each vector
        result = torch.zeros_like(reshaped)
        
        for i in range(N):
            vector = reshaped[i, :]  # [D]
            magnitude = torch.abs(vector)
            
            # Find top-k elements
            _, top_indices = torch.topk(magnitude, num_to_keep, largest=True)
            
            # Keep only top-k elements
            result[i, top_indices] = vector[top_indices]
        
        # Reshape back to original shape
        return result.view(original_shape)
    
    def compute_actual_sparsity(self, tensor: torch.Tensor) -> float:
        """
        Compute the actual sparsity rate of a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Sparsity rate (fraction of zero elements)
        """
        total_elements = tensor.numel()
        if total_elements == 0:
            return 0.0
        
        zero_elements = torch.sum(tensor == 0.0).item()
        return zero_elements / total_elements
    
    def validate_sparsity_consistency(self, 
                                    key_states: torch.Tensor,
                                    value_states: torch.Tensor,
                                    tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate that K and V tensors have consistent sparsity patterns.
        
        Args:
            key_states: [B, H, T, D] key tensor
            value_states: [B, H, T, D] value tensor
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Dictionary with validation results
        """
        if key_states.shape != value_states.shape:
            return {
                "consistent": False,
                "reason": f"Shape mismatch: K {key_states.shape} vs V {value_states.shape}"
            }
        
        # Check if zero patterns match
        k_zero_mask = torch.abs(key_states) <= tolerance
        v_zero_mask = torch.abs(value_states) <= tolerance
        
        pattern_match = torch.all(k_zero_mask == v_zero_mask).item()
        
        if pattern_match:
            k_sparsity = torch.sum(k_zero_mask).item() / k_zero_mask.numel()
            v_sparsity = torch.sum(v_zero_mask).item() / v_zero_mask.numel()
            
            return {
                "consistent": True,
                "k_sparsity": k_sparsity,
                "v_sparsity": v_sparsity,
                "sparsity_difference": abs(k_sparsity - v_sparsity)
            }
        else:
            # Count mismatched elements
            mismatch_count = torch.sum(k_zero_mask != v_zero_mask).item()
            total_elements = k_zero_mask.numel()
            
            return {
                "consistent": False,
                "reason": "Sparsity patterns don't match",
                "mismatch_count": mismatch_count,
                "mismatch_rate": mismatch_count / total_elements
            }


def create_sparsity_applier(sparsity_levels: List[float] = [0.0, 0.7, 1.0],
                          preserve_shapes: bool = True) -> SparsityClassifierApplier:
    """
    Factory function to create a sparsity classifier and applier.
    
    Args:
        sparsity_levels: Sparsity rates for each level
        preserve_shapes: Whether to preserve tensor shapes
        
    Returns:
        SparsityClassifierApplier: Configured sparsity applier instance
    """
    return SparsityClassifierApplier(sparsity_levels=sparsity_levels, 
                                   preserve_shapes=preserve_shapes)