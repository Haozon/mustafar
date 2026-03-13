"""
Global Threshold Manager for DiffSparseKV

This module implements the global threshold management system that establishes
consistent 3-level sparsity classification across all layers.

Requirements addressed: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import torch
import numpy as np
import json
import os
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import warnings


class GlobalThresholdManager:
    """
    Manages global sparsity thresholds for consistent 3-level classification across all layers.
    
    The threshold manager computes global thresholds based on flattened importance scores
    from all layers and maintains these thresholds for consistent processing.
    """
    
    def __init__(self, 
                 target_distribution: List[float] = [0.05, 0.75, 0.20],
                 sparsity_levels: List[float] = [0.0, 0.7, 1.0]):
        """
        Initialize the global threshold manager.
        
        Args:
            target_distribution: Target distribution for 3 levels [Level 0, Level 1, Level 2]
                               Default: [5%, 75%, 20%] corresponding to [0%, 70%, 100%] sparsity
            sparsity_levels: Sparsity rates for each level [0%, 70%, 100%]
        """
        if len(target_distribution) != 3 or len(sparsity_levels) != 3:
            raise ValueError("Both target_distribution and sparsity_levels must have exactly 3 elements")
        
        if abs(sum(target_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Target distribution must sum to 1.0, got {sum(target_distribution)}")
        
        self.target_distribution = np.array(target_distribution)
        self.sparsity_levels = np.array(sparsity_levels)
        
        # Validate sparsity levels are in ascending order
        if not np.all(np.diff(self.sparsity_levels) >= 0):
            raise ValueError("Sparsity levels must be in ascending order")
        
        # Computed thresholds
        self.threshold_high = None  # Level 0 vs Level 1 boundary
        self.threshold_low = None   # Level 1 vs Level 2 boundary
        
        # Statistics for validation
        self.actual_distribution = None
        self.total_tokens = 0
        
    def compute_per_layer_thresholds(self, importance_scores: torch.Tensor) -> Tuple[float, float]:
        """
        Compute per-layer thresholds for 3-level classification.
        
        This method computes thresholds for a SINGLE layer based on its importance scores.
        Each layer will have its own thresholds, allowing for layer-specific sparsity patterns.
        
        Args:
            importance_scores: Importance scores tensor for ONE layer
                             Shape: [B, H, T] for single layer
                          
        Returns:
            (threshold_high, threshold_low): Thresholds for level boundaries
                                           threshold_high: Level 0 vs Level 1 (top 5%)
                                           threshold_low: Level 1 vs Level 2 (top 80%)
                                           
        Raises:
            ValueError: If input tensor has invalid shape or contains invalid values
        """
        # Validate input - should be 3D for single layer
        if importance_scores.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, H, T] for per-layer threshold, got {importance_scores.dim()}D")
        
        # Flatten importance scores for this layer
        flattened_importance = importance_scores.flatten()
        
        # Validate input
        if flattened_importance.numel() == 0:
            raise ValueError("Empty importance tensor provided")
        
        if torch.any(torch.isnan(flattened_importance)) or torch.any(torch.isinf(flattened_importance)):
            raise ValueError("Importance scores contain NaN or infinite values")
        
        if torch.any(flattened_importance < 0):
            warnings.warn("Negative importance scores detected. This may indicate an issue with importance calculation.")
        
        # Sort importance scores in descending order
        sorted_importance, _ = torch.sort(flattened_importance, descending=True)
        n_tokens = sorted_importance.numel()
        
        # Compute cumulative percentiles based on target distribution
        # Level 0: top 5% (0% to 5%)
        # Level 1: next 75% (5% to 80%) 
        # Level 2: bottom 20% (80% to 100%)
        
        percentile_5 = int(self.target_distribution[0] * n_tokens)  # 5%
        percentile_80 = int((self.target_distribution[0] + self.target_distribution[1]) * n_tokens)  # 80%
        
        # Ensure indices are within bounds
        percentile_5 = max(0, min(percentile_5, n_tokens - 1))
        percentile_80 = max(0, min(percentile_80, n_tokens - 1))
        
        # Extract thresholds
        # threshold_high: boundary between Level 0 (0% sparsity) and Level 1 (70% sparsity)
        # threshold_low: boundary between Level 1 (70% sparsity) and Level 2 (100% sparsity)
        
        if percentile_5 < n_tokens:
            threshold_high = sorted_importance[percentile_5].item()
        else:
            threshold_high = sorted_importance[-1].item()
        
        if percentile_80 < n_tokens:
            threshold_low = sorted_importance[percentile_80].item()
        else:
            threshold_low = sorted_importance[-1].item()
        
        # Ensure threshold_high >= threshold_low (higher importance = higher threshold)
        if threshold_high < threshold_low:
            # This can happen with very uniform distributions
            # Use a small epsilon to separate them
            epsilon = (sorted_importance[0] - sorted_importance[-1]).item() * 1e-6
            threshold_high = threshold_low + epsilon
        
        return threshold_high, threshold_low
    
    def compute_global_thresholds(self, all_importance: torch.Tensor) -> Tuple[float, float]:
        """
        DEPRECATED: Use compute_per_layer_thresholds instead.
        
        This method is kept for backward compatibility but should not be used.
        Per-layer thresholds provide better layer-specific adaptation.
        """
        warnings.warn("compute_global_thresholds is deprecated. Use compute_per_layer_thresholds instead.", 
                     DeprecationWarning)
        
        # Handle different input formats
        if isinstance(all_importance, list):
            # List of per-layer tensors [B, H, T]
            flattened_importance = torch.cat([tensor.flatten() for tensor in all_importance])
        elif all_importance.dim() == 4:
            # Multi-dimensional tensor [B, H, num_layers, T]
            flattened_importance = all_importance.flatten()
        elif all_importance.dim() == 3:
            # Single layer tensor [B, H, T]
            flattened_importance = all_importance.flatten()
        elif all_importance.dim() == 1:
            # Already flattened
            flattened_importance = all_importance
        else:
            raise ValueError(f"Unsupported importance tensor shape: {all_importance.shape}")
        
        # Validate input
        if flattened_importance.numel() == 0:
            raise ValueError("Empty importance tensor provided")
        
        if torch.any(torch.isnan(flattened_importance)) or torch.any(torch.isinf(flattened_importance)):
            raise ValueError("Importance scores contain NaN or infinite values")
        
        if torch.any(flattened_importance < 0):
            warnings.warn("Negative importance scores detected. This may indicate an issue with importance calculation.")
        
        # Sort importance scores in descending order
        sorted_importance, _ = torch.sort(flattened_importance, descending=True)
        n_tokens = sorted_importance.numel()
        self.total_tokens = n_tokens
        
        # Compute cumulative percentiles based on target distribution
        # Level 0: top 5% (0% to 5%)
        # Level 1: next 75% (5% to 80%) 
        # Level 2: bottom 20% (80% to 100%)
        
        percentile_5 = int(self.target_distribution[0] * n_tokens)  # 5%
        percentile_80 = int((self.target_distribution[0] + self.target_distribution[1]) * n_tokens)  # 80%
        
        # Ensure indices are within bounds
        percentile_5 = max(0, min(percentile_5, n_tokens - 1))
        percentile_80 = max(0, min(percentile_80, n_tokens - 1))
        
        # Extract thresholds
        # threshold_high: boundary between Level 0 (0% sparsity) and Level 1 (70% sparsity)
        # threshold_low: boundary between Level 1 (70% sparsity) and Level 2 (100% sparsity)
        
        if percentile_5 < n_tokens:
            self.threshold_high = sorted_importance[percentile_5].item()
        else:
            self.threshold_high = sorted_importance[-1].item()
        
        if percentile_80 < n_tokens:
            self.threshold_low = sorted_importance[percentile_80].item()
        else:
            self.threshold_low = sorted_importance[-1].item()
        
        # Ensure threshold_high >= threshold_low (higher importance = higher threshold)
        if self.threshold_high < self.threshold_low:
            # This can happen with very uniform distributions
            # Use a small epsilon to separate them
            epsilon = (sorted_importance[0] - sorted_importance[-1]).item() * 1e-6
            self.threshold_high = self.threshold_low + epsilon
        
        # Compute actual distribution for validation
        self._compute_actual_distribution(flattened_importance)
        
        return self.threshold_high, self.threshold_low
    
    def _compute_actual_distribution(self, importance_scores: torch.Tensor) -> None:
        """
        Compute the actual distribution achieved by the computed thresholds.
        
        Args:
            importance_scores: Flattened importance scores
        """
        if self.threshold_high is None or self.threshold_low is None:
            return
        
        # Count tokens in each level
        level_0_count = torch.sum(importance_scores >= self.threshold_high).item()
        level_1_count = torch.sum((importance_scores >= self.threshold_low) & 
                                 (importance_scores < self.threshold_high)).item()
        level_2_count = torch.sum(importance_scores < self.threshold_low).item()
        
        total = importance_scores.numel()
        
        if total > 0:
            self.actual_distribution = np.array([
                level_0_count / total,
                level_1_count / total,
                level_2_count / total
            ])
        else:
            self.actual_distribution = np.array([0.0, 0.0, 0.0])
    
    def classify_importance_scores(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Classify importance scores into 3 sparsity levels using computed thresholds.
        
        Args:
            importance_scores: Importance scores tensor of any shape
            
        Returns:
            sparsity_levels: Tensor with same shape as input, containing sparsity level indices (0, 1, 2)
            
        Raises:
            RuntimeError: If thresholds have not been computed yet
        """
        if self.threshold_high is None or self.threshold_low is None:
            raise RuntimeError("Thresholds have not been computed. Call compute_global_thresholds() first.")
        
        # Initialize with Level 2 (highest sparsity)
        levels = torch.full_like(importance_scores, 2, dtype=torch.long)
        
        # Assign Level 1 (medium sparsity)
        levels[importance_scores >= self.threshold_low] = 1
        
        # Assign Level 0 (no sparsity)
        levels[importance_scores >= self.threshold_high] = 0
        
        return levels
    
    def get_sparsity_for_levels(self, levels: torch.Tensor) -> torch.Tensor:
        """
        Convert sparsity level indices to actual sparsity rates.
        
        Args:
            levels: Tensor containing sparsity level indices (0, 1, 2)
            
        Returns:
            sparsity_rates: Tensor with same shape containing sparsity rates [0.0, 0.7, 1.0]
        """
        sparsity_map = torch.tensor(self.sparsity_levels, dtype=torch.float32, device=levels.device)
        return sparsity_map[levels]
    
    def save_thresholds(self, filepath: str) -> None:
        """
        Save computed thresholds to file for persistence.
        
        Args:
            filepath: Path to save thresholds JSON file
        """
        if self.threshold_high is None or self.threshold_low is None:
            raise RuntimeError("No thresholds to save. Compute thresholds first.")
        
        threshold_data = {
            "threshold_high": float(self.threshold_high),
            "threshold_low": float(self.threshold_low),
            "target_distribution": self.target_distribution.tolist(),
            "actual_distribution": self.actual_distribution.tolist() if self.actual_distribution is not None else None,
            "sparsity_levels": self.sparsity_levels.tolist(),
            "total_tokens": self.total_tokens,
            "distribution_error": float(np.mean(np.abs(self.actual_distribution - self.target_distribution))) 
                                if self.actual_distribution is not None else None
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(threshold_data, f, indent=2)
    
    def load_thresholds(self, filepath: str) -> None:
        """
        Load thresholds from file.
        
        Args:
            filepath: Path to thresholds JSON file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Threshold file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            threshold_data = json.load(f)
        
        # Validate required fields
        required_fields = ["threshold_high", "threshold_low", "target_distribution", "sparsity_levels"]
        for field in required_fields:
            if field not in threshold_data:
                raise ValueError(f"Missing required field in threshold file: {field}")
        
        self.threshold_high = float(threshold_data["threshold_high"])
        self.threshold_low = float(threshold_data["threshold_low"])
        self.target_distribution = np.array(threshold_data["target_distribution"])
        self.sparsity_levels = np.array(threshold_data["sparsity_levels"])
        
        # Optional fields
        if "actual_distribution" in threshold_data and threshold_data["actual_distribution"] is not None:
            self.actual_distribution = np.array(threshold_data["actual_distribution"])
        
        if "total_tokens" in threshold_data:
            self.total_tokens = int(threshold_data["total_tokens"])
    
    def get_threshold_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the computed thresholds.
        
        Returns:
            Dictionary containing threshold statistics
        """
        if self.threshold_high is None or self.threshold_low is None:
            return {"status": "no_thresholds_computed"}
        
        stats = {
            "threshold_high": float(self.threshold_high),
            "threshold_low": float(self.threshold_low),
            "target_distribution": self.target_distribution.tolist(),
            "sparsity_levels": self.sparsity_levels.tolist(),
            "total_tokens": self.total_tokens
        }
        
        if self.actual_distribution is not None:
            stats.update({
                "actual_distribution": self.actual_distribution.tolist(),
                "distribution_error": float(np.mean(np.abs(self.actual_distribution - self.target_distribution))),
                "max_distribution_error": float(np.max(np.abs(self.actual_distribution - self.target_distribution))),
                "distribution_errors_per_level": (self.actual_distribution - self.target_distribution).tolist()
            })
        
        return stats
    
    def validate_thresholds(self, tolerance: float = 0.05) -> Dict[str, Any]:
        """
        Validate that computed thresholds achieve target distribution within tolerance.
        
        Args:
            tolerance: Maximum allowed deviation from target distribution
            
        Returns:
            Dictionary containing validation results
        """
        if self.actual_distribution is None:
            return {"valid": False, "reason": "No actual distribution computed"}
        
        distribution_errors = np.abs(self.actual_distribution - self.target_distribution)
        max_error = np.max(distribution_errors)
        mean_error = np.mean(distribution_errors)
        
        is_valid = max_error <= tolerance
        
        return {
            "valid": is_valid,
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "tolerance": tolerance,
            "distribution_errors": distribution_errors.tolist(),
            "target_distribution": self.target_distribution.tolist(),
            "actual_distribution": self.actual_distribution.tolist()
        }


def create_threshold_manager(target_distribution: List[float] = [0.05, 0.75, 0.20],
                           sparsity_levels: List[float] = [0.0, 0.7, 1.0]) -> GlobalThresholdManager:
    """
    Factory function to create a global threshold manager.
    
    Args:
        target_distribution: Target distribution for 3 levels
        sparsity_levels: Sparsity rates for each level
        
    Returns:
        GlobalThresholdManager: Configured threshold manager instance
    """
    return GlobalThresholdManager(target_distribution=target_distribution, 
                                sparsity_levels=sparsity_levels)