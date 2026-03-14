"""
Global Threshold Manager for DiffSparseKV

This module implements the global threshold management system that establishes
consistent multi-level sparsity classification across all layers.

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
    Manages global sparsity thresholds for consistent multi-level classification across all layers.
    
    The threshold manager computes global thresholds based on flattened importance scores
    from all layers and maintains these thresholds for consistent processing.
    """
    
    def __init__(self, 
                 target_distribution: List[float] = [0.05, 0.75, 0.20],
                 sparsity_levels: List[float] = [0.0, 0.7, 1.0]):
        """
        Initialize the global threshold manager.
        
        Args:
            target_distribution: Target distribution for each level.
            sparsity_levels: Sparsity rates for each level.
        """
        if len(target_distribution) != len(sparsity_levels):
            raise ValueError("target_distribution and sparsity_levels must have the same length")
        if len(target_distribution) < 2:
            raise ValueError("At least 2 sparsity levels are required")
        
        if abs(sum(target_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Target distribution must sum to 1.0, got {sum(target_distribution)}")
        
        self.target_distribution = np.array(target_distribution)
        self.sparsity_levels = np.array(sparsity_levels)
        
        # Validate sparsity levels are in ascending order
        if not np.all(np.diff(self.sparsity_levels) >= 0):
            raise ValueError("Sparsity levels must be in ascending order")
        
        # Computed thresholds
        self.thresholds = None
        self.threshold_high = None  # Backward-compatible alias for first boundary
        self.threshold_low = None   # Backward-compatible alias for second boundary
        
        # Statistics for validation
        self.actual_distribution = None
        self.total_tokens = 0
        
    def compute_per_layer_thresholds(self, importance_scores: torch.Tensor) -> List[float]:
        """
        Compute per-layer thresholds for multi-level classification.
        
        This method computes thresholds for a SINGLE layer based on its importance scores.
        Each layer will have its own thresholds, allowing for layer-specific sparsity patterns.
        
        Args:
            importance_scores: Importance scores tensor for ONE layer
                             Shape: [B, H, T] for single layer
                          
        Returns:
            thresholds: Threshold list of length len(levels) - 1.
                                           
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
        
        thresholds = []
        cumulative = np.cumsum(self.target_distribution)[:-1]
        for percentile in cumulative:
            index = int(percentile * n_tokens)
            index = max(0, min(index, n_tokens - 1))
            thresholds.append(sorted_importance[index].item())
        
        if thresholds:
            epsilon = max((sorted_importance[0] - sorted_importance[-1]).item() * 1e-6, 1e-12)
            for idx in range(len(thresholds) - 1):
                if thresholds[idx] < thresholds[idx + 1]:
                    thresholds[idx] = thresholds[idx + 1] + epsilon
        
        self._set_threshold_aliases(thresholds)
        return thresholds
    
    def compute_global_thresholds(self, all_importance: torch.Tensor) -> List[float]:
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
        
        thresholds = []
        cumulative = np.cumsum(self.target_distribution)[:-1]
        for percentile in cumulative:
            index = int(percentile * n_tokens)
            index = max(0, min(index, n_tokens - 1))
            thresholds.append(sorted_importance[index].item())
        
        if thresholds:
            epsilon = max((sorted_importance[0] - sorted_importance[-1]).item() * 1e-6, 1e-12)
            for idx in range(len(thresholds) - 1):
                if thresholds[idx] < thresholds[idx + 1]:
                    thresholds[idx] = thresholds[idx + 1] + epsilon
        
        self.thresholds = thresholds
        self._set_threshold_aliases(thresholds)
        
        # Compute actual distribution for validation
        self._compute_actual_distribution(flattened_importance)
        
        return thresholds

    def _set_threshold_aliases(self, thresholds: List[float]) -> None:
        self.thresholds = thresholds
        self.threshold_high = thresholds[0] if thresholds else None
        self.threshold_low = thresholds[1] if len(thresholds) > 1 else None
    
    def _compute_actual_distribution(self, importance_scores: torch.Tensor) -> None:
        """
        Compute the actual distribution achieved by the computed thresholds.
        
        Args:
            importance_scores: Flattened importance scores
        """
        if self.thresholds is None:
            return

        levels = self.classify_importance_scores(importance_scores)
        total = levels.numel()
        
        if total > 0:
            self.actual_distribution = np.array([
                torch.sum(levels == level).item() / total
                for level in range(len(self.sparsity_levels))
            ])
        else:
            self.actual_distribution = np.zeros(len(self.sparsity_levels))
    
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
        if self.thresholds is None:
            raise RuntimeError("Thresholds have not been computed. Call compute_global_thresholds() first.")

        # Start from the highest sparsity level and overwrite with denser classes.
        highest_level = len(self.sparsity_levels) - 1
        levels = torch.full_like(importance_scores, highest_level, dtype=torch.long)
        for level, threshold in enumerate(self.thresholds):
            levels[importance_scores >= threshold] = level
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
        if self.thresholds is None:
            raise RuntimeError("No thresholds to save. Compute thresholds first.")
        
        threshold_data = {
            "thresholds": [float(x) for x in self.thresholds],
            "threshold_high": float(self.threshold_high) if self.threshold_high is not None else None,
            "threshold_low": float(self.threshold_low) if self.threshold_low is not None else None,
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
        required_fields = ["target_distribution", "sparsity_levels"]
        for field in required_fields:
            if field not in threshold_data:
                raise ValueError(f"Missing required field in threshold file: {field}")

        self.target_distribution = np.array(threshold_data["target_distribution"])
        self.sparsity_levels = np.array(threshold_data["sparsity_levels"])
        if "thresholds" in threshold_data:
            thresholds = [float(x) for x in threshold_data["thresholds"]]
        else:
            thresholds = []
            if threshold_data.get("threshold_high") is not None:
                thresholds.append(float(threshold_data["threshold_high"]))
            if threshold_data.get("threshold_low") is not None:
                thresholds.append(float(threshold_data["threshold_low"]))
        self._set_threshold_aliases(thresholds)
        
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
        if self.thresholds is None:
            return {"status": "no_thresholds_computed"}
        
        stats = {
            "thresholds": [float(x) for x in self.thresholds],
            "threshold_high": float(self.threshold_high) if self.threshold_high is not None else None,
            "threshold_low": float(self.threshold_low) if self.threshold_low is not None else None,
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
