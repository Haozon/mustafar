"""
Configuration module for DiffSparseKV

Provides configuration classes and helper functions for setting up DiffSparseKV.
"""

from dataclasses import dataclass
from typing import List, Optional
from transformers import LlamaConfig


@dataclass
class DiffSparseKVConfig:
    """
    Configuration for DiffSparseKV system.
    
    Args:
        enable_diff_sparse: Whether to enable differential sparsity
        target_distribution: Target distribution for 3 sparsity levels [Level0, Level1, Level2]
        sparsity_levels: Sparsity values for each level [0.0, 0.7, 1.0]
        diff_sparse_window_size: Window size for dual-window mechanism
        obs_window_size: Observation window size for attention accumulation
        debug_diff_sparse: Enable debug logging
        use_flash_attention: Use Flash Attention for efficiency
    """
    enable_diff_sparse: bool = True
    target_distribution: List[float] = None
    sparsity_levels: List[float] = None
    diff_sparse_window_size: int = 128
    obs_window_size: int = 128
    debug_diff_sparse: bool = False
    use_flash_attention: bool = True
    level_2_mode: str = "evict"
    importance_mode: str = "attention_only"
    value_sink_keep: int = 2
    head_aggregation_mode: str = "mean"
    head_aggregation_alpha: float = 0.5
    head_disagreement_ratio: float = -1.0
<<<<<<< HEAD
    selector_mode: str = "diffsparse"
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
    
    def __post_init__(self):
        if self.target_distribution is None:
            self.target_distribution = [0.05, 0.75, 0.20]
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.0, 0.7, 1.0]
        
        # Validate
        assert len(self.target_distribution) == len(self.sparsity_levels), (
            "target_distribution and sparsity_levels must have the same length"
        )
        assert len(self.target_distribution) >= 2, "At least 2 sparsity levels are required"
        assert abs(sum(self.target_distribution) - 1.0) < 1e-6, "target_distribution must sum to 1.0"
        assert self.level_2_mode in {"evict", "zero"}, "level_2_mode must be 'evict' or 'zero'"
        assert self.importance_mode in {"attention_only", "value_aware"}, (
            "importance_mode must be 'attention_only' or 'value_aware'"
        )
        assert self.value_sink_keep >= 0, "value_sink_keep must be non-negative"
        assert self.head_aggregation_mode in {"mean", "max", "hybrid", "top2_mean"}, (
            "head_aggregation_mode must be one of mean/max/hybrid/top2_mean"
        )
        assert 0.0 <= self.head_aggregation_alpha <= 1.0, "head_aggregation_alpha must be in [0, 1]"
        assert self.head_disagreement_ratio == -1.0 or self.head_disagreement_ratio >= 1.0, (
            "head_disagreement_ratio must be -1 (disabled) or >= 1.0"
        )
<<<<<<< HEAD
        assert self.selector_mode in {"diffsparse", "snapkv"}, (
            "selector_mode must be 'diffsparse' or 'snapkv'"
        )
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
    
    def get_expected_sparsity(self) -> float:
        """Calculate expected average sparsity."""
        return sum(d * s for d, s in zip(self.target_distribution, self.sparsity_levels))


def create_diff_sparse_kv_config(
    base_config: LlamaConfig,
    enable_diff_sparse: bool = True,
    target_distribution: Optional[List[float]] = None,
    sparsity_levels: Optional[List[float]] = None,
    diff_sparse_window_size: int = 128,
    obs_window_size: int = 128,
    debug_diff_sparse: bool = False,
    use_flash_attention: bool = True,
    level_2_mode: str = "evict",
    importance_mode: str = "attention_only",
    value_sink_keep: int = 2,
    head_aggregation_mode: str = "mean",
    head_aggregation_alpha: float = 0.5,
    head_disagreement_ratio: float = -1.0,
<<<<<<< HEAD
    selector_mode: str = "diffsparse",
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
) -> LlamaConfig:
    """
    Create a LlamaConfig with DiffSparseKV settings.
    
    Args:
        base_config: Base LlamaConfig to extend
        enable_diff_sparse: Enable differential sparsity
        target_distribution: Target distribution [Level0, Level1, Level2]
        sparsity_levels: Sparsity values [0.0, 0.7, 1.0]
        diff_sparse_window_size: Window size for dual-window
        obs_window_size: Observation window size
        debug_diff_sparse: Enable debug mode
        use_flash_attention: Use Flash Attention
    
    Returns:
        LlamaConfig with DiffSparseKV attributes
    """
    if target_distribution is None:
        target_distribution = [0.05, 0.75, 0.20]
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.7, 1.0]
    
    # Add DiffSparseKV attributes to config
    base_config.enable_diff_sparse = enable_diff_sparse
    base_config.target_distribution = target_distribution
    base_config.sparsity_levels = sparsity_levels
    base_config.window_size = diff_sparse_window_size
    base_config.obs_window_size = obs_window_size
    base_config.debug_diff_sparse = debug_diff_sparse
    base_config.use_flash = use_flash_attention
    base_config.level_2_mode = level_2_mode
    base_config.importance_mode = importance_mode
    base_config.value_sink_keep = value_sink_keep
    base_config.head_aggregation_mode = head_aggregation_mode
    base_config.head_aggregation_alpha = head_aggregation_alpha
    base_config.head_disagreement_ratio = head_disagreement_ratio
<<<<<<< HEAD
    base_config.selector_mode = selector_mode
=======
>>>>>>> 34ec9a82045fc18a280c40b67c4a795e4b92dafe
    
    # Calculate expected sparsity
    expected_sparsity = sum(d * s for d, s in zip(target_distribution, sparsity_levels))
    base_config.expected_sparsity = expected_sparsity
    
    return base_config
