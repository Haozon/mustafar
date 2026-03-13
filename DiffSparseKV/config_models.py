"""
Configuration and Data Models for DiffSparseKV

This module implements the configuration classes and data models for the DiffSparseKV system,
providing structured configuration management, validation, and data persistence.

Requirements addressed: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import torch
import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import warnings


@dataclass
class DiffSparseKVConfig:
    """
    Configuration class for DiffSparseKV system with validation and default values.
    
    This class provides comprehensive configuration management for all DiffSparseKV components
    including sparsity levels, target distributions, window management, and integration settings.
    """
    
    # Core sparsity configuration
    sparsity_levels: List[float] = field(default_factory=lambda: [0.0, 0.7, 1.0])
    target_distribution: List[float] = field(default_factory=lambda: [0.05, 0.75, 0.20])
    
    # Threshold strategy configuration
    use_global_thresholds: bool = True
    use_per_layer_thresholds: bool = False
    threshold_tolerance: float = 0.05  # Maximum allowed deviation from target distribution
    
    # Window management configuration (for decoding stage)
    window_size: int = 128
    enable_dual_windows: bool = False  # Mainly for decoding phase
    attention_accumulation_steps: int = 128
    
    # Integration settings
    residual_length: int = 128  # Compatibility with existing implementations
    enable_diff_sparse: bool = True
    enable_flash_attention: bool = True
    
    # Performance and optimization settings
    preserve_tensor_shapes: bool = True
    numerical_stability_eps: float = 1e-8
    enable_statistics_tracking: bool = True
    
    # Evaluation and monitoring
    enable_performance_monitoring: bool = True
    save_threshold_statistics: bool = True
    threshold_persistence_path: Optional[str] = None
    
    # Advanced configuration
    custom_importance_calculator: Optional[str] = None  # Plugin interface for alternative methods
    custom_sparsity_applier: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate sparsity levels
        if len(self.sparsity_levels) != 3:
            raise ValueError("sparsity_levels must have exactly 3 elements")
        
        if not all(0.0 <= s <= 1.0 for s in self.sparsity_levels):
            raise ValueError("All sparsity levels must be between 0.0 and 1.0")
        
        if not all(self.sparsity_levels[i] <= self.sparsity_levels[i+1] for i in range(2)):
            raise ValueError("Sparsity levels must be in non-decreasing order")
        
        # Validate target distribution
        if len(self.target_distribution) != 3:
            raise ValueError("target_distribution must have exactly 3 elements")
        
        if not all(0.0 <= d <= 1.0 for d in self.target_distribution):
            raise ValueError("All target distribution values must be between 0.0 and 1.0")
        
        if abs(sum(self.target_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Target distribution must sum to 1.0, got {sum(self.target_distribution)}")
        
        # Validate threshold strategy
        if self.use_global_thresholds and self.use_per_layer_thresholds:
            warnings.warn("Both global and per-layer thresholds enabled. Global thresholds will take precedence.")
        
        if not self.use_global_thresholds and not self.use_per_layer_thresholds:
            raise ValueError("At least one threshold strategy must be enabled")
        
        # Validate numerical parameters
        if self.threshold_tolerance <= 0.0 or self.threshold_tolerance > 0.5:
            raise ValueError("threshold_tolerance must be between 0.0 and 0.5")
        
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if self.attention_accumulation_steps <= 0:
            raise ValueError("attention_accumulation_steps must be positive")
        
        if self.residual_length <= 0:
            raise ValueError("residual_length must be positive")
        
        if self.numerical_stability_eps <= 0.0:
            raise ValueError("numerical_stability_eps must be positive")
        
        # Validate paths
        if self.threshold_persistence_path is not None:
            try:
                Path(self.threshold_persistence_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.warn(f"Cannot create threshold persistence directory: {e}")
    
    def get_expected_compression_ratio(self) -> float:
        """
        Calculate expected compression ratio based on target distribution and sparsity levels.
        
        Returns:
            Expected compression ratio (0.0 to 1.0)
        """
        total_sparsity = sum(
            self.target_distribution[i] * self.sparsity_levels[i] 
            for i in range(3)
        )
        return total_sparsity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DiffSparseKVConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            DiffSparseKVConfig instance
        """
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DiffSparseKVConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            DiffSparseKVConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters from dictionary.
        
        Args:
            updates: Dictionary with parameter updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown configuration parameter: {key}")
        
        # Re-validate after updates
        self.validate()
    
    def create_default_threshold_path(self, base_dir: str = ".") -> str:
        """
        Create default threshold persistence path.
        
        Args:
            base_dir: Base directory for threshold files
            
        Returns:
            Default threshold file path
        """
        threshold_dir = os.path.join(base_dir, "diff_sparse_kv_thresholds")
        os.makedirs(threshold_dir, exist_ok=True)
        return os.path.join(threshold_dir, "global_thresholds.json")


@dataclass
class ImportanceScores:
    """
    Data model for importance scores with metadata and utility methods.
    
    This class encapsulates importance scores along with their metadata and provides
    utility methods for manipulation and analysis.
    """
    
    scores: torch.Tensor  # Shape: [B, H, T] or [B, H, num_layers, T]
    sequence_length: int
    num_layers: int
    num_heads: int
    batch_size: int
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    
    # Metadata
    computation_method: str = "diffkv"
    timestamp: Optional[float] = None
    layer_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate importance scores after initialization."""
        self.validate()
        
        # Set device and dtype from tensor if not provided
        if self.device is None:
            self.device = self.scores.device
        if self.dtype is None:
            self.dtype = self.scores.dtype
        
        # Set timestamp if not provided
        if self.timestamp is None:
            import time
            self.timestamp = time.time()
    
    def validate(self) -> None:
        """
        Validate importance scores and metadata.
        
        Raises:
            ValueError: If scores or metadata are invalid
        """
        if not isinstance(self.scores, torch.Tensor):
            raise ValueError("scores must be a torch.Tensor")
        
        if self.scores.dim() not in [3, 4]:
            raise ValueError(f"scores must be 3D [B, H, T] or 4D [B, H, num_layers, T], got {self.scores.dim()}D")
        
        # Validate dimensions match metadata
        if self.scores.dim() == 3:
            B, H, T = self.scores.shape
            if (B, H, T) != (self.batch_size, self.num_heads, self.sequence_length):
                raise ValueError(f"3D scores shape {(B, H, T)} doesn't match metadata "
                               f"({self.batch_size}, {self.num_heads}, {self.sequence_length})")
        else:  # 4D
            B, H, L, T = self.scores.shape
            if (B, H, L, T) != (self.batch_size, self.num_heads, self.num_layers, self.sequence_length):
                raise ValueError(f"4D scores shape {(B, H, L, T)} doesn't match metadata "
                               f"({self.batch_size}, {self.num_heads}, {self.num_layers}, {self.sequence_length})")
        
        # Validate score values
        if torch.any(torch.isnan(self.scores)) or torch.any(torch.isinf(self.scores)):
            raise ValueError("Importance scores contain NaN or infinite values")
        
        if torch.any(self.scores < 0):
            warnings.warn("Negative importance scores detected")
        
        # Validate metadata
        if self.sequence_length <= 0 or self.num_layers <= 0 or self.num_heads <= 0 or self.batch_size <= 0:
            raise ValueError("All dimension parameters must be positive")
    
    def flatten_for_thresholding(self) -> torch.Tensor:
        """
        Flatten importance scores for global threshold computation.
        
        Returns:
            Flattened tensor suitable for threshold calculation
        """
        return self.scores.flatten()
    
    def get_layer_scores(self, layer_idx: int) -> torch.Tensor:
        """
        Extract importance scores for a specific layer.
        
        Args:
            layer_idx: Layer index to extract
            
        Returns:
            Layer importance scores of shape [B, H, T]
            
        Raises:
            ValueError: If layer_idx is invalid or scores are not 4D
        """
        if self.scores.dim() != 4:
            raise ValueError("get_layer_scores() requires 4D scores tensor")
        
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        
        return self.scores[:, :, layer_idx, :]
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical summary of importance scores.
        
        Returns:
            Dictionary containing statistical measures
        """
        flattened = self.flatten_for_thresholding()
        
        stats = {
            "mean": float(torch.mean(flattened)),
            "std": float(torch.std(flattened)),
            "min": float(torch.min(flattened)),
            "max": float(torch.max(flattened)),
            "median": float(torch.median(flattened)),
            "total_elements": int(flattened.numel()),
            "zero_elements": int(torch.sum(flattened == 0.0)),
            "negative_elements": int(torch.sum(flattened < 0.0))
        }
        
        # Compute percentiles
        percentiles = [5, 25, 75, 95]
        for p in percentiles:
            stats[f"percentile_{p}"] = float(torch.quantile(flattened, p / 100.0))
        
        return stats
    
    def to_device(self, device: torch.device) -> 'ImportanceScores':
        """
        Move importance scores to specified device.
        
        Args:
            device: Target device
            
        Returns:
            New ImportanceScores instance on target device
        """
        new_scores = self.scores.to(device)
        
        return ImportanceScores(
            scores=new_scores,
            sequence_length=self.sequence_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            device=device,
            dtype=self.dtype,
            computation_method=self.computation_method,
            timestamp=self.timestamp,
            layer_indices=self.layer_indices.copy() if self.layer_indices else None
        )
    
    def clone(self) -> 'ImportanceScores':
        """Create a deep copy of the ImportanceScores."""
        return ImportanceScores(
            scores=self.scores.clone(),
            sequence_length=self.sequence_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
            computation_method=self.computation_method,
            timestamp=self.timestamp,
            layer_indices=self.layer_indices.copy() if self.layer_indices else None
        )


@dataclass
class ThresholdState:
    """
    Data model for threshold state with persistence and validation capabilities.
    
    This class manages threshold values, target distributions, and provides
    persistence and validation functionality.
    """
    
    threshold_high: float  # Level 0 vs Level 1 boundary
    threshold_low: float   # Level 1 vs Level 2 boundary
    target_distribution: List[float]
    sparsity_levels: List[float]
    
    # Validation and statistics
    actual_distribution: Optional[List[float]] = None
    total_tokens: int = 0
    computation_timestamp: Optional[float] = None
    
    # Metadata
    source_config: Optional[Dict[str, Any]] = None
    validation_tolerance: float = 0.05
    
    def __post_init__(self):
        """Validate threshold state after initialization."""
        self.validate()
        
        if self.computation_timestamp is None:
            import time
            self.computation_timestamp = time.time()
    
    def validate(self) -> None:
        """
        Validate threshold state parameters.
        
        Raises:
            ValueError: If threshold state is invalid
        """
        # Validate threshold values
        if self.threshold_high < self.threshold_low:
            raise ValueError(f"threshold_high ({self.threshold_high}) must be >= threshold_low ({self.threshold_low})")
        
        # Validate distributions
        if len(self.target_distribution) != 3:
            raise ValueError("target_distribution must have exactly 3 elements")
        
        if len(self.sparsity_levels) != 3:
            raise ValueError("sparsity_levels must have exactly 3 elements")
        
        if abs(sum(self.target_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Target distribution must sum to 1.0, got {sum(self.target_distribution)}")
        
        if self.actual_distribution is not None:
            if len(self.actual_distribution) != 3:
                raise ValueError("actual_distribution must have exactly 3 elements")
            
            if abs(sum(self.actual_distribution) - 1.0) > 1e-6:
                warnings.warn(f"Actual distribution doesn't sum to 1.0: {sum(self.actual_distribution)}")
        
        # Validate numerical parameters
        if self.validation_tolerance <= 0.0 or self.validation_tolerance > 0.5:
            raise ValueError("validation_tolerance must be between 0.0 and 0.5")
        
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
    
    def compute_distribution_error(self) -> Optional[float]:
        """
        Compute distribution error between target and actual distributions.
        
        Returns:
            Mean absolute error or None if actual distribution not available
        """
        if self.actual_distribution is None:
            return None
        
        target = np.array(self.target_distribution)
        actual = np.array(self.actual_distribution)
        
        return float(np.mean(np.abs(actual - target)))
    
    def is_valid_distribution(self, tolerance: Optional[float] = None) -> bool:
        """
        Check if actual distribution is within tolerance of target distribution.
        
        Args:
            tolerance: Custom tolerance (uses instance tolerance if None)
            
        Returns:
            True if distribution is valid, False otherwise
        """
        if self.actual_distribution is None:
            return False
        
        if tolerance is None:
            tolerance = self.validation_tolerance
        
        error = self.compute_distribution_error()
        return error is not None and error <= tolerance
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary containing validation results and statistics
        """
        report = {
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "target_distribution": self.target_distribution,
            "sparsity_levels": self.sparsity_levels,
            "total_tokens": self.total_tokens,
            "validation_tolerance": self.validation_tolerance
        }
        
        if self.actual_distribution is not None:
            error = self.compute_distribution_error()
            max_error = float(np.max(np.abs(np.array(self.actual_distribution) - np.array(self.target_distribution))))
            
            report.update({
                "actual_distribution": self.actual_distribution,
                "distribution_error": error,
                "max_distribution_error": max_error,
                "is_valid": self.is_valid_distribution(),
                "distribution_errors_per_level": (np.array(self.actual_distribution) - np.array(self.target_distribution)).tolist()
            })
        else:
            report.update({
                "actual_distribution": None,
                "distribution_error": None,
                "is_valid": False,
                "reason": "No actual distribution computed"
            })
        
        return report
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save threshold state to JSON file.
        
        Args:
            filepath: Path to save threshold file
        """
        threshold_data = {
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "target_distribution": self.target_distribution,
            "sparsity_levels": self.sparsity_levels,
            "actual_distribution": self.actual_distribution,
            "total_tokens": self.total_tokens,
            "computation_timestamp": self.computation_timestamp,
            "validation_tolerance": self.validation_tolerance,
            "source_config": self.source_config
        }
        
        # Add validation results
        if self.actual_distribution is not None:
            threshold_data["distribution_error"] = self.compute_distribution_error()
            threshold_data["is_valid"] = self.is_valid_distribution()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(threshold_data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ThresholdState':
        """
        Load threshold state from JSON file.
        
        Args:
            filepath: Path to threshold file
            
        Returns:
            ThresholdState instance
            
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
        
        return cls(
            threshold_high=float(threshold_data["threshold_high"]),
            threshold_low=float(threshold_data["threshold_low"]),
            target_distribution=threshold_data["target_distribution"],
            sparsity_levels=threshold_data["sparsity_levels"],
            actual_distribution=threshold_data.get("actual_distribution"),
            total_tokens=threshold_data.get("total_tokens", 0),
            computation_timestamp=threshold_data.get("computation_timestamp"),
            source_config=threshold_data.get("source_config"),
            validation_tolerance=threshold_data.get("validation_tolerance", 0.05)
        )
    
    def update_actual_distribution(self, 
                                 actual_distribution: List[float],
                                 total_tokens: int) -> None:
        """
        Update actual distribution and token count.
        
        Args:
            actual_distribution: Measured distribution
            total_tokens: Total number of tokens processed
        """
        if len(actual_distribution) != 3:
            raise ValueError("actual_distribution must have exactly 3 elements")
        
        self.actual_distribution = actual_distribution
        self.total_tokens = total_tokens
        
        # Update timestamp
        import time
        self.computation_timestamp = time.time()


# Factory functions for easy creation

def create_default_config() -> DiffSparseKVConfig:
    """
    Create default DiffSparseKV configuration.
    
    Returns:
        DiffSparseKVConfig with default settings
    """
    return DiffSparseKVConfig()


def create_config_for_evaluation() -> DiffSparseKVConfig:
    """
    Create configuration optimized for LongBench evaluation.
    
    Returns:
        DiffSparseKVConfig optimized for evaluation
    """
    return DiffSparseKVConfig(
        enable_dual_windows=False,  # Focus on prefill phase
        enable_performance_monitoring=True,
        save_threshold_statistics=True,
        threshold_tolerance=0.03,  # Stricter tolerance for evaluation
        enable_statistics_tracking=True
    )


def create_config_for_production() -> DiffSparseKVConfig:
    """
    Create configuration optimized for production use.
    
    Returns:
        DiffSparseKVConfig optimized for production
    """
    return DiffSparseKVConfig(
        enable_dual_windows=True,  # Full functionality
        enable_performance_monitoring=False,  # Reduce overhead
        save_threshold_statistics=False,
        enable_statistics_tracking=False,
        preserve_tensor_shapes=True
    )


def create_importance_scores_from_tensor(scores: torch.Tensor,
                                       num_layers: int,
                                       computation_method: str = "diffkv") -> ImportanceScores:
    """
    Create ImportanceScores from tensor with automatic shape inference.
    
    Args:
        scores: Importance scores tensor
        num_layers: Number of layers
        computation_method: Method used for computation
        
    Returns:
        ImportanceScores instance
    """
    if scores.dim() == 3:
        B, H, T = scores.shape
    elif scores.dim() == 4:
        B, H, _, T = scores.shape
    else:
        raise ValueError(f"Unsupported tensor dimension: {scores.dim()}")
    
    return ImportanceScores(
        scores=scores,
        sequence_length=T,
        num_layers=num_layers,
        num_heads=H,
        batch_size=B,
        computation_method=computation_method
    )


def create_threshold_state_from_computation(threshold_high: float,
                                          threshold_low: float,
                                          config: DiffSparseKVConfig,
                                          actual_distribution: Optional[List[float]] = None,
                                          total_tokens: int = 0) -> ThresholdState:
    """
    Create ThresholdState from threshold computation results.
    
    Args:
        threshold_high: High threshold value
        threshold_low: Low threshold value
        config: DiffSparseKV configuration
        actual_distribution: Measured distribution (optional)
        total_tokens: Total tokens processed
        
    Returns:
        ThresholdState instance
    """
    return ThresholdState(
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        target_distribution=config.target_distribution,
        sparsity_levels=config.sparsity_levels,
        actual_distribution=actual_distribution,
        total_tokens=total_tokens,
        source_config=config.to_dict(),
        validation_tolerance=config.threshold_tolerance
    )