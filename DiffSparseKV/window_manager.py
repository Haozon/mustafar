"""
Window Manager for DiffSparseKV Decoding Stage

This module implements the dual-window management system for the decoding stage,
managing Window_A and Window_B with attention accumulation and compression triggers.

Requirements addressed: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5

Note: This is an optional implementation as the main focus is on prefill stage processing.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import time

from config_models import DiffSparseKVConfig
from importance_calculator import DiffKVImportanceCalculator
from threshold_manager import GlobalThresholdManager
from sparsity_applier import SparsityClassifierApplier


@dataclass
class WindowState:
    """State information for a single window."""
    
    key_states: Optional[torch.Tensor] = None  # [B, H, window_size, D]
    value_states: Optional[torch.Tensor] = None  # [B, H, window_size, D]
    attention_history: Optional[torch.Tensor] = None  # [B, H, accumulation_steps, window_size]
    token_indices: List[int] = field(default_factory=list)  # Global token indices in this window
    current_size: int = 0
    is_compressed: bool = False
    compression_timestamp: Optional[float] = None
    
    def is_full(self, window_size: int) -> bool:
        """Check if window is full."""
        return self.current_size >= window_size
    
    def has_space(self, window_size: int) -> bool:
        """Check if window has space for more tokens."""
        return self.current_size < window_size
    
    def reset(self):
        """Reset window state."""
        self.key_states = None
        self.value_states = None
        self.attention_history = None
        self.token_indices = []
        self.current_size = 0
        self.is_compressed = False
        self.compression_timestamp = None


@dataclass
class AttentionAccumulator:
    """Accumulates attention weights during generation for Window_A compression."""
    
    accumulated_attention: Optional[torch.Tensor] = None  # [B, H, window_size]
    accumulation_count: int = 0
    target_accumulation_steps: int = 128
    
    def add_attention(self, attention_weights: torch.Tensor, window_a_size: int):
        """
        Add attention weights for Window_A tokens.
        
        Args:
            attention_weights: [B, H, T_q, T_k] attention weights from current step
            window_a_size: Current size of Window_A
        """
        if window_a_size == 0:
            return
        
        # Extract attention to Window_A tokens (first window_a_size tokens)
        window_a_attention = attention_weights[:, :, -1, :window_a_size]  # [B, H, window_a_size]
        
        if self.accumulated_attention is None:
            self.accumulated_attention = window_a_attention.clone()
        else:
            # Ensure shapes match (Window_A might grow)
            if self.accumulated_attention.shape[-1] < window_a_size:
                # Expand accumulated attention to match current Window_A size
                B, H, old_size = self.accumulated_attention.shape
                new_accumulated = torch.zeros(B, H, window_a_size, 
                                            device=self.accumulated_attention.device,
                                            dtype=self.accumulated_attention.dtype)
                new_accumulated[:, :, :old_size] = self.accumulated_attention
                self.accumulated_attention = new_accumulated
            
            # Add current attention
            self.accumulated_attention[:, :, :window_a_size] += window_a_attention
        
        self.accumulation_count += 1
    
    def get_average_attention(self) -> Optional[torch.Tensor]:
        """Get average accumulated attention."""
        if self.accumulated_attention is None or self.accumulation_count == 0:
            return None
        return self.accumulated_attention / self.accumulation_count
    
    def is_ready_for_compression(self) -> bool:
        """Check if enough attention has been accumulated for compression."""
        return self.accumulation_count >= self.target_accumulation_steps
    
    def reset(self):
        """Reset accumulator."""
        self.accumulated_attention = None
        self.accumulation_count = 0


class WindowManager:
    """
    Dual-window management system for DiffSparseKV decoding stage.
    
    Manages Window_A (compressible) and Window_B (protected recent tokens) with
    attention accumulation and compression triggering logic.
    """
    
    def __init__(self, 
                 config: DiffSparseKVConfig,
                 importance_calculator: Optional[DiffKVImportanceCalculator] = None,
                 threshold_manager: Optional[GlobalThresholdManager] = None,
                 sparsity_applier: Optional[SparsityClassifierApplier] = None):
        """
        Initialize window manager.
        
        Args:
            config: DiffSparseKV configuration
            importance_calculator: Optional importance calculator (created if None)
            threshold_manager: Optional threshold manager (created if None)
            sparsity_applier: Optional sparsity applier (created if None)
        """
        self.config = config
        self.window_size = config.window_size
        self.attention_accumulation_steps = config.attention_accumulation_steps
        
        # Initialize components
        self.importance_calculator = importance_calculator or DiffKVImportanceCalculator()
        self.threshold_manager = threshold_manager or GlobalThresholdManager(
            target_distribution=config.target_distribution,
            sparsity_levels=config.sparsity_levels
        )
        self.sparsity_applier = sparsity_applier or SparsityClassifierApplier(
            sparsity_levels=config.sparsity_levels
        )
        
        # Window states
        self.window_a = WindowState()  # Older tokens, compressible
        self.window_b = WindowState()  # Recent tokens, protected
        
        # Attention accumulation
        self.attention_accumulator = AttentionAccumulator(
            target_accumulation_steps=self.attention_accumulation_steps
        )
        
        # Global state
        self.total_tokens_processed = 0
        self.compression_count = 0
        self.last_compression_step = 0
        
        # Statistics
        self.compression_history: List[Dict[str, Any]] = []
        
    def add_new_token(self, 
                     key_state: torch.Tensor,
                     value_state: torch.Tensor,
                     attention_weights: Optional[torch.Tensor] = None) -> bool:
        """
        Add a new token to the window system.
        
        Args:
            key_state: [B, H, 1, D] key state for new token
            value_state: [B, H, 1, D] value state for new token
            attention_weights: [B, H, T_q, T_k] attention weights (optional)
            
        Returns:
            bool: True if compression was triggered, False otherwise
        """
        B, H, _, D = key_state.shape
        
        # Add to Window_B (recent tokens)
        if self.window_b.key_states is None:
            self.window_b.key_states = key_state.clone()
            self.window_b.value_states = value_state.clone()
        else:
            self.window_b.key_states = torch.cat([self.window_b.key_states, key_state], dim=2)
            self.window_b.value_states = torch.cat([self.window_b.value_states, value_state], dim=2)
        
        self.window_b.token_indices.append(self.total_tokens_processed)
        self.window_b.current_size += 1
        self.total_tokens_processed += 1
        
        # Accumulate attention for Window_A if available
        if attention_weights is not None and self.window_a.current_size > 0:
            self.attention_accumulator.add_attention(attention_weights, self.window_a.current_size)
        
        # Check if Window_B is full
        compression_triggered = False
        if self.window_b.is_full(self.window_size):
            compression_triggered = self._handle_window_b_full()
        
        return compression_triggered
    
    def _handle_window_b_full(self) -> bool:
        """
        Handle Window_B becoming full.
        
        Returns:
            bool: True if compression was triggered, False otherwise
        """
        compression_triggered = False
        
        # Check if Window_A exists and has accumulated enough attention
        if (self.window_a.current_size > 0 and 
            self.attention_accumulator.is_ready_for_compression()):
            
            # Trigger Window_A compression
            compression_triggered = self._compress_window_a()
        
        # Move Window_B to Window_A
        self._move_window_b_to_a()
        
        return compression_triggered
    
    def _compress_window_a(self) -> bool:
        """
        Compress Window_A using accumulated attention.
        
        Returns:
            bool: True if compression was successful, False otherwise
        """
        if self.window_a.current_size == 0:
            return False
        
        try:
            # Get accumulated attention
            accumulated_attention = self.attention_accumulator.get_average_attention()
            if accumulated_attention is None:
                warnings.warn("No accumulated attention available for Window_A compression")
                return False
            
            # Reshape attention to match importance calculator expectations
            # accumulated_attention: [B, H, window_size]
            # Need: [B, H, T_q, T_k] where T_q = T_k = window_size
            B, H, window_size = accumulated_attention.shape
            
            # Create attention matrix by expanding the accumulated attention
            # We assume each query position has uniform attention distribution
            attention_matrix = accumulated_attention.unsqueeze(2).expand(B, H, window_size, window_size)
            
            # Apply causal mask to make it valid
            causal_mask = torch.tril(torch.ones(window_size, window_size, 
                                              device=attention_matrix.device, dtype=torch.bool))
            attention_matrix = attention_matrix * causal_mask[None, None, :, :]
            
            # Normalize to make it a valid attention distribution
            attention_sums = attention_matrix.sum(dim=-1, keepdim=True)
            attention_sums = torch.clamp(attention_sums, min=1e-8)
            attention_matrix = attention_matrix / attention_sums
            
            # Calculate importance scores
            importance_scores = self.importance_calculator.compute_diffkv_importance(attention_matrix)
            
            # Use existing thresholds or compute new ones
            if (self.threshold_manager.threshold_high is None or 
                self.threshold_manager.threshold_low is None):
                # Compute thresholds from current importance scores
                self.threshold_manager.compute_global_thresholds(importance_scores)
            
            thresholds = (self.threshold_manager.threshold_high, self.threshold_manager.threshold_low)
            
            # Apply sparsity
            compressed_keys, compressed_values = self.sparsity_applier.classify_and_apply_sparsity(
                importance_scores=importance_scores,
                key_states=self.window_a.key_states,
                value_states=self.window_a.value_states,
                thresholds=thresholds
            )
            
            # Update Window_A with compressed states
            self.window_a.key_states = compressed_keys
            self.window_a.value_states = compressed_values
            self.window_a.is_compressed = True
            self.window_a.compression_timestamp = time.time()
            
            # Record compression statistics
            original_size = self.window_a.key_states.numel() + self.window_a.value_states.numel()
            compressed_size = (torch.sum(torch.abs(compressed_keys) > 1e-8).item() + 
                             torch.sum(torch.abs(compressed_values) > 1e-8).item())
            compression_ratio = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0
            
            compression_stats = {
                "step": self.total_tokens_processed,
                "window_a_size": self.window_a.current_size,
                "original_elements": original_size,
                "compressed_elements": compressed_size,
                "compression_ratio": compression_ratio,
                "attention_accumulation_steps": self.attention_accumulator.accumulation_count,
                "timestamp": self.window_a.compression_timestamp
            }
            
            self.compression_history.append(compression_stats)
            self.compression_count += 1
            self.last_compression_step = self.total_tokens_processed
            
            # Reset attention accumulator
            self.attention_accumulator.reset()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Window_A compression failed: {e}")
            return False
    
    def _move_window_b_to_a(self):
        """Move Window_B contents to Window_A."""
        # Store current Window_B as new Window_A
        self.window_a.key_states = self.window_b.key_states.clone() if self.window_b.key_states is not None else None
        self.window_a.value_states = self.window_b.value_states.clone() if self.window_b.value_states is not None else None
        self.window_a.token_indices = self.window_b.token_indices.copy()
        self.window_a.current_size = self.window_b.current_size
        self.window_a.is_compressed = False
        self.window_a.compression_timestamp = None
        
        # Reset Window_B
        self.window_b.reset()
    
    def get_current_kv_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get current complete KV cache (Window_A + Window_B).
        
        Returns:
            Tuple of (keys, values) tensors or (None, None) if empty
        """
        keys_list = []
        values_list = []
        
        # Add Window_A if it exists
        if self.window_a.key_states is not None and self.window_a.current_size > 0:
            keys_list.append(self.window_a.key_states)
            values_list.append(self.window_a.value_states)
        
        # Add Window_B if it exists
        if self.window_b.key_states is not None and self.window_b.current_size > 0:
            keys_list.append(self.window_b.key_states)
            values_list.append(self.window_b.value_states)
        
        if not keys_list:
            return None, None
        
        # Concatenate along sequence dimension
        combined_keys = torch.cat(keys_list, dim=2)
        combined_values = torch.cat(values_list, dim=2)
        
        return combined_keys, combined_values
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """
        Get current window statistics.
        
        Returns:
            Dictionary with window statistics
        """
        stats = {
            "total_tokens_processed": self.total_tokens_processed,
            "window_a": {
                "size": self.window_a.current_size,
                "is_compressed": self.window_a.is_compressed,
                "compression_timestamp": self.window_a.compression_timestamp,
                "token_indices": self.window_a.token_indices.copy()
            },
            "window_b": {
                "size": self.window_b.current_size,
                "is_full": self.window_b.is_full(self.window_size),
                "token_indices": self.window_b.token_indices.copy()
            },
            "attention_accumulator": {
                "accumulation_count": self.attention_accumulator.accumulation_count,
                "is_ready": self.attention_accumulator.is_ready_for_compression(),
                "target_steps": self.attention_accumulator.target_accumulation_steps
            },
            "compression": {
                "total_compressions": self.compression_count,
                "last_compression_step": self.last_compression_step,
                "compression_history_length": len(self.compression_history)
            }
        }
        
        return stats
    
    def get_compression_history(self) -> List[Dict[str, Any]]:
        """Get compression history."""
        return self.compression_history.copy()
    
    def should_trigger_compression(self) -> bool:
        """
        Check if compression should be triggered based on current state.
        
        Returns:
            bool: True if compression should be triggered
        """
        return (self.window_b.is_full(self.window_size) and 
                self.window_a.current_size > 0 and
                self.attention_accumulator.is_ready_for_compression())
    
    def force_compression(self) -> bool:
        """
        Force compression of Window_A regardless of accumulation state.
        
        Returns:
            bool: True if compression was successful
        """
        if self.window_a.current_size == 0:
            return False
        
        # Use current attention state even if not fully accumulated
        return self._compress_window_a()
    
    def reset_windows(self):
        """Reset both windows and attention accumulator."""
        self.window_a.reset()
        self.window_b.reset()
        self.attention_accumulator.reset()
        self.total_tokens_processed = 0
        self.compression_count = 0
        self.last_compression_step = 0
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """
        Estimate current memory usage in MB.
        
        Returns:
            Dictionary with memory usage estimates
        """
        bytes_per_element = 4  # float32
        
        window_a_elements = 0
        if self.window_a.key_states is not None:
            window_a_elements = self.window_a.key_states.numel() + self.window_a.value_states.numel()
        
        window_b_elements = 0
        if self.window_b.key_states is not None:
            window_b_elements = self.window_b.key_states.numel() + self.window_b.value_states.numel()
        
        attention_elements = 0
        if self.attention_accumulator.accumulated_attention is not None:
            attention_elements = self.attention_accumulator.accumulated_attention.numel()
        
        return {
            "window_a_mb": (window_a_elements * bytes_per_element) / (1024 * 1024),
            "window_b_mb": (window_b_elements * bytes_per_element) / (1024 * 1024),
            "attention_accumulator_mb": (attention_elements * bytes_per_element) / (1024 * 1024),
            "total_mb": ((window_a_elements + window_b_elements + attention_elements) * bytes_per_element) / (1024 * 1024)
        }
    
    def integrate_with_residual_length(self, residual_length: int) -> Dict[str, Any]:
        """
        Integrate with existing residual_length processing pattern.
        
        Args:
            residual_length: Current residual length from existing implementation
            
        Returns:
            Dictionary with integration information
        """
        # Calculate effective cache length considering windows
        effective_cache_length = self.window_a.current_size + self.window_b.current_size
        
        # Determine if window management should override residual_length behavior
        should_override = (self.config.enable_dual_windows and 
                          effective_cache_length > residual_length)
        
        integration_info = {
            "residual_length": residual_length,
            "effective_cache_length": effective_cache_length,
            "window_a_size": self.window_a.current_size,
            "window_b_size": self.window_b.current_size,
            "should_override_residual": should_override,
            "compression_active": self.window_a.is_compressed,
            "next_compression_in_steps": max(0, self.attention_accumulation_steps - self.attention_accumulator.accumulation_count)
        }
        
        return integration_info


def create_window_manager(config: DiffSparseKVConfig,
                         importance_calculator: Optional[DiffKVImportanceCalculator] = None,
                         threshold_manager: Optional[GlobalThresholdManager] = None,
                         sparsity_applier: Optional[SparsityClassifierApplier] = None) -> WindowManager:
    """
    Factory function to create a window manager.
    
    Args:
        config: DiffSparseKV configuration
        importance_calculator: Optional importance calculator
        threshold_manager: Optional threshold manager
        sparsity_applier: Optional sparsity applier
        
    Returns:
        WindowManager instance
    """
    return WindowManager(
        config=config,
        importance_calculator=importance_calculator,
        threshold_manager=threshold_manager,
        sparsity_applier=sparsity_applier
    )