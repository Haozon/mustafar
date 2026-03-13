"""
DiffSparseKV: Dual-Window Attention Mechanism for Efficient KV Cache Compression

This module implements the DiffSparseKV algorithm with dual-window mechanism for
efficient KV cache compression during inference.

Key Features:
- Dual-window mechanism (Window A: compressible, Window B: protection)
- Dynamic attention accumulation with per-token observation counting
- GQA (Grouped Query Attention) support
- Per-layer threshold management
- Independent implementation (not inheriting from MUSTAFAR)

Author: DiffSparseKV Team
Date: 2025-01-23
Version: 1.0
"""

import math
import warnings
from typing import List, Optional, Tuple, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

# Import DiffSparseKV components
try:
    from .importance_calculator import DiffKVImportanceCalculator
    from .threshold_manager import GlobalThresholdManager
    from .sparsity_applier import SparsityClassifierApplier
except ImportError:
    from importance_calculator import DiffKVImportanceCalculator
    from threshold_manager import GlobalThresholdManager
    from sparsity_applier import SparsityClassifierApplier

# Debug flag
DEBUG = False


class LlamaDiffSparseKVAttention(nn.Module):
    """
    DiffSparseKV Attention with dual-window mechanism.
    
    This class implements the complete DiffSparseKV algorithm including:
    - Prefill stage: compute thresholds and initialize dual windows
    - Decode stage: dynamic attention accumulation and compression
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Basic attention parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # DiffSparseKV parameters
        self.window_size = getattr(config, 'window_size', 32)
        self.obs_window_size = getattr(config, 'obs_window_size', 128)  # Observation window size
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)  # Use Flash Attention
        self.target_distribution = getattr(config, 'target_distribution', [0.05, 0.75, 0.20])
        self.sparsity_levels = getattr(config, 'sparsity_levels', [0.0, 0.7, 1.0])
        self.importance_mode = getattr(config, 'importance_mode', 'attention_only')
        self.value_sink_keep = getattr(config, 'value_sink_keep', 2)
        self.head_aggregation_mode = getattr(config, 'head_aggregation_mode', 'mean')
        self.head_aggregation_alpha = getattr(config, 'head_aggregation_alpha', 0.5)
        self.head_disagreement_ratio = getattr(config, 'head_disagreement_ratio', -1.0)
        
        # Validate observation window size
        if self.obs_window_size < self.window_size:
            raise ValueError(
                f"obs_window_size ({self.obs_window_size}) must be >= window_size ({self.window_size})"
            )
        
        # GQA configuration check
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        
        # DiffSparseKV components
        self.importance_calculator = DiffKVImportanceCalculator()
        self.threshold_manager = GlobalThresholdManager(
            target_distribution=self.target_distribution,
            sparsity_levels=self.sparsity_levels
        )
        self.sparsity_applier = SparsityClassifierApplier(
            sparsity_levels=self.sparsity_levels,
            level_2_mode=getattr(config, "level_2_mode", "evict"),
        )
        
        # Instance variables for state management
        self.window_state = None  # Will be initialized in prefill
        self.diff_sparse_thresholds = None  # Threshold boundaries between sparsity levels
        self.thresholds_computed = False
        self.generation_count = 0
        self.prefill_length = 0  # Track prefill sequence length for N scaling
        self.current_sequence_length = 0  # Track current total sequence length
        
        if DEBUG:
            print(f"LlamaDiffSparseKVAttention initialized:")
            print(f"  window_size: {self.window_size}")
            print(f"  obs_window_size: {self.obs_window_size}")
            print(f"  use_flash_attention: {self.use_flash_attention}")
            print(f"  num_heads: {self.num_heads}")
            print(f"  num_key_value_heads: {self.num_key_value_heads}")
            print(f"  num_key_value_groups: {self.num_key_value_groups}")

    def calculate_sparsity(self, tensor: torch.Tensor) -> float:
        """
        Calculate the sparsity of a 4D PyTorch tensor.
        
        Args:
            tensor: A 4D tensor (batch, heads, tokens, hidden_dim)
        
        Returns:
            float: The sparsity ratio (between 0 and 1)
        """
        if tensor.dim() != 4:
            raise ValueError("Input tensor must be 4D (batch, heads, tokens, hidden_dim)")
        
        total_elements = tensor.numel()
        zero_elements = torch.sum(tensor == 0).item()
        
        if DEBUG:
            print(f"Matrix size: {tensor.shape}, Zero elements: {zero_elements}, Total elements: {total_elements}")
        
        sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
        return sparsity

    def compute_observation_window_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        obs_window_size: int
    ) -> torch.Tensor:
        """
        Compute attention weights using observation window with GQA optimization.
        
        This method uses the last obs_window_size tokens as queries to compute
        attention weights for all keys. It uses the efficient "reshape queries"
        approach for GQA.
        
        Args:
            query_states: [B, H_q, seq_len, D] - Query states
            key_states: [B, H_kv, seq_len, D] - Key states
            obs_window_size: Size of observation window
            
        Returns:
            attn_weights_kv: [B, H_kv, obs_window, seq_len] - Attention weights
        """
        B, H_q, seq_len, D = query_states.shape
        H_kv = key_states.shape[1]
        
        # Extract observation window queries (last obs_window_size tokens)
        obs_queries = query_states[:, :, -obs_window_size:, :]  # [B, H_q, obs_window, D]
        
        # Reshape queries to KV head groups (GQA optimization - Method B)
        obs_queries_grouped = obs_queries.view(
            B, H_kv, self.num_key_value_groups, obs_window_size, D
        )  # [B, H_kv, group_size, obs_window, D]
        
        # Compute attention weights (利用广播机制)
        attn_weights = torch.matmul(
            obs_queries_grouped,
            key_states.unsqueeze(2).transpose(-2, -1)
        ) / math.sqrt(D)
        # Result: [B, H_kv, group_size, obs_window, seq_len]
        
        # Apply causal mask
        # Observation window is at the end of sequence
        # Query i in obs window (global position: seq_len - obs_window_size + i)
        # can see keys [0 : seq_len - obs_window_size + i + 1]
        causal_mask = torch.zeros(obs_window_size, seq_len, device=attn_weights.device, dtype=torch.bool)
        for i in range(obs_window_size):
            # Global position of this query
            global_pos = seq_len - obs_window_size + i
            # Can see keys from 0 to global_pos (inclusive)
            causal_mask[i, :global_pos + 1] = True
        
        attn_weights = attn_weights.masked_fill(
            ~causal_mask[None, None, None, :, :],
            float('-inf')
        )
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Average over group dimension
        attn_weights_kv = attn_weights.mean(dim=2)  # [B, H_kv, obs_window, seq_len]
        
        if DEBUG:
            print(f"Observation window attention computed:")
            print(f"  obs_window_size: {obs_window_size}")
            print(f"  attn_weights_kv shape: {attn_weights_kv.shape}")
        
        return attn_weights_kv

    def initialize_dual_window_after_prefill(self, prefill_keys: torch.Tensor, prefill_values: torch.Tensor):
        """
        Initialize dual-window state after prefill stage.
        
        Args:
            prefill_keys: [B, H_kv, N, D] - Prefill KV cache (already sparsified)
            prefill_values: [B, H_kv, N, D]
        """
        N = prefill_keys.shape[2]
        B, H_kv = prefill_keys.shape[:2]
        W = self.window_size
        
        if DEBUG:
            print(f"\n=== Initializing Dual Window ===")
            print(f"Prefill length: {N}, Window size: {W}")
        
        if N >= W:
            # Case A: Prefill >= W
            # Split: first (N-W) as compressed, last W as Window A
            compressed_keys = prefill_keys[:, :, :-W, :]
            compressed_values = prefill_values[:, :, :-W, :]
            window_a_keys = prefill_keys[:, :, -W:, :].clone()
            window_a_values = prefill_values[:, :, -W:, :].clone()
            window_a_size = W
            
            if DEBUG:
                print(f"Case A: Prefill >= W")
                print(f"  Compressed: {compressed_keys.shape[2]} tokens")
                print(f"  Window A: {window_a_size} tokens (full)")
        else:
            # Case B: Prefill < W
            # All tokens go to Window A (not full)
            compressed_keys = None
            compressed_values = None
            window_a_keys = prefill_keys.clone()
            window_a_values = prefill_values.clone()
            window_a_size = N
            
            if DEBUG:
                print(f"Case B: Prefill < W")
                print(f"  Window A: {window_a_size} tokens (not full, need {W - N} more)")
        
        # Window B starts empty
        window_b_keys = None
        window_b_values = None
        window_b_size = 0
        
        # Initialize accumulators (note: H_kv, not num_heads)
        accumulator = torch.zeros(B, H_kv, W, device=prefill_keys.device, dtype=torch.float32)
        token_observation_count = torch.zeros(B, H_kv, W, device=prefill_keys.device, dtype=torch.float32)
        
        # Store to instance variable
        self.window_state = {
            'compressed_keys': compressed_keys,
            'compressed_values': compressed_values,
            'window_a_keys': window_a_keys,
            'window_a_values': window_a_values,
            'window_a_size': window_a_size,
            'window_b_keys': window_b_keys,
            'window_b_values': window_b_values,
            'window_b_size': window_b_size,
            'accumulator': accumulator,
            'token_observation_count': token_observation_count
        }
        
        if DEBUG:
            print(f"Window state initialized successfully")
            print(f"=== End Initialization ===\n")

    def _trigger_compression(self):
        """
        Trigger compression of Window A and perform window sliding.
        
        This method:
        1. Normalizes accumulated attention (dynamic observation count)
        2. Classifies tokens using Prefill thresholds
        3. Applies sparsification
        4. Merges to compressed cache
        5. Slides windows: Window B → Window A
        6. Resets Window B and accumulators
        """
        ws = self.window_state
        W = self.window_size
        
        if DEBUG:
            print(f"\n=== Triggering Compression ===")
            print(f"Window A size: {ws['window_a_size']}, Window B size: {ws['window_b_size']}")
        
        # Only compress if Window A is full
        if ws['window_a_size'] == W:
            # 1. Normalize attention (dynamic observation count)
            normalized_attention = torch.zeros_like(ws['accumulator'])
            
            for i in range(W):
                # Dynamic observation count
                N_internal = W - 1 - i  # Window A internal
                N_external = ws['token_observation_count'][:, :, i]  # Actual accumulation count
                N_total = N_internal + N_external
                
                # Avoid division by zero
                N_total = torch.clamp(N_total, min=1.0)
                
                normalized_attention[:, :, i] = ws['accumulator'][:, :, i] / N_total
            
            # Apply N scaling to match prefill importance scores value range
            # Use current sequence length N (not W) for proper scaling
            N_current = self.current_sequence_length
            normalized_attention = normalized_attention * N_current
            
            if DEBUG:
                print(f"Normalized attention computed (scaled by N={N_current})")
                print(f"  Min: {normalized_attention.min().item():.6f}")
                print(f"  Max: {normalized_attention.max().item():.6f}")
                print(f"  Mean: {normalized_attention.mean().item():.6f}")
            
            # 2. Check thresholds
            if not self.thresholds_computed or self.diff_sparse_thresholds is None:
                raise RuntimeError("Thresholds not computed. Must run prefill first.")
            
            if DEBUG:
                print(f"Using thresholds: {self.diff_sparse_thresholds}")
            
            # 3. Apply sparsification
            compressed_keys, compressed_values = self.sparsity_applier.classify_and_apply_sparsity(
                importance_scores=normalized_attention,
                key_states=ws['window_a_keys'],
                value_states=ws['window_a_values'],
                thresholds=self.diff_sparse_thresholds
            )
            
            if DEBUG:
                k_sparsity = self.calculate_sparsity(compressed_keys)
                v_sparsity = self.calculate_sparsity(compressed_values)
                print(f"Compressed Window A: K sparsity={k_sparsity:.2%}, V sparsity={v_sparsity:.2%}")
            
            # 4. Merge to compressed cache
            if ws['compressed_keys'] is None:
                ws['compressed_keys'] = compressed_keys
                ws['compressed_values'] = compressed_values
            else:
                ws['compressed_keys'] = torch.cat([
                    ws['compressed_keys'], compressed_keys
                ], dim=2)
                ws['compressed_values'] = torch.cat([
                    ws['compressed_values'], compressed_values
                ], dim=2)
            
            if DEBUG:
                total_compressed = ws['compressed_keys'].shape[2]
                print(f"Total compressed cache size: {total_compressed} tokens")
        
        # 5. Window sliding: Window B → Window A
        ws['window_a_keys'] = ws['window_b_keys']
        ws['window_a_values'] = ws['window_b_values']
        ws['window_a_size'] = ws['window_b_size']
        
        # 6. Reset Window B
        ws['window_b_keys'] = None
        ws['window_b_values'] = None
        ws['window_b_size'] = 0
        
        # 7. Reset accumulators
        ws['accumulator'].zero_()
        ws['token_observation_count'].zero_()
        
        if DEBUG:
            print(f"Window sliding completed")
            print(f"  New Window A size: {ws['window_a_size']}")
            print(f"  Window B reset to empty")
            print(f"=== End Compression ===\n")

    def decode_step_with_dual_window(
        self,
        query_states: torch.Tensor,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Single decode step with dual-window mechanism (GQA support).
        
        Args:
            query_states: [B, H_q, 1, D] - Query states (num_heads)
            new_key: [B, H_kv, 1, D] - New token's key (num_key_value_heads)
            new_value: [B, H_kv, 1, D] - New token's value
            attention_mask: [B, 1, 1, T_total] or None
        
        Returns:
            attn_output: [B, H_q, 1, D]
        """
        B, H_q, _, D = query_states.shape
        _, H_kv, _, _ = new_key.shape
        W = self.window_size
        
        ws = self.window_state
        
        if DEBUG:
            print(f"\n--- Decode Step {self.generation_count + 1} ---")
            print(f"Window A size: {ws['window_a_size']}/{W}, Window B size: {ws['window_b_size']}/{W}")
        
        # 1. Add new token to Window A or Window B
        if ws['window_a_size'] < W:
            # Case B: Window A not full, add to Window A
            if ws['window_a_keys'] is None:
                ws['window_a_keys'] = new_key
                ws['window_a_values'] = new_value
            else:
                ws['window_a_keys'] = torch.cat([ws['window_a_keys'], new_key], dim=2)
                ws['window_a_values'] = torch.cat([ws['window_a_values'], new_value], dim=2)
            ws['window_a_size'] += 1
            
            if DEBUG:
                print(f"Added to Window A (filling), new size: {ws['window_a_size']}/{W}")
        else:
            # Case A: Window A full, add to Window B
            if ws['window_b_keys'] is None:
                ws['window_b_keys'] = new_key
                ws['window_b_values'] = new_value
            else:
                ws['window_b_keys'] = torch.cat([ws['window_b_keys'], new_key], dim=2)
                ws['window_b_values'] = torch.cat([ws['window_b_values'], new_value], dim=2)
            ws['window_b_size'] += 1
            
            if DEBUG:
                print(f"Added to Window B, new size: {ws['window_b_size']}/{W}")
        
        # 2. Build full KV cache
        kv_list = []
        if ws['compressed_keys'] is not None:
            kv_list.append((ws['compressed_keys'], ws['compressed_values']))
        if ws['window_a_keys'] is not None:
            kv_list.append((ws['window_a_keys'], ws['window_a_values']))
        if ws['window_b_keys'] is not None:
            kv_list.append((ws['window_b_keys'], ws['window_b_values']))
        
        full_keys = torch.cat([k for k, v in kv_list], dim=2)  # [B, H_kv, T_total, D]
        full_values = torch.cat([v for k, v in kv_list], dim=2)
        
        if DEBUG:
            print(f"Full KV cache size: {full_keys.shape[2]} tokens")
        
        # 3. Compute attention (with repeat_kv for GQA)
        attn_weights = torch.matmul(
            query_states,  # [B, H_q, 1, D]
            repeat_kv(full_keys, self.num_key_value_groups).transpose(2, 3)  # [B, H_q, T_total, D]
        ) / math.sqrt(self.head_dim)
        # attn_weights: [B, H_q, 1, T_total]
        
        # Apply mask (usually None in decode)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 4. Aggregate attention to KV head dimension (GQA)
        attn_weights_grouped = attn_weights.view(
            B, H_kv, self.num_key_value_groups, 1, -1
        )  # [B, H_kv, group_size, 1, T_total]
        
        attn_weights_kv = attn_weights_grouped.mean(dim=2)  # [B, H_kv, 1, T_total]
        
        # 5. Accumulate attention to Window A
        if ws['window_a_size'] > 0:
            # Calculate Window A position in full KV cache
            compressed_len = 0 if ws['compressed_keys'] is None else ws['compressed_keys'].shape[2]
            window_a_start = compressed_len
            window_a_end = window_a_start + ws['window_a_size']
            
            # Extract attention to Window A
            window_a_attention = attn_weights_kv[:, :, 0, window_a_start:window_a_end]  # [B, H_kv, window_a_size]
            
            # Accumulate
            ws['accumulator'][:, :, :ws['window_a_size']] += window_a_attention
            ws['token_observation_count'][:, :, :ws['window_a_size']] += 1
            
            if DEBUG:
                print(f"Accumulated attention to Window A (size: {ws['window_a_size']})")
        
        # 6. Compute attention output (using original attn_weights)
        attn_output = torch.matmul(
            attn_weights,  # [B, H_q, 1, T_total]
            repeat_kv(full_values, self.num_key_value_groups)  # [B, H_q, T_total, D]
        )  # [B, H_q, 1, D]
        
        # 7. Check if compression should be triggered
        if ws['window_a_size'] == W and ws['window_b_size'] >= W:
            if DEBUG:
                print(f"Compression condition met!")
            self._trigger_compression()
        
        return attn_output

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
        Forward pass for DiffSparseKV attention.
        
        Automatically routes to prefill or decode based on past_key_value.
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )
        
        bsz, q_len, _ = hidden_states.size()
        
        if DEBUG:
            print(f"\n{'='*60}")
            print(f"Forward pass: batch_size={bsz}, q_len={q_len}")
            print(f"past_key_value is None: {past_key_value is None}")
        
        # Linear projections
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Get sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Route to prefill or decode
        if past_key_value is None:
            # Prefill stage
            if DEBUG:
                print(f"=== PREFILL STAGE ===")
            self.generation_count = 0
            attn_output, attn_weights, past_key_value = self._prefill_with_diff_sparse(
                query_states, key_states, value_states, attention_mask, kv_seq_len
            )
        else:
            # Decode stage
            if DEBUG:
                print(f"=== DECODE STAGE (step {self.generation_count + 1}) ===")
            self.generation_count += 1
            attn_output, attn_weights, past_key_value = self._decode_with_diff_sparse(
                query_states, key_states, value_states, attention_mask, past_key_value
            )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Output projection
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        
        if DEBUG:
            print(f"{'='*60}\n")
        
        return attn_output, attn_weights, past_key_value

    def _prefill_with_diff_sparse(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_seq_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor]]:
        """
        Prefill stage with DiffSparseKV and observation window optimization.
        
        Steps:
        1. Compute attention output using Flash Attention (if enabled) or standard attention
        2. Compute observation window attention for importance calculation
        3. Compute importance scores
        4. Compute and save per-layer thresholds
        5. Apply sparsification (compress tokens before Window A)
        6. Initialize dual windows
        """
        bsz, _, q_len, _ = query_states.shape
        
        # Save prefill length for N scaling in decode
        self.prefill_length = kv_seq_len
        self.current_sequence_length = kv_seq_len
        
        if DEBUG:
            print(f"Prefill: q_len={q_len}, kv_seq_len={kv_seq_len}")
            print(f"Saved prefill_length={self.prefill_length} for N scaling")
            print(f"use_flash_attention={self.use_flash_attention}")
        
        # 1. Compute attention output
        if self.use_flash_attention:
            # Try to use Flash Attention for main computation
            try:
                from flash_attn import flash_attn_func
                
                # Flash Attention expects [B, seq_len, H, D]
                q_flash = query_states.transpose(1, 2)  # [B, seq_len, H_q, D]
                k_flash = repeat_kv(key_states, self.num_key_value_groups).transpose(1, 2)
                v_flash = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)
                
                attn_output = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    causal=True
                ).transpose(1, 2)  # [B, H_q, seq_len, D]
                
                if DEBUG:
                    print(f"Using Flash Attention for main computation")
                    
            except ImportError:
                raise ImportError(
                    "Flash Attention is required when use_flash_attention=True. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )
        else:
            # Standard attention computation
            attn_weights = torch.matmul(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                    f"but is {attn_weights.size()}"
                )
            
            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                        f"but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights,
                    torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Compute output
            attn_output = torch.matmul(
                attn_weights,
                repeat_kv(value_states, self.num_key_value_groups)
            )
            
            if DEBUG:
                print(f"Using standard attention for main computation")
        
        # 2. Compute observation window attention for importance calculation
        obs_window_size = min(self.obs_window_size, kv_seq_len)
        
        if DEBUG:
            print(f"Computing observation window attention: obs_window_size={obs_window_size}")
        obs_attn_weights_kv = self.compute_observation_window_attention(
            query_states, key_states, obs_window_size
        )  # [B, H_kv, obs_window, seq_len]
        # 3. Compute importance scores (per-head)
        importance_scores_per_head = self.importance_calculator.compute_diffkv_importance(
            obs_attn_weights_kv
        )  # Returns [B, H_kv, seq_len]
        
        if self.importance_mode == "value_aware":
            # VATP-style importance: attention-derived token score multiplied
            # by the L1 norm of the corresponding value vector.
            value_norm = value_states.abs().sum(dim=-1).to(importance_scores_per_head.dtype)
            importance_scores_per_head = importance_scores_per_head * value_norm
            
            # VATP keeps the first few sink tokens because pruning them can
            # shift the attention distribution of all remaining tokens.
            if self.value_sink_keep > 0:
                sink_keep = min(self.value_sink_keep, importance_scores_per_head.shape[-1])
                max_scores = importance_scores_per_head.max(dim=-1, keepdim=True).values
                importance_scores_per_head[..., :sink_keep] = max_scores + 1.0
        
        # 3.5. Aggregate importance scores across heads.
        mean_scores = importance_scores_per_head.mean(dim=1)
        max_scores = importance_scores_per_head.max(dim=1).values
        if self.head_aggregation_mode == "mean":
            importance_scores_aggregated = mean_scores
        elif self.head_aggregation_mode == "max":
            importance_scores_aggregated = max_scores
        elif self.head_aggregation_mode == "hybrid":
            alpha = self.head_aggregation_alpha
            importance_scores_aggregated = alpha * mean_scores + (1.0 - alpha) * max_scores
        elif self.head_aggregation_mode == "top2_mean":
            topk = min(2, importance_scores_per_head.shape[1])
            importance_scores_aggregated = importance_scores_per_head.topk(topk, dim=1).values.mean(dim=1)
        else:
            raise ValueError(f"Unsupported head_aggregation_mode: {self.head_aggregation_mode}")
        
        if self.head_disagreement_ratio >= 0.0:
            peak_ratio = max_scores / mean_scores.clamp_min(1e-6)
            importance_scores_aggregated = torch.where(
                peak_ratio >= self.head_disagreement_ratio,
                max_scores,
                importance_scores_aggregated,
            )
        
        # Expand aggregated scores back to [B, H_kv, seq_len] for compatibility
        importance_scores = importance_scores_aggregated.unsqueeze(1).expand_as(importance_scores_per_head)
        
        # 4. Compute per-layer thresholds (based on aggregated importance)
        thresholds = self.threshold_manager.compute_per_layer_thresholds(
            importance_scores_aggregated.unsqueeze(1)  # [B, 1, seq_len] - treat as single head
        )
        
        # Save to instance variable
        self.diff_sparse_thresholds = thresholds
        self.thresholds_computed = True
        
        if DEBUG:
            print(f"Thresholds computed and saved:")
            print(f"  thresholds: {thresholds}")
        
        # 5. Apply sparsification
        # Key: Only compress tokens before Window A [0 : seq_len - window_size]
        # This includes the front part of observation window
        compress_end = kv_seq_len - self.window_size
        
        if compress_end > 0:
            # Tokens to compress: [0 : compress_end]
            compress_keys = key_states[:, :, :compress_end, :]
            compress_values = value_states[:, :, :compress_end, :]
            compress_importance = importance_scores[:, :, :compress_end]
            
            # Apply sparsification
            compressed_keys, compressed_values = self.sparsity_applier.classify_and_apply_sparsity(
                compress_importance,
                compress_keys,
                compress_values,
                thresholds=thresholds
            )
            
            # Window A tokens remain dense [compress_end : seq_len]
            window_a_keys = key_states[:, :, -self.window_size:, :]
            window_a_values = value_states[:, :, -self.window_size:, :]
            
            # Concatenate
            key_states_full = torch.cat([compressed_keys, window_a_keys], dim=2)
            value_states_full = torch.cat([compressed_values, window_a_values], dim=2)
            
            if DEBUG:
                compress_k_sparsity = self.calculate_sparsity(compressed_keys)
                compress_v_sparsity = self.calculate_sparsity(compressed_values)
                print(f"Prefill sparsification:")
                print(f"  Compressed range: [0:{compress_end}] ({compress_end} tokens)")
                print(f"    K sparsity: {compress_k_sparsity:.2%}")
                print(f"    V sparsity: {compress_v_sparsity:.2%}")
                print(f"  Window A: [{compress_end}:{kv_seq_len}] ({self.window_size} tokens, dense)")
        else:
            # Sequence too short, no compression needed
            key_states_full = key_states
            value_states_full = value_states
            
            if DEBUG:
                print(f"Prefill: sequence too short, no compression")
        
        # 6. Initialize dual windows
        self.initialize_dual_window_after_prefill(key_states_full, value_states_full)
        
        # 7. Prepare past_key_value
        # IMPORTANT: the cached length used by generate() must remain the logical
        # sequence length, not the physically retained KV length after eviction.
        # Otherwise generation re-feeds a prompt suffix and shifts position_ids.
        actual_kv_seq_len = key_states_full.shape[2]
        past_key_value = (key_states_full, value_states_full, kv_seq_len)
        
        if DEBUG:
            total_k_sparsity = self.calculate_sparsity(key_states_full)
            total_v_sparsity = self.calculate_sparsity(value_states_full)
            print(f"Prefill completed:")
            print(f"  Original sequence length: {kv_seq_len} tokens")
            print(f"  Logical sequence length: {kv_seq_len} tokens")
            print(f"  Actual KV cache length: {actual_kv_seq_len} tokens")
            if actual_kv_seq_len < kv_seq_len:
                print(f"  Tokens evicted: {kv_seq_len - actual_kv_seq_len} ({(kv_seq_len - actual_kv_seq_len) / kv_seq_len * 100:.1f}%)")
            print(f"  Overall sparsity: K={total_k_sparsity:.2%}, V={total_v_sparsity:.2%}")
        
        return attn_output, None, past_key_value

    def _decode_with_diff_sparse(
        self,
        query_states: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor]]:
        """
        Decode stage - SIMPLIFIED VERSION (no dual-window, just standard attention)
        
        This version only uses the sparsified KV cache from prefill,
        but doesn't apply any additional sparsification during decode.
        
        Steps:
        1. Concatenate new token to past KV cache
        2. Compute standard attention
        3. Return updated KV cache
        """
        if DEBUG:
            print(f"\n=== DECODE STAGE (Simplified) ===")
        
        # 1. Get past KV cache
        # past_length is the logical sequence length seen by generation.
        past_keys, past_values, past_length = past_key_value
        past_kv_len = past_keys.shape[2]
        
        if DEBUG:
            print(f"Logical past length: {past_length}")
            print(f"Physical KV cache length: {past_kv_len}")
            print(f"New token: 1")
        
        # 2. Concatenate new token to past cache
        key_states = torch.cat([past_keys, key_state], dim=2)
        value_states = torch.cat([past_values, value_state], dim=2)
        kv_seq_len = key_states.shape[2]
        
        if DEBUG:
            print(f"Total KV cache length: {kv_seq_len}")
        
        # 3. Compute standard attention (with GQA)
        bsz = query_states.shape[0]
        
        attn_weights = torch.matmul(
            query_states,
            repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # 调整mask以匹配kv_seq_len
            mask_seq_len = attention_mask.shape[-1]
            
            if mask_seq_len != kv_seq_len:
                if mask_seq_len < kv_seq_len:
                    # mask太短，需要扩展
                    pad_size = kv_seq_len - mask_seq_len
                    pad_mask = torch.zeros(
                        attention_mask.shape[:-1] + (pad_size,),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=-1)
                else:
                    # mask太长，需要截断
                    attention_mask = attention_mask[..., :kv_seq_len]
            
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Compute output
        attn_output = torch.matmul(
            attn_weights,
            repeat_kv(value_states, self.num_key_value_groups)
        )
        
        # 4. Prepare past_key_value for next step
        # Keep the logical sequence length monotonic even if tokens were evicted.
        past_key_value = (key_states, value_states, past_length + key_state.shape[2])
        
        if DEBUG:
            k_sparsity = self.calculate_sparsity(key_states)
            v_sparsity = self.calculate_sparsity(value_states)
            print(f"Decode completed:")
            print(f"  KV cache sparsity: K={k_sparsity:.2%}, V={v_sparsity:.2%}")
            print(f"=== END DECODE ===\n")
        
        return attn_output, None, past_key_value
    
    def _decode_with_diff_sparse_ORIGINAL(
        self,
        query_states: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor]]:
        """
        ORIGINAL Decode stage with DiffSparseKV dual-window mechanism.
        
        This is the original implementation with dual-window.
        Currently disabled for testing.
        
        Steps:
        1. Check thresholds are computed
        2. Update current sequence length
        3. Call decode_step_with_dual_window
        4. Build full KV cache for past_key_value
        """
        # 1. Check thresholds
        if not self.thresholds_computed or self.diff_sparse_thresholds is None:
            raise RuntimeError(
                "Thresholds not computed. Must run prefill stage first before decode."
            )
        
        # 2. Update current sequence length (add 1 for the new token)
        self.current_sequence_length += 1
        
        if DEBUG:
            print(f"Decode step: current_sequence_length={self.current_sequence_length}")
        
        # 2. Decode single step with dual-window mechanism
        attn_output = self.decode_step_with_dual_window(
            query_states, key_state, value_state, attention_mask
        )
        
        # 3. Build full KV cache for past_key_value
        ws = self.window_state
        kv_list = []
        
        if ws['compressed_keys'] is not None:
            kv_list.append((ws['compressed_keys'], ws['compressed_values']))
        if ws['window_a_keys'] is not None:
            kv_list.append((ws['window_a_keys'], ws['window_a_values']))
        if ws['window_b_keys'] is not None:
            kv_list.append((ws['window_b_keys'], ws['window_b_values']))
        
        full_keys = torch.cat([k for k, v in kv_list], dim=2)
        full_values = torch.cat([v for k, v in kv_list], dim=2)
        kv_seq_len = full_keys.shape[2]
        
        # Preserve logical position growth for generation bookkeeping.
        past_key_value = (full_keys, full_values, self.current_sequence_length)
        
        if DEBUG:
            total_k_sparsity = self.calculate_sparsity(full_keys)
            total_v_sparsity = self.calculate_sparsity(full_values)
            print(f"Decode step completed:")
            print(f"  Total KV cache: {kv_seq_len} tokens")
            print(f"  Overall sparsity: K={total_k_sparsity:.2%}, V={total_v_sparsity:.2%}")
        
        return attn_output, None, past_key_value



# ============================================================================
# Decoder Layer
# ============================================================================

from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
)


class LlamaDiffSparseKVDecoderLayer(nn.Module):
    """Decoder layer with DiffSparseKV attention."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaDiffSparseKVAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask: attention mask of size `(batch_size, sequence_length)`
            output_attentions: Whether or not to return the attentions tensors
            use_cache: If set to `True`, `past_key_values` key value states are returned
            past_key_value: cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )
        
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs



# ============================================================================
# Model
# ============================================================================

from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    BaseModelOutputWithPast,
    add_start_docstrings_to_model_forward,
    LLAMA_INPUTS_DOCSTRING,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class LlamaModel_DiffSparseKV(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers with DiffSparseKV.
    
    Args:
        config: LlamaConfig
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDiffSparseKVDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        
        # embed positions
        hidden_states = inputs_embeds
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                warnings.warn(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if DEBUG:
                print(f"\n{'='*60}")
                print(f"Layer {idx}")
                print(f"{'='*60}")
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



# ============================================================================
# Causal LM
# ============================================================================

from transformers.models.llama.modeling_llama import (
    CausalLMOutputWithPast,
    replace_return_docstrings,
    _CONFIG_FOR_DOC,
    CrossEntropyLoss,
)
from transformers.cache_utils import DynamicCache


def create_diff_sparse_kv_config(
    base_config: LlamaConfig,
    enable_diff_sparse: bool = True,
    target_distribution: list = None,
    sparsity_levels: list = None,
    diff_sparse_window_size: int = 32,
    obs_window_size: int = 128,
    use_flash_attention: bool = True,
    debug_diff_sparse: bool = False
) -> LlamaConfig:
    """
    Create a LlamaConfig with DiffSparseKV parameters.
    
    Args:
        base_config: Base LlamaConfig
        enable_diff_sparse: Enable DiffSparseKV
        target_distribution: Target distribution for 3 levels [Level0, Level1, Level2]
        sparsity_levels: Sparsity levels for 3 levels [0%, X%, 100%]
        diff_sparse_window_size: Window size for dual-window mechanism (Window A/B)
        obs_window_size: Observation window size for importance calculation
        use_flash_attention: Use Flash Attention for main computation
        debug_diff_sparse: Enable debug output
    
    Returns:
        Updated LlamaConfig
    """
    if target_distribution is None:
        target_distribution = [0.05, 0.75, 0.20]
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.7, 1.0]
    
    # Validate observation window size
    if obs_window_size < diff_sparse_window_size:
        raise ValueError(
            f"obs_window_size ({obs_window_size}) must be >= window_size ({diff_sparse_window_size})"
        )
    
    # Add DiffSparseKV parameters to config
    base_config.use_diff_sparse_kv = enable_diff_sparse
    base_config.target_distribution = target_distribution
    base_config.sparsity_levels = sparsity_levels
    base_config.window_size = diff_sparse_window_size
    base_config.obs_window_size = obs_window_size
    base_config.use_flash_attention = use_flash_attention
    base_config.debug_diff_sparse = debug_diff_sparse
    
    return base_config


class LlamaForCausalLM_DiffSparseKV(LlamaPreTrainedModel):
    """LLaMA for Causal Language Modeling with DiffSparseKV."""
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_DiffSparseKV(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        
        Returns:
        
        Example:
        
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        
        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            
            if DEBUG:
                print(f"prepare_inputs_for_generation: past_length={past_length}")
            
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            
            input_ids = input_ids[:, remove_prefix_length:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
