"""
DiffSparseKV: Differential Sparse Key-Value Cache for Efficient LLM Inference

A library for applying differential sparsity to KV cache in Large Language Models,
enabling efficient long-context inference with minimal accuracy loss.

Core Components:
- ImportanceCalculator: Compute token importance scores using DiffKV method
- ThresholdManager: Manage global sparsity thresholds
- SparsityApplier: Apply differential sparsity classification and compression
- WindowManager: Manage dual-window mechanism for decoding phase

Usage:
    from diffsparsekv import DiffSparseKVConfig, create_diff_sparse_kv_config
    from diffsparsekv import LlamaForCausalLMDiffSparseKV
    
    # Create configuration
    config = create_diff_sparse_kv_config(
        base_config=base_config,
        enable_diff_sparse=True,
        target_distribution=[0.05, 0.75, 0.20],
        sparsity_levels=[0.0, 0.7, 1.0],
        diff_sparse_window_size=128
    )
    
    # Load model with DiffSparseKV
    model = LlamaForCausalLMDiffSparseKV.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16
    )
"""

__version__ = "0.1.0"
__author__ = "DiffSparseKV Team"

# Core components
from .importance_calculator import DiffKVImportanceCalculator
from .threshold_manager import GlobalThresholdManager
from .sparsity_applier import SparsityClassifierApplier
from .window_manager import WindowManager

# Configuration
from .config import DiffSparseKVConfig, create_diff_sparse_kv_config
from .budget_generator import (
    ResolvedBudgetConfig,
    list_budget_templates,
    resolve_budget_config,
)

# Model integration
from .llama_integration import (
    LlamaForCausalLM_DiffSparseKV,
    LlamaDiffSparseKVAttention,
)

# 为了兼容性，提供别名
LlamaForCausalLMDiffSparseKV = LlamaForCausalLM_DiffSparseKV

__all__ = [
    # Version
    "__version__",
    
    # Core components
    "DiffKVImportanceCalculator",
    "GlobalThresholdManager",
    "SparsityClassifierApplier",
    "WindowManager",
    
    # Configuration
    "DiffSparseKVConfig",
    "create_diff_sparse_kv_config",
    "ResolvedBudgetConfig",
    "list_budget_templates",
    "resolve_budget_config",
    
    # Model integration
    "LlamaForCausalLM_DiffSparseKV",
    "LlamaForCausalLMDiffSparseKV",  # 别名
    "LlamaDiffSparseKVAttention",
]
