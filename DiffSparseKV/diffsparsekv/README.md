# DiffSparseKV Library

Differential Sparse Key-Value Cache for Efficient LLM Inference.

## Overview

DiffSparseKV is a library that applies differential sparsity to the KV cache in Large Language Models, enabling efficient long-context inference with minimal accuracy loss.

## Core Components

### 1. ImportanceCalculator (`importance_calculator.py`)
Computes token importance scores using the DiffKV method:
- Aggregates attention weights across query heads (GQA-aware)
- Computes cumulative importance for each token
- Supports both prefill and decode phases

### 2. ThresholdManager (`threshold_manager.py`)
Manages global sparsity thresholds:
- Computes two thresholds (high and low) to classify tokens into 3 levels
- Ensures target distribution is achieved
- Validates threshold effectiveness

### 3. SparsityApplier (`sparsity_applier.py`)
Applies differential sparsity to KV cache:
- Classifies tokens into 3 sparsity levels based on importance
- Applies magnitude pruning at different rates (0%, 70%, 100%)
- Preserves tensor shapes for compatibility

### 4. WindowManager (`window_manager.py`)
Manages dual-window mechanism for decoding:
- Maintains two sliding windows (Window A and Window B)
- Accumulates attention weights for importance tracking
- Triggers compression when windows are full

### 5. LlamaIntegration (`llama_integration.py`)
Integrates DiffSparseKV into Llama models:
- Custom attention layer with DiffSparseKV support
- Handles both prefill and decode phases
- Compatible with Flash Attention

## Installation

```bash
# From DiffSparseKV directory
pip install -e .
```

Or add to your Python path:
```python
import sys
sys.path.append('/path/to/DiffSparseKV')
```

## Usage

### Basic Usage

```python
from diffsparsekv import create_diff_sparse_kv_config, LlamaForCausalLMDiffSparseKV
from transformers import LlamaConfig
import torch

# Load base config
base_config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create DiffSparseKV config
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],  # 5% Level0, 75% Level1, 20% Level2
    sparsity_levels=[0.0, 0.7, 1.0],         # 0%, 70%, 100% sparsity
    diff_sparse_window_size=128,
    obs_window_size=128,
    debug_diff_sparse=False
)

# Load model
model = LlamaForCausalLMDiffSparseKV.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Use model for generation
output = model.generate(input_ids, max_new_tokens=100)
```

### Advanced Configuration

```python
from diffsparsekv import DiffSparseKVConfig

# Create custom configuration
diff_config = DiffSparseKVConfig(
    enable_diff_sparse=True,
    target_distribution=[0.10, 0.60, 0.30],  # Custom distribution
    sparsity_levels=[0.0, 0.5, 1.0],         # Custom sparsity levels
    diff_sparse_window_size=256,             # Larger window
    obs_window_size=256,
    debug_diff_sparse=True,                  # Enable debug logging
    use_flash_attention=True
)

# Get expected sparsity
expected = diff_config.get_expected_sparsity()
print(f"Expected average sparsity: {expected:.1%}")
```

### Using Individual Components

```python
from diffsparsekv import (
    DiffKVImportanceCalculator,
    GlobalThresholdManager,
    SparsityClassifierApplier
)

# Initialize components
importance_calc = DiffKVImportanceCalculator(config)
threshold_mgr = GlobalThresholdManager(config)
sparsity_applier = SparsityClassifierApplier(config)

# Compute importance scores
importance_scores = importance_calc.compute_diffkv_importance(attention_weights)

# Compute thresholds
threshold_high, threshold_low = threshold_mgr.compute_per_layer_thresholds(
    importance_scores
)

# Apply sparsity
compressed_keys, compressed_values = sparsity_applier.classify_and_apply_sparsity(
    importance_scores,
    key_states,
    value_states,
    thresholds=(threshold_high, threshold_low)
)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_diff_sparse` | `True` | Enable differential sparsity |
| `target_distribution` | `[0.05, 0.75, 0.20]` | Target distribution for 3 levels |
| `sparsity_levels` | `[0.0, 0.7, 1.0]` | Sparsity rates for each level |
| `diff_sparse_window_size` | `128` | Window size for dual-window mechanism |
| `obs_window_size` | `128` | Observation window for attention accumulation |
| `debug_diff_sparse` | `False` | Enable debug logging |
| `use_flash_attention` | `True` | Use Flash Attention for efficiency |

## Expected Sparsity Calculation

The expected average sparsity is calculated as:

```
expected_sparsity = Σ(target_distribution[i] × sparsity_levels[i])
```

For default settings:
```
0.05 × 0.0 + 0.75 × 0.7 + 0.20 × 1.0 = 0.725 = 72.5%
```

## Architecture

```
DiffSparseKV Pipeline:

1. Prefill Phase:
   Input → Attention → ImportanceCalculator → ThresholdManager
                                            ↓
   Compressed KV ← SparsityApplier ← Thresholds

2. Decode Phase:
   New Token → WindowManager → Accumulate Attention
                             ↓
   When Full → Trigger Compression (same as prefill)
```

## Performance Tips

1. **Window Size**: Larger windows (256-512) provide better importance estimation but use more memory
2. **Target Distribution**: Conservative distributions (e.g., [0.10, 0.70, 0.20]) preserve more tokens
3. **Flash Attention**: Always enable for better performance
4. **Debug Mode**: Disable in production for better speed

## Troubleshooting

### High TTFT (Time to First Token)
- Check if prefill phase is taking too long
- Consider reducing `obs_window_size`
- Ensure Flash Attention is enabled

### Accuracy Drop
- Adjust `target_distribution` to preserve more important tokens
- Reduce sparsity levels (e.g., [0.0, 0.5, 0.9])
- Check threshold effectiveness with debug mode

### Memory Issues
- Reduce `diff_sparse_window_size`
- Increase sparsity levels for more aggressive compression

## Citation

If you use DiffSparseKV in your research, please cite:

```bibtex
@article{diffsparse2024,
  title={DiffSparseKV: Differential Sparse Key-Value Cache for Efficient LLM Inference},
  author={Your Name},
  year={2024}
}
```

## License

[Your License Here]
