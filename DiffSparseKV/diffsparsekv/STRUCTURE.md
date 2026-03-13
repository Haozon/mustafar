# DiffSparseKV Library Structure

## Directory Layout

```
DiffSparseKV/
├── diffsparsekv/                    # Core library package
│   ├── __init__.py                  # Package initialization and exports
│   ├── config.py                    # Configuration classes
│   ├── importance_calculator.py     # Token importance computation
│   ├── threshold_manager.py         # Threshold management
│   ├── sparsity_applier.py         # Sparsity application logic
│   ├── window_manager.py           # Dual-window mechanism
│   ├── llama_integration.py        # Llama model integration
│   ├── README.md                   # Library documentation
│   └── STRUCTURE.md                # This file
│
├── setup.py                        # Package installation script
├── pred_long_bench_diff_sparse.py  # LongBench evaluation script
├── eval_results.py                 # Results evaluation
├── run_baseline.sh                 # Baseline testing script
├── run_diffsparse.sh              # DiffSparseKV testing script
├── test_prefill_only.sh           # Prefill-only diagnostic script
└── eval_long_bench.sh             # Evaluation script
```

## Module Dependencies

```
config.py
    ↓
importance_calculator.py → threshold_manager.py → sparsity_applier.py
                                                         ↓
                                                   window_manager.py
                                                         ↓
                                                llama_integration.py
```

## Core Modules

### 1. `config.py`
- **Purpose**: Configuration management
- **Key Classes**: `DiffSparseKVConfig`
- **Key Functions**: `create_diff_sparse_kv_config()`
- **Dependencies**: `transformers.LlamaConfig`

### 2. `importance_calculator.py`
- **Purpose**: Compute token importance scores
- **Key Classes**: `DiffKVImportanceCalculator`
- **Key Methods**: 
  - `compute_diffkv_importance()`: Main importance computation
  - `aggregate_attention_for_kv()`: GQA-aware aggregation
- **Dependencies**: `torch`, `config`

### 3. `threshold_manager.py`
- **Purpose**: Manage sparsity thresholds
- **Key Classes**: `GlobalThresholdManager`
- **Key Methods**:
  - `compute_per_layer_thresholds()`: Compute thresholds
  - `validate_distribution()`: Check distribution accuracy
- **Dependencies**: `torch`, `config`

### 4. `sparsity_applier.py`
- **Purpose**: Apply differential sparsity
- **Key Classes**: `SparsityClassifierApplier`
- **Key Methods**:
  - `classify_and_apply_sparsity()`: Main sparsity application
  - `classify_tokens()`: Classify into 3 levels
  - `apply_magnitude_pruning()`: Apply pruning
- **Dependencies**: `torch`, `config`

### 5. `window_manager.py`
- **Purpose**: Dual-window mechanism for decoding
- **Key Classes**: `DualWindowManager`
- **Key Methods**:
  - `initialize_windows()`: Setup windows
  - `add_token()`: Add new token to window
  - `should_trigger_compression()`: Check if compression needed
- **Dependencies**: `torch`, `config`

### 6. `llama_integration.py`
- **Purpose**: Integrate DiffSparseKV into Llama
- **Key Classes**: 
  - `LlamaDiffSparseKVAttention`: Custom attention layer
  - `LlamaForCausalLMDiffSparseKV`: Full model
- **Key Methods**:
  - `forward()`: Attention forward pass
  - `_prefill_with_diff_sparse()`: Prefill phase
  - `_decode_with_diff_sparse()`: Decode phase
- **Dependencies**: All above modules, `transformers`

## Data Flow

### Prefill Phase
```
Input Tokens
    ↓
Attention Computation
    ↓
ImportanceCalculator.compute_diffkv_importance()
    ↓ (importance_scores)
ThresholdManager.compute_per_layer_thresholds()
    ↓ (threshold_high, threshold_low)
SparsityApplier.classify_and_apply_sparsity()
    ↓ (compressed_keys, compressed_values)
Output (Compressed KV Cache)
```

### Decode Phase
```
New Token
    ↓
WindowManager.add_token()
    ↓
Attention Computation
    ↓
WindowManager.accumulate_attention()
    ↓
WindowManager.should_trigger_compression()
    ↓ (if True)
[Same as Prefill Phase]
```

## Import Patterns

### For End Users
```python
# Simple import
from diffsparsekv import (
    create_diff_sparse_kv_config,
    LlamaForCausalLMDiffSparseKV
)

# Use the model
config = create_diff_sparse_kv_config(base_config, ...)
model = LlamaForCausalLMDiffSparseKV.from_pretrained(...)
```

### For Developers
```python
# Import individual components
from diffsparsekv.importance_calculator import DiffKVImportanceCalculator
from diffsparsekv.threshold_manager import GlobalThresholdManager
from diffsparsekv.sparsity_applier import SparsityClassifierApplier
from diffsparsekv.window_manager import DualWindowManager

# Use components separately
importance_calc = DiffKVImportanceCalculator(config)
threshold_mgr = GlobalThresholdManager(config)
# ...
```

## Testing Structure

```
tests/
├── test_importance_calculator.py
├── test_threshold_manager.py
├── test_sparsity_applier.py
├── test_window_manager.py
├── test_integration.py
└── test_end_to_end.py
```

## Configuration Files

### Model Configuration
```python
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0],
    diff_sparse_window_size=128,
    obs_window_size=128,
    debug_diff_sparse=False,
    use_flash_attention=True
)
```

## Extension Points

### 1. Custom Importance Calculator
```python
class CustomImportanceCalculator(DiffKVImportanceCalculator):
    def compute_diffkv_importance(self, attention_weights):
        # Custom implementation
        pass
```

### 2. Custom Sparsity Applier
```python
class CustomSparsityApplier(SparsityClassifierApplier):
    def apply_magnitude_pruning(self, tensor, sparsity_level):
        # Custom pruning strategy
        pass
```

### 3. Custom Window Manager
```python
class CustomWindowManager(DualWindowManager):
    def should_trigger_compression(self):
        # Custom compression trigger logic
        pass
```

## Performance Considerations

### Memory Usage
- **Window Size**: Larger windows use more memory but provide better importance estimation
- **Sparsity Levels**: Higher sparsity reduces memory usage
- **Flash Attention**: Reduces memory footprint significantly

### Computation Time
- **Prefill**: O(n²) for attention, O(n) for importance calculation
- **Decode**: O(n) per token, compression triggered every W tokens
- **Threshold Computation**: O(n log n) for sorting

### Optimization Tips
1. Use Flash Attention when available
2. Adjust window size based on available memory
3. Use appropriate sparsity levels for your use case
4. Disable debug mode in production

## Version History

### v0.1.0 (Current)
- Initial release
- Core DiffSparseKV implementation
- Llama integration
- Basic configuration system

### Planned Features
- [ ] Support for other model architectures (GPT, Mistral, etc.)
- [ ] Advanced threshold strategies
- [ ] Performance profiling tools
- [ ] Automatic hyperparameter tuning
- [ ] Quantization support
