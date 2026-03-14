# DiffSparseKV Usage Examples

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Configuration](#advanced-configuration)
4. [Component-Level Usage](#component-level-usage)
5. [Evaluation on LongBench](#evaluation-on-longbench)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# From DiffSparseKV directory
cd DiffSparseKV
pip install -e .
```

### Minimal Example

```python
import torch
from transformers import AutoTokenizer, LlamaConfig
from diffsparsekv import create_diff_sparse_kv_config, LlamaForCausalLMDiffSparseKV

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create config with DiffSparseKV
base_config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0]
)

# Load model
model = LlamaForCausalLMDiffSparseKV.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Basic Usage

### Example 1: Standard Generation

```python
import torch
from transformers import AutoTokenizer
from diffsparsekv import create_diff_sparse_kv_config, LlamaForCausalLMDiffSparseKV
from transformers import LlamaConfig

# Setup
model_path = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure DiffSparseKV
base_config = LlamaConfig.from_pretrained(model_path)
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],  # 5% keep all, 75% keep 30%, 20% drop all
    sparsity_levels=[0.0, 0.7, 1.0],
    diff_sparse_window_size=128,
    obs_window_size=128,
    debug_diff_sparse=False,
    use_flash_attention=True
)

# Load model
model = LlamaForCausalLMDiffSparseKV.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Generate
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Example 2: Long Context Processing

```python
import torch
from transformers import AutoTokenizer
from diffsparsekv import create_diff_sparse_kv_config, LlamaForCausalLMDiffSparseKV
from transformers import LlamaConfig

# Setup for long context
model_path = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure for long context (8K tokens)
base_config = LlamaConfig.from_pretrained(model_path)
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.10, 0.70, 0.20],  # More conservative
    sparsity_levels=[0.0, 0.7, 1.0],
    diff_sparse_window_size=256,  # Larger window for long context
    obs_window_size=256,
    use_flash_attention=True
)

# Load model
model = LlamaForCausalLMDiffSparseKV.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Long document
long_document = """
[Your long document here - can be several thousand tokens]
"""

prompt = f"{long_document}\n\nQuestion: What is the main topic?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

print(f"Input length: {inputs.input_ids.shape[1]} tokens")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Answer: {answer}")
```

---

## Advanced Configuration

### Example 3: Custom Sparsity Distribution

```python
from diffsparsekv import DiffSparseKVConfig, create_diff_sparse_kv_config
from transformers import LlamaConfig

base_config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Aggressive sparsity (80% average)
aggressive_config = create_diff_sparse_kv_config(
    base_config=base_config,
    target_distribution=[0.05, 0.50, 0.45],  # More tokens at high sparsity
    sparsity_levels=[0.0, 0.7, 1.0],
    diff_sparse_window_size=128
)
print(f"Expected sparsity: {aggressive_config.expected_sparsity:.1%}")

# Conservative sparsity (50% average)
conservative_config = create_diff_sparse_kv_config(
    base_config=base_config,
    target_distribution=[0.20, 0.60, 0.20],  # More tokens preserved
    sparsity_levels=[0.0, 0.5, 1.0],  # Lower sparsity levels
    diff_sparse_window_size=128
)
print(f"Expected sparsity: {conservative_config.expected_sparsity:.1%}")

# Balanced sparsity (70% average)
balanced_config = create_diff_sparse_kv_config(
    base_config=base_config,
    target_distribution=[0.10, 0.70, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0],
    diff_sparse_window_size=128
)
print(f"Expected sparsity: {balanced_config.expected_sparsity:.1%}")
```

### Example 4: Debug Mode

```python
from diffsparsekv import create_diff_sparse_kv_config, LlamaForCausalLMDiffSparseKV
from transformers import LlamaConfig
import torch

base_config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Enable debug mode for detailed logging
config = create_diff_sparse_kv_config(
    base_config=base_config,
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0],
    debug_diff_sparse=True,  # Enable debug logging
    use_flash_attention=True
)

model = LlamaForCausalLMDiffSparseKV.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# You'll see detailed logs during generation
# Including threshold values, distribution accuracy, etc.
```

---

## Component-Level Usage

### Example 5: Using ImportanceCalculator Directly

```python
import torch
from diffsparsekv import DiffKVImportanceCalculator
from diffsparsekv.config import DiffSparseKVConfig

# Create config
config = DiffSparseKVConfig(
    enable_diff_sparse=True,
    target_distribution=[0.05, 0.75, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0]
)

# Initialize calculator
importance_calc = DiffKVImportanceCalculator(config)

# Simulate attention weights [batch, num_kv_heads, num_q_heads_per_kv, q_len, kv_len]
batch_size = 1
num_kv_heads = 8
num_q_heads_per_kv = 4
q_len = 1
kv_len = 2048

attention_weights = torch.randn(
    batch_size, num_kv_heads, num_q_heads_per_kv, q_len, kv_len
).softmax(dim=-1)

# Compute importance scores
importance_scores = importance_calc.compute_diffkv_importance(attention_weights)

print(f"Importance scores shape: {importance_scores.shape}")  # [B, H, T]
print(f"Mean importance: {importance_scores.mean():.4f}")
print(f"Max importance: {importance_scores.max():.4f}")
```

### Example 6: Using ThresholdManager

```python
import torch
from diffsparsekv import GlobalThresholdManager
from diffsparsekv.config import DiffSparseKVConfig

config = DiffSparseKVConfig(
    target_distribution=[0.05, 0.75, 0.20],
    sparsity_levels=[0.0, 0.7, 1.0]
)

threshold_mgr = GlobalThresholdManager(config)

# Simulate importance scores
importance_scores = torch.randn(1, 8, 2048).abs()

# Compute thresholds
threshold_high, threshold_low = threshold_mgr.compute_per_layer_thresholds(
    importance_scores
)

print(f"Threshold High: {threshold_high:.4f}")
print(f"Threshold Low: {threshold_low:.4f}")

# Classify tokens
level_0_mask = importance_scores >= threshold_high
level_1_mask = (importance_scores >= threshold_low) & (importance_scores < threshold_high)
level_2_mask = importance_scores < threshold_low

print(f"Level 0 (keep all): {level_0_mask.sum().item()} tokens")
print(f"Level 1 (70% sparse): {level_1_mask.sum().item()} tokens")
print(f"Level 2 (drop all): {level_2_mask.sum().item()} tokens")
```

### Example 7: Using SparsityApplier

```python
import torch
from diffsparsekv import SparsityClassifierApplier
from diffsparsekv.config import DiffSparseKVConfig

config = DiffSparseKVConfig(
    sparsity_levels=[0.0, 0.7, 1.0]
)

sparsity_applier = SparsityClassifierApplier(config)

# Simulate KV cache
batch_size, num_heads, seq_len, head_dim = 1, 8, 2048, 128
key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
importance_scores = torch.randn(batch_size, num_heads, seq_len).abs()

# Apply sparsity
compressed_keys, compressed_values = sparsity_applier.classify_and_apply_sparsity(
    importance_scores,
    key_states,
    value_states,
    thresholds=(0.5, 0.2)  # Example thresholds
)

print(f"Original shape: {key_states.shape}")
print(f"Compressed shape: {compressed_keys.shape}")
print(f"Non-zero ratio: {(compressed_keys != 0).float().mean():.2%}")
```

---

## Evaluation on LongBench

### Example 8: Running LongBench Evaluation

```python
# Use the provided script
import subprocess

result = subprocess.run([
    "python", "pred_long_bench_diff_sparse.py",
    "--model_name_or_path", "/path/to/Llama-2-7b-hf",
    "--mode", "diff_sparse_kv",
    "--diff_sparse_target_distribution", "0.05,0.75,0.20",
    "--diff_sparse_sparsity_levels", "0.0,0.7,1.0",
    "--diff_sparse_window_size", "128",
    "--obs_window_size", "128",
    "--use_flash_attention", "true",
    "--e", "0"
], capture_output=True, text=True)

print(result.stdout)
```

Or use the shell script:

```bash
cd DiffSparseKV
bash run_diffsparse.sh
```

---

## Troubleshooting

### Issue 1: High TTFT (Time to First Token)

**Problem**: First token takes too long to generate.

**Solution**:
```python
# Reduce observation window size
config = create_diff_sparse_kv_config(
    base_config=base_config,
    obs_window_size=64,  # Reduced from 128
    diff_sparse_window_size=64
)
```

### Issue 2: Accuracy Drop

**Problem**: Model accuracy decreases significantly.

**Solution**:
```python
# Use more conservative sparsity
config = create_diff_sparse_kv_config(
    base_config=base_config,
    target_distribution=[0.15, 0.65, 0.20],  # Preserve more tokens
    sparsity_levels=[0.0, 0.5, 0.9]  # Lower sparsity levels
)
```

### Issue 3: Out of Memory

**Problem**: GPU runs out of memory.

**Solution**:
```python
# Use more aggressive sparsity
config = create_diff_sparse_kv_config(
    base_config=base_config,
    target_distribution=[0.05, 0.50, 0.45],  # More aggressive
    sparsity_levels=[0.0, 0.8, 1.0],  # Higher sparsity
    diff_sparse_window_size=64  # Smaller window
)
```

### Issue 4: Import Errors

**Problem**: Cannot import diffsparsekv modules.

**Solution**:
```bash
# Make sure you're in the right directory
cd /path/to/DiffSparseKV

# Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="/path/to/DiffSparseKV:$PYTHONPATH"
```

---

## Performance Tips

1. **Always use Flash Attention** for better performance
2. **Adjust window size** based on your memory constraints
3. **Tune target distribution** for your specific use case
4. **Disable debug mode** in production
5. **Use appropriate dtype** (float16 recommended)

## Next Steps

- Check [README.md](README.md) for detailed documentation
- See [STRUCTURE.md](STRUCTURE.md) for architecture details
- Run benchmarks with `run_diffsparse.sh`
- Compare with baseline using `run_baseline.sh`
