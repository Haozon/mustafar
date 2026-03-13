# LeanSparseKV: 自适应稀疏KV缓存系统

## 项目概述

LeanSparseKV实现了一个基于token重要性的5级自适应稀疏化系统，通过DiffKV算法动态分配不同的稀疏度等级，在保持模型精度的同时显著减少KV缓存的内存占用。

## 核心特性

- **5级稀疏度分配**: [0%, 50%, 70%, 90%, 100%] 根据token重要性自适应分配
- **DiffKV重要性评估**: 基于attention权重的归一化重要性分数计算
- **分位数阈值校准**: 通过校准数据集自动计算最优阈值参数
- **目标稀疏度控制**: 精确控制平均稀疏度（默认70%）
- **幅度剪枝**: 在每个稀疏度等级内保留最大幅度的元素

## 文件结构

```
LeanSparseKV/
├── README.md                           # 项目说明文档
├── architecture.md                     # 详细技术架构文档
├── config_example.yaml                 # 配置文件示例
├── calibrate_sparsity_thresholds.py    # 核心：阈值校准脚本
├── validate_thresholds.py              # 阈值验证脚本
├── quantile_consistency_analysis.py    # 分位数一致性分析
└── run_threshold_calibration.sh        # 完整校准流程脚本
```

## 快速开始

### 1. 环境要求

```bash
pip install torch transformers numpy matplotlib seaborn tqdm
```

### 2. 阈值校准

```bash
# 使用默认配置校准阈值
./run_threshold_calibration.sh /path/to/model

# 或者手动运行
python calibrate_sparsity_thresholds.py \
    --model_path /path/to/model \
    --dataset math \
    --num_samples 200 \
    --target_sparsity 0.70
```

### 3. 验证阈值

```bash
python validate_thresholds.py \
    --thresholds_file results/thresholds.json \
    --model_path /path/to/model \
    --dataset math
```

## 核心算法流程

### 1. DiffKV重要性分数计算

```python
def compute_diffkv_importance(attention_weights):
    # [B, H, T_q, T_k] -> [B, H, T_k]
    importance = attention_weights.mean(dim=2)
    # 序列长度归一化
    normalized_importance = importance * T_k
    return normalized_importance
```

### 2. 5级稀疏度分配

```python
# 基于阈值分配稀疏度等级
if score >= α_h:     level = 0 (0%稀疏)   # 最重要
elif score >= α_mh:  level = 1 (50%稀疏)  # 重要
elif score >= α_m:   level = 2 (70%稀疏)  # 中等
elif score >= α_ml:  level = 3 (90%稀疏)  # 低重要性
else:                level = 4 (100%稀疏) # 完全剪枝
```

### 3. 幅度剪枝

```python
# 在每个稀疏度等级内保留最大幅度的元素
num_to_keep = int((1 - sparsity_level) * D)
_, top_indices = torch.topk(torch.abs(vector), num_to_keep)
sparse_vector = vector * magnitude_mask
```

## 配置选项

### 目标分布配置

```yaml
# 标准分布 (数学推理任务)
target_distribution: [0.10, 0.20, 0.30, 0.20, 0.20]  # 平均69%稀疏

# 保守分布 (高精度要求)
target_distribution: [0.15, 0.25, 0.30, 0.20, 0.10]  # 平均61.5%稀疏

# 激进分布 (长文本场景)
target_distribution: [0.08, 0.17, 0.30, 0.22, 0.23]  # 平均75%稀疏
```

### 粒度选择

- `per_layer`: 每层使用相同阈值（推荐）
- `per_head`: 每个attention head独立阈值

## 使用示例

### 在推理中应用稀疏化

```python
import json
import torch

# 加载预计算的阈值
with open('thresholds.json', 'r') as f:
    threshold_data = json.load(f)
    thresholds = threshold_data['thresholds']

def apply_sparsity_to_kv_cache(model, layer_id, K, V, attention_weights):
    # 计算重要性分数
    importance_scores = compute_diffkv_importance(attention_weights)
    
    # 获取该层的阈值
    layer_thresholds = thresholds[str(layer_id)]
    α_h = layer_thresholds['alpha_h']
    α_mh = layer_thresholds['alpha_mh']
    α_m = layer_thresholds['alpha_m']
    α_ml = layer_thresholds['alpha_ml']
    
    # 分配稀疏度等级
    sparsity_map = assign_sparsity_levels(importance_scores, (α_h, α_mh, α_m, α_ml))
    
    # 应用幅度剪枝
    K_sparse = apply_magnitude_sparsity(K, sparsity_map)
    V_sparse = apply_magnitude_sparsity(V, sparsity_map)
    
    return K_sparse, V_sparse
```

## 性能指标

### 校准质量指标

- **稀疏度误差**: < 2% 为优秀
- **分布误差**: < 0.05 为优秀
- **层间一致性**: CV < 0.2 为可接受

### 内存节省

- **70%稀疏度**: 理论上节省70%的KV缓存内存
- **自适应分配**: 重要token保留更多信息，提高精度保持

## 支持的模型

- Llama-2-7B
- Llama-3-8B-Instruct
- Mistral-7B-Instruct-v0.2
- 其他基于transformer的因果语言模型

## 注意事项

1. **校准数据集**: 使用与目标任务相似的数据进行校准
2. **样本数量**: 200-500个样本通常足够获得稳定的阈值
3. **序列长度**: 校准时的序列长度应覆盖实际应用范围
4. **验证重要性**: 校准后务必在验证集上测试效果

## 引用

如果使用本项目，请引用相关论文：

```bibtex
@article{mustafar2024,
  title={Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference},
  author={...},
  journal={arXiv preprint arXiv:2505.22913},
  year={2024}
}
```

## 许可证

本项目遵循MIT许可证。