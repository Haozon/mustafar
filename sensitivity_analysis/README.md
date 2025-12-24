# 层级KV Cache敏感度分析 - 贪心搜索方法

## 📋 概述

这是一个快速验证版本的贪心搜索实现，用于分析LLM各层对KV Cache稀疏化的敏感度。

### 核心特性

- ✅ **Per-layer稀疏度配置**：每层独立的稀疏度设置
- ✅ **K和V统一稀疏度**：简化搜索空间，K和V使用相同稀疏度
- ✅ **Loss评估**：使用Cross Entropy Loss作为评估指标（更直接反映模型性能）
- ✅ **贪心搜索策略**：迭代选择最优层进行稀疏化
- ✅ **完整可视化**：自动生成分析图表

## 🚀 快速开始

### 1. 运行分析

```bash
# 方式1: 使用脚本（推荐，默认使用loss评估）
bash sensitivity_analysis/run_warm_start_search.sh

# 方式2: 使用专门的loss评估脚本
bash sensitivity_analysis/run_loss_based_search.sh

# 方式3: 直接运行Python
python sensitivity_analysis/greedy_search_simple.py --eval_metric loss
```

### 2. 查看结果

```bash
# 可视化结果
python sensitivity_analysis/visualize_results.py

# 查看JSON结果
cat sensitivity_results/greedy_search_results.json
```

## ⚙️ 配置参数

在 `greedy_search_simple.py` 的 `main()` 函数中修改配置：

```python
config = GreedySearchConfig(
    model_path="/home/zh/model/Meta-Llama-3-8B-Instruct",  # 模型路径
    initial_sparsity=0.4,      # 初始稀疏度
    step_size=0.05,            # 稀疏度增加步长
    target_sparsity=0.7,       # 目标稀疏度
    num_samples=30,            # 验证样本数（快速验证用30）
    max_length=512,            # 序列长度
    eval_metric="loss",        # 评估指标：loss（推荐）或ppl
    output_dir="./sensitivity_results"  # 输出目录
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_sparsity` | 0.4 | 初始稀疏度，所有层从此开始 |
| `step_size` | 0.05 | 每次迭代的稀疏度增加量 |
| `target_sparsity` | 0.7 | 目标稀疏度，达到后停止 |
| `num_samples` | 30 | 验证样本数，越多越准确但越慢 |
| `max_length` | 512 | 输入序列最大长度 |
| `eval_metric` | "loss" | 评估指标，loss更直接反映性能 |

### Loss vs PPL 评估指标

- **Loss (推荐)**：直接使用Cross Entropy Loss，更准确反映模型性能变化
- **PPL**：Perplexity，是Loss的指数形式，传统上更常用但可能放大数值差异

## 📊 输出结果

### 1. JSON结果文件

`sensitivity_results/greedy_search_results.json` 包含：

```json
{
  "config": {
    "model_path": "...",
    "target_compression": 0.5,
    "num_layers": 32
  },
  "sparsity_config": {
    "0": 0.3,
    "1": 0.5,
    ...
  },
  "sensitivity_scores": {
    "0": 0.85,
    "1": 0.42,
    ...
  },
  "iteration_history": [...]
}
```

### 2. 可视化图表

`sensitivity_results/greedy_search_visualization.png` 包含4个子图：

1. **Layer-wise Sparsity Configuration**: 每层的稀疏度配置
2. **Layer Sensitivity Scores**: 每层的敏感度分数
3. **Search Progress**: 迭代过程中perplexity和retention的变化
4. **Sparsity vs Sensitivity**: 稀疏度与敏感度的相关性

## 🔍 结果解读

### 敏感度分数

- **高敏感度（>0.7）**：该层对稀疏化敏感，应保留更多信息（低稀疏度）
- **中敏感度（0.3-0.7）**：该层可以适度稀疏化
- **低敏感度（<0.3）**：该层对稀疏化鲁棒，可以激进压缩（高稀疏度）

### 典型模式

根据文档，通常会观察到：

- **早期层（0-5）**：高敏感度，学习基础特征
- **中间层（10-20）**：低敏感度，存在冗余
- **后期层（28-31）**：高敏感度，任务特定处理

## ⏱️ 预期运行时间

在A100 GPU上，Llama-3-8B模型：

| 配置 | 样本数 | 候选数 | 预计时间 |
|------|--------|--------|----------|
| 快速验证 | 30 | 4 | ~40分钟 |
| 标准配置 | 50 | 4 | ~1小时 |
| 完整分析 | 100 | 5 | ~2.5小时 |

## 🛠️ 故障排除

### 内存不足

如果遇到OOM错误：

```python
# 减少样本数
num_samples=20

# 减少序列长度
max_length=256
```

### 速度太慢

```python
# 减少候选数
sparsity_levels=[0.0, 0.5, 0.7]

# 减少样本数
num_samples=20
```

### 结果不稳定

```python
# 增加样本数
num_samples=50

# 使用更细粒度的候选
sparsity_levels=[0.0, 0.2, 0.4, 0.6, 0.8]
```

## 📝 下一步

完成贪心搜索后，可以：

1. **对比其他方法**：实现梯度方法、重构误差方法
2. **验证结果**：在LongBench上测试找到的配置
3. **优化配置**：基于敏感度分数手动调整
4. **扩展实验**：测试不同模型、不同任务

## 🔗 相关文件

- `greedy_search_simple.py`: 主实现文件
- `visualize_results.py`: 可视化工具
- `run_greedy_search.sh`: 运行脚本
- `../doc/层级KV敏感度分析方案与实际效果调研.md`: 理论文档
