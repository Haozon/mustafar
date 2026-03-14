# 层级KV Cache敏感度分析 - 实施计划

## 📋 目标

为Llama-3-8B模型的每一层分配最优的KV Cache稀疏度，使得在达到目标压缩率的同时，最小化性能损失。

## 🎯 问题定义

**给定**：
- L=32层的Llama-3-8B模型
- 目标保留率：50% (compression ratio = 0.5)
- 稀疏度候选集：[0.0, 0.3, 0.5, 0.7]
- 验证集：wikitext-2 (10-30样本)

**目标**：
找到每层的最优稀疏度配置 `{s_1*, s_2*, ..., s_32*}`，使得：
- 平均保留率 = 50%
- Perplexity最小

## 📊 方法对比

| 方法 | 时间成本 | 内存成本 | 准确性 | 实现难度 | 推荐度 |
|------|---------|---------|--------|---------|--------|
| **贪心搜索** | 2-3h | 低 | 高 | 中 | ⭐⭐⭐⭐⭐ |
| 梯度方法 | 10-20min | 高 | 中 | 高 | ⭐⭐⭐⭐ |
| 重构误差 | 4-8h | 低 | 高 | 中 | ⭐⭐⭐ |
| 统计启发式 | 5-10min | 低 | 低 | 低 | ⭐⭐ |

## 🚀 推荐方案：贪心搜索（快速版）

### 为什么选择贪心搜索？

1. **直观易懂**：每步选择"性价比"最高的层
2. **不需要梯度**：实现简单，适用性广
3. **效果好**：经验上接近最优解
4. **可解释**：每层的选择有明确理由

### 核心算法

```
迭代过程：
1. 初始化：所有层稀疏度 = 0.0
2. 每次迭代：
   - 尝试所有层的下一个稀疏度级别
   - 计算效率 η = |压缩增益| / perplexity增量
   - 选择效率最高的层
   - 更新该层稀疏度
3. 直到达到目标保留率

敏感度计算：
- 被频繁选择且效率低的层 → 高敏感度
- σ_l = Σ (1 / η_l)
```

### 快速配置（推荐）

```python
config = FastGreedyConfig(
    model_path="/home/zh/model/Meta-Llama-3-8B-Instruct",
    sparsity_levels=[0.0, 0.5, 0.7],  # 3个候选
    target_compression=0.5,            # 保留50%
    num_samples=10,                    # 10个样本
    max_length=256,                    # 短序列
    batch_size=2                       # 批量处理
)
```

**预期时间**：2-3小时

### 加速策略

| 策略 | 加速比 | 影响 |
|------|--------|------|
| 样本数：100→10 | 10× | 结果可能有噪声，但足够验证 |
| 序列长度：512→256 | 2× | 对短文本任务影响小 |
| 候选数：5→3 | 1.5× | 粒度较粗，可后续细化 |
| 批量处理：1→2 | 1.3× | 提高GPU利用率 |

**总加速比**：~20×（从40小时→2小时）

## 📝 实施步骤

### 阶段1：快速验证（30分钟）

**目标**：验证代码正确性，估算完整时间

```bash
# 只测试5层
python sensitivity_analysis/test_5_layers.py
```

**配置**：
- 测试层：[0, 8, 16, 24, 31]
- 样本数：5
- 候选数：3

**输出**：
- 5层的敏感度排序
- 单次评估时间
- 预估完整实验时间

### 阶段2：完整搜索（2-3小时）

**目标**：获得所有32层的敏感度配置

```bash
# 运行快速贪心搜索
bash sensitivity_analysis/run_fast.sh
```

**输出**：
- `sensitivity_results/fast_greedy_results.json`
- 每层的稀疏度配置
- 每层的敏感度分数

### 阶段3：结果验证（30分钟）

**目标**：用更多样本验证最终配置

```bash
# 验证最终配置
python sensitivity_analysis/validate_config.py \
    --config sensitivity_results/fast_greedy_results.json \
    --num_samples 50
```

### 阶段4：可视化分析（10分钟）

```bash
# 生成可视化图表
python sensitivity_analysis/visualize_results.py \
    sensitivity_results/fast_greedy_results.json
```

**输出**：
- 每层稀疏度配置图
- 敏感度热力图
- 迭代过程曲线

## 📊 预期结果

### 典型敏感度模式

```
高敏感度层（应保留更多信息）：
- 早期层 (0-5)：学习基础特征
- 后期层 (28-31)：任务特定处理

低敏感度层（可激进压缩）：
- 中间层 (10-20)：存在冗余
```

### 示例配置

```json
{
  "sparsity_config": {
    "0": 0.3,   // 早期层：低稀疏度
    "8": 0.5,
    "16": 0.7,  // 中间层：高稀疏度
    "24": 0.5,
    "31": 0.3   // 后期层：低稀疏度
  },
  "sensitivity_scores": {
    "0": 0.85,  // 高敏感度
    "16": 0.23, // 低敏感度
    "31": 0.91  // 高敏感度
  }
}
```

## 🔍 结果验证

### 1. 检查敏感度模式

```python
# 应该观察到U型曲线
# 早期和后期层敏感度高，中间层低
```

### 2. 与uniform sparsity对比

```bash
# 测试uniform配置
python test_uniform.py --sparsity 0.5

# 测试per-layer配置
python test_perlayer.py --config results.json

# 对比性能
# Per-layer应该比uniform好5-10%
```

### 3. 稳定性检查

```bash
# 运行3次，检查结果一致性
for i in {1..3}; do
    python greedy_search_fast.py --seed $i
done
```

## ⚠️ 常见问题

### Q1: 内存不足

**解决方案**：
```python
# 减少样本数
num_samples = 5

# 减少序列长度
max_length = 128

# 使用gradient checkpointing
model.gradient_checkpointing_enable()
```

### Q2: 速度太慢

**解决方案**：
```python
# 进一步减少样本
num_samples = 5

# 减少候选数
sparsity_levels = [0.0, 0.7]

# 只测试部分层
test_layers = range(0, 32, 2)  # 每隔一层测试
```

### Q3: 结果不稳定

**解决方案**：
```python
# 增加样本数
num_samples = 20

# 多次运行取平均
# 固定随机种子
torch.manual_seed(42)
```

## 📈 后续工作

完成贪心搜索后，可以：

1. **对比其他方法**
   - 实现梯度方法（10分钟）
   - 对比结果一致性

2. **在LongBench上验证**
   - 使用找到的配置
   - 测试实际任务性能

3. **优化配置**
   - 基于敏感度手动调整
   - 尝试不同的目标压缩率

4. **扩展实验**
   - 测试不同模型（Llama-2-7B）
   - 测试不同任务

## 🎓 理论背景

### 贪心搜索的理论基础

**效率指标**：
```
η_l = |ΔC| / ΔL
```
- ΔC：压缩率变化
- ΔL：perplexity变化
- η_l：单位损失换取的压缩增益

**敏感度定义**：
```
σ_l = Σ (1 / η_l)
```
- 效率低 → 敏感度高
- 被频繁选择 → 敏感度累积

### 为什么有效？

1. **局部最优接近全局最优**：层间相对独立
2. **贪心策略合理**：优先压缩"便宜"的层
3. **经验验证**：在多个模型上效果好

## 📚 参考资料

- 文档：`doc/层级KV敏感度分析方案与实际效果调研.md`
- 代码：`sensitivity_analysis/greedy_search_fast.py`
- 加速指南：`sensitivity_analysis/SPEEDUP_GUIDE.md`

## ✅ 检查清单

开始实验前，确认：

- [ ] 模型路径正确：`/home/zh/model/Meta-Llama-3-8B-Instruct`
- [ ] GPU可用：`nvidia-smi`
- [ ] 依赖已安装：`pip install -r requirements.txt`
- [ ] 磁盘空间充足：至少10GB
- [ ] 时间充裕：预留3-4小时

开始实验：

```bash
# 1. 快速验证（30分钟）
python sensitivity_analysis/test_5_layers.py

# 2. 完整搜索（2-3小时）
bash sensitivity_analysis/run_fast.sh

# 3. 可视化结果（10分钟）
python sensitivity_analysis/visualize_results.py
```

祝实验顺利！🎉
