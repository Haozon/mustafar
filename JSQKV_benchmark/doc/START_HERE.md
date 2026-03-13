# 🚀 JSQKV Benchmark - 开始使用

## ✅ 配置已验证通过！

你的 JSQKV Benchmark 已经配置完成并通过验证。

## 📋 当前配置

### 模型
- **Llama-2 7B**: 输入2048 + 输出2048 tokens
- **Llama-3 8B Instruct**: 输入4096 + 输出4096 tokens

### 测试配置
1. **dense**: Dense baseline (无稀疏，无量化)
2. **sparse_50**: 50%稀疏度
3. **sparse_70**: 70%稀疏度
4. **sparse_50_quant_2bit**: 50%稀疏 + 2bit量化
5. **sparse_70_quant_2bit**: 70%稀疏 + 2bit量化

### 绘图方案
1. **scheme_1**: Dense vs Sparse-50% vs Sparse-50%-Quant (复现论文图7)
2. **scheme_2**: Dense vs Sparse-70% vs Sparse-70%-Quant
3. **scheme_3**: 不同稀疏度对比
4. **scheme_4**: 量化效果对比

### 测试参数
- Batch sizes: [1, 2, 4, 6, 8]
- 重复次数: 3次
- 预热tokens: 10

## 🎯 快速开始

### 方式1: 一键运行（推荐）

```bash
# 激活环境
conda activate mustar

# 进入目录
cd JSQKV_benchmark

# 运行完整测试
bash run_benchmark.sh
```

这会自动完成所有测试并生成PDF图表。

### 方式2: 分步运行

```bash
# 激活环境
conda activate mustar
cd JSQKV_benchmark

# 步骤1: 运行benchmark测试
python benchmark_throughput.py

# 步骤2: 生成图表
python plot_throughput.py
```

### 方式3: 小规模测试（推荐先运行）

```bash
conda activate mustar
cd JSQKV_benchmark

# 只测试一个模型和配置（快速验证）
python benchmark_throughput.py --models llama2_7b --configs sparse_50

# 生成对应图表
python plot_throughput.py --schemes scheme_1
```

## 📊 查看结果

测试完成后，结果保存在：

```bash
# 查看数据文件
ls results/raw_data/*.json

# 查看PDF图表
ls results/plots/*.pdf

# 用PDF阅读器打开
evince results/plots/throughput_comparison_50.pdf
# 或
xdg-open results/plots/throughput_comparison_50.pdf
```

## 🔧 自定义配置

### 修改稀疏度

编辑 `benchmark_config.yaml`：

```yaml
test_configs:
  sparse_50:
    k_sparsity: 0.6  # 改为60%
    v_sparsity: 0.6
```

### 添加新配置

```yaml
test_configs:
  sparse_80:
    k_sparsity: 0.8
    v_sparsity: 0.8
    use_quant: false
    display_name: "Mustafar-KV-80%"
```

### 修改batch size

```yaml
batch_sizes: [1, 2, 4]  # 减少测试规模
```

### 修改重复次数

```yaml
num_repeats: 1  # 加快测试速度
```

修改后记得验证配置：

```bash
python test_config.py
```

## 📚 文档索引

- **START_HERE.md** (本文件): 快速开始
- **QUICKSTART.md**: 快速开始指南
- **README.md**: 详细文档
- **EXAMPLES.md**: 12个使用示例
- **STRUCTURE.md**: 目录结构说明
- **SUMMARY.md**: 项目总结

## 💡 常用命令

```bash
# 验证配置
python test_config.py

# 完整测试
bash run_benchmark.sh

# 只测试特定模型
python benchmark_throughput.py --models llama2_7b

# 只测试特定配置
python benchmark_throughput.py --configs sparse_50 sparse_70

# 只生成特定图表
python plot_throughput.py --schemes scheme_1

# 只绘图不测试（使用已有结果）
bash run_benchmark.sh --skip-benchmark

# 只测试不绘图
bash run_benchmark.sh --skip-plot
```

## ⚠️ 注意事项

1. **显存**: Dense模型在大batch size时可能OOM，可以先测试稀疏配置
2. **时间**: 完整测试可能需要数小时，建议先小规模测试
3. **环境**: 确保已激活 `mustar` conda环境
4. **路径**: 确保模型路径正确（已在配置中设置）

## 🐛 故障排除

### 显存不足
```bash
# 减小batch size
vim benchmark_config.yaml
# 修改: batch_sizes: [1, 2, 4]
```

### 测试时间太长
```bash
# 减少重复次数
vim benchmark_config.yaml
# 修改: num_repeats: 1
```

### 模型路径错误
```bash
# 检查模型路径
ls /home/zh/model/Llama-2-7b-hf
ls /home/zh/model/Meta-Llama-3-8B-Instruct
```

## 🎉 推荐工作流

```bash
# 1. 激活环境
conda activate mustar
cd JSQKV_benchmark

# 2. 验证配置
python test_config.py

# 3. 小规模测试（验证流程，约5-10分钟）
python benchmark_throughput.py --models llama2_7b --configs sparse_50

# 4. 查看结果
ls results/raw_data/
ls results/plots/

# 5. 确认无误后，运行完整测试（可能需要数小时）
bash run_benchmark.sh

# 6. 查看所有生成的PDF图表
ls results/plots/*.pdf
```

## 📈 预期输出

测试完成后会看到：

```
============================================================
BENCHMARK RESULTS SUMMARY
============================================================

llama2_7b:
----------------------------------------------------------------------

  dense:
    BS=1: 75.23 tokens/sec, Memory=12.34 GB
    BS=2: 125.67 tokens/sec, Memory=15.67 GB
    ...

  sparse_50:
    BS=1: 78.45 tokens/sec, Memory=8.12 GB
    BS=2: 135.89 tokens/sec, Memory=10.45 GB
    ...

  sparse_50_quant_2bit:
    BS=1: 82.34 tokens/sec, Memory=5.67 GB
    BS=2: 145.23 tokens/sec, Memory=7.23 GB
    ...
```

## 🎨 生成的图表

- `throughput_comparison_50.pdf`: 50%稀疏度对比（复现论文图7）
- `throughput_comparison_70.pdf`: 70%稀疏度对比
- `throughput_comparison_sparsity.pdf`: 不同稀疏度对比
- `throughput_comparison_quantization.pdf`: 量化效果对比

所有图表都是PDF矢量图，适合论文使用。

## 🚀 现在开始！

```bash
conda activate mustar
cd JSQKV_benchmark
bash run_benchmark.sh
```

祝测试顺利！如有问题，请查看其他文档或提交issue。
