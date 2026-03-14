# JSQKV Benchmark 快速开始指南

## 1. 验证配置

首先测试配置文件是否正确：

```bash
cd JSQKV_benchmark
python test_config.py
```

如果看到 `✅ Configuration is valid!`，说明配置正确。

## 2. 运行完整测试

### 方式A: 一键运行（推荐）

```bash
bash run_benchmark.sh
```

这会自动完成：
1. 运行所有benchmark测试
2. 生成所有对比图

### 方式B: 分步运行

```bash
# 步骤1: 运行benchmark测试
python benchmark_throughput.py

# 步骤2: 生成图表
python plot_throughput.py
```

## 3. 快速测试（小规模验证）

如果想先快速验证流程，可以只测试一个模型和配置：

```bash
# 只测试 Llama-2 7B 的 sparse_50 配置
python benchmark_throughput.py --models llama2_7b --configs sparse_50

# 生成对应的图表
python plot_throughput.py --schemes scheme_1
```

## 4. 查看结果

测试完成后，结果保存在：

- **数据**: `results/raw_data/llama2_7b_results.json`
- **图表**: `results/plots/throughput_comparison_50.pdf`

## 5. 修改配置

### 修改稀疏度

编辑 `benchmark_config.yaml`：

```yaml
test_configs:
  sparse_50:
    k_sparsity: 0.6  # 改为60%稀疏度
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

### 添加新的对比图

```yaml
plot_schemes:
  my_scheme:
    name: "my_comparison"
    title: "My Custom Comparison"
    configs: ["dense", "sparse_50", "sparse_80"]
    styles:
      dense:
        color: "#FF69B4"
        marker: "^"
        linestyle: "--"
      sparse_50:
        color: "#2E8B57"
        marker: "s"
        linestyle: "-"
      sparse_80:
        color: "#4169E1"
        marker: "o"
        linestyle: "-"
```

## 6. 常见问题

### Q: 显存不足怎么办？

A: 减小batch size或跳过dense配置：

```bash
# 只测试稀疏配置
python benchmark_throughput.py --configs sparse_50 sparse_70
```

### Q: 测试时间太长？

A: 减少重复次数，编辑 `benchmark_config.yaml`：

```yaml
num_repeats: 1  # 改为1次
batch_sizes: [1, 2, 4]  # 减少batch size
```

### Q: 如何只重新生成图表？

A: 使用 `--skip-benchmark` 选项：

```bash
bash run_benchmark.sh --skip-benchmark
```

或直接运行：

```bash
python plot_throughput.py
```

## 7. 推荐工作流

```bash
# 1. 验证配置
python test_config.py

# 2. 小规模测试（验证流程）
python benchmark_throughput.py --models llama2_7b --configs sparse_50

# 3. 确认无误后，运行完整测试
bash run_benchmark.sh

# 4. 查看结果
ls results/plots/*.pdf
```

## 8. 输出示例

测试完成后会看到类似输出：

```
============================================================
BENCHMARK RESULTS SUMMARY
============================================================

llama2_7b:
----------------------------------------------------------------------

  dense:
    BS=1: 75.23 tokens/sec, Memory=12.34 GB
    BS=2: 125.67 tokens/sec, Memory=15.67 GB
    BS=4: 185.43 tokens/sec, Memory=21.23 GB

  sparse_50:
    BS=1: 78.45 tokens/sec, Memory=8.12 GB
    BS=2: 135.89 tokens/sec, Memory=10.45 GB
    BS=4: 220.34 tokens/sec, Memory=14.67 GB

  sparse_50_quant_2bit:
    BS=1: 82.34 tokens/sec, Memory=5.67 GB
    BS=2: 145.23 tokens/sec, Memory=7.23 GB
    BS=4: 265.78 tokens/sec, Memory=9.89 GB
```

## 9. 下一步

- 查看 `README.md` 了解详细文档
- 修改 `benchmark_config.yaml` 自定义测试
- 查看生成的PDF图表

祝测试顺利！🚀
