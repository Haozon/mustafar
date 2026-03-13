# JSQKV Benchmark

吞吐量测试框架，用于对比不同稀疏度和量化配置下的模型性能。

## 目录结构

```
JSQKV_benchmark/
├── benchmark_config.yaml           # 配置文件（可修改稀疏度、量化等参数）
├── benchmark_throughput.py         # 主测试脚本
├── plot_throughput.py              # 绘图脚本（生成PDF矢量图）
├── run_benchmark.sh                # 一键运行脚本
├── utils/
│   ├── config_loader.py            # 配置加载器
│   ├── model_loader.py             # 模型加载器
│   └── metrics.py                  # 性能指标计算
└── results/
    ├── raw_data/                   # JSON结果数据
    └── plots/                      # PDF图表
```

## 快速开始

### 1. 配置测试参数

编辑 `benchmark_config.yaml` 文件：

```yaml
# 修改稀疏度
test_configs:
  sparse_50:
    k_sparsity: 0.5  # 修改为你想要的稀疏度
    v_sparsity: 0.5
    
# 添加新的测试配置
  sparse_80:
    k_sparsity: 0.8
    v_sparsity: 0.8
    use_quant: false
    display_name: "Mustafar-KV-80%"
```

### 2. 运行测试

#### 方式1: 一键运行（推荐）

```bash
cd JSQKV_benchmark
bash run_benchmark.sh
```

#### 方式2: 分步运行

```bash
# 运行benchmark测试
python benchmark_throughput.py

# 生成图表
python plot_throughput.py
```

### 3. 查看结果

- **数据文件**: `results/raw_data/*.json`
- **图表文件**: `results/plots/*.pdf`

## 高级用法

### 测试特定模型

```bash
# 只测试 Llama-2 7B
python benchmark_throughput.py --models llama2_7b

# 测试多个模型
python benchmark_throughput.py --models llama2_7b llama3_8b
```

### 测试特定配置

```bash
# 只测试 dense 和 sparse_50
python benchmark_throughput.py --configs dense sparse_50

# 只测试量化配置
python benchmark_throughput.py --configs sparse_50_quant_2bit sparse_70_quant_2bit
```

### 生成特定图表

```bash
# 只生成 scheme_1 的图表
python plot_throughput.py --schemes scheme_1

# 生成多个方案的图表
python plot_throughput.py --schemes scheme_1 scheme_2
```

### 使用自定义配置文件

```bash
python benchmark_throughput.py --config my_config.yaml
python plot_throughput.py --config my_config.yaml
```

## 配置文件说明

### 模型配置

```yaml
models:
  llama2_7b:
    path: "/path/to/model"           # 模型路径
    input_length: 2048                # 输入序列长度
    output_length: 2048               # 输出序列长度
    display_name: "Llama-2 7B"       # 显示名称
```

### 测试配置

```yaml
test_configs:
  config_name:
    k_sparsity: 0.5                  # Key稀疏度 (0.0-1.0)
    v_sparsity: 0.5                  # Value稀疏度 (0.0-1.0)
    use_quant: true                  # 是否使用量化
    quant_bits: 2                    # 量化位宽
    display_name: "显示名称"          # 图例中的名称
```

### 绘图方案

```yaml
plot_schemes:
  scheme_name:
    name: "output_filename"          # 输出文件名（不含扩展名）
    title: "图表标题"                 # 图表标题
    configs: ["config1", "config2"]  # 要对比的配置列表
    styles:                          # 每个配置的绘图样式
      config1:
        color: "#FF69B4"             # 颜色（十六进制）
        marker: "^"                  # 标记样式
        linestyle: "--"              # 线型
```

## 添加新的测试配置

### 示例1: 添加80%稀疏度配置

在 `benchmark_config.yaml` 中添加：

```yaml
test_configs:
  sparse_80:
    k_sparsity: 0.8
    v_sparsity: 0.8
    use_quant: false
    display_name: "Mustafar-KV-80%"
  
  sparse_80_quant_2bit:
    k_sparsity: 0.8
    v_sparsity: 0.8
    use_quant: true
    quant_bits: 2
    display_name: "Mustafar-KV-80%-Quant-2bit"
```

### 示例2: 添加新的绘图方案

```yaml
plot_schemes:
  scheme_5:
    name: "throughput_comparison_80"
    title: "Throughput comparison (80% sparsity)"
    configs: ["dense", "sparse_80", "sparse_80_quant_2bit"]
    styles:
      dense:
        color: "#FF69B4"
        marker: "^"
        linestyle: "--"
      sparse_80:
        color: "#2E8B57"
        marker: "s"
        linestyle: "-"
      sparse_80_quant_2bit:
        color: "#FF8C00"
        marker: "o"
        linestyle: "--"
```

## 测试参数

- **batch_sizes**: `[1, 2, 4, 6, 8]` - 测试的批次大小
- **num_repeats**: `3` - 每个配置重复测试次数
- **warmup_tokens**: `10` - 预热token数量

## 输出指标

每个测试配置会输出以下指标：

- **Throughput**: 吞吐量 (tokens/second)
- **TTFT**: Time to First Token (ms)
- **TPOT**: Time per Output Token (ms)
- **Peak Memory**: 峰值显存占用 (GB)
- **Batch Time**: 批次生成时间 (seconds)

## 故障排除

### 显存不足 (OOM)

如果遇到显存不足，可以：

1. 减小 batch_sizes: `[1, 2, 4]`
2. 减小序列长度
3. 跳过 dense 配置的大 batch size 测试

### 模型加载失败

检查模型路径是否正确：

```yaml
models:
  llama2_7b:
    path: "/home/zh/model/Llama-2-7b-hf"  # 确保路径正确
```

### 缺少依赖

安装必要的Python包：

```bash
pip install torch transformers pyyaml matplotlib
```

## 注意事项

1. **测试时间**: 完整测试可能需要数小时，建议先用小规模配置测试
2. **显存占用**: Dense模型在大batch size时显存占用较高
3. **结果备份**: 每次运行会自动备份带时间戳的结果文件
4. **PDF质量**: 生成的PDF为矢量图，适合论文使用

## 示例工作流

```bash
# 1. 修改配置文件
vim benchmark_config.yaml

# 2. 先测试一个小配置验证
python benchmark_throughput.py --models llama2_7b --configs sparse_50

# 3. 确认无误后运行完整测试
bash run_benchmark.sh

# 4. 查看结果
ls results/plots/*.pdf
```

## 联系方式

如有问题，请查看项目文档或提交issue。
