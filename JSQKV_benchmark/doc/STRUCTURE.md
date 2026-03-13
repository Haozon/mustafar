# JSQKV Benchmark 目录结构

```
JSQKV_benchmark/
│
├── benchmark_config.yaml           # 主配置文件（修改这里配置测试参数）
│   ├── models                      # 模型配置（路径、序列长度）
│   ├── test_configs                # 测试配置（稀疏度、量化）
│   ├── plot_schemes                # 绘图方案（对比组合）
│   └── batch_sizes                 # 测试参数
│
├── benchmark_throughput.py         # 主测试脚本
│   └── 功能：运行吞吐量测试，保存JSON结果
│
├── plot_throughput.py              # 绘图脚本
│   └── 功能：读取JSON结果，生成PDF矢量图
│
├── run_benchmark.sh                # 一键运行脚本
│   └── 功能：自动运行测试+绘图
│
├── test_config.py                  # 配置验证脚本
│   └── 功能：验证配置文件正确性
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── config_loader.py            # 配置加载器
│   ├── model_loader.py             # 模型加载器
│   └── metrics.py                  # 性能指标计算
│
├── results/                        # 结果目录
│   ├── raw_data/                   # JSON数据文件
│   │   ├── llama2_7b_results.json
│   │   └── llama3_8b_results.json
│   └── plots/                      # PDF图表文件
│       ├── throughput_comparison_50.pdf
│       ├── throughput_comparison_70.pdf
│       └── ...
│
├── README.md                       # 详细文档
├── QUICKSTART.md                   # 快速开始指南
├── STRUCTURE.md                    # 本文件（目录结构说明）
└── .gitignore                      # Git忽略文件

```

## 文件说明

### 配置文件

- **benchmark_config.yaml**: 核心配置文件
  - 定义测试模型（路径、序列长度）
  - 定义测试配置（稀疏度、量化参数）
  - 定义绘图方案（对比组合、样式）
  - 定义测试参数（batch size、重复次数）

### 主要脚本

- **benchmark_throughput.py**: 测试脚本
  - 加载模型（Dense/Sparse/Quant）
  - 测量吞吐量、TTFT、TPOT、内存
  - 保存JSON结果

- **plot_throughput.py**: 绘图脚本
  - 读取JSON结果
  - 根据方案配置生成对比图
  - 输出PDF矢量图

- **run_benchmark.sh**: 一键运行
  - 自动运行测试
  - 自动生成图表
  - 支持参数选项

- **test_config.py**: 配置验证
  - 检查配置文件语法
  - 验证配置完整性
  - 显示配置摘要

### 工具模块

- **utils/config_loader.py**
  - 加载YAML配置
  - 验证配置完整性
  - 提供配置访问接口

- **utils/model_loader.py**
  - 根据配置加载模型
  - 支持Dense/Sparse/Quant三种模式
  - 自动配置模型参数

- **utils/metrics.py**
  - 测量吞吐量
  - 测量TTFT/TPOT
  - 测量内存占用

### 结果目录

- **results/raw_data/**
  - 存储JSON格式的原始数据
  - 每个模型一个文件
  - 自动备份带时间戳的版本

- **results/plots/**
  - 存储PDF矢量图
  - 每个方案一个文件
  - 同时生成PNG预览图

## 数据流

```
benchmark_config.yaml
        ↓
benchmark_throughput.py
        ↓
results/raw_data/*.json
        ↓
plot_throughput.py
        ↓
results/plots/*.pdf
```

## 使用流程

1. **配置**: 编辑 `benchmark_config.yaml`
2. **验证**: 运行 `python test_config.py`
3. **测试**: 运行 `python benchmark_throughput.py`
4. **绘图**: 运行 `python plot_throughput.py`
5. **查看**: 打开 `results/plots/*.pdf`

或者直接运行：`bash run_benchmark.sh`

## 扩展性

### 添加新模型

在 `benchmark_config.yaml` 的 `models` 部分添加：

```yaml
models:
  my_model:
    path: "/path/to/model"
    input_length: 2048
    output_length: 2048
    display_name: "My Model"
```

### 添加新配置

在 `benchmark_config.yaml` 的 `test_configs` 部分添加：

```yaml
test_configs:
  my_config:
    k_sparsity: 0.8
    v_sparsity: 0.8
    use_quant: true
    quant_bits: 4
    display_name: "My Config"
```

### 添加新绘图方案

在 `benchmark_config.yaml` 的 `plot_schemes` 部分添加：

```yaml
plot_schemes:
  my_scheme:
    name: "my_plot"
    title: "My Comparison"
    configs: ["config1", "config2"]
    styles:
      config1:
        color: "#FF0000"
        marker: "o"
        linestyle: "-"
```

## 依赖关系

```
benchmark_throughput.py
    ├── utils/config_loader.py
    ├── utils/model_loader.py
    └── utils/metrics.py

plot_throughput.py
    └── utils/config_loader.py

run_benchmark.sh
    ├── benchmark_throughput.py
    └── plot_throughput.py
```

## 注意事项

1. 所有配置通过 `benchmark_config.yaml` 管理
2. 结果自动保存，支持增量测试
3. 支持灵活的绘图组合
4. PDF矢量图适合论文使用
5. 自动备份结果文件
