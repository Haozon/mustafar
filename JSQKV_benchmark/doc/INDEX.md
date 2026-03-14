# JSQKV Benchmark - 文件索引

## 📖 文档文件（按阅读顺序）

### 1. 快速开始
- **START_HERE.md** ⭐ - 从这里开始！配置已验证，可直接使用
- **QUICKSTART.md** - 快速开始指南，5分钟上手

### 2. 详细文档
- **README.md** - 完整的使用文档和API说明
- **EXAMPLES.md** - 12个实用示例，涵盖各种使用场景

### 3. 参考文档
- **STRUCTURE.md** - 目录结构和文件说明
- **SUMMARY.md** - 项目总结和功能概览
- **INDEX.md** (本文件) - 文件索引

## 🔧 核心文件

### 配置文件
- **benchmark_config.yaml** - 主配置文件
  - 定义模型（路径、序列长度）
  - 定义测试配置（稀疏度、量化）
  - 定义绘图方案（对比组合）
  - 定义测试参数（batch size等）

### 主要脚本
- **benchmark_throughput.py** - 吞吐量测试脚本
  - 加载模型
  - 运行benchmark
  - 保存JSON结果

- **plot_throughput.py** - 绘图脚本
  - 读取JSON结果
  - 生成PDF矢量图
  - 支持多种对比方案

- **run_benchmark.sh** - 一键运行脚本
  - 自动运行测试
  - 自动生成图表
  - 支持命令行参数

- **test_config.py** - 配置验证脚本
  - 验证配置语法
  - 检查配置完整性
  - 显示配置摘要

## 🛠️ 工具模块

### utils/
- **utils/__init__.py** - 模块初始化
- **utils/config_loader.py** - 配置加载器
  - 加载YAML配置
  - 验证配置完整性
  - 提供配置访问接口

- **utils/model_loader.py** - 模型加载器
  - 加载Dense模型
  - 加载Sparse模型
  - 加载Quant模型

- **utils/metrics.py** - 性能指标计算
  - 测量吞吐量
  - 测量TTFT/TPOT
  - 测量内存占用

## 📊 结果目录

### results/raw_data/
- 存储JSON格式的测试结果
- 每个模型一个文件
- 自动备份带时间戳的版本

### results/plots/
- 存储PDF矢量图
- 每个方案一个文件
- 同时生成PNG预览图

## 📝 其他文件

- **.gitignore** - Git忽略规则
- **results/raw_data/.gitkeep** - 保持目录结构
- **results/plots/.gitkeep** - 保持目录结构

## 🎯 文件用途速查

### 我想...

#### 开始使用
→ 阅读 **START_HERE.md**

#### 快速上手
→ 阅读 **QUICKSTART.md**

#### 查看示例
→ 阅读 **EXAMPLES.md**

#### 了解详细功能
→ 阅读 **README.md**

#### 修改配置
→ 编辑 **benchmark_config.yaml**

#### 验证配置
→ 运行 `python test_config.py`

#### 运行测试
→ 运行 `bash run_benchmark.sh`

#### 只测试不绘图
→ 运行 `python benchmark_throughput.py`

#### 只绘图不测试
→ 运行 `python plot_throughput.py`

#### 查看结果
→ 查看 `results/raw_data/*.json` 和 `results/plots/*.pdf`

#### 了解目录结构
→ 阅读 **STRUCTURE.md**

#### 了解项目概况
→ 阅读 **SUMMARY.md**

## 📚 推荐阅读顺序

### 新手用户
1. START_HERE.md - 快速开始
2. QUICKSTART.md - 快速指南
3. EXAMPLES.md - 查看示例
4. README.md - 深入了解

### 高级用户
1. SUMMARY.md - 项目概览
2. STRUCTURE.md - 目录结构
3. benchmark_config.yaml - 配置文件
4. utils/*.py - 源代码

### 故障排除
1. START_HERE.md - 常见问题
2. README.md - 故障排除章节
3. EXAMPLES.md - 相关示例

## 🔍 快速搜索

### 配置相关
- 修改稀疏度: benchmark_config.yaml → test_configs
- 修改batch size: benchmark_config.yaml → batch_sizes
- 添加新模型: benchmark_config.yaml → models
- 添加新方案: benchmark_config.yaml → plot_schemes

### 命令相关
- 验证配置: `python test_config.py`
- 完整测试: `bash run_benchmark.sh`
- 测试特定模型: `python benchmark_throughput.py --models MODEL`
- 生成特定图表: `python plot_throughput.py --schemes SCHEME`

### 结果相关
- JSON数据: results/raw_data/
- PDF图表: results/plots/
- 查看数据: `cat results/raw_data/llama2_7b_results.json`
- 查看图表: `ls results/plots/*.pdf`

## 📦 完整文件列表

```
JSQKV_benchmark/
├── benchmark_config.yaml           # 配置文件
├── benchmark_throughput.py         # 测试脚本
├── plot_throughput.py              # 绘图脚本
├── run_benchmark.sh                # 一键运行
├── test_config.py                  # 配置验证
├── START_HERE.md                   # 快速开始 ⭐
├── QUICKSTART.md                   # 快速指南
├── README.md                       # 详细文档
├── EXAMPLES.md                     # 使用示例
├── STRUCTURE.md                    # 目录结构
├── SUMMARY.md                      # 项目总结
├── INDEX.md                        # 本文件
├── .gitignore                      # Git配置
├── utils/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── model_loader.py
│   └── metrics.py
└── results/
    ├── raw_data/
    │   └── .gitkeep
    └── plots/
        └── .gitkeep
```

## 🎓 学习路径

### 路径1: 快速使用（10分钟）
1. START_HERE.md
2. 运行 `bash run_benchmark.sh`
3. 查看结果

### 路径2: 深入学习（30分钟）
1. START_HERE.md
2. QUICKSTART.md
3. EXAMPLES.md
4. 尝试修改配置
5. 运行测试

### 路径3: 完全掌握（1小时）
1. 阅读所有文档
2. 查看源代码
3. 尝试各种配置
4. 自定义绘图方案

## 💡 提示

- ⭐ 标记的文件是最重要的
- 所有文档都有详细的代码示例
- 配置文件有详细的注释
- 遇到问题先查看 START_HERE.md 的故障排除部分

## 🚀 现在开始

```bash
# 1. 阅读快速开始
cat START_HERE.md

# 2. 激活环境
conda activate mustar

# 3. 进入目录
cd JSQKV_benchmark

# 4. 验证配置
python test_config.py

# 5. 运行测试
bash run_benchmark.sh
```

祝使用愉快！🎉
