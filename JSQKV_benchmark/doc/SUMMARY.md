# JSQKV Benchmark 项目总结

## 项目概述

JSQKV Benchmark 是一个用于测试和对比不同稀疏度和量化配置下 LLM 模型吞吐量性能的完整框架。

## 核心功能

### 1. 灵活的配置系统
- ✅ YAML配置文件，易于修改
- ✅ 支持多种稀疏度配置（50%, 70%, 自定义）
- ✅ 支持量化配置（2-bit, 4-bit等）
- ✅ 支持多种绘图方案组合

### 2. 自动化测试
- ✅ 自动加载模型（Dense/Sparse/Quant）
- ✅ 自动测量吞吐量、TTFT、TPOT、内存
- ✅ 自动保存JSON结果
- ✅ 支持增量测试

### 3. 可视化
- ✅ 自动生成PDF矢量图
- ✅ 支持多种对比方案
- ✅ 可自定义绘图样式
- ✅ 同时生成PNG预览图

### 4. 易用性
- ✅ 一键运行脚本
- ✅ 配置验证工具
- ✅ 详细的文档和示例
- ✅ 灵活的命令行参数

## 文件清单

### 核心文件
```
benchmark_config.yaml           # 主配置文件
benchmark_throughput.py         # 测试脚本
plot_throughput.py              # 绘图脚本
run_benchmark.sh                # 一键运行
test_config.py                  # 配置验证
```

### 工具模块
```
utils/config_loader.py          # 配置加载
utils/model_loader.py           # 模型加载
utils/metrics.py                # 性能测量
```

### 文档
```
README.md                       # 详细文档
QUICKSTART.md                   # 快速开始
EXAMPLES.md                     # 使用示例
STRUCTURE.md                    # 目录结构
SUMMARY.md                      # 本文件
```

## 默认配置

### 模型
- Llama-2 7B: 2048+2048 tokens
- Llama-3 8B: 4096+4096 tokens

### 测试配置
- Dense (baseline)
- Sparse-50% (50%稀疏度)
- Sparse-70% (70%稀疏度)
- Sparse-50%-Quant-2bit (50%稀疏+2bit量化)
- Sparse-70%-Quant-2bit (70%稀疏+2bit量化)

### 绘图方案
- Scheme 1: Dense vs Sparse-50% vs Sparse-50%-Quant
- Scheme 2: Dense vs Sparse-70% vs Sparse-70%-Quant
- Scheme 3: 不同稀疏度对比
- Scheme 4: 量化效果对比

### 测试参数
- Batch sizes: [1, 2, 4, 6, 8]
- Num repeats: 3
- Warmup tokens: 10

## 使用流程

### 标准流程
```bash
1. python test_config.py          # 验证配置
2. bash run_benchmark.sh          # 运行测试
3. ls results/plots/*.pdf         # 查看结果
```

### 自定义流程
```bash
1. vim benchmark_config.yaml      # 修改配置
2. python test_config.py          # 验证配置
3. python benchmark_throughput.py # 运行测试
4. python plot_throughput.py      # 生成图表
```

## 输出结果

### 数据文件
```
results/raw_data/
├── llama2_7b_results.json
├── llama3_8b_results.json
└── *_results_20260128_*.json  # 带时间戳的备份
```

### 图表文件
```
results/plots/
├── throughput_comparison_50.pdf
├── throughput_comparison_70.pdf
├── throughput_comparison_sparsity.pdf
├── throughput_comparison_quantization.pdf
└── *.png  # PNG预览图
```

## 性能指标

每个测试配置输出：
- **Throughput**: 吞吐量 (tokens/second)
- **TTFT**: Time to First Token (ms)
- **TPOT**: Time per Output Token (ms)
- **Peak Memory**: 峰值显存 (GB)
- **Batch Time**: 批次时间 (seconds)

## 扩展性

### 添加新模型
在 `benchmark_config.yaml` 中添加模型配置即可

### 添加新稀疏度
在 `benchmark_config.yaml` 中添加测试配置即可

### 添加新对比图
在 `benchmark_config.yaml` 中添加绘图方案即可

### 修改测试参数
在 `benchmark_config.yaml` 中修改 batch_sizes、num_repeats 等

## 优势特点

1. **配置驱动**: 所有参数通过配置文件管理，无需修改代码
2. **模块化设计**: 清晰的模块划分，易于维护和扩展
3. **自动化**: 一键运行，自动保存结果
4. **灵活性**: 支持任意组合的对比测试
5. **可复现**: 固定随机种子，结果可复现
6. **专业输出**: PDF矢量图，适合论文使用

## 依赖要求

```
Python 3.8+
torch
transformers
pyyaml
matplotlib
```

## 注意事项

1. **显存**: Dense模型在大batch size时显存占用较高
2. **时间**: 完整测试可能需要数小时
3. **路径**: 确保模型路径正确
4. **备份**: 结果自动备份，不会覆盖

## 快速参考

### 常用命令
```bash
# 验证配置
python test_config.py

# 完整测试
bash run_benchmark.sh

# 测试特定配置
python benchmark_throughput.py --configs sparse_50

# 生成特定图表
python plot_throughput.py --schemes scheme_1

# 只绘图不测试
bash run_benchmark.sh --skip-benchmark
```

### 配置修改
```yaml
# 修改稀疏度
test_configs:
  my_config:
    k_sparsity: 0.6  # 自定义稀疏度

# 修改batch size
batch_sizes: [1, 2, 4]  # 减少测试规模

# 修改重复次数
num_repeats: 1  # 加快测试速度
```

## 文档索引

- **快速开始**: 查看 `QUICKSTART.md`
- **详细文档**: 查看 `README.md`
- **使用示例**: 查看 `EXAMPLES.md`
- **目录结构**: 查看 `STRUCTURE.md`
- **项目总结**: 本文件 `SUMMARY.md`

## 项目状态

✅ 核心功能完成
✅ 文档完善
✅ 示例丰富
✅ 可直接使用

## 下一步

1. 运行 `python test_config.py` 验证配置
2. 运行小规模测试验证流程
3. 根据需要修改配置
4. 运行完整测试
5. 查看生成的PDF图表

祝测试顺利！🚀
