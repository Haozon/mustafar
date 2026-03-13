# JSQKV Benchmark 使用示例

## 示例1: 复现论文图7（50%稀疏度对比）

### 配置
使用默认的 `scheme_1` 配置：

```yaml
plot_schemes:
  scheme_1:
    name: "throughput_comparison_50"
    title: "Throughput comparison (50% sparsity)"
    configs: ["dense", "sparse_50", "sparse_50_quant_2bit"]
```

### 运行
```bash
# 完整测试
bash run_benchmark.sh

# 或者分步运行
python benchmark_throughput.py --configs dense sparse_50 sparse_50_quant_2bit
python plot_throughput.py --schemes scheme_1
```

### 预期结果
生成 `results/plots/throughput_comparison_50.pdf`，包含：
- Dense (粉色三角虚线)
- Mustafar-KV-50% (绿色方块实线)
- Mustafar-KV-50%-Quant-2bit (橙色圆点虚线)

---

## 示例2: 对比70%稀疏度

### 配置
使用 `scheme_2` 配置：

```yaml
plot_schemes:
  scheme_2:
    name: "throughput_comparison_70"
    title: "Throughput comparison (70% sparsity)"
    configs: ["dense", "sparse_70", "sparse_70_quant_2bit"]
```

### 运行
```bash
python benchmark_throughput.py --configs dense sparse_70 sparse_70_quant_2bit
python plot_throughput.py --schemes scheme_2
```

### 预期结果
生成 `results/plots/throughput_comparison_70.pdf`

---

## 示例3: 对比不同稀疏度（无量化）

### 配置
使用 `scheme_3` 配置：

```yaml
plot_schemes:
  scheme_3:
    name: "throughput_comparison_sparsity"
    title: "Throughput comparison (different sparsity levels)"
    configs: ["dense", "sparse_50", "sparse_70"]
```

### 运行
```bash
python benchmark_throughput.py --configs dense sparse_50 sparse_70
python plot_throughput.py --schemes scheme_3
```

### 预期结果
对比Dense vs 50%稀疏 vs 70%稀疏的性能差异

---

## 示例4: 量化效果对比

### 配置
使用 `scheme_4` 配置：

```yaml
plot_schemes:
  scheme_4:
    name: "throughput_comparison_quantization"
    title: "Throughput comparison (quantization effect)"
    configs: ["sparse_50", "sparse_50_quant_2bit", "sparse_70", "sparse_70_quant_2bit"]
```

### 运行
```bash
python benchmark_throughput.py --configs sparse_50 sparse_50_quant_2bit sparse_70 sparse_70_quant_2bit
python plot_throughput.py --schemes scheme_4
```

### 预期结果
对比量化前后的性能提升

---

## 示例5: 添加自定义稀疏度（80%）

### 步骤1: 修改配置文件

编辑 `benchmark_config.yaml`，添加：

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

### 步骤2: 验证配置
```bash
python test_config.py
```

### 步骤3: 运行测试
```bash
python benchmark_throughput.py --configs dense sparse_80 sparse_80_quant_2bit
python plot_throughput.py --schemes scheme_5
```

---

## 示例6: 只测试单个模型

### 只测试 Llama-2 7B
```bash
python benchmark_throughput.py --models llama2_7b
```

### 只测试 Llama-3 8B
```bash
python benchmark_throughput.py --models llama3_8b
```

---

## 示例7: 快速验证流程（小规模测试）

### 修改配置
临时修改 `benchmark_config.yaml`：

```yaml
batch_sizes: [1, 2]  # 只测试2个batch size
num_repeats: 1       # 只重复1次
```

### 运行
```bash
python benchmark_throughput.py --models llama2_7b --configs sparse_50
python plot_throughput.py --schemes scheme_1
```

---

## 示例8: 生成多个对比图

### 一次生成所有图表
```bash
# 先运行所有测试
python benchmark_throughput.py

# 生成所有方案的图表
python plot_throughput.py
```

### 只生成特定图表
```bash
# 只生成scheme_1和scheme_2
python plot_throughput.py --schemes scheme_1 scheme_2
```

---

## 示例9: 使用已有结果重新绘图

如果已经运行过测试，只想修改绘图样式：

### 步骤1: 修改样式
编辑 `benchmark_config.yaml` 中的 `styles` 部分：

```yaml
plot_schemes:
  scheme_1:
    styles:
      dense:
        color: "#FF0000"  # 改为红色
        marker: "x"       # 改为叉号
```

### 步骤2: 重新绘图
```bash
python plot_throughput.py --schemes scheme_1
```

不需要重新运行测试！

---

## 示例10: 对比所有配置

### 添加综合对比方案

编辑 `benchmark_config.yaml`：

```yaml
plot_schemes:
  scheme_all:
    name: "throughput_comparison_all"
    title: "Throughput comparison (all configurations)"
    configs: ["dense", "sparse_50", "sparse_70", "sparse_50_quant_2bit", "sparse_70_quant_2bit"]
    styles:
      dense:
        color: "#808080"
        marker: "^"
        linestyle: "--"
      sparse_50:
        color: "#2E8B57"
        marker: "s"
        linestyle: "-"
      sparse_70:
        color: "#4169E1"
        marker: "o"
        linestyle: "-"
      sparse_50_quant_2bit:
        color: "#FF8C00"
        marker: "s"
        linestyle: "--"
      sparse_70_quant_2bit:
        color: "#DC143C"
        marker: "o"
        linestyle: "--"
```

### 运行
```bash
python plot_throughput.py --schemes scheme_all
```

---

## 示例11: 测试不同量化位宽

### 添加4-bit量化配置

编辑 `benchmark_config.yaml`：

```yaml
test_configs:
  sparse_50_quant_4bit:
    k_sparsity: 0.5
    v_sparsity: 0.5
    use_quant: true
    quant_bits: 4
    display_name: "Mustafar-KV-50%-Quant-4bit"

plot_schemes:
  scheme_quant_bits:
    name: "throughput_comparison_quant_bits"
    title: "Throughput comparison (different quantization bits)"
    configs: ["sparse_50", "sparse_50_quant_2bit", "sparse_50_quant_4bit"]
    styles:
      sparse_50:
        color: "#2E8B57"
        marker: "s"
        linestyle: "-"
      sparse_50_quant_2bit:
        color: "#FF8C00"
        marker: "o"
        linestyle: "--"
      sparse_50_quant_4bit:
        color: "#DC143C"
        marker: "^"
        linestyle: ":"
```

### 运行
```bash
python benchmark_throughput.py --configs sparse_50 sparse_50_quant_2bit sparse_50_quant_4bit
python plot_throughput.py --schemes scheme_quant_bits
```

---

## 示例12: 批量测试脚本

创建自定义测试脚本 `my_test.sh`：

```bash
#!/bin/bash

# 测试所有50%稀疏度配置
echo "Testing 50% sparsity configurations..."
python benchmark_throughput.py --configs dense sparse_50 sparse_50_quant_2bit

# 测试所有70%稀疏度配置
echo "Testing 70% sparsity configurations..."
python benchmark_throughput.py --configs dense sparse_70 sparse_70_quant_2bit

# 生成所有对比图
echo "Generating plots..."
python plot_throughput.py

echo "Done!"
```

运行：
```bash
chmod +x my_test.sh
./my_test.sh
```

---

## 常用命令速查

```bash
# 验证配置
python test_config.py

# 完整测试
bash run_benchmark.sh

# 只测试不绘图
bash run_benchmark.sh --skip-plot

# 只绘图不测试
bash run_benchmark.sh --skip-benchmark

# 测试特定模型
python benchmark_throughput.py --models llama2_7b

# 测试特定配置
python benchmark_throughput.py --configs sparse_50 sparse_70

# 生成特定图表
python plot_throughput.py --schemes scheme_1 scheme_2

# 查看结果
ls results/plots/*.pdf
ls results/raw_data/*.json
```

---

## 故障排除示例

### 问题1: 显存不足

```bash
# 解决方案：只测试小batch size
# 修改 benchmark_config.yaml
batch_sizes: [1, 2, 4]  # 去掉6和8
```

### 问题2: 测试时间太长

```bash
# 解决方案：减少重复次数
# 修改 benchmark_config.yaml
num_repeats: 1  # 改为1次
```

### 问题3: 某个配置失败

```bash
# 解决方案：跳过失败的配置
python benchmark_throughput.py --configs sparse_50 sparse_70
# 不包含失败的配置
```

---

希望这些示例能帮助你快速上手！🚀
