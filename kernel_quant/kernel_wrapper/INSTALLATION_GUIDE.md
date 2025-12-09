# Mustafar Quantized Kernel 安装指南

## 环境要求

- CUDA 11.0+
- PyTorch 1.12+
- Python 3.7+
- GCC 7.0+
- GPU 架构: SM 80+ (Ampere 或更新)

## 安装步骤

### 1. 进入 kernel_wrapper 目录

```bash
cd kernel_quant/kernel_wrapper
```

### 2. 编译并安装

```bash
python setup.py install
```

或者使用开发模式（推荐用于调试）：

```bash
python setup.py develop
```

### 3. 验证安装

```python
import mustafar_package_quant
print(dir(mustafar_package_quant))
# 应该看到: ['mustafar_key_formulation_quant', 'mustafar_value_formulation_quant']
```

## 运行测试

返回项目根目录并运行测试：

```bash
cd ../..
python test_mustafar_key_formulation_quant.py
```

## 常见问题

### 1. CUDA 架构不匹配

如果遇到架构错误，修改 `setup.py` 中的 `-arch=sm_XX` 参数：

- V100: `sm_70`
- A100: `sm_80`
- RTX 3090: `sm_86`
- H100: `sm_90`

### 2. 编译错误

确保 CUDA 和 PyTorch 版本兼容：

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### 3. 内存不足

减小测试中的 batch_size 或 seq_len。

## 性能优化

- 使用 `--use_fast_math` 编译选项（已默认启用）
- 根据 GPU 调整 tile 配置
- 使用 nsight 进行性能分析

## 卸载

```bash
pip uninstall mustafar_package_quant
```
