# Mustafar 量化版本快速开始指南

## 5 分钟快速上手

### 步骤 1: 编译内核

```bash
./build_quant_kernel.sh
```

预期输出：
```
✓ CUDA 版本: release 11.x
✓ PyTorch 版本: x.x.x
✓ CUDA 可用: True
开始编译...
编译完成！
✓ mustafar_package_quant 导入成功
```

### 步骤 2: 运行测试

```bash
python test_mustafar_key_formulation_quant.py
```

预期输出：
```
=== 模型参数 ===
batch_size: 2
num_heads: 4
head_dim: 128
seq_len (compressed): 256
量化位宽: 2 bits

压缩后内存占用:
原始: 0.50 MB
压缩后: 0.10 MB
压缩比: 20.00%

✓ mustafar_key_formulation_quant 调用成功
✓ 输出形状正确
✓ 结果与参考实现接近

✓ 测试完成
```

### 步骤 3: 在你的代码中使用

```python
import torch
from kernel_quant.compression_quant import convert_key_batched_quant
import mustafar_package_quant

# 1. 压缩 Key Cache
k_cache = torch.randn(8, 256, 128, dtype=torch.float16, device='cuda')
bitmaps, tile_offsets, packed_quant, scales, zeros = \
    convert_key_batched_quant(k_cache)

# 2. 准备 Query
query = torch.randn(8, 8, 128, dtype=torch.float16, device='cuda')

# 3. 执行稀疏注意力计算
output = mustafar_package_quant.mustafar_key_formulation_quant(
    bitmaps,           # [B, num_tiles] int64
    packed_quant,      # [total_bytes] uint8
    tile_offsets,      # [B, num_tiles] int32
    scales,            # [B, num_tiles] float32
    zeros,             # [B, num_tiles] float32
    query,             # [batch, query_len, head_dim] float16
    M_Global=256,      # Key 序列长度
    K_Global=128,      # head 维度
    Batch_Size=8,      # batch 大小
    num_key_value_groups=1,  # GQA 组数
    bit=2,             # 量化位宽
    capacity=4         # 每字节容纳的量化值数
)

print(f"Output shape: {output.shape}")  # [8, 8, 256]
```

## 常见问题

### Q1: 编译失败怎么办？

**A**: 检查 CUDA 和 PyTorch 版本是否兼容：

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

如果版本不匹配，重新安装对应版本的 PyTorch。

### Q2: 如何调整 GPU 架构？

**A**: 编辑 `kernel_quant/kernel_wrapper/setup.py`，修改 `-arch=sm_XX`：

```python
'-arch=sm_80',  # A100
# '-arch=sm_86',  # RTX 3090
# '-arch=sm_70',  # V100
```

### Q3: 内存不足怎么办？

**A**: 减小测试参数：

```python
# 在 test_mustafar_key_formulation_quant.py 中修改
batch_size = 1      # 减小 batch
seq_len = 128       # 减小序列长度
```

### Q4: 量化误差太大怎么办？

**A**: 考虑使用 4-bit 量化（需要修改代码）或使用原版 float16 实现。

### Q5: 如何卸载？

**A**: 
```bash
pip uninstall mustafar_package_quant
```

## 性能优化建议

### 1. 选择合适的对齐

在 `compression_quant.py` 中调整 `align_bytes`：

```python
align_bytes = 4  # 默认是 1，改为 4 可能提升性能
```

### 2. 调整量化位宽

```python
bit = 4  # 使用 4-bit 量化（需要修改内核代码）
```

### 3. 使用 nsight 分析

```bash
nsys profile python test_mustafar_key_formulation_quant.py
```

## 下一步

- 阅读 [kernel_quant/README.md](kernel_quant/README.md) 了解详细实现
- 阅读 [QUANTIZATION_IMPLEMENTATION_SUMMARY.md](QUANTIZATION_IMPLEMENTATION_SUMMARY.md) 了解技术细节
- 查看 [kernel_quant/kernel_wrapper/INSTALLATION_GUIDE.md](kernel_quant/kernel_wrapper/INSTALLATION_GUIDE.md) 了解安装选项

## 获取帮助

如果遇到问题：

1. 检查 [kernel_quant/README.md](kernel_quant/README.md) 的"限制与注意事项"部分
2. 查看测试脚本 `test_mustafar_key_formulation_quant.py` 的示例用法
3. 对比原版实现 `kernel/` 和量化版 `kernel_quant/` 的差异

## 贡献

欢迎提交 Issue 和 Pull Request！
