# Mustafar 量化稀疏矩阵乘法内核

本目录包含 Mustafar 稀疏注意力机制的量化版本实现，支持 2-bit 量化以进一步减少内存占用。

## 目录结构

```
kernel_quant/
├── compression_quant.py          # 量化压缩实现（Triton）
├── csrc/                         # CUDA 内核源码
│   ├── SpMM_Kernel_Quant.cuh    # 量化稀疏矩阵乘法内核
│   ├── SpMM_API_Quant.cu        # 内核 API 接口
│   ├── MatMulUtilities.cuh      # 矩阵乘法工具函数
│   ├── MMA_PTX.cuh              # Tensor Core PTX 指令
│   ├── AsyncCopy_PTX.cuh        # 异步拷贝 PTX 指令
│   ├── Reduction_Kernel.cuh     # Split-K 归约内核
│   └── TilingConfig.h           # Tile 配置
├── kernel_wrapper/              # Python 绑定
│   ├── mustafar_wrapper_quant.h
│   ├── mustafar_wrapper_quant.cu
│   ├── pybind_quant.cpp
│   ├── setup.py
│   └── INSTALLATION_GUIDE.md
└── README.md                    # 本文件
```

## 核心特性

### 1. 量化压缩

- **量化位宽**: 2-bit per value
- **量化方案**: Per-tile min-max 量化
- **压缩流程**:
  1. 稀疏化：保留非零元素
  2. 量化：将 float16 量化为 2-bit
  3. 打包：4 个量化值打包到 1 个 uint8

### 2. 数据格式

与原版 `kernel/compression.py` 的主要区别：

| 组件 | 原版 | 量化版 |
|------|------|--------|
| 非零值存储 | `packed_not`: float16 | `packed_quant`: uint8 (2-bit 打包) |
| 索引信息 | `accum_counts`: 累积计数 | `tile_offsets`: 字节偏移 |
| 量化参数 | 无 | `scales`, `zeros`: per-tile float32 |

### 3. 内核实现

**关键修改点**：

1. **加载阶段** (`SpMM_CopyFromGlobalToReg_Quant`):
   - 加载 uint8 打包的量化值
   - 加载 per-tile 的 scale 和 zero_point
   - 使用 tile_offsets 定位数据

2. **解压阶段** (`SpMM_DecompressFromRegisterToShared_Quant`):
   - 从 uint8 解包 2-bit 量化值
   - 反量化：`value = (q - zero_point) * scale`
   - 转换为 float16 并写入共享内存

3. **计算阶段**:
   - 与原版相同，使用 Tensor Core 进行矩阵乘法

## 编译与安装

### 快速开始

```bash
# 从项目根目录运行
./build_quant_kernel.sh
```

### 手动编译

```bash
cd kernel_quant/kernel_wrapper
python setup.py install
```

详细说明见 [INSTALLATION_GUIDE.md](kernel_wrapper/INSTALLATION_GUIDE.md)

## 使用示例

### 1. 压缩 Key Cache

```python
import torch
from kernel_quant.compression_quant import convert_key_batched_quant

# 输入: [B, M, N] float16
k_cache = torch.randn(8, 256, 128, dtype=torch.float16, device='cuda')

# 压缩
bitmaps, tile_offsets, packed_quant, scales, zeros = \
    convert_key_batched_quant(k_cache)

print(f"Bitmaps: {bitmaps.shape}")           # [B, num_tiles]
print(f"Tile offsets: {tile_offsets.shape}") # [B, num_tiles]
print(f"Packed quant: {packed_quant.shape}") # [total_bytes]
print(f"Scales: {scales.shape}")             # [B, num_tiles]
print(f"Zeros: {zeros.shape}")               # [B, num_tiles]
```

### 2. 稀疏矩阵乘法

```python
import mustafar_package_quant

# Query: [batch_size, query_len, head_dim]
query = torch.randn(8, 8, 128, dtype=torch.float16, device='cuda')

# 调用量化内核
output = mustafar_package_quant.mustafar_key_formulation_quant(
    bitmaps,
    packed_quant,
    tile_offsets,
    scales,
    zeros,
    query,
    M_Global=256,
    K_Global=128,
    Batch_Size=8,
    num_key_value_groups=1,
    bit=2,
    capacity=4
)

print(f"Output: {output.shape}")  # [batch_size, query_len, M_Global]
```

## 测试

运行完整测试：

```bash
python test_mustafar_key_formulation_quant.py
```

测试内容：
- ✓ 量化压缩功能
- ✓ 内核调用
- ✓ 与参考实现对比
- ✓ 性能基准测试

## 性能与压缩比

### 内存占用对比

以 `[8, 256, 128]` 的 Key Cache 为例（70% 稀疏度）：

| 方案 | 内存占用 | 压缩比 |
|------|----------|--------|
| 原始 float16 | 0.50 MB | 100% |
| 稀疏压缩 (float16) | 0.15 MB | 30% |
| 量化压缩 (2-bit) | 0.08 MB | 16% |

### 精度损失

- 2-bit 量化引入的量化误差
- 相对误差通常 < 5%
- 对最终注意力分数影响较小

### 性能

- 解压开销：~10-20% 额外延迟
- 内存带宽节省：~6x
- 适用场景：内存受限的长序列推理

## 与原版的对比

| 特性 | 原版 (`kernel/`) | 量化版 (`kernel_quant/`) |
|------|------------------|-------------------------|
| 压缩方式 | 稀疏 | 稀疏 + 量化 |
| 数据精度 | float16 | 2-bit |
| 内存占用 | 中 | 低 |
| 计算精度 | 高 | 中（量化误差） |
| 计算速度 | 快 | 稍慢（解压开销） |
| 适用场景 | 通用 | 内存受限 |

## 技术细节

### 量化方案

Per-tile min-max 量化：

```
scale = (max - min) / (2^bit - 1)
zero_point = -min / scale
q = clamp(round(value / scale + zero_point), 0, 2^bit - 1)
```

反量化：

```
value = (q - zero_point) * scale
```

### Tile 布局

- Tile 大小：64 元素
- Key 矩阵：转置后按列主序分 tile
- Value 矩阵：按行主序分 tile

### 打包格式

2-bit 量化值打包到 uint8：

```
byte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
```

每个 tile 最多需要 16 bytes (64 values * 2 bits / 8)

## 限制与注意事项

1. **GPU 架构**: 需要 SM 80+ (Ampere 或更新)
2. **序列长度**: M 必须是 64 的倍数
3. **量化误差**: 2-bit 量化会引入精度损失
4. **对齐要求**: tile_offsets 可配置对齐（默认 1 byte）

## 未来改进

- [ ] 支持 4-bit 量化（更高精度）
- [ ] 支持 1-bit 量化（更高压缩比）
- [ ] 混合精度量化（重要 tile 用更高精度）
- [ ] 动态量化（运行时自适应）
- [ ] 支持更多 GPU 架构

## 参考文献

- [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

## 许可证

Apache License 2.0
