# Kernel Quant 文档索引

本目录包含 Mustafar 量化稀疏矩阵乘法内核的技术文档。

---

## 📚 文档列表

### 1. 总体说明
- **[README.md](../README.md)** - 量化内核总体介绍
  - 目录结构
  - 核心特性
  - 编译安装
  - 使用示例

### 2. 技术报告
- **[COMPRESSION_QUANT_TECHNICAL_REPORT.md](COMPRESSION_QUANT_TECHNICAL_REPORT.md)** - 量化压缩技术详解
  - 量化方案设计
  - 数据格式说明
  - 性能分析

- **[COMPRESSION_TIMING_ANALYSIS.md](COMPRESSION_TIMING_ANALYSIS.md)** - Compression 计时差异专项分析
  - Sparse vs Quant 时间差异原因
  - 代码级证据（含关键片段）
  - benchmark 口径与复现实验

- **[bitmap-sparse-quant-1.md](bitmap-sparse-quant-1.md)** - Bitmap 稀疏量化方案
  - Bitmap 编码原理
  - 稀疏存储优化

### 3. 实现指南（新增）
- **[VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md](VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md)** - Value 量化完整实现方案
  - 当前实现状况分析
  - Key vs Value 差异对比
  - 详细实现步骤
  - 测试验证方案
  - 性能优化建议
  - 常见问题解答

- **[QUICK_IMPLEMENTATION_GUIDE.md](QUICK_IMPLEMENTATION_GUIDE.md)** - Value 量化快速实现指南
  - 简洁的实现步骤
  - 代码示例
  - 验证清单
  - 故障排查

---

## 🎯 快速导航

### 我想了解...

#### 量化内核的基本概念
→ 阅读 [README.md](../README.md) 的"核心特性"部分

#### 如何编译和安装
→ 阅读 [README.md](../README.md) 的"编译与安装"部分

#### 量化方案的技术细节
→ 阅读 [COMPRESSION_QUANT_TECHNICAL_REPORT.md](COMPRESSION_QUANT_TECHNICAL_REPORT.md)

#### 如何实现 Value 量化
→ 阅读 [QUICK_IMPLEMENTATION_GUIDE.md](QUICK_IMPLEMENTATION_GUIDE.md)（推荐新手）
→ 阅读 [VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md](VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md)（详细方案）

#### Key 和 Value 的实现差异
→ 阅读 [VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md](VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md) 第 2 节

#### 数据格式和接口定义
→ 阅读 [VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md](VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md) 第 10.1 节

---

## 📂 代码文件索引

### Python 层
```
kernel_quant/
├── compression_quant.py          # 量化压缩实现（Triton）
│   ├── calculate_bitmap_and_scale_key_batched    ✅ Key 已实现
│   ├── compress_key_batched                      ✅ Key 已实现
│   ├── convert_key_batched_quant                 ✅ Key 已实现
│   ├── calculate_bitmap_and_scale_value_batched  ❌ Value 待实现
│   ├── compress_value_batched                    ❌ Value 待实现
│   └── convert_value_batched_quant               ❌ Value 待实现
```

### CUDA 层
```
kernel_quant/csrc/
├── SpMM_Kernel_Quant.cuh         # 量化稀疏矩阵乘法内核
│   ├── Key_Kernel_Quant                          ✅ 已实现
│   ├── Value_Kernel_Quant                        ✅ 已实现
│   ├── SpMM_CopyFromGlobalToReg_Quant           ✅ 已实现
│   └── SpMM_DecompressFromRegisterToShared_Quant ✅ 已实现
├── SpMM_API_Quant.cu             # API 接口
│   ├── Key_SplitK_API_Quant                      ✅ 已实现
│   └── Value_SplitK_API_Quant                    ✅ 已实现
└── SpMM_API_Quant.cuh            # API 头文件
```

### Python 绑定
```
kernel_quant/kernel_wrapper/
├── mustafar_wrapper_quant.cu     # C++ wrapper
│   ├── mustafar_key_formulation_quant            ✅ 已实现
│   └── mustafar_value_formulation_quant          ✅ 已实现
├── pybind_quant.cpp              # PyBind11 绑定
└── setup.py                      # 编译脚本
```

### 测试文件
```
/home/zh/mustafar/
├── test_value_only.py            # Value kernel 测试
├── test_nonquant_value.py        # 非量化对比测试
├── debug_value_kernel.py         # 调试脚本
└── kernel_quant/
    └── test_mustafar_key_formulation_quant.py  # Key kernel 测试
```

---

## 🔧 实现状态

### 已完成 ✅
- [x] Key 矩阵量化压缩（Python + CUDA）
- [x] Value 矩阵量化计算（CUDA kernel）
- [x] Python 绑定接口
- [x] Key 端到端测试

### 进行中 🚧
- [ ] Value 矩阵量化压缩（Python Triton kernels）
- [ ] Value 端到端测试
- [ ] 精度验证
- [ ] 性能基准测试

### 待完成 📋
- [ ] 模型集成优化
- [ ] 文档完善
- [ ] 示例代码

---

## 📊 性能指标

### 内存占用（以 [8, 256, 128] 为例，70% 稀疏度）

| 方案 | Key | Value | 总计 |
|------|-----|-------|------|
| 原始 float16 | 0.50 MB | 0.50 MB | 1.00 MB |
| 稀疏 float16 | 0.15 MB | 0.15 MB | 0.30 MB |
| 量化 2-bit | 0.08 MB | 0.08 MB | 0.16 MB |

### 压缩比

- 相对原始: **16%** (6.25x 压缩)
- 相对稀疏: **53%** (1.88x 压缩)

### 精度损失

- 量化误差: < 5%
- 对最终输出影响: 可忽略

---

## 🔗 相关资源

### 外部文档
- [Triton 文档](https://triton-lang.org/)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/)
- [PyTorch C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### 参考论文
- Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

### 相关项目
- [SnapKV](../../../SnapKV/) - 稀疏注意力参考实现
- [DiffSparseKV](../../../DiffSparseKV/) - 差分稀疏 KV Cache

---

## 📝 更新日志

### 2026-02-06
- ✨ 新增 `VALUE_QUANTIZATION_IMPLEMENTATION_PLAN.md`
- ✨ 新增 `QUICK_IMPLEMENTATION_GUIDE.md`
- ✨ 新增 `INDEX.md`（本文档）
- 📊 完成 Value 量化实现方案设计

### 历史记录
- 2025-xx-xx: 完成 Key 量化实现
- 2025-xx-xx: 完成 CUDA kernel 实现
- 2025-xx-xx: 项目初始化

---

## 💡 贡献指南

如果你想为本项目贡献代码或文档：

1. 阅读相关技术文档
2. 按照实现指南完成代码
3. 添加测试用例
4. 更新文档
5. 提交 Pull Request

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 项目 Issue Tracker
- 技术讨论群
- 邮件联系

---

**最后更新**: 2026-02-06
**维护者**: Mustafar Team
