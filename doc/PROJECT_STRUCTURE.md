# Mustafar 项目结构整理说明

## 项目概述

Mustafar是一个高效的稀疏注意力机制CUDA实现，本次整理将所有编译产物统一放置到`build`目录中，使项目结构更加清晰和规范。

## 新的目录结构

```
mustafar/
├── build/                          # 🆕 统一构建目录
│   ├── kernel/                     # CUDA kernel编译产物
│   │   ├── libSpMM_API.so         # 核心CUDA共享库
│   │   ├── SpMM_API.o             # 编译的目标文件
│   │   ├── SpMM_API.cuh           # 头文件
│   │   └── Makefile               # 构建文件
│   └── python_ext/                # Python扩展编译产物
│       ├── mustafar_package.cpython-311-x86_64-linux-gnu.so  # Python扩展
│       ├── lib.linux-x86_64-cpython-311/                     # 编译库
│       ├── temp.linux-x86_64-cpython-311/                    # 临时文件
│       └── mustafar_batched_spmv_package.egg-info/           # 包信息
│
├── kernel/                         # 内核源代码目录
│   ├── csrc/                       # CUDA源文件
│   │   ├── SpMM_API.cu            # 主要的SpMM API实现
│   │   ├── SpMM_Kernel.cuh        # SpMM内核头文件
│   │   ├── MMA_PTX.cuh            # Matrix Multiply-Accumulate PTX
│   │   ├── AsyncCopy_PTX.cuh      # 异步拷贝PTX
│   │   ├── Reduction_Kernel.cuh   # 归约内核
│   │   ├── MatMulUtilities.cuh    # 矩阵乘法工具
│   │   └── TilingConfig.h         # 分块配置
│   ├── kernel_wrapper/            # Python绑定
│   │   ├── pybind.cpp             # PyBind11绑定代码
│   │   ├── mustafar_wrapper.cu    # CUDA wrapper实现
│   │   ├── mustafar_wrapper.h     # wrapper头文件
│   │   ├── setup.py               # 🔄 已更新：使用新build路径
│   │   └── build/                 # 旧的构建目录（已迁移）
│   ├── compression.py             # Triton压缩函数实现
│   └── build/                     # 旧的构建目录（已迁移）
│
├── models/                         # 模型实现
│   ├── llama_mustafar_kernel.py   # 使用kernel的LLaMA实现
│   ├── llama_mustafar_Kt_Mag_Vt_Mag.py  # 不同配置的实现
│   └── ...                        # 其他模型变体
│
├── utils/                          # 工具函数
├── config/                         # 配置文件
├── longbench/                      # LongBench评测
├── outputs/                        # 输出目录 
├── pred/                           # 预测结果
├── figs/                           # 图片资源
│
├── build_kernels.sh               # 🆕 统一构建脚本
├── setup_paths.py                 # 🆕 Python路径自动配置
├── PROJECT_STRUCTURE.md           # 🆕 本文档
│
├── mustafar_latency_validation.py # 🔄 已更新：使用新路径
├── attention_kernel_test_correct.py # 🔄 已更新：使用新路径
├── mem_spd_test.py                # 🔄 已更新：使用新路径
├── pred_long_bench.py             # 🔄 已更新：使用新路径
│
├── eval_long_bench.py             # 评测脚本
├── requirements.txt               # Python依赖
├── README.md                      # 项目说明
└── LICENSE                        # 许可证
```

## 主要改进

### 1. 统一构建目录
- **之前**: 编译产物分散在`kernel/build/`和`kernel/kernel_wrapper/build/`等多个位置
- **现在**: 所有编译产物统一放在`build/`目录下，分为`kernel/`和`python_ext/`两个子目录

### 2. 自动路径配置
- **创建了`setup_paths.py`**: 自动配置Python导入路径，优先使用`build/`目录中的编译产物
- **更新了所有Python脚本**: 使用`import setup_paths`自动配置路径

### 3. 统一构建脚本
- **创建了`build_kernels.sh`**: 一键构建所有组件，支持清理和重新构建
- **智能路径检测**: 自动检测Makefile和构建依赖

## 使用方法

### 构建项目

```bash
# 完整构建
./build_kernels.sh

# 清理并重新构建  
./build_kernels.sh clean
./build_kernels.sh
```

### Python脚本使用

**方法1: 自动路径配置（推荐）**
```python
import setup_paths  # 自动配置路径，放在脚本最开头

# 然后正常导入
import mustafar_package
from compression import convert_key_batched, convert_value_batched
```

**方法2: 手动设置路径**
```python
import sys
sys.path.insert(0, 'build/python_ext')
sys.path.insert(0, 'kernel')

# 然后正常导入
import mustafar_package
from compression import convert_key_batched, convert_value_batched
```

**方法3: 环境变量**
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/build/python_ext:$(pwd)/kernel"
python your_script.py
```

### 验证安装

```bash
python setup_paths.py
```

如果看到以下输出表示配置成功：
```
✓ mustafar_package imported successfully
✓ compression functions imported successfully  
✓ All imports successful
```

## 已更新的脚本

以下脚本已经更新为使用新的路径配置：

1. **mustafar_latency_validation.py** - 延迟验证实验
2. **attention_kernel_test_correct.py** - 注意力内核测试
3. **mem_spd_test.py** - 内存速度测试
4. **pred_long_bench.py** - LongBench预测脚本

所有这些脚本现在都使用`import setup_paths`来自动配置路径。

## 兼容性说明

- **向后兼容**: 如果`build/`目录不存在，`setup_paths.py`会自动回退到使用原有的`kernel/kernel_wrapper/`路径
- **渐进迁移**: 可以逐步迁移脚本，新旧路径可以并存

## 清理说明

可以安全删除以下旧的构建目录：
- `kernel/build/`
- `kernel/kernel_wrapper/build/`
- `kernel/kernel_wrapper/*.so`
- `kernel/kernel_wrapper/*.egg-info`

但建议先运行`./build_kernels.sh`确保新的构建系统工作正常。

## 故障排除

### 1. 导入错误
```bash
python setup_paths.py
```
检查路径配置是否正确。

### 2. 构建失败
- 检查CUDA版本和GPU架构兼容性
- 检查PyTorch版本和CUDA版本匹配
- 查看`build_kernels.sh`中的错误信息

### 3. 路径问题
如果遇到导入问题，可以手动设置环境变量：
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/build/python_ext:$(pwd)/kernel"
```

## 总结

本次整理实现了：
- ✅ 统一的构建目录结构
- ✅ 自动化的构建脚本
- ✅ 智能的路径配置
- ✅ 向后兼容性
- ✅ 清晰的使用文档

项目现在更加规范和易于维护，编译产物不会再散落在各个目录中。