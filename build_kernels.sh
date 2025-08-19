#!/bin/bash
# Mustafar Kernel Build Script
# 统一编译脚本，将所有编译产物放置到build目录

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Mustafar Kernel Build Script${NC}"
echo -e "${BLUE}========================================${NC}"

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
KERNEL_BUILD_DIR="${BUILD_DIR}/kernel"
PYTHON_EXT_BUILD_DIR="${BUILD_DIR}/python_ext"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo -e "${YELLOW}Build directory: ${BUILD_DIR}${NC}"

# 创建构建目录
echo -e "${BLUE}Creating build directories...${NC}"
mkdir -p "${KERNEL_BUILD_DIR}"
mkdir -p "${PYTHON_EXT_BUILD_DIR}"

# 清理旧的构建产物 (可选)
if [ "$1" = "clean" ]; then
    echo -e "${YELLOW}Cleaning old build artifacts...${NC}"
    rm -rf "${BUILD_DIR}"/*
    rm -rf kernel/build/*
    rm -rf kernel/kernel_wrapper/build
    rm -f kernel/kernel_wrapper/*.so
    rm -rf kernel/kernel_wrapper/*.egg-info
    echo -e "${GREEN}Clean completed.${NC}"
    exit 0
fi

# 1. 编译核心CUDA kernel
echo -e "${BLUE}Building CUDA SpMM kernel...${NC}"
cd "${PROJECT_ROOT}/kernel"

# 检查是否存在Makefile
if [ -f "Makefile" ]; then
    echo -e "${YELLOW}Using existing Makefile in kernel directory${NC}"
    make clean && make
    
    # 复制编译产物到统一构建目录
    cp build/* "${KERNEL_BUILD_DIR}/" 2>/dev/null || true
else
    echo -e "${YELLOW}Makefile not found, using direct nvcc compilation${NC}"
    # 直接使用nvcc编译
    nvcc -shared -Xcompiler -fPIC -o "${KERNEL_BUILD_DIR}/libSpMM_API.so" \
         csrc/SpMM_API.cu \
         -I./csrc \
         -O3 \
         --use_fast_math \
         -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_89,code=sm_89
fi

echo -e "${GREEN}CUDA kernel build completed.${NC}"

# 2. 编译Python扩展
echo -e "${BLUE}Building Python extension...${NC}"
cd "${PROJECT_ROOT}/kernel/kernel_wrapper"

# 设置构建目录环境变量
export BUILD_DIR="${PYTHON_EXT_BUILD_DIR}"

# 清理旧的Python扩展构建
rm -rf build *.so *.egg-info

# 使用setup.py编译，指定构建目录
python setup.py build_ext --build-lib="${PYTHON_EXT_BUILD_DIR}" --build-temp="${PYTHON_EXT_BUILD_DIR}/temp"

# 复制最终的.so文件到构建目录根部，方便导入
find "${PYTHON_EXT_BUILD_DIR}" -name "*.so" -exec cp {} "${PYTHON_EXT_BUILD_DIR}/" \;

echo -e "${GREEN}Python extension build completed.${NC}"

# 3. 验证构建结果
echo -e "${BLUE}Verifying build results...${NC}"

KERNEL_SO="${KERNEL_BUILD_DIR}/libSpMM_API.so"
PYTHON_SO="${PYTHON_EXT_BUILD_DIR}/mustafar_package.cpython-311-x86_64-linux-gnu.so"

if [ -f "${KERNEL_SO}" ]; then
    echo -e "${GREEN}✓ CUDA kernel library found: ${KERNEL_SO}${NC}"
else
    echo -e "${RED}✗ CUDA kernel library not found!${NC}"
    exit 1
fi

if [ -f "${PYTHON_SO}" ]; then
    echo -e "${GREEN}✓ Python extension found: ${PYTHON_SO}${NC}"
else
    echo -e "${RED}✗ Python extension not found!${NC}"
    exit 1
fi

# 4. 更新Python路径设置
echo -e "${BLUE}Updating Python path configuration...${NC}"
cat > "${PROJECT_ROOT}/setup_paths.py" << EOF
#!/usr/bin/env python3
"""
Mustafar Python Path Setup
自动配置Python导入路径，使用build目录中的编译产物
"""

import sys
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
BUILD_DIR = PROJECT_ROOT / "build"

# 添加构建目录到Python路径
PYTHON_EXT_DIR = BUILD_DIR / "python_ext"
KERNEL_DIR = PROJECT_ROOT / "kernel"

# 确保路径存在
if PYTHON_EXT_DIR.exists():
    sys.path.insert(0, str(PYTHON_EXT_DIR))
    print(f"Added to Python path: {PYTHON_EXT_DIR}")

if KERNEL_DIR.exists():
    sys.path.insert(0, str(KERNEL_DIR))
    print(f"Added to Python path: {KERNEL_DIR}")

# 验证导入
def verify_imports():
    """验证关键模块是否可以正常导入"""
    try:
        import mustafar_package
        print("✓ mustafar_package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import mustafar_package: {e}")
        return False

if __name__ == "__main__":
    print("Mustafar Python Path Setup")
    print("=" * 40)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Build directory: {BUILD_DIR}")
    print("=" * 40)
    
    success = verify_imports()
    if success:
        print("✓ All imports successful")
    else:
        print("✗ Import verification failed")
        sys.exit(1)
EOF

echo -e "${GREEN}Python path setup created: ${PROJECT_ROOT}/setup_paths.py${NC}"

# 5. 生成使用说明
echo -e "${BLUE}Generating usage instructions...${NC}"
cat > "${PROJECT_ROOT}/BUILD_USAGE.md" << EOF
# Mustafar Build System Usage

## 目录结构

\`\`\`
mustafar/
├── build/                     # 统一构建目录
│   ├── kernel/               # CUDA kernel编译产物
│   │   ├── libSpMM_API.so   # 核心CUDA库
│   │   └── ...
│   └── python_ext/          # Python扩展编译产物
│       ├── mustafar_package.cpython-311-x86_64-linux-gnu.so
│       └── ...
├── kernel/                   # 内核源代码
│   ├── csrc/                # CUDA源文件
│   ├── kernel_wrapper/      # Python绑定
│   └── compression.py       # 压缩函数
├── models/                  # 模型实现
├── build_kernels.sh         # 统一构建脚本
└── setup_paths.py          # Python路径配置
\`\`\`

## 构建命令

### 完整构建
\`\`\`bash
./build_kernels.sh
\`\`\`

### 清理并重新构建
\`\`\`bash
./build_kernels.sh clean
./build_kernels.sh
\`\`\`

## 使用方式

### 方法1: 在Python脚本中使用路径设置
\`\`\`python
# 在你的Python脚本开头添加：
import sys
sys.path.insert(0, 'build/python_ext')
sys.path.insert(0, 'kernel')

# 然后正常导入
import mustafar_package
from compression import convert_key_batched, convert_value_batched
\`\`\`

### 方法2: 使用自动路径配置
\`\`\`python
# 在你的Python脚本开头添加：
import setup_paths  # 自动配置路径

# 然后正常导入
import mustafar_package
from compression import convert_key_batched, convert_value_batched
\`\`\`

### 方法3: 设置环境变量
\`\`\`bash
export PYTHONPATH="\$PYTHONPATH:\$(pwd)/build/python_ext:\$(pwd)/kernel"
python your_script.py
\`\`\`

## 验证安装

运行以下命令验证构建是否成功：
\`\`\`bash
python setup_paths.py
\`\`\`

或者在Python中：
\`\`\`python
import setup_paths
setup_paths.verify_imports()
\`\`\`

## 故障排除

1. **CUDA编译错误**: 检查CUDA版本和GPU架构兼容性
2. **Python扩展编译错误**: 检查PyTorch和CUDA版本匹配
3. **导入错误**: 确保Python路径正确设置

## 清理构建产物

\`\`\`bash
./build_kernels.sh clean
\`\`\`

这将清理所有构建产物，包括build目录和kernel目录中的临时文件。
EOF

echo -e "${GREEN}Build usage documentation created: ${PROJECT_ROOT}/BUILD_USAGE.md${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Build completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Test the build: ${GREEN}python setup_paths.py${NC}"
echo -e "2. Read usage guide: ${GREEN}cat BUILD_USAGE.md${NC}"
echo -e "3. Update your scripts to use: ${GREEN}import setup_paths${NC}"
echo -e "${BLUE}========================================${NC}"