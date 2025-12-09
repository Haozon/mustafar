#!/bin/bash

# Mustafar Quantized Kernel 编译脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  Mustafar Quantized Kernel 编译脚本"
echo "=========================================="
echo ""

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保 CUDA 已正确安装"
    exit 1
fi

echo "✓ CUDA 版本:"
nvcc --version | grep "release"
echo ""

# 检查 PyTorch
echo "检查 PyTorch..."
python -c "import torch; print(f'✓ PyTorch 版本: {torch.__version__}'); print(f'✓ CUDA 可用: {torch.cuda.is_available()}'); print(f'✓ CUDA 版本: {torch.version.cuda}')" || {
    echo "错误: PyTorch 未正确安装或不支持 CUDA"
    exit 1
}
echo ""

# 进入 kernel_wrapper 目录
cd kernel_wrapper

echo "开始编译..."
echo ""

# 清理旧的编译文件
if [ -d "build" ]; then
    echo "清理旧的编译文件..."
    rm -rf build
fi

if [ -d "mustafar_package_quant.egg-info" ]; then
    rm -rf mustafar_package_quant.egg-info
fi

# 编译
echo "执行编译..."
python setup.py install

echo ""
echo "=========================================="
echo "  编译完成！"
echo "=========================================="
echo ""

# 返回项目根目录
cd ../..

# 验证安装
echo "验证安装..."
python -c "
import mustafar_package_quant
print('✓ mustafar_package_quant 导入成功')
print('可用函数:')
for name in dir(mustafar_package_quant):
    if not name.startswith('_'):
        print(f'  - {name}')
" || {
    echo "警告: 模块导入失败，请检查编译日志"
    exit 1
}

echo ""
echo "=========================================="
echo "  安装验证成功！"
echo "=========================================="
