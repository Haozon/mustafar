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

# 确保路径存在，如果不存在则使用原路径
if PYTHON_EXT_DIR.exists():
    sys.path.insert(0, str(PYTHON_EXT_DIR))
    print(f"Added to Python path: {PYTHON_EXT_DIR}")
else:
    # 使用原有的kernel_wrapper目录
    original_wrapper = KERNEL_DIR / "kernel_wrapper"
    if original_wrapper.exists():
        sys.path.insert(0, str(original_wrapper))
        print(f"Added to Python path: {original_wrapper}")

if KERNEL_DIR.exists():
    sys.path.insert(0, str(KERNEL_DIR))
    print(f"Added to Python path: {KERNEL_DIR}")

# 验证导入
def verify_imports():
    """验证关键模块是否可以正常导入"""
    success = True
    
    try:
        import mustafar_package
        print("✓ mustafar_package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mustafar_package: {e}")
        success = False
    
    try:
        from compression import convert_key_batched, convert_value_batched
        print("✓ compression functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import compression functions: {e}")
        success = False
    
    return success

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