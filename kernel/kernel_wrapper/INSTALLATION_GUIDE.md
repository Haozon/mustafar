# Mustafar Kernel 安装指南

## 问题描述
安装 mustafar_package 后，导入时出现 GLIBCXX 版本不匹配错误：
```
ImportError: /home/zh/miniconda3/envs/mustar/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

## 根本原因
编译时链接了系统版本的 libstdc++.so.6，但运行时 conda 环境优先使用自己的旧版本 libstdc++.so.6，缺少所需的 GLIBCXX_3.4.32 符号。

## 安装步骤

### 1. 编译安装
```bash
cd /home/zh/mustafar/kernel/kernel_wrapper
python setup.py build_ext --inplace
```

### 2. 设置环境变量（临时解决方案）
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zh/miniconda3/envs/mustar/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH
```

### 3. 验证安装
```bash
python -c "import mustafar_package; print('Import successful')"
```

## 永久解决方案

### 方法一：更新 conda 环境的 libstdc++
```bash
conda install -c conda-forge libstdcxx-ng
```

### 方法二：修改 ~/.bashrc 或 ~/.zshrc
```bash
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/zh/miniconda3/envs/mustar/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
