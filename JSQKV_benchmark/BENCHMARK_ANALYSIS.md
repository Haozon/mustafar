# JSQKV Benchmark 分析报告

## 问题1：为什么稀疏70%与50%的区别不大？

### 数据分析

#### Llama3-8B 吞吐量对比 (Tokens/Second)

| Batch Size | Dense  | Sparse 50% | Sparse 70% | 50% vs Dense | 70% vs Dense |
|------------|--------|------------|------------|--------------|--------------|
| 1          | 37.08  | 17.83      | 16.62      | -51.9%       | -55.2%       |
| 2          | 52.86  | 34.08      | 33.17      | -35.5%       | -37.3%       |
| 4          | 67.72  | 68.40      | 66.74      | +1.0%        | -1.4%        |
| 6          | 71.93  | 94.35      | 99.11      | +31.2%       | +37.8%       |
| 8          | 74.11  | 127.07     | 131.14     | +71.5%       | +77.0%       |

### 关键发现

1. **小batch性能下降**：Batch size 1-2时，稀疏化反而比dense慢
2. **大batch才有收益**：Batch size 6-8时，稀疏化有明显加速（70%+）
3. **50%和70%差异小**：两种稀疏度的性能差异仅3-6%

### 可能原因

#### 1. Kernel Overhead占主导（小batch）
- 稀疏矩阵乘法有固定的启动开销
- 小batch时，计算量少，overhead占比大
- Dense kernel更优化，延迟更低

#### 2. 内存带宽瓶颈（大batch）
- 70%稀疏度理论上应该比50%快40%
- 实际只快3-6%，说明不是计算瓶颈
- 可能是内存访问模式导致的带宽利用率问题

#### 3. 稀疏格式开销
- Bitmap + 压缩数据的访问模式可能不够优化
- 70%稀疏度时，数据更分散，cache miss更多
- 间接寻址的开销可能抵消了计算减少的收益

### 建议优化方向

1. **优化小batch性能**
   - 减少kernel启动开销
   - 考虑fusion更多操作
   - 对小batch使用不同的kernel实现

2. **改进内存访问模式**
   - 优化稀疏数据的存储布局
   - 提高cache命中率
   - 考虑使用shared memory缓存

3. **动态选择策略**
   - Batch size < 4: 使用dense
   - Batch size >= 4: 使用sparse
   - 根据实际硬件特性调整阈值

---

## 问题2：为什么没有量化数据？

### 问题诊断

检查结果文件发现：
```json
"sparse_50_quant_2bit": {},
"sparse_70_quant_2bit": {}
```

量化配置是空的，说明测试失败或被跳过。

### 可能原因

#### 1. 量化kernel未安装
```bash
# 检查是否安装了量化kernel
ls -la kernel_quant/kernel_wrapper/*.so
```

如果没有 `.so` 文件，需要编译安装：
```bash
cd kernel_quant/kernel_wrapper
bash ../build_quant_kernel.sh
```

#### 2. 运行时错误被捕获
benchmark脚本中有 try-except，错误可能被静默捕获了。

#### 3. CUDA版本不兼容
量化kernel可能需要特定的CUDA版本。

### 解决方案

#### 方案1：检查kernel安装状态
```bash
cd ~/mustafar
python3 << 'EOF'
import sys
sys.path.append('kernel_quant')
try:
    from compression_quant import compress_kv_cache_quant
    print("✅ 量化kernel已安装")
except ImportError as e:
    print(f"❌ 量化kernel未安装: {e}")
EOF
```

#### 方案2：单独测试量化配置
```bash
cd JSQKV_benchmark
python benchmark_throughput.py \
    --models llama3_8b \
    --configs sparse_50_quant_2bit \
    --output-dir results/raw_data
```

这样可以看到详细的错误信息。

#### 方案3：重新编译量化kernel
```bash
cd ~/mustafar/kernel_quant
bash build_quant_kernel.sh
```

#### 方案4：修改benchmark脚本，显示详细错误
在 `benchmark_throughput.py` 的 `run_single_benchmark` 函数中，
将 `except Exception as e:` 改为更详细的错误处理：

```python
except Exception as e:
    print(f"❌ Error during benchmark: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # 如果是量化配置，给出特别提示
    if test_config.get('use_quant', False):
        print("\n⚠️  量化测试失败，可能原因：")
        print("  1. 量化kernel未安装")
        print("  2. CUDA版本不兼容")
        print("  3. 模型路径错误")
        print("\n请运行以下命令检查：")
        print("  cd kernel_quant && bash build_quant_kernel.sh")
    
    return None
```

### 下一步行动

1. **立即执行**：检查kernel安装状态
2. **如果未安装**：编译安装量化kernel
3. **重新运行**：只测试量化配置，观察错误信息
4. **根据错误**：针对性解决问题

---

## 总结

### 性能问题
- 50%和70%稀疏度差异小是正常的，主要受内存带宽限制
- 小batch时稀疏化不划算，建议动态选择策略

### 量化问题
- 量化测试失败，需要检查kernel安装
- 建议先单独测试量化配置，获取详细错误信息

### 优化建议
1. 针对不同batch size使用不同策略
2. 优化稀疏kernel的内存访问模式
3. 确保量化kernel正确安装和配置
