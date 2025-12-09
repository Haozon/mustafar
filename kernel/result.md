(mustar) zh@nku-PowerEdge-R750:~/mustafar/kernel$ python test_mustafar_key_formulation.py 
✓ mustafar_package 导入成功
依赖导入完成

=== 模型参数 ===
batch_size: 2
num_heads: 4
num_key_value_heads: 4
head_dim: 128
seq_len (compressed): 256
total_batch_kv: 8
total_batch_size: 8
num_key_value_groups: 1

K Cache 形状: torch.Size([8, 256, 128])
非零元素比例: 29.69%
数据类型: torch.float16

调用 convert_key_batched...
k_bitmaps 形状: torch.Size([8, 512])
k_accum_counts 形状: torch.Size([8, 513])
k_packed_not_batched 长度: 8
k_nz_offset 形状: torch.Size([8])

压缩后内存占用:
原始: 0.50 MB
压缩后: 0.22 MB
压缩比: 44.60%

Query 形状: torch.Size([2, 4, 1, 128])
Padded Query 形状: torch.Size([8, 8, 128])
Query 数据类型: torch.float16

调用 mustafar_package.mustafar_key_formulation...
✓ 调用成功
输出形状: torch.Size([8, 8, 256])
输出数据类型: torch.float16
输出统计:
  最大值: 0.000000
  最小值: 0.000000
  平均值: 0.000000
  标准差: 0.000000

测试普通密集矩阵乘法（Ground Truth）...
密集矩阵乘法输出形状: torch.Size([8, 8, 256])
密集矩阵乘法输出统计:
  最大值: 26.546875
  最小值: -26.828125
  平均值: 0.028259
  标准差: 2.146484

测试稀疏 Python 参考实现...
稀疏参考实现输出形状: torch.Size([8, 8, 256])
稀疏参考实现输出统计:
  最大值: 24.859375
  最小值: -19.593750
  平均值: 0.020660
  标准差: 2.183594

============================================================
对比分析
============================================================

【1】CUDA 稀疏实现 vs 稀疏参考实现
CUDA 结果样本 (前5个值): tensor([0., 0., 0., 0., 0.], device='cuda:0', dtype=torch.float16)
稀疏参考样本 (前5个值): tensor([ 2.7910, -1.1484, -3.8008,  3.7695,  3.0391], device='cuda:0',
       dtype=torch.float16)
差异统计:
  最大差异: 24.859375
  平均差异: 4.922359
  中位数差异: 4.156250
相对误差:
  最大相对误差: 100.00%
  平均相对误差: 100.00%
是否接近 (rtol=1e-2, atol=1e-3): False

============================================================
【2】CUDA 稀疏实现 vs 密集矩阵乘法（Ground Truth）
CUDA 结果样本 (前5个值): tensor([0., 0., 0., 0., 0.], device='cuda:0', dtype=torch.float16)
密集矩阵样本 (前5个值): tensor([ 3.2324, -4.1094,  3.8105,  3.9355, -3.3164], device='cuda:0',
       dtype=torch.float16)
差异统计:
  最大差异: 26.828125
  平均差异: 4.824267
  中位数差异: 3.978516
相对误差:
  最大相对误差: 100.00%
  平均相对误差: 100.00%
是否接近 (rtol=1e-2, atol=1e-3): False

============================================================
【3】稀疏参考实现 vs 密集矩阵乘法（Ground Truth）
稀疏参考样本 (前5个值): tensor([ 2.7910, -1.1484, -3.8008,  3.7695,  3.0391], device='cuda:0',
       dtype=torch.float16)
密集矩阵样本 (前5个值): tensor([ 3.2324, -4.1094,  3.8105,  3.9355, -3.3164], device='cuda:0',
       dtype=torch.float16)
差异统计 (稀疏误差):
  最大差异: 30.773438
  平均差异: 6.870632
  中位数差异: 5.750000
相对误差:
  最大相对误差: 398490.01%
  平均相对误差: 778.08%
============================================================

性能测试...

性能对比 (10 次迭代):
  CUDA 稀疏实现:      0.0889 ms
  稀疏参考实现:       1467.3862 ms
  密集矩阵乘法:       0.4245 ms

加速比:
  CUDA vs 稀疏参考:   16509.29x
  CUDA vs 密集矩阵:   4.78x
  密集矩阵 vs 稀疏参考: 3456.51x

============================================================
测试总结
============================================================

模型配置:
  Batch size: 2
  Num heads: 4
  Head dim: 128
  Compressed seq len: 256
  Sparsity: 70.0%

压缩效果:
  原始内存: 0.50 MB
  压缩后内存: 0.22 MB
  压缩比: 44.60%

功能验证:
  ✓ mustafar_key_formulation 调用成功
  ✓ 输出形状正确: torch.Size([8, 8, 256])
  ⚠ CUDA 实现与稀疏参考实现有差异
  ⚠ CUDA 实现与密集矩阵乘法有差异（预期有稀疏误差）

✓ 测试完成
============================================================