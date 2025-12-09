# 量化压缩：从 uint8 到 uint32 的迁移总结

## 修改日期
2024年（根据当前时间）

## 修改原因
- **问题**：使用 uint8 存储时，Triton kernel 中多个 lane 并行写入同一字节的不同位，导致覆盖
- **根本原因**：`tl.store` 不是原子操作，`tl.atomic_or` 不支持 uint8
- **解决方案**：改用 uint32 存储，支持原子操作

## 核心变化

### 数据格式
| 项目 | 原方案 (uint8) | 新方案 (uint32) |
|------|---------------|----------------|
| 存储单元 | uint8 (1 字节) | uint32 (4 字节) |
| 每单元存储量化值数 | 4 个 (8 bits / 2 bits) | 16 个 (32 bits / 2 bits) |
| 每 tile 最大存储单元数 | 16 | 4 |
| 原子操作支持 | ❌ | ✅ |
| 对齐要求 | 需要手动对齐到 4 字节 | 天然 4 字节对齐 |

### 空间开销
- **70% 稀疏度，平均 19 个非零值/tile**
  - uint8 方案：`ceil(19/4) = 5` 字节 → 对齐后 8 字节
  - uint32 方案：`ceil(19/16) = 2` 个 uint32 = 8 字节
  - **结论：空间开销相同！**

## 修改文件清单

### 1. Python 层（`kernel_quant/compression_quant.py`）
- ✅ 修改 `capacity = 16`（从 4 改为 16）
- ✅ 修改 buffer 类型为 `torch.uint32`（从 `torch.uint8`）
- ✅ 修改 Triton kernel 使用 `tl.atomic_or` 进行原子写入
- ✅ 简化对齐逻辑（uint32 天然对齐）
- ✅ 修改偏移计算为 uint32 单位（从字节单位）

### 2. CUDA Kernel（`kernel_quant/csrc/SpMM_Kernel_Quant.cuh`）
- ✅ 修改 `SpMM_CopyFromGlobalToReg_Quant` 函数签名
  - `const uint8_t* GlobalPTR_quant` → `const uint32_t* GlobalPTR_quant`
  - 注释更新：说明 uint32 存储格式
- ✅ 修改加载逻辑
  - 直接加载 uint32，无需 `reinterpret_cast`
  - 简化对齐检查（天然对齐）
- ✅ 修改解包逻辑
  - 从 uint32 中提取 2-bit 量化值
  - 更新位偏移计算
- ✅ 修改 `Key_Kernel_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`
- ✅ 修改 `Value_Kernel_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`

### 3. CUDA API（`kernel_quant/csrc/SpMM_API_Quant.cu`）
- ✅ 修改 `Key_SplitK_Kernel_Ex_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`
- ✅ 修改 `Key_SplitK_API_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`
- ✅ 修改 `Value_SplitK_Kernel_Ex_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`
- ✅ 修改 `Value_SplitK_API_Quant` 函数签名
  - `const uint8_t* NZ_quant` → `const uint32_t* NZ_quant`

### 4. Python 测试（`test_mustafar_key_formulation_quant.py`）
- ✅ 修改 `dequantize_tile` 函数
  - 参数：`packed_bytes` → `packed_units`
  - 类型：uint8 → uint32
  - capacity：4 → 16
- ✅ 修改 `reconstruct_sparse_key_matrix_quant` 函数
  - 偏移：`byte_offset` → `uint32_offset`
  - 单位：`bytes_needed` → `units_needed`
  - capacity：4 → 16
- ✅ 修改 `sparse_matmul_reference_quant` 函数
  - capacity 默认值：4 → 16
- ✅ 修改 `main` 函数
  - capacity：4 → 16
  - 内存计算：`k_packed_quant.numel() * 1` → `k_packed_quant.numel() * 4`

## 技术细节

### Triton Kernel 写入逻辑
```python
# 计算目标 uint32 索引与位移
uint32_idx = tile_uint32_offset + (gidx // capacity)
bit_shift = (gidx % capacity) * bit

# 构造要写入的 uint32 值
value_to_write = tl.cast(q_int << bit_shift, tl.uint32)

# 使用原子 OR 操作写入
tl.atomic_or(packed_not_ptr + uint32_idx, value_to_write, mask=mask_valid)
```

### CUDA Kernel 加载逻辑
```cuda
// 直接加载 uint32，天然对齐
for (int j = 0; j < MAX_UINT32_PER_TILE; j++) {
    if (j < units_needed) {
        Registers_quant[i * 32 + j] = GlobalPTR_quant[uint32_offset + j];
    }
}
```

### CUDA Kernel 解包逻辑
```cuda
// 从 uint32 中提取 2-bit 量化值
int unit_idx = j / capacity;           // 第几个 uint32
int bit_offset = (j % capacity) * bit; // 在 uint32 内的位偏移
uint32_t packed_unit = quant_units[unit_idx];
uint32_t q_value = (packed_unit >> bit_offset) & mask;
```

## 优势总结

### ✅ 正确性
- 原子操作保证并发写入无覆盖
- 避免了 uint8 的数据竞争问题

### ✅ 性能
- 空间开销几乎相同（对齐后）
- 向量化读取性能优秀
- 原子操作开销可接受

### ✅ 可维护性
- 代码逻辑更清晰
- 天然对齐，无需额外处理
- 类型一致性更好

### ✅ 可扩展性
- 易于支持其他量化位宽（1-bit, 4-bit）
- 统一的存储格式

## 下一步

1. **编译验证**
   ```bash
   conda activate mustafar
   bash kernel_quant/build_quant_kernel.sh
   ```

2. **功能测试**
   ```bash
   python test_mustafar_key_formulation_quant.py
   ```

3. **性能测试**
   - 对比 uint8 vs uint32 的实际性能
   - 测试不同稀疏度下的压缩比
   - 测试端到端推理速度

## 预期结果

- ✅ 编译成功，无错误
- ✅ 功能测试通过，结果与参考实现接近
- ✅ 内存对齐错误消失
- ✅ 压缩比与 uint8 方案相同
- ✅ 性能可接受（略有原子操作开销）
