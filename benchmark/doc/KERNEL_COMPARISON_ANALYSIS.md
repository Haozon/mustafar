# 稀疏算子 vs 稀疏+量化算子对比分析

## 📊 核心区别总结

| 特性 | 无量化版本 | 量化版本 |
|------|-----------|---------|
| **Kernel 调用** | `mustafar_package.mustafar_key_formulation` | `mustafar_quant_cuda.mustafar_quant_sparse_forward` |
| **压缩函数** | `compression.convert_key_batched` | `compression_quant.convert_key_batched_quant` |
| **数据结构** | `[bmps, idxs, nzs, nz_offset]` | `[bmps, tile_offsets, [packed_quant], None, scales, zeros]` |
| **量化参数** | 无 | `scales, zeros` (per-tile) |
| **数据类型** | FP16 | 2-bit (打包成 uint32) |

---

## 🔍 详细对比

### **1. Key Attention 计算**

#### **无量化版本：**
```python
# 调用稀疏矩阵乘法 kernel
att_compressed = mustafar_package.mustafar_key_formulation(
    k_compressed[0],      # bitmaps
    torch.cat(k_compressed[2]),  # nzs (FP16 非零值)
    k_compressed[1],      # idxs
    k_compressed[3],      # nz_offset
    padded_query,         # query
    compressed_length,    # M
    model_dim,            # K
    total_batch_size,     # batch
    self.num_key_value_groups
)
```

**数据流：**
```
Query (FP16) × Compressed Key (FP16) → Attention Scores (FP16)
```

#### **量化版本：**
```python
# 调用量化稀疏矩阵乘法 kernel
k_nz_quant = k_compressed[2][0].to(torch.int32)  # 2-bit 打包数据

att_compressed = mustafar_quant_cuda.mustafar_quant_sparse_forward(
    k_compressed[0],            # bitmaps
    k_nz_quant,                 # NZ_quant (uint32, 打包的2-bit)
    k_compressed[1],            # tile_offsets
    k_scales.flatten().to(torch.float32),  # scales (per-tile)
    k_offsets.flatten().to(torch.float32), # zeros (per-tile)
    padded_query,               # query (FP16)
    compressed_length,          # M
    model_dim,                  # K
    total_batch_size,           # batch
    self.num_key_value_groups,
    self.quant_bits,            # 2
    16                          # capacity (16个2-bit值/uint32)
)
```

**数据流：**
```
Query (FP16) × [Compressed Key (2-bit) + Scales + Zeros] 
    → 反量化 (FP16) 
    → Attention Scores (FP16)
```

---

### **2. Value Attention 计算**

#### **无量化版本：**
```python
attn_output_compressed = mustafar_package.mustafar_value_formulation(
    v_compressed[0],      # bitmaps
    torch.cat(v_compressed[2]),  # nzs (FP16)
    v_compressed[1],      # idxs
    v_compressed[3],      # nz_offset
    padded_score,         # attention weights
    self.Reduction_Workspace,
    model_dim,
    compressed_length,
    total_batch_size,
    self.num_key_value_groups
)
```

#### **量化版本：**
```python
v_nz_quant = v_compressed[2][0].to(torch.int32)

attn_output_compressed = mustafar_quant_cuda.mustafar_quant_sparse_value_forward(
    v_compressed[0],            # bitmaps
    v_nz_quant,                 # NZ_quant (uint32)
    v_compressed[1],            # tile_offsets
    v_scales.flatten().to(torch.float32),  # scales
    v_offsets.flatten().to(torch.float32), # zeros
    padded_score,               # attention weights
    self.Reduction_Workspace,
    model_dim,                  # 输出维度
    compressed_length,          # K维度
    total_batch_size,
    self.num_key_value_groups,
    self.quant_bits,
    16
)
```

---

### **3. 压缩函数对比**

#### **无量化版本：**
```python
# Key 压缩
k_bmps, k_idxs, k_nzs = compression.convert_key_batched(
    k_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim)
)

# 返回：
# - k_bmps: bitmap (uint64)
# - k_idxs: tile indices (int32)
# - k_nzs: 非零值列表 (FP16) [batch个tensor]
```

#### **量化版本：**
```python
# Key 压缩 + 量化
k_bmps, k_tile_offsets, k_packed_quant, k_scales, k_zeros = \
    compression_quant.convert_key_batched_quant(
        k_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim)
    )

# 返回：
# - k_bmps: bitmap (uint64)
# - k_tile_offsets: tile偏移 (int32)
# - k_packed_quant: 打包的2-bit数据 (uint32)
# - k_scales: 量化scale (FP32) [B, num_tiles]
# - k_zeros: 量化zero-point (FP32) [B, num_tiles]
```

---

### **4. 数据结构对比**

#### **无量化版本的 past_key_value：**
```python
past_key_value = (
    k_compressed,      # [bmps, idxs, nzs_list, nz_offset]
    k_local_window,    # FP16 tensor
    v_compressed,      # [bmps, idxs, nzs_list, nz_offset]
    v_local_window,    # FP16 tensor
    compressed_length, # int
    kv_seq_len        # int
)
```

#### **量化版本的 past_key_value：**
```python
past_key_value = (
    k_compressed,      # [bmps, tile_offsets, [packed_quant], None, scales, zeros]
    k_local_window,    # FP16 tensor
    v_compressed,      # [bmps, tile_offsets, [packed_quant], None, scales, zeros]
    v_local_window,    # FP16 tensor
    compressed_length, # int
    k_scales,          # FP32 tensor [B, num_tiles]
    k_offsets,         # FP32 tensor [B, num_tiles]
    v_scales,          # FP32 tensor [B, num_tiles]
    v_offsets,         # FP32 tensor [B, num_tiles]
    kv_seq_len        # int
)
```

---

## 🎯 关键发现

### **1. 量化增加了额外的计算步骤**

**无量化流程：**
```
Sparse MatMul: Query × Compressed_Key → Attention
```

**量化流程：**
```
Dequantize: Compressed_Key_2bit + Scales + Zeros → Key_FP16
Sparse MatMul: Query × Key_FP16 → Attention
```

**额外开销：反量化操作**

---

### **2. 量化参数存储**

每个 tile (64 维) 需要：
- 1 个 scale (FP32, 4 bytes)
- 1 个 zero-point (FP32, 4 bytes)

**总开销：**
```python
num_tiles = (compressed_length * model_dim) // 64
scale_memory = num_tiles * 4 bytes
zero_memory = num_tiles * 4 bytes
total_quant_params = num_tiles * 8 bytes
```

**示例（compressed_length=256, model_dim=128）：**
```
num_tiles = 256 * 128 / 64 = 512
quant_params_memory = 512 * 8 = 4 KB (per batch)
```

---

### **3. 数据打包方式**

**2-bit 量化打包：**
```
16 个 2-bit 值 → 1 个 uint32
原始: [v0, v1, ..., v15] (每个 2-bit)
打包: uint32 = v0 | (v1<<2) | (v2<<4) | ... | (v15<<30)
```

**解包（在 kernel 中）：**
```cuda
uint32_t packed = packed_quant[idx];
uint8_t v0 = (packed >> 0) & 0x3;   // 取最低2位
uint8_t v1 = (packed >> 2) & 0x3;   // 取次低2位
...
uint8_t v15 = (packed >> 30) & 0x3; // 取最高2位
```

---

## 🔧 性能瓶颈分析

### **为什么量化版本更慢？**

#### **1. 反量化开销（主要瓶颈）**
```cuda
// 在 kernel 中，每次访问都需要：
float value = (quantized_value * scale) + zero_point;
```

**估算：**
- 每个非零值需要：1次解包 + 1次乘法 + 1次加法
- 如果有 N 个非零值，总计：3N 次操作
- 相比 FP16 直接访问，慢 **2-3x**

#### **2. 内存访问模式**
```
无量化：直接读取 FP16 值（2 bytes）
量化：读取 uint32（4 bytes）+ 解包 + 读取 scale（4 bytes）+ 读取 zero（4 bytes）
     = 12 bytes 总访问量 vs 2 bytes
```

**内存带宽增加 6x！**

#### **3. Kernel 调用次数**
从 benchmark 数据：
- 无量化：compress 调用 ~171 次
- 量化：compress 调用 ~1,152 次（增加 6.7x）
- 量化：quantize kernel 调用 ~229,440 次（新增）

---

## 💡 优化建议

### **优先级 1：融合反量化到 SpMM kernel**
```cuda
// 当前（慢）：分两步
dequantize(compressed_key) → key_fp16
spmm(query, key_fp16) → attention

// 优化（快）：融合
spmm_with_dequant(query, compressed_key, scales, zeros) → attention
```

**预期收益：消除反量化的内存访问开销，快 2-3x**

### **优先级 2：优化量化参数访问**
```cuda
// 当前：每个值都读取 scale/zero
for (int i = 0; i < nnz; i++) {
    float scale = scales[tile_id];
    float zero = zeros[tile_id];
    value = (quant[i] * scale) + zero;
}

// 优化：使用 shared memory 缓存
__shared__ float shared_scales[MAX_TILES];
__shared__ float shared_zeros[MAX_TILES];
// 一次性加载到 shared memory
// 后续访问从 shared memory 读取
```

**预期收益：减少全局内存访问，快 1.5-2x**

### **优先级 3：批处理量化操作**
```python
# 当前：逐个 token 量化
for token in tokens:
    quantize(token)

# 优化：批量量化
quantize_batch(all_tokens)
```

**预期收益：减少 kernel launch 开销，快 1.3-1.5x**

---

## 📈 预期优化效果

| 优化方案 | 当前 TPOT | 优化后 TPOT | 加速比 |
|---------|----------|------------|--------|
| **当前** | 82.78 ms | - | 1.0x |
| **融合反量化** | 82.78 ms | 40-50 ms | 1.7-2.1x |
| **+ 优化参数访问** | 40-50 ms | 30-35 ms | 1.3-1.7x |
| **+ 批处理** | 30-35 ms | 25-30 ms | 1.2-1.4x |
| **总计** | 82.78 ms | **25-30 ms** | **2.8-3.3x** |

**目标：让量化版本比无量化版本快 2x！**

---

## ✅ 结论

1. **量化版本的主要瓶颈是反量化开销**
2. **需要融合反量化到 SpMM kernel 中**
3. **优化后，量化版本有望达到：**
   - TPOT: 25-30 ms（比无量化快 2x）
   - 内存: 节省 15-18%
   - TTFT: 已经快 49%（保持）
