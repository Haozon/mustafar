#pragma once 
#include <torch/extension.h>
#include <cuda_runtime.h> 

torch::Tensor mustafar_key_formulation_quant(
    torch::Tensor bmp,              // torch.int64
    torch::Tensor NZ_quant,         // torch.int32 (packed quant values, uint32 bit layout)
    torch::Tensor tile_offsets,     // torch.int32 (tile uint32 偏移)
    torch::Tensor scales,           // torch.float16 (per-tile scale)
    torch::Tensor zeros,            // torch.float16 (per-tile zero_point)
    torch::Tensor B,                // torch.float16
    int M_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups,
    int bit,                        // 量化位宽
    int capacity,                   // 每字节容纳的量化值数
    int dequant_mode = 0            // 0: speed, 1: memory
);

torch::Tensor mustafar_key_formulation_quant_meta(
    torch::Tensor bmp,
    torch::Tensor NZ_quant,
    torch::Tensor tile_offsets,
    torch::Tensor tile_counts,
    torch::Tensor tile_units,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor B,
    int M_Global,
    int K_Global,
    int Batch_Size,
    int num_key_value_groups,
    int bit,
    int capacity,
    int dequant_mode = 0
);

torch::Tensor mustafar_value_formulation_quant(
    torch::Tensor bmp,
    torch::Tensor NZ_quant,
    torch::Tensor tile_offsets,
    torch::Tensor tile_counts,      // torch.int32
    torch::Tensor tile_units,       // torch.int32
    torch::Tensor scales,           // torch.float16
    torch::Tensor zeros,            // torch.float16
    torch::Tensor B,
    torch::Tensor Reduction_Workspace,
    int M_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups,
    int bit,
    int capacity,
    int dequant_mode = 0,           // 0: speed, 1: memory
    int split_k = 1,
    int value_tile_config = 0
);

torch::Tensor mustafar_value_formulation_quant_decode_n1(
    torch::Tensor bmp,
    torch::Tensor NZ_quant,
    torch::Tensor tile_offsets,
    torch::Tensor tile_counts,
    torch::Tensor tile_units,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor score,
    torch::Tensor Reduction_Workspace,
    int M_Global,
    int K_Global,
    int Batch_Size,
    int num_key_value_groups,
    int bit,
    int capacity,
    int dequant_mode = 0,
    int split_k = 1
);
