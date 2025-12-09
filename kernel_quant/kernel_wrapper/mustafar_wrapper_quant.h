#pragma once 
#include <torch/extension.h>
#include <cuda_runtime.h> 

torch::Tensor mustafar_key_formulation_quant(
    torch::Tensor bmp,              // torch.int64
    torch::Tensor NZ_quant,         // torch.uint8 (量化值)
    torch::Tensor tile_offsets,     // torch.int32 (tile 字节偏移)
    torch::Tensor scales,           // torch.float32 (per-tile scale)
    torch::Tensor zeros,            // torch.float32 (per-tile zero_point)
    torch::Tensor B,                // torch.float16
    int M_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups,
    int bit,                        // 量化位宽
    int capacity                    // 每字节容纳的量化值数
);

torch::Tensor mustafar_value_formulation_quant(
    torch::Tensor bmp,
    torch::Tensor NZ_quant,
    torch::Tensor tile_offsets,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor B,
    torch::Tensor Reduction_Workspace,
    int M_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups,
    int bit,
    int capacity
);
