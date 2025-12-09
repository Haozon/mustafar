#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mustafar_wrapper_quant.h"
#include "SpMM_API_Quant.cuh"

torch::Tensor mustafar_key_formulation_quant(
    torch::Tensor bmp,
    torch::Tensor NZ_quant,
    torch::Tensor tile_offsets,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor B,
    int M_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups,
    int bit,
    int capacity
) 
{
    // 检查设备
    if (B.device() != bmp.device() || B.device() != NZ_quant.device() || 
        B.device() != tile_offsets.device() || B.device() != scales.device() || 
        B.device() != zeros.device()) {
        throw std::runtime_error("All input tensors must be on the same device.");
    }
    
    // 检查数据类型
    if (B.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor B must be of type float16.");
    }
    if (NZ_quant.dtype() != at::kInt) {
        throw std::runtime_error("Tensor NZ_quant must be of type uint32.");
    }
    if (bmp.dtype() != at::kLong) {
        throw std::runtime_error("Tensor bmp must be of type int64.");
    }
    if (tile_offsets.dtype() != at::kInt) {
        throw std::runtime_error("Tensor tile_offsets must be of type int32.");
    }
    if (scales.dtype() != at::kFloat) {
        throw std::runtime_error("Tensor scales must be of type float32.");
    }
    if (zeros.dtype() != at::kFloat) {
        throw std::runtime_error("Tensor zeros must be of type float32.");
    }

    TORCH_CHECK(
        bmp.is_contiguous() && NZ_quant.is_contiguous() && 
        tile_offsets.is_contiguous() && B.is_contiguous() &&
        scales.is_contiguous() && zeros.is_contiguous(),
        "All tensors must be contiguous."
    );

    TORCH_CHECK(
        bmp.is_cuda() && NZ_quant.is_cuda() && tile_offsets.is_cuda() && 
        B.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
        "All tensors must be on CUDA device."
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    auto C = torch::zeros({Batch_Size, 8, M_Global}, B.options());
    
    // 转换指针
    bmp = bmp.to(at::kUInt64);
    const uint64_t* bmp_cuda_ptr = bmp.data_ptr<uint64_t>();
    
    const int32_t* NZ_quant_aten_ptr = NZ_quant.data_ptr<int32_t>();
    const uint32_t* NZ_quant_cuda_ptr = reinterpret_cast<const uint32_t*>(NZ_quant_aten_ptr);
    
    const int32_t* tile_offsets_aten_ptr = tile_offsets.data_ptr<int32_t>();
    const uint32_t* tile_offsets_cuda_ptr = reinterpret_cast<const uint32_t*>(tile_offsets_aten_ptr);
    
    const float* scales_cuda_ptr = scales.data_ptr<float>();
    const float* zeros_cuda_ptr = zeros.data_ptr<float>();
    
    const at::Half* B_aten_ptr = B.data_ptr<at::Half>();
    const half* B_cuda_ptr = reinterpret_cast<const half*>(B_aten_ptr);
    
    at::Half* C_aten_ptr = C.data_ptr<at::Half>();
    half* C_cuda_ptr = reinterpret_cast<half*>(C_aten_ptr);

    // 调用 CUDA kernel
    Key_SplitK_API_Quant(
        stream,
        static_cast<half*>(nullptr),
        bmp_cuda_ptr,
        NZ_quant_cuda_ptr,
        tile_offsets_cuda_ptr,
        scales_cuda_ptr,
        zeros_cuda_ptr,
        B_cuda_ptr,
        C_cuda_ptr,
        M_Global,
        8,  // N_Global
        K_Global,
        static_cast<half*>(nullptr),
        1,  // Split_K
        Batch_Size, 
        num_key_value_groups,
        bit,
        capacity
    );

    return C;
}

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
) 
{
    // 检查设备
    if (B.device() != bmp.device() || B.device() != NZ_quant.device() || 
        B.device() != tile_offsets.device() || B.device() != scales.device() || 
        B.device() != zeros.device()) {
        throw std::runtime_error("All input tensors must be on the same device.");
    }
    
    // 检查数据类型
    if (B.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor B must be of type float16.");
    }
    if (NZ_quant.dtype() != at::kInt) {
        throw std::runtime_error("Tensor NZ_quant must be of type uint32.");
    }
    if (bmp.dtype() != at::kLong) {
        throw std::runtime_error("Tensor bmp must be of type int64.");
    }
    if (tile_offsets.dtype() != at::kInt) {
        throw std::runtime_error("Tensor tile_offsets must be of type int32.");
    }
    if (scales.dtype() != at::kFloat) {
        throw std::runtime_error("Tensor scales must be of type float32.");
    }
    if (zeros.dtype() != at::kFloat) {
        throw std::runtime_error("Tensor zeros must be of type float32.");
    }

    TORCH_CHECK(
        bmp.is_contiguous() && NZ_quant.is_contiguous() && 
        tile_offsets.is_contiguous() && scales.is_contiguous() && 
        zeros.is_contiguous(),
        "All tensors must be contiguous."
    );

    TORCH_CHECK(
        bmp.is_cuda() && NZ_quant.is_cuda() && tile_offsets.is_cuda() && 
        B.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
        "All tensors must be on CUDA device."
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int Split_K = 1;
    auto C = torch::zeros({Batch_Size, 8, M_Global}, B.options());
    
    // 转换指针
    bmp = bmp.to(at::kUInt64);
    const uint64_t* bmp_cuda_ptr = bmp.data_ptr<uint64_t>();
    
    const int32_t* NZ_quant_aten_ptr = NZ_quant.data_ptr<int32_t>();
    const uint32_t* NZ_quant_cuda_ptr = reinterpret_cast<const uint32_t*>(NZ_quant_aten_ptr);
    
    const int32_t* tile_offsets_aten_ptr = tile_offsets.data_ptr<int32_t>();
    const uint32_t* tile_offsets_cuda_ptr = reinterpret_cast<const uint32_t*>(tile_offsets_aten_ptr);
    
    const float* scales_cuda_ptr = scales.data_ptr<float>();
    const float* zeros_cuda_ptr = zeros.data_ptr<float>();
    
    const at::Half* B_aten_ptr = B.data_ptr<at::Half>();
    const half* B_cuda_ptr = reinterpret_cast<const half*>(B_aten_ptr);
    
    at::Half* C_aten_ptr = C.data_ptr<at::Half>();
    half* C_cuda_ptr = reinterpret_cast<half*>(C_aten_ptr);
    
    at::Half* Reduction_Workspace_aten_ptr = Reduction_Workspace.data_ptr<at::Half>();
    half* Reduction_Workspace_cuda_ptr = reinterpret_cast<half*>(Reduction_Workspace_aten_ptr);

    // 调用 CUDA kernel
    Value_SplitK_API_Quant(
        stream,
        static_cast<half*>(nullptr),
        bmp_cuda_ptr,
        NZ_quant_cuda_ptr,
        tile_offsets_cuda_ptr,
        scales_cuda_ptr,
        zeros_cuda_ptr,
        B_cuda_ptr,
        C_cuda_ptr,
        M_Global,
        8,  // N_Global
        K_Global,
        Reduction_Workspace_cuda_ptr,
        Split_K,
        Batch_Size, 
        num_key_value_groups,
        bit,
        capacity
    );

    return C;
}
