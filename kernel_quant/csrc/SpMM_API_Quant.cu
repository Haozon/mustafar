/***************************************************************************
 * Quantized Sparse Matrix Multiplication API
 * Based on SpMM_API.cu with 2-bit quantization support
 ***************************************************************************/

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel_Quant.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Key矩阵的量化 Split-K 稀疏矩阵乘法内核启动器
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void Key_SplitK_Kernel_Ex_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,  // 改为 uint32*
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    const half*  B,
    half*        Reduction_Workspace,
    const int    M_Global,
    const int    N_Global,
    const int    K_Global,
    int          Split_K, 
    const int    Batch_Size, 
    const int    num_key_value_groups,
    int          bit,
    int          capacity)
{
    Split_K = 1;  // Key 矩阵强制 Split_K = 1
    
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    
    cudaFuncSetAttribute(
        Key_Kernel_Quant<TilingConfig, SparseKernelConfig>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    int dimN = max(N_Global / TilingConfig::TILE_N, 1);
    int dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    Key_Kernel_Quant<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ_quant, tile_offsets, scales, zeros,
        B, Reduction_Workspace, M_Global, N_Global, K_Global, 1, 
        Batch_Size, num_key_value_groups, bit, capacity);
}

/**
 * @brief Key矩阵量化稀疏矩阵乘法的主要 API 接口
 */
cudaError_t Key_SplitK_API_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,  // 改为 uint32*
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    const half*  B,
    half*        C,
    const int    M_Global,
    const int    N_Global,
    const int    K_Global,
    half*        Reduction_Workspace,
    int          Split_K,
    const int    Batch_Size, 
    const int    num_key_value_groups,
    int          bit,
    int          capacity)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API_Quant.cu/Key_SplitK_API_Quant(): Entering API----\n");
    printf("M: %d, N: %d, K: %d, SplitK: %d, bit: %d, capacity: %d\n", 
           M_Global, N_Global, K_Global, Split_K, bit, capacity);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    
    switch (N_Global) {
        case 8:
            Key_SplitK_Kernel_Ex_Quant<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, 
                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;
    
    if (Split_K == 1)
        return Error;
    
    return cudaGetLastError();
}

/**
 * @brief Value矩阵的量化 Split-K 稀疏矩阵乘法内核启动器
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void Value_SplitK_Kernel_Ex_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    const half*  B,
    half*        Reduction_Workspace,
    const int    M_Global,
    const int    N_Global,
    const int    K_Global,
    int          Split_K, 
    const int    Batch_Size, 
    const int    num_key_value_groups,
    int          bit,
    int          capacity)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    
    cudaFuncSetAttribute(
        Value_Kernel_Quant<TilingConfig, SparseKernelConfig>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    int dimN = max(N_Global / TilingConfig::TILE_N, 1);
    int dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    Value_Kernel_Quant<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ_quant, tile_offsets, scales, zeros,
        B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, 
        Batch_Size, num_key_value_groups, bit, capacity);
}

/**
 * @brief Value矩阵量化稀疏矩阵乘法的主要 API 接口
 */
cudaError_t Value_SplitK_API_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,
    const uint32_t* tile_offsets,
    const float* scales,
    const float* zeros,
    const half*  B,
    half*        C,
    const int    M_Global,
    const int    N_Global,
    const int    K_Global,
    half*        Reduction_Workspace,
    int          Split_K,
    const int    Batch_Size, 
    const int    num_key_value_groups,
    int          bit,
    int          capacity)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API_Quant.cu/Value_SplitK_API_Quant(): Entering API----\n");
    printf("M: %d, N: %d, K: %d, SplitK: %d, bit: %d, capacity: %d\n", 
           M_Global, N_Global, K_Global, Split_K, bit, capacity);
    assert(K_Global % TILE_K == 0);
#endif
    
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1) {
        SpMM_SplitK_OutputPTR = C;
    } else {
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    }
    
    switch (N_Global) {
        case 8:
            Value_SplitK_Kernel_Ex_Quant<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, 
                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;
    
    if (Split_K == 1)
        return Error;
    
    dim3 GridDim((M_Global * N_Global) / 256, 1, Batch_Size);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(
        C, Reduction_Workspace, M_Global, N_Global, Split_K, Batch_Size);
    
    return cudaGetLastError();
}
