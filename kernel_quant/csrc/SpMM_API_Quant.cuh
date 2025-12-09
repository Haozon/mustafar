/***************************************************************************
 * Quantized Sparse Matrix Multiplication API Header
 * Based on SpMM_API.cuh with 2-bit quantization support
 ***************************************************************************/

#ifndef SPMM_API_QUANT_CUH
#define SPMM_API_QUANT_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @brief Key矩阵的量化 Split-K 稀疏矩阵乘法内核启动器模板
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void Key_SplitK_Kernel_Ex_Quant(
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
    int          capacity);

/**
 * @brief Key矩阵量化稀疏矩阵乘法的主要 API 接口
 * 
 * @param stream CUDA stream
 * @param A 输入矩阵 A (half precision)
 * @param bmp 位图索引
 * @param NZ_quant 量化后的非零元素 (2-bit)
 * @param tile_offsets Tile偏移量
 * @param scales 量化缩放因子
 * @param zeros 量化零点
 * @param B 输入矩阵 B (half precision)
 * @param C 输出矩阵 C (half precision)
 * @param M_Global M维度
 * @param N_Global N维度
 * @param K_Global K维度
 * @param Reduction_Workspace 中间结果工作空间
 * @param Split_K Split-K参数
 * @param Batch_Size 批次大小
 * @param num_key_value_groups KV组数量
 * @param bit 量化位数 (默认2)
 * @param capacity Tile容量
 */
cudaError_t Key_SplitK_API_Quant(
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
    int          capacity);

/**
 * @brief Value矩阵的量化 Split-K 稀疏矩阵乘法内核启动器模板
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
    int          capacity);

/**
 * @brief Value矩阵量化稀疏矩阵乘法的主要 API 接口
 * 
 * @param stream CUDA stream
 * @param A 输入矩阵 A (half precision)
 * @param bmp 位图索引
 * @param NZ_quant 量化后的非零元素 (2-bit)
 * @param tile_offsets Tile偏移量
 * @param scales 量化缩放因子
 * @param zeros 量化零点
 * @param B 输入矩阵 B (half precision)
 * @param C 输出矩阵 C (half precision)
 * @param M_Global M维度
 * @param N_Global N维度
 * @param K_Global K维度
 * @param Reduction_Workspace 中间结果工作空间
 * @param Split_K Split-K参数
 * @param Batch_Size 批次大小
 * @param num_key_value_groups KV组数量
 * @param bit 量化位数 (默认2)
 * @param capacity Tile容量
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
    int          capacity);

#endif // SPMM_API_QUANT_CUH
