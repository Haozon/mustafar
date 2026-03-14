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

constexpr int VALUE_TILE_CONFIG_AUTO = 0;
constexpr int VALUE_TILE_CONFIG_TILE64 = 1;
constexpr int VALUE_TILE_CONFIG_TILE128 = 2;
constexpr int VALUE_TILE_CONFIG_FUSED = 3;

__device__ __forceinline__ uint64_t shfl_sync_u64(unsigned mask, uint64_t value, int src_lane)
{
    uint32_t lo = static_cast<uint32_t>(value);
    uint32_t hi = static_cast<uint32_t>(value >> 32);
    lo = __shfl_sync(mask, lo, src_lane);
    hi = __shfl_sync(mask, hi, src_lane);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

template<int DequantMode>
__global__ void Value_Kernel_Quant_FusedN8(
    const uint64_t* bmp,
    const uint32_t* NZ_quant,
    const uint32_t* tile_offsets,
    const uint32_t* tile_counts,
    const half* scales,
    const half* zeros,
    const half* B,
    half* Output,
    const int M_Global,
    const int K_Global,
    int Split_K,
    const int num_key_value_groups)
{
    const int batch_id = blockIdx.z;
    const int group_id = blockIdx.z / num_key_value_groups;
    const int split_id = blockIdx.y;
    const int col_tile = blockIdx.x;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int tiles_per_token = M_Global / 64;
    const int total_tiles_per_group = (M_Global * K_Global) / 64;
    const uint64_t* bmp_group = bmp + group_id * total_tiles_per_group;
    const uint32_t* offset_group = tile_offsets + group_id * total_tiles_per_group;
    const uint32_t* count_group = tile_counts + group_id * total_tiles_per_group;
    const half* scale_group = scales + group_id * total_tiles_per_group;
    const half* zero_group = zeros + group_id * total_tiles_per_group;
    const half* score_batch = B + batch_id * K_Global * 8;

    const int NumKBlock = K_Global / 64;
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock = AverageNumKBlock * Split_K;
    const int PaddingKBlock = RoundedKBlock - NumKBlock;
    const int NumIter = (split_id == (Split_K - 1)) ? (AverageNumKBlock - PaddingKBlock) : AverageNumKBlock;
    const int token_start = split_id * AverageNumKBlock * 64;
    const int token_count = NumIter * 64;
    const unsigned full_mask = 0xffffffffu;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int pos0 = lane_id;
    const int pos1 = lane_id + WARP_SIZE;

    for (int tok = 0; tok < token_count; ++tok) {
        const int global_token = token_start + tok;
        const int tile_idx = global_token * tiles_per_token + col_tile;
        uint32_t nnz_tile = 0;
        uint32_t tile_offset = 0;
        uint64_t bmp_val = 0;
        float scale_f = 0.0f;
        float zero_f = 0.0f;
        float score_f = 0.0f;

        if (lane_id == 0) {
            nnz_tile = count_group[tile_idx];
            if (nnz_tile > 0) {
                tile_offset = offset_group[tile_idx];
                bmp_val = bmp_group[tile_idx];
                scale_f = __half2float(scale_group[tile_idx]);
                zero_f = __half2float(zero_group[tile_idx]);
            }
            score_f = __half2float(score_batch[warp_id * K_Global + global_token]);
        }
        nnz_tile = __shfl_sync(full_mask, nnz_tile, 0);
        if (nnz_tile == 0) {
            continue;
        }
        tile_offset = __shfl_sync(full_mask, tile_offset, 0);
        bmp_val = shfl_sync_u64(full_mask, bmp_val, 0);
        scale_f = __shfl_sync(full_mask, scale_f, 0);
        zero_f = __shfl_sync(full_mask, zero_f, 0);
        score_f = __shfl_sync(full_mask, score_f, 0);

        const uint64_t bmp_rev = __brevll(bmp_val);

        const uint64_t bit0 = 1ull << pos0;
        if (bmp_rev & bit0) {
            const uint64_t prefix0 = (pos0 == 0) ? 0ull : (bmp_rev & (bit0 - 1ull));
            const uint32_t rank0 = __popcll(prefix0);
            const uint32_t packed0 = NZ_quant[tile_offset + (rank0 >> 4)];
            const uint32_t q0 = (packed0 >> ((rank0 & 15) << 1)) & 0x3u;
            acc0 += score_f * ((static_cast<float>(q0) - zero_f) * scale_f);
        }

        const uint64_t bit1 = 1ull << pos1;
        if (bmp_rev & bit1) {
            const uint64_t prefix1 = bmp_rev & (bit1 - 1ull);
            const uint32_t rank1 = __popcll(prefix1);
            const uint32_t packed1 = NZ_quant[tile_offset + (rank1 >> 4)];
            const uint32_t q1 = (packed1 >> ((rank1 & 15) << 1)) & 0x3u;
            acc1 += score_f * ((static_cast<float>(q1) - zero_f) * scale_f);
        }
    }

    half* output_batch = Output + batch_id * M_Global * 8 * Split_K;
    half* output_split = output_batch + split_id * (M_Global * 8);
    half* output_tile = output_split + col_tile * 64;
    output_tile[pos0 + warp_id * M_Global] = __float2half_rn(acc0);
    output_tile[pos1 + warp_id * M_Global] = __float2half_rn(acc1);
}

/**
 * @brief Key矩阵的量化 Split-K 稀疏矩阵乘法内核启动器
 */
template<typename TilingConfig, typename SparseKernelConfig, bool Fast2Bit, int DequantMode>
static void Key_SplitK_Kernel_Ex_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,  // 改为 uint32*
    const uint32_t* tile_offsets,
    const half* scales,
    const half* zeros,
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
        Key_Kernel_Quant<TilingConfig, SparseKernelConfig, Fast2Bit, DequantMode>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    int dimN = max(N_Global / TilingConfig::TILE_N, 1);
    int dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    Key_Kernel_Quant<TilingConfig, SparseKernelConfig, Fast2Bit, DequantMode><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
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
    const half* scales,
    const half* zeros,
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
    int          capacity,
    int          dequant_mode)
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
            if (bit == 2 && capacity == 16) {
                if (dequant_mode == DEQUANT_MODE_MEMORY) {
                    Key_SplitK_Kernel_Ex_Quant<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_MEMORY>(
                        stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                        B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                        Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                } else {
                    Key_SplitK_Kernel_Ex_Quant<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_SPEED>(
                        stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                        B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                        Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                }
            } else {
                if (dequant_mode == DEQUANT_MODE_MEMORY) {
                    Key_SplitK_Kernel_Ex_Quant<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_MEMORY>(
                        stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                        B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                        Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                } else {
                    Key_SplitK_Kernel_Ex_Quant<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_SPEED>(
                        stream, A, bmp, NZ_quant, tile_offsets, scales, zeros,
                        B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                        Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                }
            }
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
template<typename TilingConfig, typename SparseKernelConfig, bool Fast2Bit, int DequantMode>
static void Value_SplitK_Kernel_Ex_Quant(
    cudaStream_t stream,
    const half*  A,
    const uint64_t* bmp, 
    const uint32_t* NZ_quant,
    const uint32_t* tile_offsets,
    const uint32_t* tile_counts,
    const uint32_t* tile_units,
    const half* scales,
    const half* zeros,
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
        Value_Kernel_Quant<TilingConfig, SparseKernelConfig, Fast2Bit, DequantMode>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    int dimN = max(N_Global / TilingConfig::TILE_N, 1);
    int dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    Value_Kernel_Quant<TilingConfig, SparseKernelConfig, Fast2Bit, DequantMode><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
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
    const uint32_t* tile_counts,
    const uint32_t* tile_units,
    const half* scales,
    const half* zeros,
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
    int          capacity,
    int          dequant_mode,
    int          value_tile_config)
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
    
    int effective_tile_config = value_tile_config;
    if (effective_tile_config == VALUE_TILE_CONFIG_AUTO) {
        effective_tile_config = (Split_K >= 4) ? VALUE_TILE_CONFIG_TILE64 : VALUE_TILE_CONFIG_TILE128;
    }

    switch (N_Global) {
        case 8:
            switch (effective_tile_config) {
                case VALUE_TILE_CONFIG_FUSED: {
                    dim3 GridDim(M_Global / 64, Split_K, Batch_Size);
                    dim3 BlockDim(8 * WARP_SIZE, 1, 1);
                    if (dequant_mode == DEQUANT_MODE_MEMORY) {
                        Value_Kernel_Quant_FusedN8<DEQUANT_MODE_MEMORY><<<GridDim, BlockDim, 0, stream>>>(
                            bmp, NZ_quant, tile_offsets, tile_counts, scales, zeros,
                            B, SpMM_SplitK_OutputPTR, M_Global, K_Global, Split_K, num_key_value_groups);
                    } else {
                        Value_Kernel_Quant_FusedN8<DEQUANT_MODE_SPEED><<<GridDim, BlockDim, 0, stream>>>(
                            bmp, NZ_quant, tile_offsets, tile_counts, scales, zeros,
                            B, SpMM_SplitK_OutputPTR, M_Global, K_Global, Split_K, num_key_value_groups);
                    }
                    break;
                }
                case VALUE_TILE_CONFIG_TILE64:
                    if (bit == 2 && capacity == 16) {
                        if (dequant_mode == DEQUANT_MODE_MEMORY) {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<1, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_MEMORY>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        } else {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<1, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_SPEED>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        }
                    } else {
                        if (dequant_mode == DEQUANT_MODE_MEMORY) {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<1, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_MEMORY>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        } else {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<1, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_SPEED>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        }
                    }
                    break;
                case VALUE_TILE_CONFIG_TILE128:
                default:
                    if (bit == 2 && capacity == 16) {
                        if (dequant_mode == DEQUANT_MODE_MEMORY) {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_MEMORY>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        } else {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>, true, DEQUANT_MODE_SPEED>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        }
                    } else {
                        if (dequant_mode == DEQUANT_MODE_MEMORY) {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_MEMORY>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        } else {
                            Value_SplitK_Kernel_Ex_Quant<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>, false, DEQUANT_MODE_SPEED>(
                                stream, A, bmp, NZ_quant, tile_offsets, tile_counts, tile_units, scales, zeros,
                                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global,
                                Split_K, Batch_Size, num_key_value_groups, bit, capacity);
                        }
                    }
                    break;
            }
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
