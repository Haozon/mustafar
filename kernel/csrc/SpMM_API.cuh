#ifndef SPMM_API_CUH
#define SPMM_API_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

cudaError_t Key_SplitK_API(
    cudaStream_t stream,
    const half* A,
    const uint64_t* bmp,
    const uint4* NZ,
    const uint32_t* idx,
    const uint32_t* NZ_offset,
    const half* B,
    half* C,
    const int M_Global,
    const int N_Global,
    const int K_Global,
    half* Reduction_Workspace,
    int Split_K,
    const int Batch_Size,
    const int num_key_value_groups);

cudaError_t Value_SplitK_API(
    cudaStream_t stream,
    const half* A,
    const uint64_t* bmp,
    const uint4* NZ,
    const uint32_t* idx,
    const uint32_t* NZ_offset,
    const half* B,
    half* C,
    const int M_Global,
    const int N_Global,
    const int K_Global,
    half* Reduction_Workspace,
    int Split_K,
    const int Batch_Size,
    const int num_key_value_groups);

#endif
