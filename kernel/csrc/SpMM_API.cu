/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

/*
函数功能总结：
1. print_packed_halfs
功能: 调试工具，解包并打印32位整数中的两个half值
用途: 验证数据打包/解包的正确性
2. Key_SplitK_Kernel_Ex
功能: Key矩阵计算的内核启动器
特点: 强制Split_K=1，专门优化Q×K^T计算
配置: 使用较大的tile配置以适应Key矩阵特性
3. Key_SplitK_API
功能: Key矩阵计算的主要API接口
特点: 简化的Split-K处理，主要用于注意力分数计算
验证: 严格的参数检查（M必须是256的倍数）
4. Value_SplitK_Kernel_Ex
功能: Value矩阵计算的内核启动器
特点: 支持真正的Split-K优化
配置: 使用较小的tile配置以适应Value矩阵特性
5. Value_SplitK_API
功能: Value矩阵计算的主要API接口
特点: 完整的Split-K支持，包括自动归约
灵活性: 对M维度的限制较少，支持更多样的矩阵尺寸
这些API函数构成了稀疏注意力计算的完整接口，分别处理注意力机制的两个关键步骤：Q×K^T和Attention×V。
*/

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @brief 打印打包的半精度浮点数
 * 
 * 该函数用于调试目的，将一个32位整数中打包的两个16位半精度浮点数解包并打印。
 * 在CUDA中，通常将两个half值打包到一个uint32中以提高内存效率。
 * 
 * @param packed_value 包含两个half值的32位整数
 *                     - 低16位：第一个half值
 *                     - 高16位：第二个half值
 */
void print_packed_halfs(uint32_t packed_value) {
    // Extract the first half (lower 16 bits)
    // 提取第一个half值（低16位）
    half first_half = (half)(packed_value & 0xFFFF);  // Mask to get the lower 16 bits

    // Extract the second half (upper 16 bits)
    // 提取第二个half值（高16位）
    half second_half = (half)((packed_value >> 16) & 0xFFFF);  // Shift right and mask to get the upper 16 bits

    // Print the two half values
    // 打印两个half值（转换为float以便阅读）
    printf("First half: %f\n", __half2float(first_half));  // Convert half to float for readable output
    printf("Second half: %f\n", __half2float(second_half));
}

/**
 * @brief Key矩阵的Split-K稀疏矩阵乘法内核启动器
 * 
 * 该函数负责配置和启动Key_Kernel，处理注意力机制中Q×K^T的计算。
 * 使用Split-K技术来提高GPU的SM（流式多处理器）占用率。
 * 
 * 主要功能：
 * - 配置共享内存大小
 * - 计算网格和块的维度
 * - 启动CUDA内核
 * 
 * @param stream CUDA流，用于异步执行
 * @param A 稀疏矩阵A（实际未直接使用，通过压缩格式访问）
 * @param bmp 位图数组，标识稀疏矩阵中非零元素的位置
 * @param NZ 非零元素值数组（uint4格式，每个包含4个uint32）
 * @param idx 索引数组，指向每个tile的非零元素起始位置
 * @param NZ_offset 每个批次/组的非零元素偏移量数组
 * @param B 密集矩阵B（Key矩阵）
 * @param Reduction_Workspace 用于存储中间结果的工作空间
 * @param M_Global 矩阵A的行数
 * @param N_Global 矩阵B的列数
 * @param K_Global 矩阵A的列数/矩阵B的行数
 * @param Split_K Split-K优化的分割数（在此函数中强制设为1）
 * @param Batch_Size 批处理大小
 * @param num_key_value_groups GQA中的键值组数
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void Key_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint64_t* bmp, 
                                  const uint4* NZ,
                                  //const uint32_t* NZ, 
                                  const uint32_t* idx,
                                  //const uint32_t* bmp_idx_offset, 
                                  const uint32_t* NZ_offset,
                                  //const uint4* Compressed_A,
                                  //const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K, 
                                  const int    Batch_Size, 
                                  const int    num_key_value_groups)
{
    // 强制设置Split_K为1（Key矩阵通常不需要Split-K优化）
    Split_K = 1;
    // 计算所需的共享内存大小
    // 需要存储：矩阵A的tile (TILE_M × TILE_K) + 矩阵B的tile (TILE_N × TILE_K)
    // 双缓冲需要×2，还需要存储输出结果C的tile
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    // 设置内核的动态共享内存大小
    cudaFuncSetAttribute(
        Key_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    // 计算网格维度
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
        //fatter N size might benefit from dimN larger than 1. (1 is the preset for Coruscant)
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    //dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
        //each M tiled row handled by SplitK TBs.
   // 网格配置：(dimN, dimM, Batch_Size)
    // - dimN: 处理N维度的块数
    // - dimM: 处理M维度的块数  
    // - Batch_Size: 批处理维度
    dim3 GridDim(dimN, dimM, Batch_Size);
    // 块配置：每个块包含多个warp
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    

    //std::cout << "----SpMM_SplitK_Kernel_Ex(): Shared Memory Size: " << SHMEM_SZ << " Bytes" << std::endl;
    //std::cout << "----SpMM_SplitK_Kernel_Ex(): GridDim: " << dimN << "x" << dimM << " BlockDim: " << WARP_SIZE * TilingConfig::BLOCK_WARPS << "x1x1" << std::endl;
        // GridDim: 1x196: (7168/256) * 7(Split_K)
    // stream is just the GPU job_ID.
    // 启动Key内核
    Key_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ, idx, /*bmp_idx_offset,*/ NZ_offset, //Compressed_A, TileOffsets, 
        B, Reduction_Workspace, M_Global, N_Global, K_Global, 1, Batch_Size, num_key_value_groups); //explicitly set Split_K to 1. 
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
/**
 * @brief Key矩阵稀疏矩阵乘法的主要API接口
 * 
 * 这是Key矩阵计算的主要入口点，处理注意力机制中的Q×K^T运算。
 * 支持批处理、GQA（分组查询注意力）和可选的Split-K优化。
 * 
 * 工作流程：
 * 1. 参数验证和调试信息输出
 * 2. 根据Split-K设置选择输出位置
 * 3. 根据N_Global大小选择合适的tile配置
 * 4. 启动相应的内核
 * 5. 如果需要，执行Split-K归约
 * 
 * @param stream CUDA流
 * @param A 稀疏矩阵A
 * @param bmp 位图数组
 * @param NZ 非零元素数组
 * @param idx 索引数组
 * @param NZ_offset 非零元素偏移数组
 * @param B 密集矩阵B
 * @param C 输出矩阵C
 * @param M_Global 矩阵维度M
 * @param N_Global 矩阵维度N
 * @param K_Global 矩阵维度K
 * @param Reduction_Workspace Split-K归约工作空间
 * @param Split_K Split-K分割数
 * @param Batch_Size 批处理大小
 * @param num_key_value_groups GQA组数
 * @return cudaError_t CUDA错误码
 */
cudaError_t Key_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ,
                            //const uint32_t* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            //const uint4* Compressed_A,
                            //const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K, //given that this is always 1. 
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API.cu/SpMM_SplitK_API(): Entering SpMM_SplitK_API----\n");
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    // 参数验证
    assert(K_Global % TILE_K == 0); // K必须是TILE_K的倍数
    assert(M_Global % 256 == 0); // M必须是256的倍数
#endif
    // 根据Split_K选择输出位置
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C; // 直接输出到最终结果
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    //printf("Beginning of SpMM_SplitK_Kernel_Ex, N_Global is %d\n", N_Global); donghyeon: it's just the input.
    // 根据N_Global大小选择合适的tile配置
    switch (N_Global) {

        case 8:
            // 对于N=8的情况，使用TilingConfig<4, 1, 1, 1>配置
            // 参数含义：TILE_M=4*64=256, TILE_N=1*16=16, BLOCK_ROW_WARPS=1, BLOCK_COL_WARPS=1
            Key_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx, NZ_offset,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups);
        // 可以添加更多case来支持不同的N_Global大小
            break;

    }
    // 检查内核执行错误
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;
    // 如果Split_K=1，无需归约，直接返回
    if (Split_K == 1)
        return Error;
    //dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    //dim3 BlockDim(WARP_SIZE, 1, 1);
    //SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    // 如果Split_K>1，需要执行归约操作（当前代码中被注释掉）
    // 这里应该调用SplitK_Reduction内核来合并Split-K的结果
    return cudaGetLastError();
}

/**
 * @brief Value矩阵的Split-K稀疏矩阵乘法内核启动器
 * 
 * 该函数负责配置和启动Value_Kernel，处理注意力机制中Attention×V的计算。
 * 与Key_SplitK_Kernel_Ex类似，但针对Value矩阵的特殊需求进行了优化。
 * 
 * 主要区别：
 * - 支持真正的Split-K优化（不强制设为1）
 * - 使用不同的tile配置
 * - 可能需要后续的归约操作
 * 
 * 参数含义与Key_SplitK_Kernel_Ex相同
 */
template<typename TilingConfig, typename SparseKernelConfig>
static void Value_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint64_t* bmp, 
                                  const uint4* NZ,
                                  //const uint32_t* NZ, 
                                  const uint32_t* idx,
                                  //const uint32_t* bmp_idx_offset, 
                                  const uint32_t* NZ_offset,
                                  //const uint4* Compressed_A,
                                  //const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K, 
                                  const int    Batch_Size, 
                                  const int    num_key_value_groups)
{
    //Split_K = 1;
    // 注意：这里没有强制设置Split_K=1，允许真正的Split-K优化
    
    // 计算共享内存大小（与Key内核相同）
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    // 设置Value内核的共享内存大小
    cudaFuncSetAttribute(
        Value_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    //printf("DEBUG: testing if this is reflected to pip\n");
    // 计算网格维度（与Key内核相同）
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
        //fatter N size might benefit from dimN larger than 1. (1 is the preset for Coruscant)
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    //dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
        //each M tiled row handled by SplitK TBs.
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    //std::cout << "----SpMM_SplitK_Kernel_Ex(): Shared Memory Size: " << SHMEM_SZ << " Bytes" << std::endl;
    //if DEBUG: std::cout << "----SpMM_SplitK_Kernel_Ex(): GridDim: " << dimN << "x" << dimM << "x" << Batch_Size << " BlockDim: " << WARP_SIZE * TilingConfig::BLOCK_WARPS << "x1x1" << std::endl;
        // GridDim: 1x196: (7168/256) * 7(Split_K)
    // stream is just the GPU job_ID.
    // 启动Value内核（注意这里传递真实的Split_K值）
    Value_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ, idx, /*bmp_idx_offset,*/ NZ_offset, //Compressed_A, TileOffsets, 
        B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups); //explicitly set Split_K to 1. AHH [05/22]
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
/**
 * @brief Value矩阵稀疏矩阵乘法的主要API接口
 * 
 * 这是Value矩阵计算的主要入口点，处理注意力机制中的Attention×V运算。
 * 与Key_SplitK_API类似，但支持真正的Split-K优化和后续的归约操作。
 * 
 * 主要特点：
 * - 支持Split-K优化以提高大矩阵的计算效率
 * - 当Split_K>1时，自动执行归约操作合并结果
 * - 使用不同的tile配置以适应Value矩阵的特性
 * 
 * 工作流程：
 * 1. 参数验证（M_Global不要求是256的倍数）
 * 2. 选择输出位置（直接输出或工作空间）
 * 3. 启动Value内核
 * 4. 如果Split_K>1，执行SplitK_Reduction归约
 * 
 * 参数含义与Key_SplitK_API相同
 */
cudaError_t Value_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ,
                            //const uint32_t* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            //const uint4* Compressed_A,
                            //const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K, //given that this is always 1. 
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API.cu/SpMM_SplitK_API(): Entering SpMM_SplitK_API----\n");
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    //assert(M_Global % 256 == 0);
    // 注意：Value矩阵不要求M_Global是256的倍数
#endif
    // 选择输出位置
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1){
        //printf("Split_K is 1, so no reduction is needed\n");
        SpMM_SplitK_OutputPTR = C; // 直接输出到最终结果
    }
    else{
        //printf("Reduction Workspace is selected as output location\n");
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    }
    // 根据N_Global选择tile配置
    switch (N_Global) {
       
        case 8:
            // 对于Value矩阵，使用TilingConfig<2, 1, 1, 1>配置
            // 参数含义：TILE_M=2*64=128, TILE_N=1*16=16, BLOCK_ROW_WARPS=1, BLOCK_COL_WARPS=1
            // 相比Key矩阵使用更小的TILE_M，可能是为了适应Value矩阵的内存访问模式
            //SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
            Value_SplitK_Kernel_Ex<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx, NZ_offset,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups);
            break;
        // 可以添加更多case支持不同的N_Global大小
    }
    // 检查内核执行错误
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;
    // 如果Split_K=1，无需归约
    if (Split_K == 1)
        return Error;
   
    //cudaStreamSynchronize(stream);
    //if DEBUG: printf("Starting Reduction with Split_K: %d, Warp_Size: %d\n", Split_K, WARP_SIZE);
    //dim3 GridDim((M_Global * N_Global) / 256, 1, Batch_Size);
    // 执行Split-K归约操作
    // 网格配置：每个块处理256个元素，需要(M_Global * N_Global) / 256个块
    dim3 GridDim((M_Global * N_Global) / 256, 1, Batch_Size);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    // 启动归约内核，将Split_K个部分结果合并为最终结果
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K, Batch_Size);
    return cudaGetLastError();
}
