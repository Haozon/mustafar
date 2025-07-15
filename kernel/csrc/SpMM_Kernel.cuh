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
    SpMM_CopyFromGlobalToReg: 从全局内存加载压缩的稀疏矩阵数据到寄存器
    SpMM_InitSharedMemory: 高效地将共享内存初始化为零
    SpMM_DecompressFromRegisterToShared: 将压缩的稀疏数据解压并存储到共享内存
    Key_Kernel: 专门处理Key矩阵的稀疏矩阵乘法内核
    Value_Kernel: 专门处理Value矩阵的稀疏矩阵乘法内核
    这些函数共同实现了高效的稀疏矩阵乘法运算，特别针对Transformer模型中的注意力机制进行了优化。
*/


#include "MatMulUtilities.cuh"
#include <vector>

#define DEBUG 0
#define DEBUG2 0
#define DEBUG1 0


/**
 * @brief 从全局内存加载稀疏矩阵数据到寄存器
 * 
 * 该函数负责将压缩存储的稀疏矩阵数据从全局内存加载到线程的寄存器中。
 * 每个线程处理2个位图（bitmap），每个位图对应一列的稀疏数据。
 * 
 * @param Registers_nz 存储非零元素值的寄存器数组（64个uint32）
 * @param Registers_bmp 存储位图的寄存器数组（2个uint64）
 * @param Registers_nnz 存储非零元素数量的寄存器数组（2个uint32）
 * @param GlobalPTR_nz 指向全局内存中非零元素数据的指针
 * @param GlobalPTR_bmp 指向全局内存中位图数据的指针
 * @param GlobalPTR_nnz 指向全局内存中非零元素索引的指针
 * @param nnz_tile0 第一个tile的非零元素数量
 * @param nnz_tile1 第二个tile的非零元素数量
 * @param startTileIdx 起始tile索引
 */
template<typename TilingConfig, typename SparseKernelConfig>

__device__ __forceinline__ void SpMM_CopyFromGlobalToReg(//uint32_t* Registers_nz,
                                                         uint32_t    Registers_nz[64],
                                                         uint64_t*    Registers_bmp,
                                                         uint32_t*    Registers_nnz,
                                                         //const uint32_t* GlobalPTR_nz,
                                                         const uint4* GlobalPTR_nz,
                                                         const uint64_t* GlobalPTR_bmp,
                                                         const uint32_t* GlobalPTR_nnz, 
                                                         uint32_t* nnz_tile0, 
                                                         uint32_t* nnz_tile1,
                                                         int startTileIdx) 
{
    // 每个位图最多64个非零元素，除以2（half精度）再除以4（uint4打包）= 8
    constexpr int MAX_NZ_PER_BMP_div_2_4 = 8; //first divide by 2 for half, then divide by 4 for uint4. : 64 / 8 = 8
   
    // Each thread handles 2 bitmaps (each of a column)
    // 每个线程处理2个位图（每个对应一列）
    #if DEBUG2
        if (blockIdx.x == 0 && blockIdx.y == 383 && threadIdx.x == 127) { //[7168, 7168, 8]  //383
            printf("------Check inside Reg load...\n");
            printf("StartTileIdx: %d\n", startTileIdx);
            printf("bmp0: %u\n", GlobalPTR_bmp[startTileIdx]);
            printf("nnz0: %u\n", GlobalPTR_nnz[startTileIdx]);
        }
    #endif
#pragma unroll     
    for (int i = 0; i < 2; i++) {
        int globalTileIdx = startTileIdx + i;
        // Load bitmap
        // 加载位图和非零元素索引
        Registers_bmp[i] = GlobalPTR_bmp[globalTileIdx];
        Registers_nnz[i] = GlobalPTR_nnz[globalTileIdx]; 

        // Load non-zero values into the register
        // 计算位图中非零元素的数量（使用popcount指令）
        uint32_t num_nz_per_bitmap = __popcll(Registers_bmp[i]);
        if (i){
            *nnz_tile1 = num_nz_per_bitmap; // 第二个tile的非零元素数量
        }
        else{
            *nnz_tile0 = num_nz_per_bitmap; // 第一个tile的非零元素数量
        }

        // Load non-zero elements (half precision) into the register
        // 加载非零元素值（half精度）到寄存器
#pragma unroll 
        for (int j = 0; j < MAX_NZ_PER_BMP_div_2_4 ; j++) { //8 iterations to copy the 4 x packed two fp16s.
            //loading Vectors 
            // 加载向量化数据（uint4包含4个uint32）
            if (j <= num_nz_per_bitmap / 8 ) {
            //if (j < num_nz_per_bitmap / 8 ) {
                //**Registers_nnz is in 'uint32' units. 
                // 从全局内存加载4个uint32值
                Registers_nz[i * 32 + j * 4 + 0] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].x; // load nz
                Registers_nz[i * 32 + j * 4 + 1] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].y; // load nz
                Registers_nz[i * 32 + j * 4 + 2] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].z; // load nz
                Registers_nz[i * 32 + j * 4 + 3] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].w; // load nz
            }
        }
    }
}

// Init Shared Memory to 0
/**
 * @brief 初始化共享内存为0
 * 
 * 该函数将共享内存中用于存储矩阵A的区域初始化为0。
 * 使用异步拷贝指令cp_async_ignore_src来高效地将内存清零。
 * 
 * @param SharedPTR 指向共享内存的指针
 */
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    // 确保TILE_M能被BLOCK_WARPS整除
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    // 假设TILE_K为64
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    // 确保每个线程的迭代次数计算正确
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        // 使用异步拷贝指令将共享内存清零（16字节对齐）
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}

/**
 * @brief 从寄存器解压稀疏数据到共享内存
 * 
 * 该函数将存储在寄存器中的压缩稀疏矩阵数据解压并写入共享内存。
 * 使用位图来确定非零元素在原始矩阵中的位置，然后将非零元素放置到正确的位置。
 * 
 * @param SharedPTR 指向共享内存的指针
 * @param Registers_nz 存储非零元素值的寄存器数组
 * @param Registers_bmp 存储位图的寄存器数组
 * @param nnz_tile0 第一个tile的非零元素数量
 * @param nnz_tile1 第二个tile的非零元素数量
 * @param TB_ROW 线程块内的行索引
 * @param TB_COL 线程块内的列索引
 */
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR,
                                                                    uint32_t Registers_nz[64],
                                                                    uint64_t* Registers_bmp,
                                                                    uint32_t* nnz_tile0, 
                                                                    uint32_t* nnz_tile1,
                                                                    int TB_ROW, 
                                                                    int TB_COL)
                                                                    //int tileIdx)
{
    //tildIdx = 2*tid = nth 64x1 tile to start with. 
//entire smem space is 256x64. 
    // 计算在共享内存中的起始位置（整个共享内存空间是256x64）
int tile_element_start = TB_ROW * 64 * 64 + TB_COL * 2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
         // Reinterpret Registers_nz as half*
        // 将寄存器中的uint32数据重新解释为half*
        half* nz_values = reinterpret_cast<half*>(Registers_nz+i*32);

        uint64_t bmp = Registers_bmp[i]; // 当前处理的位图
        int pos1 = 0;  // Initialize pos1 before processing rows // 位置计数器

        // Precompute tile positions
        // 预计算tile位置
        int fuk = tile_element_start + i;
        //int tileCol = 64 * (tileIdx + i);

        uint32_t nnz_tile = i? *nnz_tile1 : *nnz_tile0; // 获取对应tile的非零元素数量


    #pragma unroll
        for (int j = 0; j < 64; j++){
            if (j == nnz_tile){
                // 处理完所有非零元素后退出，线程变为非活跃状态
                break; //becomes inactive thread, waits for other threads to finish. 
            }
            // 找到位图中下一个设置的位（从左开始计数前导零）
            pos1 = __clzll(bmp); 
            // 清除已处理的位
            bmp &= ~(0x8000000000000000 >> pos1);
            
            // 计算在共享内存中的输出索引
            int output_idx = fuk + (pos1 << 6);  // pos1 * 64
            SharedPTR[output_idx] = nz_values[j]; // 将非零元素写入共享内存

            pos1++;
        }
    }
}


/**
 * @brief Key矩阵的稀疏矩阵乘法内核
 * 
 * 该内核执行稀疏矩阵A与密集矩阵B的乘法运算，专门用于处理Key矩阵。
 * 支持批处理、分组查询注意力(GQA)和Split-K优化。
 * 
 * 主要特点：
 * - 使用双缓冲技术进行流水线优化
 * - 支持异步内存拷贝
 * - 使用Tensor Core进行高效计算
 * - 支持批处理和GQA
 * 
 * @param A 稀疏矩阵A（未直接使用，通过压缩格式访问）
 * @param bmp 位图数组，标识稀疏矩阵中的非零元素位置
 * @param NZ 非零元素值数组
 * @param idx 索引数组，指向每个tile的非零元素起始位置
 * @param NZ_offset 每个批次的非零元素偏移量
 * @param B 密集矩阵B
 * @param Reduction_Workspace 输出结果工作空间
 * @param M_Global 矩阵A的行数
 * @param N_Global 矩阵B的列数
 * @param K_Global 矩阵A的列数/矩阵B的行数
 * @param Split_K Split-K优化的分割数
 * @param Batch_Size 批处理大小
 * @param num_key_value_groups GQA中的键值组数
 */
template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
//__maxnreg__(255)
Key_Kernel(const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K,
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{    
    // 批处理和GQA相关的索引计算
    const int mustafar_batch_id = blockIdx.z; // 当前批次ID
    const int mustafar_group_id = blockIdx.z / num_key_value_groups; // GQA组ID
    // Access batched data using offsets
    // 根据偏移量访问批处理数据
    const uint4* NZ_batch = NZ + NZ_offset[mustafar_group_id]; 
    //const uint32_t* idx_batch = idx + bmp_idx_offset[mustafar_batch_id];
    const uint32_t* idx_batch = idx + mustafar_group_id * (1 + M_Global * K_Global / 64); //because idx has 1 extra element per batch. 
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);

    // Access B and C with strides
    // 访问B矩阵和C矩阵的批处理数据
    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;
    //const half* B_batch = B + mustafar_batch_id * K_Global; //note that spmm_debug has not been updated yet. 
    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global;

    // Split-K相关的计算
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); //M_Global / TILE_M: tiling the M dimension of Matrix A.
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x; //block DimX is 1 for skinny matrices (see SpMM_API/line 42)
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);  //blockIdx.y % (num M Tile rows): wrap around num_tile_rows
        //i.e., TB0, TB(num M tile rows), TB(2*num M tile rows) .. handle the first M tile row
    //
    // K维度的分块计算
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    #if DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) { //[7168, 7168, 8]
            printf("------K dimension related Debugging info...\n");
            printf("NumKBlock: %d\n", NumKBlock); //112: how many iterations it takes to finish computing that output tile
            printf("AverageNumKBlock: %d\n", AverageNumKBlock); //16: 
            printf("RoundedKBlock: %d\n", RoundedKBlock); //112: related to the padding
            printf("PaddingKBlock: %d\n", PaddingKBlock); //0: re  lated to the padding
            printf("NumIter: %d\n", NumIter); //16: thus the final conclusion
        }
    #endif
    // 寄存器变量声明
    //the following will reside in SMSP regfile
    uint64_t Registers_bmp[2];  //4 regs // 存储位图
    uint32_t Registers_nnz[2];  //2 regs // 存储非零元素索引
    uint32_t Registers_nz[64];  //64 regs // Enough to hold non-zero values for 2 tiles // 存储非零元素值
    uint32_t nnz_tile0; // 两个tile的非零元素数量
    uint32_t nnz_tile1;

    // 共享内存
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned 

    // Warp and lane identification.
    // Warp和线程标识
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    // 计算warp在tile中的位置
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //    printf("#1 TilingConfig::WARP_COL_TENSORS: %d\n", TilingConfig::WARP_COL_TENSORS); //1 for sub-16, 2 for 32. 
    //}
    // 寄存器数组用于存储矩阵片段
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];//[8][4] = 32 uint32 
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; //[8][4] = 32 uint32
    // copying B tile from GlobalMemory to SharedMemory
    //const half* BTileGlobalPTR = //B was supposed to be col-major. 
    //    B + Tile_Start_N * K_Global
    //    + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //
    // B矩阵tile的全局内存指针
    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;

    //my definition ~ see whiteboard and paper
    // 计算稀疏矩阵的tile索引
    //int BaseTileIdx = y * (32 * K_Global / 8) + BatchID * K_Global / (8*Split_K); //For original 8x8 tile
    //int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K; //For 1-64 col tiles. 
    int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K; //For 1-64 col tiles. new ver (2/7) -> hm looks correct? 
    //below changed to allow the column-wise bitmap format. 
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;
    //int StartTileIdx = BaseTileIdx + TB_Row * K_Global / 8 + TB_Col * 2; 
    int StartTileIdx = BaseTileIdx + TB_Row * K_Global + TB_Col * 2;
    //int StartTileIdx = BaseTileIdx + tid_times_2 -2;
    //int tileIdx = 2 * tid; // for 64x1 local index for DecompressFromRegisterToShared (64x64)

    // 第一次加载稀疏数据到寄存器
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp, 
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0, 
                                                                &nnz_tile1,
                                                                StartTileIdx); 
    // 初始化共享内存并异步加载B矩阵
    SpMM_InitSharedMemory<TilingConfig>(smem); //rst_smem
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); //ld_dense: this is async, defined in MatMulUtilies.cuh
    cp_async_group_commit();
    
    // 初始化累加器矩阵C为0
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // [4*4][8 in TilingConfig] = 64 floats
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();

    // 解压稀疏数据到共享内存
    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    //SharedPTR,
                                                                    smem,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0, 
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2 
    //
    cp_async_wait_group<0>();
    __syncthreads();
     #if DEBUG
        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        if ( (threadIdx.x == 31 | threadIdx.x == 127) && blockIdx.x == 0 && blockIdx.y == 0){ //Debugging 256x64 for rectangular sanity
        //if (blockIdx.x == 0 && blockIdx.y == 0){
        //if (blockIdx.x == 127){
                printf("---Exit SpMM Decompression...\n \
                For thread %d, blockIdx.x: %d, blockIdx.y: %d, mustafar_batch_id: %d\n \
                StartTileIdx, the access index for bmp and nnz: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, mustafar_batch_id, StartTileIdx);
            }
        __syncthreads(); // only for debugging
    #endif
    //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
    StartTileIdx +=64; // 更新到下一个tile


//
// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
// 主计算循环（双缓冲流水线）
#pragma unroll(1) //unroll exactly once.
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) { //remove the last iteration and move computation to epilogue
        
        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);

        // double buffer
        // 双缓冲指针设置
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); //place for 256x64 A and 64x16 B (or TileN=32)
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        // 异步加载下一个tile的数据
        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); //rst_smem
        cp_async_group_commit();
        
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp,
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0,
                                                                &nnz_tile1, 
                                                                StartTileIdx); 

        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //ld_dense
        cp_async_group_commit();

        // 使用当前数据进行计算
        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
        //
        // 等待数据加载完成并解压到共享内存
        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    smem_write_PTR,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0,
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2
            
            //smem_write_PTR,
            //Registers_GlobalToShared,
            //NNZ_ThreadLocal1,
            //smem_write_PTR + TilingConfig::TILE_M * TILE_K / 2,
            //Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            //NNZ_ThreadLocal2); //extract 
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
        //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
        StartTileIdx += 64;

    }
    
    
    //add epliogue
    // 尾声：处理最后一个tile
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
    __syncthreads();
    //end of epliogue

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    // 将计算结果从寄存器存储到共享内存
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    //half* BlockGlobalPTR =
    //    Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
    //half* BlockGlobalPTR = C_batch + Tile_Start_M + Tile_Start_N * M_Global;
    // 将结果从共享内存写入全局内存
    half* BlockGlobalPTR =
        C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;

    #if DEBUG
        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        if ( (threadIdx.x == 31 | threadIdx.x == 127) && blockIdx.x == 0 && blockIdx.y == 0){ //Debugging 256x64 for rectangular sanity
        //if (blockIdx.x == 0 && blockIdx.y == 0){
        //if (blockIdx.x == 127){
                printf("---Exit StoreToSharedMemoryFromRegister(), Entering write to global memory...\n \
                For thread %d, blockIdx.x: %d, blockIdx.y: %d, mustafar_batch_id: %d\n \
                StartTileIdx, the access index for bmp and nnz: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, mustafar_batch_id, StartTileIdx);
            }
        __syncthreads(); // only for debugging
    #endif
    
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}

/**
 * @brief Value矩阵的稀疏矩阵乘法内核
 * 
 * 该内核执行稀疏矩阵A与密集矩阵B的乘法运算，专门用于处理Value矩阵。
 * 与Key_Kernel类似，但在索引计算和内存访问模式上有所不同，适应Value矩阵的特殊需求。
 * 
 * 主要区别：
 * - 不同的tile索引计算方式
 * - 不同的StartTileIdx更新策略
 * - 支持Split-K的结果累积
 * 
 * 参数含义与Key_Kernel相同
 */
template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
//__maxnreg__(255)
Value_Kernel(const half*  A,
                            const uint64_t* bmp, 
                            //const uint32_t* NZ,
                            const uint4* NZ, 
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


    //batched SpMV ID from Z dimension
    //const int mustafar_batch_id = blockIdx.z;
    //Logic for supporting GQA. 
        //Multiple Batches work on same compressed KV, but use different vector and output location.
    // 批处理和GQA索引计算（与Key_Kernel相同）
    const int mustafar_batch_id = blockIdx.z; //Batch num
    const int mustafar_group_id = blockIdx.z / num_key_value_groups; //GQA number
    // Access batched data using offsets
    const uint4* NZ_batch = NZ + NZ_offset[mustafar_group_id]; 
    //const uint32_t* idx_batch = idx + bmp_idx_offset[mustafar_batch_id];
    const uint32_t* idx_batch = idx + mustafar_group_id * (1 + M_Global * K_Global / 64); //because idx has 1 extra element per batch. 
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);

    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;
    // 注意：Value内核的输出包含Split_K维度
    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global * Split_K;


    // Split-K和tile计算（与Key_Kernel相同）
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); //M_Global / TILE_M: tiling the M dimension of Matrix A.
        //(M_Global / TilingConfig::TILE_M) is 1 for Value formulation. 
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x; //block DimX is 1 for skinny matrices (see SpMM_API/line 42)
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);  //blockIdx.y % (num M Tile rows): wrap around num_tile_rows
        //i.e., TB0, TB(num M tile rows), TB(2*num M tile rows) .. handle the first M tile row
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;


    //the following will reside in SMSP regfile
    // 寄存器和共享内存声明（与Key_Kernel相同）
    uint64_t Registers_bmp[2];  //4 regs
    uint32_t Registers_nnz[2];  //2 regs
    uint32_t Registers_nz[64];  //64 regs // Enough to hold non-zero values for 2 tiles 
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned 

    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //    printf("#1 TilingConfig::WARP_COL_TENSORS: %d\n", TilingConfig::WARP_COL_TENSORS); //1 for sub-16, 2 for 32. 
    //}
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];//[8][4] = 32 uint32 
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; //[8][4] = 32 uint32
    // copying B tile from GlobalMemory to SharedMemory
    //const half* BTileGlobalPTR = //B was supposed to be col-major. 
    //    B + Tile_Start_N * K_Global
    //    + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //

    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;


    // Value矩阵特有的tile索引计算
    int BaseTileIdx = BatchID * (M_Global/64) * (K_Global / Split_K); // y ==0 when M_GLOBAL = TILE_M = 128.
        //This works because: K_Global/SplitK already accounts for each 'tile' 
    
    //below changed to allow the column-wise bitmap format. 
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;


    //code for value SpMV
    // Value SpMV的特殊索引计算
    int StartTileIdx = BaseTileIdx + TB_Row * 64 + TB_Col * 2;


    // 执行与Key_Kernel相似的计算流程
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp, 
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0, 
                                                                &nnz_tile1,
                                                                StartTileIdx); 


    SpMM_InitSharedMemory<TilingConfig>(smem); //rst_smem
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); //, true); //ld_dense: this is async, defined in MatMulUtilies.cuh
    
    //CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
    //    smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, true, K_Global); //ld_dense: this is async, defined in MatMulUtilies.cuh
    cp_async_group_commit();
    // 初始化累加器
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // [4*4][8 in TilingConfig] = 64 floats
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();

    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    //SharedPTR,
                                                                    smem,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0, 
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2 
    //
    cp_async_wait_group<0>();
    __syncthreads();
    // Value矩阵特有的索引更新方式
    StartTileIdx += M_Global;



    // 主计算循环（双缓冲流水线）
#pragma unroll(1) //unroll exactly once.
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) { //remove the last iteration and move computation to epilogue
        

        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);

        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); //place for 256x64 A and 64x16 B (or TileN=32)
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); //rst_smem
        cp_async_group_commit();

        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp,
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0,
                                                                &nnz_tile1, 
                                                                StartTileIdx); 

        // Copying B Tile
        //CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        ////    smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy, K_Global);  //ld_dense
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //ld_dense
        cp_async_group_commit();


        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    smem_write_PTR,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0,
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2
            

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
        //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
         // code for key SpMV
        //StartTileIdx +=64;
        // code for value SpMV
        // Value矩阵特有的索引更新
        StartTileIdx += M_Global;

    }
    

    
    //add epliogue
    // 尾声处理
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
    __syncthreads();
    //end of epliogue

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    // 存储结果
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();

    // BatchID在这里实际上是Split-K的编号    
    half* BlockGlobalPTR =
        C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
        //BatchID is effectively the SplitK number. 
        


//int RDWS_write_offset = BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global + mustafar_batch_id * M_Global * N_Global * Split_K;

#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS){  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) { // j-th row
           BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
        }
    }
}

