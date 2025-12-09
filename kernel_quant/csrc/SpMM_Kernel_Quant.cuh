/***************************************************************************
 * Quantized Sparse Matrix Multiplication Kernel
 * Based on the original SpMM_Kernel.cuh with 2-bit quantization support
 ***************************************************************************/

#include "MatMulUtilities.cuh"
#include <vector>

#define DEBUG 0
#define DEBUG2 0
#define DEBUG1 0

/**
 * @brief 从全局内存加载量化的稀疏矩阵数据到寄存器
 * 
 * 与原版的主要区别：
 * 1. 加载 uint32 打包的量化值到寄存器中
 * 2. 加载 per-tile 的 scale 和 zero_point
 * 3. 使用 tile_offsets 表示每个tile所在的 uint32 偏移
 * 
 * 重要：使用 uint32 存储格式
 * - 每个 uint32 存储 16 个 2-bit 量化值
 * - 天然 4 字节对齐，无需额外处理
 * - 每个 tile 最多需要 4 个 uint32 (64 / 16 = 4)
 */
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg_Quant(
    uint32_t    Registers_quant[64],  // 存储量化值（uint32 数组）
    uint64_t*   Registers_bmp,
    uint32_t*   Registers_tile_offset,  // tile 的 uint32 偏移
    float*      Registers_scale,        // per-tile scale
    float*      Registers_zero,         // per-tile zero_point
    const uint32_t* GlobalPTR_quant,    // 量化值的全局指针 (uint32*)
    const uint64_t* GlobalPTR_bmp,
    const uint32_t* GlobalPTR_tile_offset,
    const float* GlobalPTR_scale,
    const float* GlobalPTR_zero,
    uint32_t* nnz_tile0, 
    uint32_t* nnz_tile1,
    int startTileIdx,
    int bit,           // 量化位宽 (2)
    int capacity       // 每 uint32 容纳的量化值数 (16)
) 
{
    constexpr int MAX_UINT32_PER_TILE = 4;  // 64 values / 16 values per uint32 = 4 uint32s max
   
#pragma unroll     
    for (int i = 0; i < 2; i++) {
        int globalTileIdx = startTileIdx + i;
        
        // 加载 bitmap
        Registers_bmp[i] = GlobalPTR_bmp[globalTileIdx];
        
        // 加载 tile offset (uint32 偏移)
        Registers_tile_offset[i] = GlobalPTR_tile_offset[globalTileIdx];
        
        // 加载 scale 和 zero_point
        Registers_scale[i] = GlobalPTR_scale[globalTileIdx];
        Registers_zero[i] = GlobalPTR_zero[globalTileIdx];
        
        // 计算非零元素数量
        uint32_t num_nz_per_bitmap = __popcll(Registers_bmp[i]);
        if (i) {
            *nnz_tile1 = num_nz_per_bitmap;
        } else {
            *nnz_tile0 = num_nz_per_bitmap;
        }
        
        // 计算该 tile 需要的 uint32 数量
        uint32_t units_needed = (num_nz_per_bitmap + capacity - 1) / capacity;
        
        // 加载量化值（uint32 数组）
        uint32_t uint32_offset = Registers_tile_offset[i];
        
#pragma unroll 
        for (int j = 0; j < MAX_UINT32_PER_TILE; j++) {
            if (j < units_needed) {
                // 直接加载 uint32，天然对齐
                Registers_quant[i * 32 + j] = GlobalPTR_quant[uint32_offset + j];
            } else {
                // 清零未使用的寄存器
                Registers_quant[i * 32 + j] = 0;
            }
        }
    }
}

// Init Shared Memory to 0 (与原版相同)
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}

/**
 * @brief 从寄存器解压并反量化稀疏数据到共享内存
 * 
 * 关键修改：
 * 1. 从 uint32 解包量化值（每个 uint32 存 16 个 2-bit 值）
 * 2. 应用反量化：value = (q - zero_point) * scale
 * 3. 转换为 float16 并写入共享内存
 */
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared_Quant(
    half* __restrict__ SharedPTR,
    uint32_t Registers_quant[64],
    uint64_t* Registers_bmp,
    float* Registers_scale,
    float* Registers_zero,
    uint32_t* nnz_tile0, 
    uint32_t* nnz_tile1,
    int TB_ROW, 
    int TB_COL,
    int bit,        // 量化位宽 (2)
    int capacity    // 每 uint32 容纳的量化值数 (16)
)
{
    int tile_element_start = TB_ROW * 64 * 64 + TB_COL * 2;
    
#pragma unroll
    for (int i = 0; i < 2; i++) {
        // 获取当前 tile 的 bitmap, scale, zero_point
        uint64_t bmp = Registers_bmp[i];
        float scale = Registers_scale[i];
        float zero_point = Registers_zero[i];
        uint32_t nnz_tile = i ? *nnz_tile1 : *nnz_tile0;
        
        // 直接使用 uint32 数组
        uint32_t* quant_units = Registers_quant + i * 32;
        
        int pos1 = 0;
        int fuk = tile_element_start + i;
        uint32_t mask = (1 << bit) - 1;  // 2-bit: mask = 0b11
        
#pragma unroll
        for (int j = 0; j < 64; j++) {
            if (j == nnz_tile) {
                break;
            }
            
            // 找到下一个非零位置
            pos1 = __clzll(bmp);
            bmp &= ~(0x8000000000000000ULL >> pos1);
            
            // 从打包的 uint32 中提取量化值
            int unit_idx = j / capacity;           // 第几个 uint32
            int bit_offset = (j % capacity) * bit; // 在 uint32 内的位偏移
            uint32_t packed_unit = quant_units[unit_idx];
            uint32_t q_value = (packed_unit >> bit_offset) & mask;
            
            // 反量化：dequant_value = (q - zero_point) * scale
            float dequant_value = (static_cast<float>(q_value) - zero_point) * scale;
            
            // 转换为 half 并写入共享内存
            int output_idx = fuk + (pos1 << 6);
            SharedPTR[output_idx] = __float2half(dequant_value);
            
            pos1++;
        }
    }
}


/**
 * @brief Key矩阵的量化稀疏矩阵乘法内核
 * 
 * 主要修改：
 * 1. 使用量化数据格式（uint8 + scale + zero_point）
 * 2. 在解压阶段进行反量化
 * 3. 其余计算流程与原版相同
 */
template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
Key_Kernel_Quant(const half*  A,
                 const uint64_t* bmp, 
                 const uint32_t* NZ_quant,     // 量化值 (uint32)
                 const uint32_t* tile_offsets, // tile uint32 偏移
                 const float* scales,          // per-tile scale
                 const float* zeros,           // per-tile zero_point
                 const half*  B,
                 half*        Reduction_Workspace,
                 const int    M_Global,
                 const int    N_Global,
                 const int    K_Global,
                 int          Split_K,
                 const int    Batch_Size, 
                 const int    num_key_value_groups,
                 int          bit,             // 量化位宽
                 int          capacity)        // 每 uint32 容纳的量化值数
{    
    const int mustafar_batch_id = blockIdx.z;
    const int mustafar_group_id = blockIdx.z / num_key_value_groups;
    
    // 访问批处理数据
    // tile_offsets 已经包含了全局 uint32 偏移信息
    const uint32_t* NZ_quant_batch = NZ_quant;
    const uint32_t* tile_offsets_batch = tile_offsets + mustafar_group_id * (M_Global * K_Global / 64);
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);
    const float* scales_batch = scales + mustafar_group_id * (M_Global * K_Global / 64);
    const float* zeros_batch = zeros + mustafar_group_id * (M_Global * K_Global / 64);
    
    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;
    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global;

    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    
    const int NumKBlock        = K_Global / TILE_K;
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;

    // 寄存器变量（量化版本）
    uint64_t Registers_bmp[2];
    uint32_t Registers_tile_offset[2];
    float    Registers_scale[2];
    float    Registers_zero[2];
    uint32_t Registers_quant[64];  // 存储打包的量化值
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];

    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    
    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;

    int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K;
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;
    int StartTileIdx = BaseTileIdx + TB_Row * K_Global + TB_Col * 2;

    // 第一次加载（量化版本）
    SpMM_CopyFromGlobalToReg_Quant<TilingConfig, SparseKernelConfig>(
        Registers_quant,
        Registers_bmp,
        Registers_tile_offset,
        Registers_scale,
        Registers_zero,
        NZ_quant_batch,
        bmp_batch,
        tile_offsets_batch,
        scales_batch,
        zeros_batch,
        &nnz_tile0, 
        &nnz_tile1,
        StartTileIdx,
        bit,
        capacity
    );
    
    SpMM_InitSharedMemory<TilingConfig>(smem);
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    
    // 初始化累加器
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    
    cp_async_wait_group<1>();
    __syncthreads();

    // 解压并反量化到共享内存
    SpMM_DecompressFromRegisterToShared_Quant<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_quant,
        Registers_bmp,
        Registers_scale,
        Registers_zero,
        &nnz_tile0, 
        &nnz_tile1,
        TB_Row, 
        TB_Col,
        bit,
        capacity
    );
    
    cp_async_wait_group<0>();
    __syncthreads();
    
    StartTileIdx += 64;

    // 主计算循环
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) {
        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + 
                                     BatchID * AverageNumKBlock * TILE_K + 
                                     ((tile_id_k + 1) * TILE_K);

        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();
        
        SpMM_CopyFromGlobalToReg_Quant<TilingConfig, SparseKernelConfig>(
            Registers_quant,
            Registers_bmp,
            Registers_tile_offset,
            Registers_scale,
            Registers_zero,
            NZ_quant_batch,
            bmp_batch,
            tile_offsets_batch,
            scales_batch,
            zeros_batch,
            &nnz_tile0,
            &nnz_tile1, 
            StartTileIdx,
            bit,
            capacity
        );

        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
        
        cp_async_wait_group<1>();
        __syncthreads();
        
        SpMM_DecompressFromRegisterToShared_Quant<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_quant,
            Registers_bmp,
            Registers_scale,
            Registers_zero,
            &nnz_tile0,
            &nnz_tile1,
            TB_Row, 
            TB_Col,
            bit,
            capacity
        );
            
        cp_async_wait_group<0>();
        __syncthreads();
        StartTileIdx += 64;
    }
    
    // 尾声
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
    __syncthreads();

    // 存储结果
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    
    half* BlockGlobalPTR = C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;

#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}

/**
 * @brief Value矩阵的量化稀疏矩阵乘法内核
 * 
 * 与 Key_Kernel_Quant 类似，但使用 Value 矩阵的索引计算方式
 */
template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
Value_Kernel_Quant(const half*  A,
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
    const int mustafar_batch_id = blockIdx.z;
    const int mustafar_group_id = blockIdx.z / num_key_value_groups;
    
    const uint32_t* NZ_quant_batch = NZ_quant;
    const uint32_t* tile_offsets_batch = tile_offsets + mustafar_group_id * (M_Global * K_Global / 64);
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);
    const float* scales_batch = scales + mustafar_group_id * (M_Global * K_Global / 64);
    const float* zeros_batch = zeros + mustafar_group_id * (M_Global * K_Global / 64);

    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;
    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global * Split_K;

    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    
    const int NumKBlock        = K_Global / TILE_K;
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;

    uint64_t Registers_bmp[2];
    uint32_t Registers_tile_offset[2];
    float    Registers_scale[2];
    float    Registers_zero[2];
    uint32_t Registers_quant[64];
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];

    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    
    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;

    // Value 矩阵特有的索引计算
    int BaseTileIdx = BatchID * (M_Global/64) * (K_Global / Split_K);
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;
    int StartTileIdx = BaseTileIdx + TB_Row * 64 + TB_Col * 2;

    SpMM_CopyFromGlobalToReg_Quant<TilingConfig, SparseKernelConfig>(
        Registers_quant,
        Registers_bmp,
        Registers_tile_offset,
        Registers_scale,
        Registers_zero,
        NZ_quant_batch,
        bmp_batch,
        tile_offsets_batch,
        scales_batch,
        zeros_batch,
        &nnz_tile0, 
        &nnz_tile1,
        StartTileIdx,
        bit,
        capacity
    );

    SpMM_InitSharedMemory<TilingConfig>(smem);
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    
    cp_async_wait_group<1>();
    __syncthreads();

    SpMM_DecompressFromRegisterToShared_Quant<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_quant,
        Registers_bmp,
        Registers_scale,
        Registers_zero,
        &nnz_tile0, 
        &nnz_tile1,
        TB_Row, 
        TB_Col,
        bit,
        capacity
    );
    
    cp_async_wait_group<0>();
    __syncthreads();
    
    // Value 矩阵特有的索引更新
    StartTileIdx += M_Global;

#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) {
        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + 
                                     BatchID * AverageNumKBlock * TILE_K + 
                                     ((tile_id_k + 1) * TILE_K);

        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();

        SpMM_CopyFromGlobalToReg_Quant<TilingConfig, SparseKernelConfig>(
            Registers_quant,
            Registers_bmp,
            Registers_tile_offset,
            Registers_scale,
            Registers_zero,
            NZ_quant_batch,
            bmp_batch,
            tile_offsets_batch,
            scales_batch,
            zeros_batch,
            &nnz_tile0,
            &nnz_tile1, 
            StartTileIdx,
            bit,
            capacity
        );

        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);

        cp_async_wait_group<1>();
        __syncthreads();
        
        SpMM_DecompressFromRegisterToShared_Quant<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_quant,
            Registers_bmp,
            Registers_scale,
            Registers_zero,
            &nnz_tile0,
            &nnz_tile1,
            TB_Row, 
            TB_Col,
            bit,
            capacity
        );

        cp_async_wait_group<0>();
        __syncthreads();
        StartTileIdx += M_Global;
    }
    
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
    __syncthreads();

    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();

    half* BlockGlobalPTR = C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;

#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS) {
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) {
           BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
        }
    }
}
