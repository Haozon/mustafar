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
#include "TilingConfig.h"

/**
 * 从共享内存加载数据片段到寄存器（用于矩阵A）
 * @param NumOfTensors 张量数量（WARP_ROW_TENSORS）
 * @param Registers 输出寄存器数组，每个张量4个寄存器
 * @param smem 共享内存指针
 * @param warp_start_row warp在矩阵中的起始行
 * @param k_offset K维度的偏移量
 */
template<int NumOfTensors> //WARP_ROW_TENSORS
__device__ __forceinline__ void FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                              half* __restrict__ smem,
                                                              int warp_start_row,
                                                              int k_offset)
{
    // 获取线程在warp中的ID（0-31）
    int lane_id = threadIdx.x % 32;
    // 计算线程在MMA操作中的行索引
    int i       = lane_id % MMA_M;
    // 计算线程在MMA操作中的列索引
    int j       = lane_id / MMA_M;
    
    // 计算共享内存地址：基址 + 行偏移 + 列偏移
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    // 将通用地址转换为共享内存地址
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    // 循环加载所有张量的数据
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // 使用ldmatrix指令从共享内存加载4x4的矩阵片段到寄存器
        // ldmatrix.sync.aligned.x4.m8n8: 加载8x8矩阵的4个片段
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        // 移动到下一个张量的共享内存位置
        smem_local_ptr += TILE_K * MMA_M * sizeof(half);
    }
}

/**
 * 从共享内存加载数据片段到寄存器（用于矩阵B）
 * @param NumOfTensors 张量数量
 * @param N8 N维度除以8的值（TilingConfig::WARP_COL_TENSORS）
 * @param Registers 输出寄存器数组
 * @param smem 共享内存指针
 * @param warp_start_row warp在矩阵中的起始行
 * @param k_offset K维度的偏移量
 */
template<int NumOfTensors, int N8> //TilingConfig::WARP_COL_TENSORS,
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    // 获取线程在warp中的ID
    int      lane_id             = threadIdx.x % 32;
    // 计算基础行索引
    int      i                   = lane_id % 8;
    // 创建行排列掩码，用于XOR操作实现数据重排
    uint32_t Mask_RowPermutation = i << 4;

    // 如果线程ID大于15，调整行索引
    if (lane_id > 15)
        i += 8;
    // 计算列索引：根据lane_id的位模式确定
    int j = (lane_id % 16) >= 8 ? 1 : 0;

    // 计算共享内存地址
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    // 应用行排列掩码，用于优化内存访问模式
    smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;

    // 循环加载所有张量的数据
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // 使用ldmatrix指令加载矩阵B的数据片段
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        // 移动到下一个张量的位置（按N维度步进）
        smem_local_ptr += TILE_K * MMA_N * sizeof(half);
    }
}

/**
 * 执行混合精度矩阵乘法累加操作
 * 使用Tensor Core进行16x8x16的MMA操作
 * @param c 输出累加器寄存器数组（FP32格式）
 * @param a 输入矩阵A的寄存器指针（FP16格式）
 * @param b 输入矩阵B的寄存器指针（FP16格式）
 */
__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    // 使用mma.sync指令执行矩阵乘法累加
    // m16n8k16: 16x8的输出矩阵，K维度为16
    // row.col: A矩阵行主序，B矩阵列主序
    // f32.f16.f16.f32: 输出FP32，输入A和B为FP16，累加器为FP32
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"      // 输出：4个FP32累加器
                 "{ %4, %5, %6, %7 },"     // 输入A：4个FP16寄存器
                 "{ %8, %9 },"             // 输入B：2个FP16寄存器（列主序）
                 "{ %10, %11, %12, %13 };" // 输入累加器：4个FP32寄存器
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])  // 输出约束
                 : "r"(a[0]),   // 矩阵A的寄存器
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),   // 矩阵B的寄存器（列主序布局）
                   "r"(b[1]),
                   "r"(c[0]),   // 累加器输入（与输出相同，实现累加）
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}