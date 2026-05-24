#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

__device__ inline void __tl_ptr_copy_elem(void* dst, const void* src, int bytes) {
  char* d = (char*)dst;
  const char* s = (const char*)src;
  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }
}

extern "C" __global__ void dsa_stage1_kernel(float* __restrict__ D, float* __restrict__ D1, const float* __restrict__ IndexMask, const float* __restrict__ IndexScores, const half_t* __restrict__ K, float* __restrict__ M, float* __restrict__ M1, const half_t* __restrict__ Q);
extern "C" __global__ void __launch_bounds__(256, 1) dsa_stage1_kernel(float* __restrict__ D, float* __restrict__ D1, const float* __restrict__ IndexMask, const float* __restrict__ IndexScores, const half_t* __restrict__ K, float* __restrict__ M, float* __restrict__ M1, const half_t* __restrict__ Q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[43008];
  float scores_f[64];
  float row_max_local_frag[8];
  float row_sum_local_frag[8];
  float idx_scores_f[64];
  float row_max_local_frag_1[8];
  float row_sum_local_frag_1[8];
  if (((int)threadIdx.x) < 128) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = -0x1.fffffep+127f/*-3.402823e+38*/;
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)] = -0x1.fffffep+127f/*-3.402823e+38*/;
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 256)] = 0x0p+0f/*0.000000e+00*/;
  }
  __syncthreads();
  if (0x1.2a05f2p+33f/*1.000000e+10*/ < IndexMask[0]) {
    ((float*)buf_dyn_shmem)[384] = (((float*)buf_dyn_shmem)[384] + 0x1.79ca10c924223p-67f/*1.000000e-20*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (i < 1) {
      ((uint4*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = *(uint4*)(Q + ((((((int)threadIdx.x) >> 2) * 64) + (((int)blockIdx.z) * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    } else {
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      ((uint4*)buf_dyn_shmem)[(((i * 256) + ((int)threadIdx.x)) + 128)] = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    scores_f[i_1] = 0x0p+0f/*0.000000e+00*/;
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    if ((((int)threadIdx.x) & 7) < 4) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 5120)) = ((uint4*)buf_dyn_shmem)[((((i_2 * 128) + ((((int)threadIdx.x) >> 3) * 4)) + (((int)threadIdx.x) & 7)) + 128)];
    } else {
      half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 5120)) = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    if (i_3 < 16) {
      ((half_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (i_3 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13312)] = K[(((((((int)threadIdx.x) & 127) * 64) + (((int)blockIdx.z) * 32)) + (i_3 * 2)) + (((int)threadIdx.x) >> 7))];
    } else {
      ((half_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 127) >> 6) * 4096) + (i_3 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 13312)] = half_t(0x0p+0f/*0.000000e+00*/);
    }
  }
  {
    half_t A_local[32];
    half_t B_local[16];
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 5120)])), (&(A_local[(i_4 * 8)])));
      }
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 7) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 13312)])), (&(B_local[(i_5 * 8)])));
      }
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_f + ((i_6 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_f + (((i_6 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    float condval;
    if ((((((int)threadIdx.x) & 63) < 32) && ((((((((int)threadIdx.x) >> 6) * 32) + (((i_7 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_7 >> 4) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))))) {
      condval = (scores_f[i_7] * 0x1.6a09e667f3bccp-3f/*1.767767e-01*/);
    } else {
      condval = -0x1.fffffep+127f/*-3.402823e+38*/;
    }
    scores_f[i_7] = condval;
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    row_max_local_frag[i_8] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      row_max_local_frag[i_8] = max(row_max_local_frag[i_8], scores_f[(((((i_8 >> 1) * 16) + ((rv & 3) * 4)) + ((i_8 & 1) * 2)) + (rv >> 2))]);
    }
    row_max_local_frag[i_8] = tl::AllReduce<tl::MaxOp, 256, 64, 0, tl::NamedBarrier<256>>::run(row_max_local_frag[i_8], (&(((float*)buf_dyn_shmem)[512])));
    row_max_local_frag[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(row_max_local_frag[i_8]);
  }
  __syncthreads();
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 6)) == 0) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 64) + (i_9 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)] = row_max_local_frag[i_9];
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 640)] = ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)];
    float condval_1;
    if ((max(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)], ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)]) <= -0x1.fffffep+127f/*-3.402823e+38*/)) {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_1 = max(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)], ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)]);
    }
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)] = condval_1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 32; ++i_10) {
    float2 __1;
    float2 __2;
      float2 v_ = *(float2*)(scores_f + (i_10 * 2));
      float2 v__1 = make_float2(((float*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_10 >> 3) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 384)], ((float*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_10 >> 3) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 384)]);
      *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    __1.x = __expf(__2.x);
    __1.y = __expf(__2.y);
    *(float2*)(scores_f + (i_10 * 2)) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 8; ++i_11) {
    row_sum_local_frag[i_11] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
      row_sum_local_frag[i_11] = (row_sum_local_frag[i_11] + scores_f[(((((i_11 >> 1) * 16) + ((rv_1 & 3) * 4)) + ((i_11 & 1) * 2)) + (rv_1 >> 2))]);
    }
    row_sum_local_frag[i_11] = tl::AllReduce<tl::SumOp, 256, 64, 0, tl::NamedBarrier<256>>::run(row_sum_local_frag[i_11], (&(((float*)buf_dyn_shmem)[768])));
    row_sum_local_frag[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(row_sum_local_frag[i_11]);
  }
  __syncthreads();
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 6)) == 0) {
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 64) + (i_12 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 768)] = row_sum_local_frag[i_12];
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 256)] = ((((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 256)] * __expf((((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 640)] - ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 384)]))) + ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 768)]);
  }
  if (((int)blockIdx.z) == 0) {
    #pragma unroll
    for (int i_13 = 0; i_13 < 64; ++i_13) {
      idx_scores_f[i_13] = -0x1.fffffep+127f/*-3.402823e+38*/;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      float broadcast_var_2 = -0x1.fffffep+127f/*-3.402823e+38*/;
      ulonglong4 condval_2;
      if ((i_14 < 4)) {
        condval_2 = tl::load_global_256(&(*(ulonglong4*)(IndexScores + ((i_14 * 2048) + (((int)threadIdx.x) * 8)))));
      } else {
        condval_2 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2));
      }
      *(ulonglong4*)(idx_scores_f + (i_14 * 8)) = condval_2;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 8; ++i_15) {
      row_max_local_frag_1[i_15] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
        row_max_local_frag_1[i_15] = max(row_max_local_frag_1[i_15], idx_scores_f[((i_15 * 8) + rv_2)]);
      }
      row_max_local_frag_1[i_15] = tl::AllReduce<tl::MaxOp, 16, 1, 0, tl::NamedBarrier<256>>::run(row_max_local_frag_1[i_15]);
    }
    if ((((int)threadIdx.x) % 16) == 0) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        ((float*)buf_dyn_shmem)[(((i_16 * 16) + (((int)threadIdx.x) >> 4)) + 512)] = row_max_local_frag_1[i_16];
      }
    }
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 640)] = ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)];
      float condval_3;
      if ((max(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)], ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)]) <= -0x1.fffffep+127f/*-3.402823e+38*/)) {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_3 = max(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)], ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)]);
      }
      ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = condval_3;
    }
  }
  __syncthreads();
  if (((int)blockIdx.z) == 0) {
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      for (int vec = 0; vec < 2; ++vec) {
        float4 __3;
        float4 __4;
          float4 v__2 = *(float4*)(idx_scores_f + ((i_17 * 8) + (vec * 4)));
          float4 v__3 = make_float4(((float*)buf_dyn_shmem)[(((i_17 * 16) + (((int)threadIdx.x) >> 4)) + 128)], ((float*)buf_dyn_shmem)[(((i_17 * 16) + (((int)threadIdx.x) >> 4)) + 128)], ((float*)buf_dyn_shmem)[(((i_17 * 16) + (((int)threadIdx.x) >> 4)) + 128)], ((float*)buf_dyn_shmem)[(((i_17 * 16) + (((int)threadIdx.x) >> 4)) + 128)]);
          *(float2*)(&(__4.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__4.z)) = tl::sub2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
        __3.x = __expf(__4.x);
        __3.y = __expf(__4.y);
        __3.z = __expf(__4.z);
        __3.w = __expf(__4.w);
        *(float4*)(idx_scores_f + ((i_17 * 8) + (vec * 4))) = __3;
      }
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      row_sum_local_frag_1[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        row_sum_local_frag_1[i_18] = (row_sum_local_frag_1[i_18] + idx_scores_f[((i_18 * 8) + rv_3)]);
      }
      row_sum_local_frag_1[i_18] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<256>>::run(row_sum_local_frag_1[i_18]);
    }
    if ((((int)threadIdx.x) % 16) == 0) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 8; ++i_19) {
        ((float*)buf_dyn_shmem)[(((i_19 * 16) + (((int)threadIdx.x) >> 4)) + 768)] = row_sum_local_frag_1[i_19];
      }
    }
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = ((((float*)buf_dyn_shmem)[((int)threadIdx.x)] * __expf((((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 640)] - ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)]))) + ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 768)]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    for (int i_20 = 0; i_20 < 128; ++i_20) {
      if (i_20 < 64) {
        M[((((int)blockIdx.z) * 64) + i_20)] = ((float*)buf_dyn_shmem)[(i_20 + 384)];
        D[((((int)blockIdx.z) * 64) + i_20)] = ((float*)buf_dyn_shmem)[(i_20 + 256)];
        if (((int)blockIdx.z) == 0) {
          M1[i_20] = ((float*)buf_dyn_shmem)[(i_20 + 128)];
          D1[i_20] = ((float*)buf_dyn_shmem)[i_20];
        }
      }
    }
  }
}

