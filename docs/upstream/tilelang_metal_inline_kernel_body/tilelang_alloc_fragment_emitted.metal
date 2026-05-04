#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void gemm_kernel_kernel(  device const float4* A [[ buffer(0) ]],
  device const float4* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float buf_dyn_shmem[32768];
  simdgroup_float8x8 rC[64];
  rC[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[2] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[3] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[4] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[5] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[6] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[7] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[8] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[9] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[10] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[11] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[12] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[13] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[14] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[15] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[16] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[17] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[18] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[19] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[20] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[21] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[22] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[23] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[24] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[25] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[26] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[27] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[28] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[29] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[30] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[31] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[32] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[33] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[34] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[35] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[36] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[37] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[38] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[39] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[40] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[41] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[42] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[43] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[44] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[45] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[46] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[47] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[48] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[49] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[50] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[51] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[52] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[53] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[54] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[55] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[56] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[57] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[58] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[59] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[60] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[61] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[62] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  rC[63] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  for (int i = 0; i < 8; ++i) {
    *(threadgroup float4*)(buf_dyn_shmem + ((((i * 512) + (((int)threadIdx.x) * 4)) / 4) * 4)) = A[(((((((int)blockIdx.y) * 8192) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) / 4)];
  }
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(threadgroup float4*)(buf_dyn_shmem + (((((i_1 * 512) + (((int)threadIdx.x) * 4)) + 4096) / 4) * 4)) = B[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)];
  }
  simdgroup_float8x8 A_local[4];
  simdgroup_float8x8 B_local[4];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ki = 0; ki < 8; ++ki) {
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      simdgroup_load(A_local[i_2], (&(buf_dyn_shmem[(((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_2 * 512)) + (ki * 8))])), 64, 0, (bool)0);
    }
    for (int j = 0; j < 4; ++j) {
      simdgroup_load(B_local[j], (&(buf_dyn_shmem[((((ki * 512) + ((((int)threadIdx.x) >> 6) * 32)) + (j * 8)) + 4096)])), 64, 0, (bool)0);
    }
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        simdgroup_multiply_accumulate(rC[((i_3 * 4) + j_1)], A_local[i_3], B_local[j_1], rC[((i_3 * 4) + j_1)]);
      }
    }
  }
  simdgroup_store(rC[0], (&(C[((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32))])), 128, 0, (bool)0);
  simdgroup_store(rC[1], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 8)])), 128, 0, (bool)0);
  simdgroup_store(rC[2], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 16)])), 128, 0, (bool)0);
  simdgroup_store(rC[3], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 24)])), 128, 0, (bool)0);
  simdgroup_store(rC[4], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 1024)])), 128, 0, (bool)0);
  simdgroup_store(rC[5], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 1032)])), 128, 0, (bool)0);
  simdgroup_store(rC[6], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 1040)])), 128, 0, (bool)0);
  simdgroup_store(rC[7], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 1048)])), 128, 0, (bool)0);
  simdgroup_store(rC[8], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 2048)])), 128, 0, (bool)0);
  simdgroup_store(rC[9], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 2056)])), 128, 0, (bool)0);
  simdgroup_store(rC[10], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 2064)])), 128, 0, (bool)0);
  simdgroup_store(rC[11], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 2072)])), 128, 0, (bool)0);
  simdgroup_store(rC[12], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 3072)])), 128, 0, (bool)0);
  simdgroup_store(rC[13], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 3080)])), 128, 0, (bool)0);
  simdgroup_store(rC[14], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 3088)])), 128, 0, (bool)0);
  simdgroup_store(rC[15], (&(C[(((((((int)blockIdx.y) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + 3096)])), 128, 0, (bool)0);
}



