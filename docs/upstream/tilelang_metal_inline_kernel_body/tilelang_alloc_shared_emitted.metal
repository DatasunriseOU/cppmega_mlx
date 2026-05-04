#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void add_kernel_kernel(  device const float* A [[ buffer(0) ]],
  device const float* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float buf_dyn_shmem[128];
  buf_dyn_shmem[((int)threadIdx.x)] = A[((int)threadIdx.x)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 128; ++i) {
    C[i] = (buf_dyn_shmem[i] + B[i]);
  }
}



