// Function: main_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

static inline void __tl_ptr_copy_elem(device void* dst, device const void* src, int bytes) {
  device char* d = (device char*)dst;
  device const char* s = (device const char*)src;
  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }
}
static inline void __tl_ptr_copy_elem(threadgroup void* dst, device const void* src, int bytes) {
  threadgroup char* d = (threadgroup char*)dst;
  device const char* s = (device const char*)src;
  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }
}
static inline void __tl_ptr_copy_elem(device void* dst, threadgroup const void* src, int bytes) {
  device char* d = (device char*)dst;
  threadgroup const char* s = (threadgroup const char*)src;
  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }
}
static inline void __tl_ptr_copy_elem(threadgroup void* dst, threadgroup const void* src, int bytes) {
  threadgroup char* d = (threadgroup char*)dst;
  threadgroup const char* s = (threadgroup const char*)src;
  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }
}

namespace tl {
struct SumOp {
  template <typename T> inline T operator()(T x, T y) const { return x + y; }
};
struct MulOp {
  template <typename T> inline T operator()(T x, T y) const { return x * y; }
};
struct MaxOp {
  template <typename T> inline T operator()(T x, T y) const { return y < x ? x : y; }
};
struct MinOp {
  template <typename T> inline T operator()(T x, T y) const { return y > x ? x : y; }
};
struct BitAndOp {
  template <typename T> inline T operator()(T x, T y) const { return x & y; }
};
struct BitOrOp {
  template <typename T> inline T operator()(T x, T y) const { return x | y; }
};
struct BitXorOp {
  template <typename T> inline T operator()(T x, T y) const { return x ^ y; }
};
template <typename T, int rows_per_threadgroup, int cols>
struct RowReduceSumContiguousInnermost {
  static_assert(rows_per_threadgroup > 0,
                "rows_per_threadgroup must be positive");
  static_assert(cols > 0, "cols must be positive");
  enum { simdgroup_size = 32 };
  static inline void run(device const T* A, device T* B, uint block_id,
                         uint tid, uint rows) {
    const uint row_in_group = tid / uint(simdgroup_size);
    const uint lane = tid & uint(simdgroup_size - 1);
    if (row_in_group >= uint(rows_per_threadgroup)) {
      return;
    }
    const uint row = block_id * uint(rows_per_threadgroup) + row_in_group;
    if (row >= rows) {
      return;
    }
    T acc = T(0);
    for (uint col = lane; col < uint(cols); col += uint(simdgroup_size)) {
      acc += A[row * uint(cols) + col];
    }
    T total = simd_sum(acc);
    if (lane == 0) {
      B[row] = total;
    }
  }
};
struct SyncThreadsBarrier {
  template <int phase = 0> static inline void sync() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
};
template <int all_threads> struct NamedBarrier {
  template <int phase = 0> static inline void sync() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
};
template <class Reducer, int threads, int scale, int thread_offset,
          class Barrier, int batch_size, int workspace_stride>
struct AllReduce;
template <class Reducer>
struct SimdgroupIntraReduce {
  template <typename T>
  static inline T run(T x) {
    x = Reducer()(x, simd_shuffle_xor(x, uint(16)));
    x = Reducer()(x, simd_shuffle_xor(x, uint(8)));
    x = Reducer()(x, simd_shuffle_xor(x, uint(4)));
    x = Reducer()(x, simd_shuffle_xor(x, uint(2)));
    x = Reducer()(x, simd_shuffle_xor(x, uint(1)));
    return x;
  }
};
template <>
struct SimdgroupIntraReduce<SumOp> {
  template <typename T>
  static inline T run(T x) {
    return simd_sum(x);
  }
};
template <class Reducer, int threads, int thread_offset,
          class Barrier, int batch_size, int workspace_stride>
struct AllReduceSimdgroupCross {
  enum { simdgroup_size = 32 };
  enum { simdgroup_count = threads / simdgroup_size };
  enum { final_slot = simdgroup_count };
  template <typename T>
  static inline T reduce_simdgroup(T x) {
    return SimdgroupIntraReduce<Reducer>::run(x);
  }
  template <typename T>
  static inline T reduce_partials(T x, uint lane) {
    if (lane < uint(simdgroup_count)) {
      if (simdgroup_count >= 32) {
        x = Reducer()(x, simd_shuffle_xor(x, uint(16)));
      }
      if (simdgroup_count >= 16) {
        x = Reducer()(x, simd_shuffle_xor(x, uint(8)));
      }
      if (simdgroup_count >= 8) {
        x = Reducer()(x, simd_shuffle_xor(x, uint(4)));
      }
      if (simdgroup_count >= 4) {
        x = Reducer()(x, simd_shuffle_xor(x, uint(2)));
      }
      if (simdgroup_count >= 2) {
        x = Reducer()(x, simd_shuffle_xor(x, uint(1)));
      }
    }
    return x;
  }
  template <typename T>
  static inline T run(T x, uint tid, threadgroup T* red_buf) {
    const int local_tid = int(tid) - thread_offset;
    const uint lane = uint(local_tid & (simdgroup_size - 1));
    const uint simdgroup_id = uint(local_tid >> 5);
    x = reduce_simdgroup(x);
    if (lane == 0) {
      red_buf[simdgroup_id] = x;
    }
    Barrier::template sync<1>();
    T result = red_buf[0];
    if (simdgroup_id == 0) {
      result = lane < uint(simdgroup_count) ? red_buf[lane] : red_buf[0];
      result = reduce_partials(result, lane);
      if (lane == 0) {
        red_buf[final_slot] = result;
      }
    }
    Barrier::template sync<2>();
    return red_buf[final_slot];
  }
  template <typename T>
  static inline void run_batch(thread T* x, uint tid,
                               threadgroup T* red_buf) {
    const int local_tid = int(tid) - thread_offset;
    const uint lane = uint(local_tid & (simdgroup_size - 1));
    const uint simdgroup_id = uint(local_tid >> 5);
    for (int i = 0; i < batch_size; ++i) {
      T partial = reduce_simdgroup(x[i]);
      x[i] = partial;
    }
    for (int i = 0; i < batch_size; ++i) {
      if (lane == 0) {
        const int batch_offset = i * workspace_stride;
        red_buf[simdgroup_id + batch_offset] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + batch_offset]
                       : red_buf[batch_offset];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + batch_offset] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      const int batch_offset = i * workspace_stride;
      x[i] = red_buf[final_slot + batch_offset];
    }
  }
  template <typename T>
  static inline void run_batch(threadgroup T* x, uint tid,
                               threadgroup T* red_buf) {
    const int local_tid = int(tid) - thread_offset;
    const uint lane = uint(local_tid & (simdgroup_size - 1));
    const uint simdgroup_id = uint(local_tid >> 5);
    for (int i = 0; i < batch_size; ++i) {
      T partial = reduce_simdgroup(x[i]);
      x[i] = partial;
    }
    for (int i = 0; i < batch_size; ++i) {
      if (lane == 0) {
        const int batch_offset = i * workspace_stride;
        red_buf[simdgroup_id + batch_offset] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + batch_offset]
                       : red_buf[batch_offset];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + batch_offset] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      const int batch_offset = i * workspace_stride;
      x[i] = red_buf[final_slot + batch_offset];
    }
  }
};
template <class Reducer, int threads, int scale, int thread_offset,
          class Barrier, int batch_size, int workspace_stride,
          bool done>
struct AllReduceStep;
template <class Reducer, int threads, int scale, int thread_offset,
          class Barrier, int batch_size, int workspace_stride>
struct AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,
                     batch_size, workspace_stride, true> {
  template <typename T>
  static inline T run(T x, uint tid, threadgroup T* red_buf = nullptr) {
    return x;
  }
  template <typename T>
  static inline void run_batch(thread T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {}
  template <typename T>
  static inline void run_batch(threadgroup T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {}
};
template <class Reducer, int threads, int scale, int thread_offset,
          class Barrier, int batch_size, int workspace_stride>
struct AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,
                     batch_size, workspace_stride, false> {
  enum { offset = threads / 2 };
  template <typename T>
  static inline T run(T x, uint tid, threadgroup T* red_buf = nullptr) {
    const int local_tid = int(tid) - thread_offset;
    if (offset >= 32) {
      Barrier::template sync<1>();
      red_buf[local_tid] = x;
      Barrier::template sync<2>();
      x = Reducer()(x, red_buf[local_tid ^ offset]);
    } else {
      x = Reducer()(x, simd_shuffle_xor(x, uint(offset)));
    }
    if (offset == scale) {
      return x;
    }
    return AllReduce<Reducer, offset, scale, thread_offset, Barrier,
                     batch_size, workspace_stride>::run(x, tid, red_buf);
  }
  template <typename T>
  static inline void run_batch(thread T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {
    const int local_tid = int(tid) - thread_offset;
    if (offset >= 32) {
      Barrier::template sync<1>();
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        red_buf[local_tid + batch_offset] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) + batch_offset]);
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], simd_shuffle_xor(x[i], uint(offset)));
      }
    }
    if (offset != scale) {
      AllReduce<Reducer, offset, scale, thread_offset, Barrier,
                batch_size, workspace_stride>::run_batch(x, tid, red_buf);
    }
  }
  template <typename T>
  static inline void run_batch(threadgroup T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {
    const int local_tid = int(tid) - thread_offset;
    if (offset >= 32) {
      Barrier::template sync<1>();
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        red_buf[local_tid + batch_offset] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        const int batch_offset = i * workspace_stride;
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) + batch_offset]);
      }
    } else {
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], simd_shuffle_xor(x[i], uint(offset)));
      }
    }
    if (offset != scale) {
      AllReduce<Reducer, offset, scale, thread_offset, Barrier,
                batch_size, workspace_stride>::run_batch(x, tid, red_buf);
    }
  }
};
template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads % scale == 0,
                "tl::AllReduce<>: threads must be divisible by scale");
  static_assert((threads & (threads - 1)) == 0,
                "tl::AllReduce<>: threads must be a power of two");
  template <typename T>
  static inline T run(T x, uint tid, threadgroup T* red_buf = nullptr) {
    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&
        workspace_stride >= threads) {
      return AllReduceSimdgroupCross<Reducer, threads, thread_offset,
          Barrier, batch_size, workspace_stride>::run(x, tid, red_buf);
    }
    return AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,
                         batch_size, workspace_stride,
                         (threads == scale)>::run(x, tid, red_buf);
  }
  template <typename T>
  static inline void run_batch(thread T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {
    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&
        workspace_stride >= threads) {
      AllReduceSimdgroupCross<Reducer, threads, thread_offset,
          Barrier, batch_size, workspace_stride>::run_batch(x, tid, red_buf);
      return;
    }
    AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,
                  batch_size, workspace_stride,
                  (threads == scale)>::run_batch(x, tid, red_buf);
  }
  template <typename T>
  static inline void run_batch(threadgroup T* x, uint tid,
                               threadgroup T* red_buf = nullptr) {
    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&
        workspace_stride >= threads) {
      AllReduceSimdgroupCross<Reducer, threads, thread_offset,
          Barrier, batch_size, workspace_stride>::run_batch(x, tid, red_buf);
      return;
    }
    AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,
                  batch_size, workspace_stride,
                  (threads == scale)>::run_batch(x, tid, red_buf);
  }
};
} /* namespace tl */

struct tvm_bfloat16 {
  ushort bits;
  tvm_bfloat16() = default;
  tvm_bfloat16(float value) {
    uint raw = as_type<uint>(value);
    uint lsb = (raw >> 16) & 1u;
    bits = ushort((raw + 0x7fffu + lsb) >> 16);
  }
  operator float() const {
    return as_type<float>(uint(bits) << 16);
  }
};
static inline float __tvm_bfloat16_to_float(thread const tvm_bfloat16& value) {
  return as_type<float>(uint(value.bits) << 16);
}
static inline float __tvm_bfloat16_to_float(device const tvm_bfloat16& value) {
  return as_type<float>(uint(value.bits) << 16);
}
static inline float __tvm_bfloat16_to_float(threadgroup const tvm_bfloat16& value) {
  return as_type<float>(uint(value.bits) << 16);
}
static inline float __tvm_bfloat16_to_float(constant const tvm_bfloat16& value) {
  return as_type<float>(uint(value.bits) << 16);
}

namespace tl {
static inline float AtomicAdd(device float* address, float val,
                              int memory_order = 0) {
  (void)memory_order;
  device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(address);
  uint old_bits = atomic_load_explicit(bits, memory_order_relaxed);
  while (true) {
    float old_val = as_type<float>(old_bits);
    uint new_bits = as_type<uint>(old_val + val);
    uint expected = old_bits;
    if (atomic_compare_exchange_weak_explicit(
            bits, &expected, new_bits, memory_order_relaxed,
            memory_order_relaxed)) {
      return old_val;
    }
    old_bits = expected;
  }
}
static inline int AtomicAdd(device int* address, int val,
                            int memory_order = 0) {
  (void)memory_order;
  return atomic_fetch_add_explicit(
      reinterpret_cast<device atomic_int*>(address), val,
      memory_order_relaxed);
}
static inline uint AtomicAdd(device uint* address, uint val,
                             int memory_order = 0) {
  (void)memory_order;
  return atomic_fetch_add_explicit(
      reinterpret_cast<device atomic_uint*>(address), val,
      memory_order_relaxed);
}
static inline float AtomicAdd(threadgroup float* address, float val,
                              int memory_order = 0) {
  (void)memory_order;
  threadgroup atomic_uint* bits =
      reinterpret_cast<threadgroup atomic_uint*>(address);
  uint old_bits = atomic_load_explicit(bits, memory_order_relaxed);
  while (true) {
    float old_val = as_type<float>(old_bits);
    uint new_bits = as_type<uint>(old_val + val);
    uint expected = old_bits;
    if (atomic_compare_exchange_weak_explicit(
            bits, &expected, new_bits, memory_order_relaxed,
            memory_order_relaxed)) {
      return old_val;
    }
    old_bits = expected;
  }
}
static inline int AtomicAdd(threadgroup int* address, int val,
                            int memory_order = 0) {
  (void)memory_order;
  return atomic_fetch_add_explicit(
      reinterpret_cast<threadgroup atomic_int*>(address), val,
      memory_order_relaxed);
}
static inline uint AtomicAdd(threadgroup uint* address, uint val,
                             int memory_order = 0) {
  (void)memory_order;
  return atomic_fetch_add_explicit(
      reinterpret_cast<threadgroup atomic_uint*>(address), val,
      memory_order_relaxed);
}
static inline ushort __tl_bf16_round(float value) {
  uint raw = as_type<uint>(value);
  uint lsb = (raw >> 16) & 1u;
  return ushort((raw + 0x7fffu + lsb) >> 16);
}
static inline float AtomicAdd(device tvm_bfloat16* address, float val,
                              int memory_order = 0) {
  (void)memory_order;
  uintptr_t addr = reinterpret_cast<uintptr_t>(address);
  uint shift = uint(addr & 2u) * 8u;  // 0 for low half, 16 for high
  device atomic_uint* word = reinterpret_cast<device atomic_uint*>(
      addr & ~uintptr_t(3u));
  uint old_word = atomic_load_explicit(word, memory_order_relaxed);
  while (true) {
    ushort old_bits = ushort((old_word >> shift) & 0xffffu);
    float old_val = as_type<float>(uint(old_bits) << 16);
    ushort new_bits = __tl_bf16_round(old_val + val);
    uint new_word = (old_word & ~(0xffffu << shift)) |
                    (uint(new_bits) << shift);
    uint expected = old_word;
    if (atomic_compare_exchange_weak_explicit(
            word, &expected, new_word, memory_order_relaxed,
            memory_order_relaxed)) {
      return old_val;
    }
    old_word = expected;
  }
}
static inline float AtomicAdd(threadgroup tvm_bfloat16* address,
                              float val, int memory_order = 0) {
  (void)memory_order;
  uintptr_t addr = reinterpret_cast<uintptr_t>(address);
  uint shift = uint(addr & 2u) * 8u;
  threadgroup atomic_uint* word =
      reinterpret_cast<threadgroup atomic_uint*>(addr & ~uintptr_t(3u));
  uint old_word = atomic_load_explicit(word, memory_order_relaxed);
  while (true) {
    ushort old_bits = ushort((old_word >> shift) & 0xffffu);
    float old_val = as_type<float>(uint(old_bits) << 16);
    ushort new_bits = __tl_bf16_round(old_val + val);
    uint new_word = (old_word & ~(0xffffu << shift)) |
                    (uint(new_bits) << shift);
    uint expected = old_word;
    if (atomic_compare_exchange_weak_explicit(
            word, &expected, new_word, memory_order_relaxed,
            memory_order_relaxed)) {
      return old_val;
    }
    old_word = expected;
  }
}
} /* namespace tl */

kernel void main_kernel(  device half* B [[ buffer(0) ]],
  device half* C [[ buffer(1) ]],
  device half* D [[ buffer(2) ]],
  device half* cb [[ buffer(3) ]],
  device half* dA_cumsum [[ buffer(4) ]],
  device float* dA_cumsum_y [[ buffer(5) ]],
  device float* dC [[ buffer(6) ]],
  device float* dD [[ buffer(7) ]],
  device float* dchunk_states [[ buffer(8) ]],
  device float* dinp [[ buffer(9) ]],
  device half* dout [[ buffer(10) ]],
  device half* dt [[ buffer(11) ]],
  device float* dx [[ buffer(12) ]],
  device float* dz [[ buffer(13) ]],
  device float* prev_states [[ buffer(14) ]],
  device half* x [[ buffer(15) ]],
  device half* y [[ buffer(16) ]],
  device half* z [[ buffer(17) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup uchar buf_dyn_shmem[29184];
  thread float dD_local[1];
  thread float acc[1];
  thread half4 C_local_cast_1[1];
  thread float4 dacs_local_cast_2[1];
  thread half4 opBh_local_cast[1];
  simdgroup_float8x8 dchunk_frag[32];
  thread float dsd[1];
  thread float accn[1];
  thread float cdiag[1];
  thread float acc_1[1];
  int cse_v26 = ((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 64));
  if (((int)threadIdx.x) < 64) {
    ((threadgroup float*)buf_dyn_shmem)[((int)threadIdx.x)] = ((float)dA_cumsum[(cse_v26 + ((int)threadIdx.x))]);
    ((threadgroup float*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)] = 0.000000e+00f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  dD_local[0] = 0.000000e+00f;
  int cse_v2 = (((int)blockIdx.x) * 524288);
  int cse_v21 = (((int)blockIdx.y) * 64);
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float z_v = ((float)z[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))]);
    float cse_v11 = (z_v * -1.000000e+00f);
    float cse_v4 = (1.000000e+00f + exp(cse_v11));
    float cse_v9 = (z_v / cse_v4);
    float y_v = ((float)y[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))]);
    float dout_v = ((float)dout[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))]);
    float dgate = (dout_v * y_v);
    float cse_v13 = (dout_v * cse_v9);
    float cse_v12 = (1.000000e+00f / cse_v4);
    dz[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))] = (dgate * (cse_v12 * (1.000000e+00f + (z_v * (1.000000e+00f - cse_v12)))));
    float d_v = ((float)D[((int)blockIdx.y)]);
    float x_v = ((float)x[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))]);
    dx[((((cse_v2 + (i * 16384)) + ((((int)threadIdx.x) / 64) * 8192)) + cse_v21) + (((int)threadIdx.x) % 64))] = (d_v * cse_v13);
    dD_local[0] = (dD_local[0] + (cse_v13 * x_v));
    ((threadgroup half*)buf_dyn_shmem)[(((i * 128) + ((int)threadIdx.x)) + 256)] = ((half)cse_v13);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tl::AtomicAdd((&(dD[((int)blockIdx.y)])), dD_local[0]);
  int cse_v4_1 = (((int)blockIdx.x) * 33554432);
  int cse_v20 = (((int)blockIdx.y) * 4096);
  for (int _tmp = 0; _tmp < 2048; ++_tmp) {
    dinp[((((cse_v4_1 + ((_tmp >> 5) * 524288)) + cse_v20) + ((_tmp & 31) * 128)) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int _tmp_1 = 0; _tmp_1 < 32; ++_tmp_1) {
    acc[0] = 0.000000e+00f;
    int cse_v12_1 = (_tmp_1 * 128);
    for (int pp = 0; pp < 64; ++pp) {
      acc[0] = (acc[0] + (((float)((threadgroup half*)buf_dyn_shmem)[(((cse_v12_1 + ((((int)threadIdx.x) / 64) * 64)) + pp) + 256)]) * ((float)x[(((cse_v2 + ((((int)threadIdx.x) % 64) * 8192)) + cse_v21) + pp)])));
    }
    ((threadgroup half*)buf_dyn_shmem)[((cse_v12_1 + ((int)threadIdx.x)) + 4352)] = ((half)acc[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int cse_v3 = (((int)blockIdx.x) * 32768);
  int cse_v11_1 = (((int)blockIdx.y) >> 4);
  int cse_v23 = (cse_v11_1 * 64);
  int cse_v29 = (cse_v2 + cse_v20);
  for (int _tmp_2 = 0; _tmp_2 < 2; ++_tmp_2) {
    int cse_v19 = (_tmp_2 * 32);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      int4 v__1 = int4(((((((cse_v3 + (i_1 * 8192)) + ((((int)threadIdx.x) / 8) * 512)) + cse_v23) + cse_v19) + ((((int)threadIdx.x) % 8) * 4)))+(1*0), ((((((cse_v3 + (i_1 * 8192)) + ((((int)threadIdx.x) / 8) * 512)) + cse_v23) + cse_v19) + ((((int)threadIdx.x) % 8) * 4)))+(1*1), ((((((cse_v3 + (i_1 * 8192)) + ((((int)threadIdx.x) / 8) * 512)) + cse_v23) + cse_v19) + ((((int)threadIdx.x) % 8) * 4)))+(1*2), ((((((cse_v3 + (i_1 * 8192)) + ((((int)threadIdx.x) / 8) * 512)) + cse_v23) + cse_v19) + ((((int)threadIdx.x) % 8) * 4)))+(1*3));
      C_local_cast_1[0] = (half4(C[v__1[0]],C[v__1[1]],C[v__1[2]],C[v__1[3]]));
      dacs_local_cast_2[0] = float4(((threadgroup float*)buf_dyn_shmem)[((i_1 * 16) + (((int)threadIdx.x) / 8))], ((threadgroup float*)buf_dyn_shmem)[((i_1 * 16) + (((int)threadIdx.x) / 8))], ((threadgroup float*)buf_dyn_shmem)[((i_1 * 16) + (((int)threadIdx.x) / 8))], ((threadgroup float*)buf_dyn_shmem)[((i_1 * 16) + (((int)threadIdx.x) / 8))]);
      float broadcast_var = 1.442695e+00f;
      opBh_local_cast[0] = ((half4)(((float4)C_local_cast_1[0]) * exp2((dacs_local_cast_2[0] * float4(1.442695e+00f, 1.442695e+00f, 1.442695e+00f, 1.442695e+00f)))));
      *(threadgroup half4*)(((threadgroup half*)buf_dyn_shmem) + (((i_1 * 512) + (((int)threadIdx.x) * 4)) + 8448)) = opBh_local_cast[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dchunk_frag[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[2] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[3] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[4] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[5] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[6] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[7] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[8] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[9] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[10] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[11] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[12] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[13] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[14] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[15] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[16] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[17] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[18] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[19] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[20] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[21] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[22] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[23] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[24] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[25] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[26] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[27] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[28] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[29] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[30] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    dchunk_frag[31] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    simdgroup_half8x8 A_local[4];
    simdgroup_half8x8 B_local[2];
    for (int ki = 0; ki < 8; ++ki) {
      long cse_v5 = ((long)ki);
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        simdgroup_load(A_local[i_2], (&(((threadgroup half*)buf_dyn_shmem)[((((cse_v5 * (long)512) + (((((int)threadIdx.x) & (long)63) >> (long)5) * (long)32)) + ((long)(i_2 * 8))) + (long)256)])), 64, 0, (bool)1);
      }
      for (int j = 0; j < 2; ++j) {
        simdgroup_load(B_local[j], (&(((threadgroup half*)buf_dyn_shmem)[((((cse_v5 * (long)256) + ((((int)threadIdx.x) >> (long)6) * (long)16)) + ((long)(j * 8))) + (long)8448)])), 32, 0, (bool)0);
      }
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          int cse_v27 = ((i_3 * 2) + j_1);
          simdgroup_multiply_accumulate(dchunk_frag[cse_v27], A_local[i_3], B_local[j_1], dchunk_frag[cse_v27]);
        }
      }
    }
    simdgroup_store(dchunk_frag[0], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4224)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[1], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4232)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[2], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4480)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[3], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4488)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[4], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4736)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[5], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4744)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[6], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 4992)])), 32, 0, (bool)0);
    simdgroup_store(dchunk_frag[7], (&(((threadgroup float*)buf_dyn_shmem)[(((((((int)threadIdx.x) % 64) / 32) * 1024) + ((((int)threadIdx.x) / 64) * 16)) + 5000)])), 32, 0, (bool)0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int simd_cast_i = 0; simd_cast_i < 16; ++simd_cast_i) {
      ((threadgroup half*)buf_dyn_shmem)[(((simd_cast_i * 128) + ((int)threadIdx.x)) + 12544)] = ((half)((threadgroup float*)buf_dyn_shmem)[(((simd_cast_i * 128) + ((int)threadIdx.x)) + 4224)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      C_local_cast_1[0] = *(threadgroup half4*)(((threadgroup half*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 12544));
      dacs_local_cast_2[0] = ((float4)C_local_cast_1[0]);
      *(device float4*)(dchunk_states + ((((cse_v29 + (i_4 * 1024)) + ((((int)threadIdx.x) / 8) * 64)) + cse_v19) + ((((int)threadIdx.x) % 8) * 4))) = dacs_local_cast_2[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int ll = 0; ll < 64; ++ll) {
    float sd = exp2((((threadgroup float*)buf_dyn_shmem)[ll] * 1.442695e+00f));
    dsd[0] = 0.000000e+00f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (((int)threadIdx.x) == 0) {
      for (int nn = 0; nn < 64; ++nn) {
        accn[0] = 0.000000e+00f;
        int cse_v10 = (ll * 64);
        for (int pp_1 = 0; pp_1 < 64; ++pp_1) {
          float cs = prev_states[((cse_v29 + (pp_1 * 64)) + nn)];
          accn[0] = (accn[0] + (((float)((threadgroup half*)buf_dyn_shmem)[((cse_v10 + pp_1) + 256)]) * cs));
        }
        float c_v = ((float)C[(((cse_v3 + (ll * 512)) + cse_v23) + nn)]);
        cdiag[0] = 0.000000e+00f;
        for (int ss = 0; ss < (ll + 1); ++ss) {
          float lmat = exp2(((((threadgroup float*)buf_dyn_shmem)[ll] - ((threadgroup float*)buf_dyn_shmem)[ss]) * 1.442695e+00f));
          float dt_s = ((float)dt[(cse_v26 + ss)]);
          float b_v = ((float)B[(((cse_v3 + (ss * 512)) + cse_v23) + nn)]);
          cdiag[0] = (cdiag[0] + (((lmat * dt_s) * ((float)((threadgroup half*)buf_dyn_shmem)[((cse_v10 + ss) + 4352)])) * b_v));
        }
        dC[(((cse_v2 + (ll * 8192)) + cse_v21) + nn)] = ((accn[0] * sd) + cdiag[0]);
        dsd[0] = (dsd[0] + (accn[0] * c_v));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (((int)threadIdx.x) == 0) {
      int cse_v1 = (ll + 64);
      ((threadgroup float*)buf_dyn_shmem)[cse_v1] = (((threadgroup float*)buf_dyn_shmem)[cse_v1] + (dsd[0] * sd));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int _tmp_3 = 0; _tmp_3 < 32; ++_tmp_3) {
    for (int nn_1 = 0; nn_1 < 64; ++nn_1) {
      acc_1[0] = 0.000000e+00f;
      int cse_v6 = (_tmp_3 * 2);
      for (int ll_1 = (cse_v6 + (((int)threadIdx.x) >> 6)); ll_1 < 64; ++ll_1) {
        float lmat_1 = exp2(((((threadgroup float*)buf_dyn_shmem)[ll_1] - ((threadgroup float*)buf_dyn_shmem)[(cse_v6 + (((int)threadIdx.x) / 64))]) * 1.442695e+00f));
        half condval;
        if ((((ll_1 >> 6) + ((int)blockIdx.x)) < 8)) {
          condval = C[(((cse_v3 + (ll_1 * 512)) + cse_v23) + nn_1)];
        } else {
          condval = 0.000000e+00h;
        }
        float c_v_1 = ((float)condval);
        acc_1[0] = (acc_1[0] + ((((float)((threadgroup half*)buf_dyn_shmem)[(((ll_1 * 64) + (((int)threadIdx.x) % 64)) + 256)]) * c_v_1) * lmat_1));
      }
      dinp[(((((cse_v4_1 + (_tmp_3 * 1048576)) + ((((int)threadIdx.x) / 64) * 524288)) + cse_v20) + ((((int)threadIdx.x) % 64) * 64)) + nn_1)] = (dinp[(((((cse_v4_1 + (_tmp_3 * 1048576)) + ((((int)threadIdx.x) / 64) * 524288)) + cse_v20) + ((((int)threadIdx.x) % 64) * 64)) + nn_1)] + acc_1[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    int cse_v15 = (i_5 * 128);
    float cb_v = ((float)cb[(((cse_v3 + (cse_v11_1 * 4096)) + cse_v15) + ((int)threadIdx.x))]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int cse_v7 = (i_5 * 2);
    float lmat_2 = exp2(((((threadgroup float*)buf_dyn_shmem)[(cse_v7 + (((int)threadIdx.x) / 64))] - ((threadgroup float*)buf_dyn_shmem)[(((int)threadIdx.x) % 64)]) * 1.442695e+00f));
    float dt_s_1 = ((float)dt[(cse_v26 + (((int)threadIdx.x) % 64))]);
    float condval_1;
    if (((((int)threadIdx.x) & 63) < (cse_v7 + (((int)threadIdx.x) >> 6)))) {
      condval_1 = 1.000000e+00f;
    } else {
      condval_1 = 0.000000e+00f;
    }
    float dseg = ((((((float)((threadgroup half*)buf_dyn_shmem)[((cse_v15 + ((int)threadIdx.x)) + 4352)]) * cb_v) * lmat_2) * dt_s_1) * condval_1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tl::AtomicAdd((&(((threadgroup float*)buf_dyn_shmem)[((((long)cse_v7) + (((int)threadIdx.x) >> (long)6)) + (long)64)])), dseg);
    tl::AtomicAdd((&(((threadgroup float*)buf_dyn_shmem)[((((int)threadIdx.x) & (long)63) + (long)64)])), (dseg * -1.000000e+00f));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)threadIdx.x) < 64) {
    dA_cumsum_y[(cse_v26 + ((int)threadIdx.x))] = ((threadgroup float*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)];
  }
}


