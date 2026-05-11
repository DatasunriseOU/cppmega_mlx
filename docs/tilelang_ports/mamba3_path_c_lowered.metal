// === Path C (TileLang DSL) lowered MSL ===
// Bench shape: B=2 T=512 H=4 P=32 N=64

// ---- Forward ----
// Function: fwd_kernel
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
        red_buf[simdgroup_id + i * workspace_stride] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + i * workspace_stride]
                       : red_buf[i * workspace_stride];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + i * workspace_stride] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      x[i] = red_buf[final_slot + i * workspace_stride];
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
        red_buf[simdgroup_id + i * workspace_stride] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + i * workspace_stride]
                       : red_buf[i * workspace_stride];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + i * workspace_stride] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      x[i] = red_buf[final_slot + i * workspace_stride];
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
        red_buf[local_tid + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) +
                                      i * workspace_stride]);
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
        red_buf[local_tid + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) +
                                      i * workspace_stride]);
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
kernel void fwd_kernel(  device float* A [[ buffer(0) ]],
  device float* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  device float* D [[ buffer(3) ]],
  device float* dt [[ buffer(4) ]],
  device float* h0 [[ buffer(5) ]],
  device float* h_last [[ buffer(6) ]],
  device float* x [[ buffer(7) ]],
  device float* y [[ buffer(8) ]],
  device float* z [[ buffer(9) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  thread float h_state[64];
  float y_acc = 0.000000e+00f;
  int cse_v2 = (((int)threadIdx.x) * 64);
  for (int n = 0; n < 64; ++n) {
    h_state[n] = h0[(cse_v2 + n)];
  }
  for (int t = 0; t < 512; ++t) {
    int cse_v1 = (((int)threadIdx.x) & 127);
    int cse_v3 = (((int)threadIdx.x) >> 7);
    int cse_v4 = (cse_v1 >> 5);
    int cse_v6 = (((cse_v3 * 2048) + (t * 4)) + cse_v4);
    float A_val = A[cse_v6];
    float dt_val = dt[cse_v6];
    float cse_v2_1 = (A_val * dt_val);
    float decay = exp(cse_v2_1);
    int cse_v5 = (((cse_v3 * 65536) + (t * 128)) + cse_v1);
    float x_val = x[cse_v5];
    float z_val = z[cse_v5];
    y_acc = 0.000000e+00f;
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      int cse_v7 = ((((cse_v3 * 131072) + (t * 256)) + (cse_v4 * 64)) + n_1);
      float new_h = ((decay * h_state[n_1]) + (x_val * B[cse_v7]));
      h_state[n_1] = new_h;
      y_acc = (y_acc + (new_h * C[cse_v7]));
    }
    float D_h = D[cse_v4];
    float y_skipped = (y_acc + (D_h * x_val));
    float cse_v1_1 = (z_val * -1.000000e+00f);
    float sig_z = (1.000000e+00f / (1.000000e+00f + exp(cse_v1_1)));
    y[cse_v5] = ((z_val * sig_z) * y_skipped);
  }
  for (int n_2 = 0; n_2 < 64; ++n_2) {
    h_last[(cse_v2 + n_2)] = h_state[n_2];
  }
}

// ---- Backward ----
// Function: bwd_kernel
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
        red_buf[simdgroup_id + i * workspace_stride] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + i * workspace_stride]
                       : red_buf[i * workspace_stride];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + i * workspace_stride] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      x[i] = red_buf[final_slot + i * workspace_stride];
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
        red_buf[simdgroup_id + i * workspace_stride] = x[i];
      }
    }
    Barrier::template sync<1>();
    if (simdgroup_id == 0) {
      for (int i = 0; i < batch_size; ++i) {
        T result = lane < uint(simdgroup_count)
                       ? red_buf[lane + i * workspace_stride]
                       : red_buf[i * workspace_stride];
        result = reduce_partials(result, lane);
        if (lane == 0) {
          red_buf[final_slot + i * workspace_stride] = result;
        }
      }
    }
    Barrier::template sync<2>();
    for (int i = 0; i < batch_size; ++i) {
      x[i] = red_buf[final_slot + i * workspace_stride];
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
        red_buf[local_tid + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) +
                                      i * workspace_stride]);
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
        red_buf[local_tid + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) +
                                      i * workspace_stride]);
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
kernel void bwd_kernel(  device float* A [[ buffer(0) ]],
  device float* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  device float* D [[ buffer(3) ]],
  device float* dA_partial [[ buffer(4) ]],
  device float* dB_partial [[ buffer(5) ]],
  device float* dC_partial [[ buffer(6) ]],
  device float* dD_partial [[ buffer(7) ]],
  device float* ddt_partial [[ buffer(8) ]],
  device float* dh0 [[ buffer(9) ]],
  device float* dt [[ buffer(10) ]],
  device float* dx [[ buffer(11) ]],
  device float* dy [[ buffer(12) ]],
  device float* dz [[ buffer(13) ]],
  device float* h0 [[ buffer(14) ]],
  device float* x [[ buffer(15) ]],
  device float* z [[ buffer(16) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  thread float h_state[64];
  thread float dh[64];
  float dD_acc = 0.000000e+00f;
  float inv_decay = 0.000000e+00f;
  float y_state = 0.000000e+00f;
  float dx_inp = 0.000000e+00f;
  float d_decay = 0.000000e+00f;
  int cse_v3 = (((int)threadIdx.x) * 64);
  for (int n = 0; n < 64; ++n) {
    h_state[n] = h0[(cse_v3 + n)];
  }
  int cse_v4 = (((int)threadIdx.x) & 127);
  int cse_v5 = (((int)threadIdx.x) >> 7);
  int cse_v6 = (cse_v4 >> 5);
  int cse_v7 = (cse_v5 * 65536);
  int cse_v8 = (cse_v5 * 131072);
  int cse_v9 = (cse_v5 * 2048);
  int cse_v11 = (cse_v6 * 64);
  for (int t = 0; t < 512; ++t) {
    int cse_v13 = ((cse_v9 + (t * 4)) + cse_v6);
    float A_val = A[cse_v13];
    float dt_val = dt[cse_v13];
    float cse_v1 = (A_val * dt_val);
    float decay = exp(cse_v1);
    float x_val = x[((cse_v7 + (t * 128)) + cse_v4)];
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      float new_h = ((decay * h_state[n_1]) + (x_val * B[(((cse_v8 + (t * 256)) + cse_v11) + n_1)]));
      h_state[n_1] = new_h;
    }
  }
  for (int n_2 = 0; n_2 < 64; ++n_2) {
    dh[n_2] = 0.000000e+00f;
  }
  dD_acc = 0.000000e+00f;
  float D_h = D[cse_v6];
  for (int r = 0; r < 512; ++r) {
    int cse_v14 = (((cse_v9 + cse_v6) + 2044) - (r * 4));
    float A_val_1 = A[cse_v14];
    float dt_val_1 = dt[cse_v14];
    float cse_v3_1 = (A_val_1 * dt_val_1);
    float decay_1 = exp(cse_v3_1);
    inv_decay = (1.000000e+00f / decay_1);
    int cse_v15 = (((cse_v7 + cse_v4) + 65408) - (r * 128));
    float x_val_1 = x[cse_v15];
    float z_val = z[cse_v15];
    float dY = dy[cse_v15];
    y_state = 0.000000e+00f;
    int cse_v1_1 = (r * 256);
    int cse_v12 = (cse_v8 + cse_v11);
    for (int n_3 = 0; n_3 < 64; ++n_3) {
      y_state = (y_state + (h_state[n_3] * C[(((cse_v12 + n_3) + 130816) - cse_v1_1)]));
    }
    float y_skipped = (y_state + (D_h * x_val_1));
    float cse_v2 = (z_val * -1.000000e+00f);
    float sig_z = (1.000000e+00f / (1.000000e+00f + exp(cse_v2)));
    float silu_z = (z_val * sig_z);
    float silu_dz = (sig_z * (1.000000e+00f + (z_val * (1.000000e+00f - sig_z))));
    float d_silu = (dY * y_skipped);
    float d_y_skipped = (dY * silu_z);
    dz[cse_v15] = (d_silu * silu_dz);
    dD_acc = (dD_acc + (d_y_skipped * x_val_1));
    for (int n_4 = 0; n_4 < 64; ++n_4) {
      dh[n_4] = (dh[n_4] + (d_y_skipped * C[(((cse_v12 + n_4) + 130816) - cse_v1_1)]));
    }
    dx_inp = 0.000000e+00f;
    d_decay = 0.000000e+00f;
    int cse_v2_1 = (r * 8192);
    int cse_v10 = ((cse_v5 * 4194304) + (cse_v4 * 64));
    if (r == 511) {
      for (int n_5 = 0; n_5 < 64; ++n_5) {
        float B_val = B[(((cse_v12 + n_5) + 130816) - cse_v1_1)];
        int cse_v16 = (((cse_v10 + n_5) + 4186112) - cse_v2_1);
        dC_partial[cse_v16] = (d_y_skipped * h_state[n_5]);
        dB_partial[cse_v16] = (dh[n_5] * x_val_1);
        dx_inp = (dx_inp + (dh[n_5] * B_val));
        d_decay = (d_decay + (dh[n_5] * h0[(cse_v3 + n_5)]));
      }
    } else {
      for (int n_6 = 0; n_6 < 64; ++n_6) {
        float B_val_1 = B[(((cse_v12 + n_6) + 130816) - cse_v1_1)];
        int cse_v17 = (((cse_v10 + n_6) + 4186112) - cse_v2_1);
        dC_partial[cse_v17] = (d_y_skipped * h_state[n_6]);
        dB_partial[cse_v17] = (dh[n_6] * x_val_1);
        dx_inp = (dx_inp + (dh[n_6] * B_val_1));
        float h_prev = ((h_state[n_6] - (x_val_1 * B_val_1)) * inv_decay);
        d_decay = (d_decay + (dh[n_6] * h_prev));
        h_state[n_6] = h_prev;
      }
    }
    float dx_skip = (d_y_skipped * D_h);
    dx[cse_v15] = ((d_y_skipped * D_h) + dx_inp);
    float d_logdecay = (d_decay * decay_1);
    dA_partial[cse_v15] = (d_logdecay * dt_val_1);
    ddt_partial[cse_v15] = (d_logdecay * A_val_1);
    for (int n_7 = 0; n_7 < 64; ++n_7) {
      dh[n_7] = (dh[n_7] * decay_1);
    }
  }
  for (int n_8 = 0; n_8 < 64; ++n_8) {
    dh0[(cse_v3 + n_8)] = dh[n_8];
  }
  dD_partial[((int)threadIdx.x)] = dD_acc;
}
