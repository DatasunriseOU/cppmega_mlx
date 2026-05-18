// === Path C (TileLang DSL) lowered MSL ===
// Bench shape: B=1 T=2048 H=112 P=64 N=64

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
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 gridThreadIdx [[thread_position_in_grid]]
) {
  int grid_tid = gridThreadIdx.x;
  thread float h_state[64];
  float decay = 0.000000e+00f;
  float y_acc = 0.000000e+00f;
  int cse_v1 = (gridThreadIdx.x / 64);
  float D_h = D[cse_v1];
  int cse_v2 = (gridThreadIdx.x * 64);
  for (int n = 0; n < 64; ++n) {
    h_state[n] = h0[(cse_v2 + n)];
  }
  for (int t = 0; t < 2048; ++t) {
    int cse_v4 = ((t * 112) + cse_v1);
    float A_val = A[cse_v4];
    float dt_val = dt[cse_v4];
    decay = exp((A_val * dt_val));
    int cse_v3 = (t * 7168);
    int cse_v5 = (cse_v3 + gridThreadIdx.x);
    float x_val = x[cse_v5];
    float z_val = z[cse_v5];
    y_acc = 0.000000e+00f;
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      int cse_v6 = ((cse_v3 + (cse_v1 * 64)) + n_1);
      float B_val = B[cse_v6];
      float C_val = C[cse_v6];
      float new_h = ((decay * h_state[n_1]) + (x_val * B_val));
      h_state[n_1] = new_h;
      y_acc = (y_acc + (new_h * C_val));
    }
    float y_skipped = (y_acc + (D_h * x_val));
    float cse_v1_1 = (z_val * -1.000000e+00f);
    float cse_v1_2 = (1.000000e+00f / (1.000000e+00f + exp(cse_v1_1)));
    y[cse_v5] = ((z_val * cse_v1_2) * y_skipped);
  }
  for (int n_2 = 0; n_2 < 64; ++n_2) {
    h_last[(cse_v2 + n_2)] = h_state[n_2];
  }
}

// ---- Backward ----
// Function: bwd_partial_kernel
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
kernel void bwd_partial_kernel(  device float* A [[ buffer(0) ]],
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
  device float* h_snap [[ buffer(14) ]],
  device float* x [[ buffer(15) ]],
  device float* z [[ buffer(16) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint3 gridThreadIdx [[thread_position_in_grid]]
) {
  int grid_tid = gridThreadIdx.x;
  int p = 0;
  thread float h_state[64];
  thread float dh[64];
  float dD_acc = 0.000000e+00f;
  float decay = 0.000000e+00f;
  float y_state = 0.000000e+00f;
  float dx_inp = 0.000000e+00f;
  float d_decay = 0.000000e+00f;
  p = (gridThreadIdx.x & 63);
  int cse_v1 = (gridThreadIdx.x / 64);
  int cse_v4 = (cse_v1 * 4096);
  for (int n = 0; n < 64; ++n) {
    float condval;
    if (((0 <= p) && (p < 64))) {
      condval = h_snap[(((cse_v4 + (p * 64)) + n) + 939524096)];
    } else {
      condval = 0.000000e+00f;
    }
    h_state[n] = condval;
    dh[n] = 0.000000e+00f;
  }
  dD_acc = 0.000000e+00f;
  float D_h = D[cse_v1];
  int cse_v5 = (cse_v1 * 64);
  for (int rt = 0; rt < 2048; ++rt) {
    int cse_v7 = ((cse_v1 + 229264) - (rt * 112));
    float A_val = A[cse_v7];
    float dt_val = dt[cse_v7];
    decay = exp((A_val * dt_val));
    int cse_v2 = (rt * 7168);
    float condval_1;
    if (((0 <= p) && (p < 64))) {
      condval_1 = x[(((cse_v5 + p) + 14672896) - cse_v2)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    float x_val = condval_1;
    float condval_2;
    if (((0 <= p) && (p < 64))) {
      condval_2 = z[(((cse_v5 + p) + 14672896) - cse_v2)];
    } else {
      condval_2 = 0.000000e+00f;
    }
    float z_val = condval_2;
    float condval_3;
    if (((0 <= p) && (p < 64))) {
      condval_3 = dy[(((cse_v5 + p) + 14672896) - cse_v2)];
    } else {
      condval_3 = 0.000000e+00f;
    }
    float dY = condval_3;
    y_state = 0.000000e+00f;
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      y_state = (y_state + (h_state[n_1] * C[(((cse_v5 + n_1) + 14672896) - cse_v2)]));
    }
    float y_skipped = (y_state + (D_h * x_val));
    float cse_v1_1 = (z_val * -1.000000e+00f);
    float cse_v1_2 = (1.000000e+00f / (1.000000e+00f + exp(cse_v1_1)));
    float cse_v2_1 = (z_val * cse_v1_2);
    float cse_v5_1 = (cse_v1_2 * (1.000000e+00f + (z_val * (1.000000e+00f - cse_v1_2))));
    float d_silu = (dY * y_skipped);
    float cse_v3 = (dY * cse_v2_1);
    if (0 <= p) {
      if (p < 64) {
        dz[(((cse_v5 + p) + 14672896) - cse_v2)] = (d_silu * cse_v5_1);
      }
    }
    dD_acc = (dD_acc + (cse_v3 * x_val));
    dx_inp = 0.000000e+00f;
    d_decay = 0.000000e+00f;
    for (int n_2 = 0; n_2 < 64; ++n_2) {
      int cse_v8 = (((cse_v5 + n_2) + 14672896) - cse_v2);
      float C_val = C[cse_v8];
      float B_val = B[cse_v8];
      int cse_v3_1 = (rt * 458752);
      float condval_4;
      if (((0 <= p) && (p < 64))) {
        condval_4 = h_snap[((((cse_v4 + (p * 64)) + n_2) + 939065344) - cse_v3_1)];
      } else {
        condval_4 = 0.000000e+00f;
      }
      float h_prev = condval_4;
      float dh_n = (dh[n_2] + (cse_v3 * C_val));
      if (0 <= p) {
        if (p < 64) {
          int cse_v6 = (cse_v4 + (n_2 * 64));
          dC_partial[(((cse_v6 + p) + 939065344) - cse_v3_1)] = (cse_v3 * h_state[n_2]);
          dB_partial[(((cse_v6 + p) + 939065344) - cse_v3_1)] = (dh_n * x_val);
        }
      }
      dx_inp = (dx_inp + (dh_n * B_val));
      d_decay = (d_decay + (dh_n * h_prev));
      dh[n_2] = (dh_n * decay);
      h_state[n_2] = h_prev;
    }
    float cse_v4_1 = (cse_v3 * D_h);
    if (0 <= p) {
      if (p < 64) {
        dx[(((cse_v5 + p) + 14672896) - cse_v2)] = (cse_v4_1 + dx_inp);
      }
    }
    float d_logdecay = (d_decay * decay);
    if (0 <= p) {
      if (p < 64) {
        dA_partial[(((cse_v5 + p) + 14672896) - cse_v2)] = (d_logdecay * dt_val);
        ddt_partial[(((cse_v5 + p) + 14672896) - cse_v2)] = (d_logdecay * A_val);
      }
    }
  }
  if (0 <= p) {
    for (int n_3 = 0; n_3 < 64; ++n_3) {
      if (p < 64) {
        dh0[((cse_v4 + (p * 64)) + n_3)] = dh[n_3];
      }
    }
    if (p < 64) {
      dD_partial[(cse_v5 + p)] = dD_acc;
    }
  }
}
