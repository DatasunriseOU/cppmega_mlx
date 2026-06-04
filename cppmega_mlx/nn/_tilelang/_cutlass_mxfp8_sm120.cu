// ===========================================================================
// Standalone CUTLASS >= 4.5.1 SM120/SM121 block-scaled MXFP8 x MXFP8 GEMM.
//
// LEVER 3 (lever="cutlass-mxfp8"). Native TN block-scaled MXFP8 MMA on gb10
// (GB10/DGX-Spark, sm_121 == cc 12.1) via the CUTLASS C++ COLLECTIVE BUILDER
// path (arch::Sm120, shared by SM121). This is the SAME C++ route measured at
// ~188 TFLOPS FP8 on a real DGX Spark (NVIDIA Dev Forum thread 359960) and is
// UNAFFECTED by the Python-DSL ptxas-reject bug (CUTLASS issue #3227, which is
// nvvm/MLIR-lowering only).
//
// Lifted from CUTLASS v4.5.1 example
//   examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu
// but specialised to PURE MXFP8 x MXFP8 (ElementB = mx_float8_t<float_e4m3_t>,
// NOT the mxfp6 of 79c) -> kind::mxf8f6f4.block_scale e4m3 x e4m3, fp32 accum.
//
// SIDE-CHECKOUT BUILD (header-only; does NOT touch the live tilelang
// 3rdparty/cutlass == 4.1.0):
//   git clone --depth 1 --branch v4.5.1 \
//       https://github.com/NVIDIA/cutlass.git /home/dave/source/cutlass-451
//   /usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
//     -gencode arch=compute_121a,code=sm_121a \
//     -I/home/dave/source/cutlass-451/include \
//     -I/home/dave/source/cutlass-451/tools/util/include \
//     --expt-relaxed-constexpr -lineinfo \
//     _cutlass_mxfp8_sm120.cu -o _cutlass_mxfp8_sm120.so
//
// MUST be sm_121a (the 'a' suffix). Plain sm_121 / sm_120 hits CUTLASS issue
// #2820 "arch conditional MMA used without targeting appropriate compute
// capability" + #3227 gpu_arch_map default (12,1)->sm_121 (not sm_121a).
//
// RULE #1 (NO silent fallback): every cudaError / CUTLASS status is RETURNED as
// a nonzero code with the failing site + cudaGetErrorString in the C error-string
// accessor. The Python driver RAISES on any nonzero. There is NO bf16/cuBLAS
// fallback in this kernel — real MXFP8 MMA or a hard error.
// ===========================================================================

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

// ---------------------------------------------------------------------------
// Compile-time arch guard. The block-scaled SM120 MMA atom (MXF8F6F4MMAOP)
// only exists in CUTLASS >= 4.5.0/4.5.1. If the side-checkout headers are too
// old (or nvcc was not pointed at sm_121a) these macros are undefined and we
// fail the build LOUDLY here rather than silently emitting a wrong kernel.
// ---------------------------------------------------------------------------
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && \
    !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#  error "CUTLASS SM120/SM121 block-scaled MMA not supported by these headers. \
Need CUTLASS >= 4.5.1 (side-checkout v4.5.1) AND nvcc -gencode \
arch=compute_121a,code=sm_121a. RULE #1: refusing to build a non-block-scaled \
kernel that would silently run the wrong path."
#endif

namespace {

using namespace cute;

// --- Element types (the MXFP8 x MXFP8 contract) ---------------------------
// mx_float8_t<float_e4m3_t> couples the e4m3 element payload with its E8M0
// (ue8m0) block-32 scale, exactly as CUTLASS example 79c declares ElementA.
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;  // pure MXFP8 (79c uses mxfp6 here)
using ElementC   = cutlass::bfloat16_t;   // output / source
using ElementD   = cutlass::bfloat16_t;   // output
using ElementAccumulator = float;
using ElementCompute      = float;

// TN layout is the ONLY layout the sm120_blockscaled_mma_builder asserts
// (UmmaMajorA == K && UmmaMajorB == K): A row-major (K-major), B col-major
// (K-major). This matches our existing fp8 amax block-scale producer.
using LayoutATag = cutlass::layout::RowMajor;     // A: M-major rows, K contiguous
using LayoutBTag = cutlass::layout::ColumnMajor;  // B: N-major cols, K contiguous
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<typename ElementA::DataType>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<typename ElementB::DataType>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

// --- Arch + op class -------------------------------------------------------
// arch::Sm120 is the consumer-Blackwell ("geforce") tag SHARED by SM121
// (GB10). OpClassBlockScaledTensorOp selects the warp-level f8f6f4 block-scale
// atom (NOT the SM100 datacenter tcgen05/TMEM path SM121 lacks).
using ArchTag    = cutlass::arch::Sm120;
using OpClass    = cutlass::arch::OpClassBlockScaledTensorOp;

// 128x128x128 mainloop tile + cluster 1x1x1 (SM121 has no multicast/2-SM MMA;
// the builder asserts ClusterShape == 1). N >= 32 asserted; 128 satisfies it.
using MmaTileShape     = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using KernelSchedule    = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule  = cutlass::epilogue::collective::EpilogueScheduleAuto;

// --- Collective epilogue ---------------------------------------------------
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutCTag, AlignC,
        ElementD, LayoutDTag, AlignD,
        EpilogueSchedule
    >::CollectiveOp;

// --- Collective mainloop (sm120_blockscaled_mma_builder.inl) ---------------
// StageCountAutoCarveout subtracts the epilogue's SMEM from the budget so the
// pipelined stages fit under GB10's 101,376-byte dynamic-SMEM hard cap.
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, LayoutATag, AlignA,
        ElementB, LayoutBTag, AlignB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,   // ProblemShape (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue,
    void  // default tile scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Strides for the dense operands / output.
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// Block-scale (E8M0) layout config: builds LayoutSFA/LayoutSFB atoms.
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Thread-local error string for the host accessor.
char g_last_error[512] = {0};

inline void set_error(const char* where, cudaError_t e) {
  std::snprintf(g_last_error, sizeof(g_last_error),
                "[%s] cudaError %d: %s", where, int(e), cudaGetErrorString(e));
}
inline void set_error_status(const char* where, cutlass::Status s) {
  std::snprintf(g_last_error, sizeof(g_last_error),
                "[%s] cutlass::Status %d: %s", where, int(s),
                cutlassGetStatusString(s));
}

}  // anonymous namespace

extern "C" {

// Return the last error message (NULL-terminated). Valid until the next call.
const char* cppmega_mxfp8_last_error() { return g_last_error; }

// Return the byte sizes the Python driver must allocate for the packed E8M0
// scale-factor tensors (SFA, SFB) for a given (M, N, K). The driver lays the
// per-32-block ue8m0 exponent bytes from fp8_amax into these buffers using the
// SAME atom ordering the kernel expects (Sm1xxBlkScaledConfig). We expose the
// shapes here so the Python side never has to re-derive the CUTLASS atom layout.
//
// Returns 0 on success; nonzero on bad shape (RULE #1: no silent clamp).
int cppmega_mxfp8_sf_sizes(int M, int N, int K,
                           int64_t* sfa_bytes, int64_t* sfb_bytes) {
  if (M <= 0 || N <= 0 || K <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[sf_sizes] non-positive shape M=%d N=%d K=%d", M, N, K);
    return 1;
  }
  if (K % 32 != 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[sf_sizes] K=%d not a multiple of the 32-element MXFP8 "
                  "block; refusing to round (RULE #1).", K);
    return 2;
  }
  if (N < 32) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[sf_sizes] N=%d < 32; sm120_blockscaled builder asserts "
                  "N>=32.", N);
    return 3;
  }
  auto problem = cute::make_shape(M, N, K, 1);
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);
  // cute::cosize == number of scale-factor elements (one ue8m0 byte each).
  *sfa_bytes = static_cast<int64_t>(cute::cosize(layout_sfa));
  *sfb_bytes = static_cast<int64_t>(cute::cosize(layout_sfb));
  return 0;
}

// Native MXFP8 x MXFP8 GEMM: D(M,N) = A(M,K) @ B(N,K)^T, both e4m3 with E8M0
// (ue8m0) block-32 scales. TN layout (A row-major, B col-major). All pointers
// are raw CUdeviceptr (device memory). C may be NULL (alpha*A@B + 0).
//
// Returns 0 on success; nonzero (with cppmega_mxfp8_last_error populated) on
// ANY failure. RULE #1: no fallback path — a failure here propagates to a
// Python raise.
int cppmega_mxfp8_gemm_sm121(
    const void* A_e4m3, const void* SFA_e8m0,
    const void* B_e4m3, const void* SFB_e8m0,
    const void* C, void* D,
    int M, int N, int K,
    float alpha, float beta,
    void* stream_raw) {

  g_last_error[0] = '\0';

  if (A_e4m3 == nullptr || B_e4m3 == nullptr || D == nullptr ||
      SFA_e8m0 == nullptr || SFB_e8m0 == nullptr) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[gemm] null operand pointer (A=%p SFA=%p B=%p SFB=%p D=%p)",
                  A_e4m3, SFA_e8m0, B_e4m3, SFB_e8m0, D);
    return 10;
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[gemm] non-positive shape M=%d N=%d K=%d", M, N, K);
    return 11;
  }
  if (K % 32 != 0 || N < 32) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "[gemm] shape violates MXFP8 block-scaled contract "
                  "(K%%32==0 && N>=32): M=%d N=%d K=%d", M, N, K);
    return 12;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);

  // Dense strides (TN). L (batch) = 1.
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto problem = cute::make_shape(M, N, K, 1);
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

  using ElementAData = typename ElementA::DataType;
  using ElementBData = typename ElementB::DataType;
  using ElementSF    = typename ElementA::ScaleFactorType;  // float_ue8m0_t

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {  // MainloopArguments
          reinterpret_cast<const ElementAData*>(A_e4m3), stride_a,
          reinterpret_cast<const ElementBData*>(B_e4m3), stride_b,
          reinterpret_cast<const ElementSF*>(SFA_e8m0), layout_sfa,
          reinterpret_cast<const ElementSF*>(SFB_e8m0), layout_sfb
      },
      {  // EpilogueArguments
          {alpha, beta},
          reinterpret_cast<const ElementC*>(C), stride_c,
          reinterpret_cast<ElementD*>(D), stride_d
      }
  };

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    set_error_status("can_implement", status);
    return 20;
  }

  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes > 0) {
    cudaError_t e = cudaMalloc(&workspace, workspace_bytes);
    if (e != cudaSuccess) { set_error("cudaMalloc(workspace)", e); return 21; }
  }

  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    set_error_status("initialize", status);
    if (workspace) cudaFree(workspace);
    return 22;
  }

  // Run. GemmUniversalAdapter::run() opts the kernel into the required dynamic
  // SMEM via cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)
  // internally; if the carved tile still exceeds GB10's 101,376-byte cap the
  // launch returns cudaErrorInvalidValue / OutOfResources here and we RAISE.
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    set_error_status("run", status);
    if (workspace) cudaFree(workspace);
    return 23;
  }

  cudaError_t e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    set_error("cudaStreamSynchronize", e);
    if (workspace) cudaFree(workspace);
    return 24;
  }

  if (workspace) {
    e = cudaFree(workspace);
    if (e != cudaSuccess) { set_error("cudaFree(workspace)", e); return 25; }
  }
  return 0;
}

}  // extern "C"
