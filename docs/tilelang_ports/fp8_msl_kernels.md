# FP8 MSL Kernels (vendored)

- Module: cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py
- Tests: tests/test_fp8_msl_kernels.py
- Bench: scripts/bench_fp8_msl_kernels.py -> bench/tilelang_ports/fp8_msl_kernels.json

## Source attribution

The Metal Shading Language sources embedded in the module are direct,
near-verbatim ports of the kernels published in two upstream repositories.

- AppMana/mps-fp8-for-torch-and-comfyui-python-package
  (commit a902571eca5362f5e2496cf33dcce52c8bac6a15) -- LUT-based decode
  and vectorized matmul. License: Apache 2.0.
- audiohacking/fp8-mps-metal
  (commit d4fbd40c48aa2a243e600d06627c7dd818150636) -- earlier branchy
  decode variant kept for reference. License: MIT (declared in README).

The vendored MSL is the AppMana variant: 256-entry LUT decode in MSL
constant memory plus integer-bit-manipulation encode (no log2 / exp2
transcendentals on the hot path). Both upstream projects target torch.mps
via torch.mps.compile_shader; this module re-hosts the same MSL via
mx.fast.metal_kernel so the kernels run inside MLX with no PyTorch
dependency.

## Why direct vendoring

Apple Silicon (through M5 / MSL 4.0) has no native FP8 hardware path:

- No float8_e4m3 type in MSL.
- No simdgroup_matrix<float8> MMA instruction.
- TileLang TVM-Metal codegen errors on float8_e4m3 -> Metal type
  conversion (3rdparty/tvm/src/target/source/codegen_metal.cc:271).

Path B therefore treats FP8 as **storage-only**:

1. Quantize: float -> uint8 via the integer-bit-manipulation encode with
   round-half-to-even (banker's rounding).
2. Dequantize: uint8 -> float via a 256-entry LUT in MSL constant memory
   (single load, no branching).
3. Matmul: dequant in-register, fp32 fma accumulation, fp32 output.

Path C now uses the same storage-only premise for the receipted TileLang
128x128x128 per-tensor case, but takes the faster route through TileLang's
Metal FP8 GEMM lowering: decode the FP8 shared tiles once into `threadgroup
half`, synchronize, then feed the existing FP16 `simdgroup_matrix` MMA
sequence. This is still not native FP8 MMA; it is FP8 storage plus FP16 MMA.

## Kernel surface

| Kernel | Inputs | Outputs | Notes |
| --- | --- | --- | --- |
| `fp8_to_half_kernel` | `(uint8[N])` | `fp16[N]` | direct LUT lookup |
| `half_to_fp8_kernel` | `(fp16[N])` | `uint8[N]` | banker's rounding |
| `fp8_scaled_matmul_kernel` | `(uint8[M,K], uint8[N,K], float[1\|M], float[1\|N], uint32 mode)` | `fp32[M,N]` | per-tensor or per-channel scale |
| `fp8_scaled_vecmat_kernel` | `(uint8[K], uint8[N,K], float[1], float[1\|N], uint32 mode)` | `fp32[N]` | SIMD reduction across K, M=1 only |

The Python wrappers add shape / dtype validation and lazy fallback to
mx.from_fp8 + mx.matmul when Metal is unavailable.

## Autograd

`fp8_scaled_matmul` is wrapped in `mx.custom_function` with a manual VJP
that dequantizes the inputs and uses `mx.matmul` to compute fp32
gradients (the upstream kernels are forward-only). The FP8 cast itself
is treated as a non-differentiable boundary -- gradients into the uint8
buffers are returned as zeros, gradients into the scale factors come
from the chain rule.

`fp8_scaled_vecmat` is forward-only (used at inference time on M=1
batches; backward should use `fp8_scaled_matmul` with M=1 instead).

## Bench (Davids-Mac-Studio.local, MLX 0.31.1)

From bench/tilelang_ports/fp8_msl_kernels.json (10 iters, 3 warmups):

| Shape | Op | Median ms | Notes |
| --- | --- | ---: | --- |
| 4096x4096x4096 | `half_to_fp8` (A) | 1.115 | encode bandwidth-bound |
| 4096x4096x4096 | `fp8_to_half` (A) | 0.639 | decode bandwidth-bound |
| 4096x4096x4096 | `fp8_scaled_matmul` | 228.227 | LUT decode + fp32 fma |
| 4096x4096x4096 | `mx.matmul fp16` (baseline) | 9.785 | simdgroup MMA |
| 4096x4096x4096 | `fp8_scaled_vecmat` (M=1) | 0.263 | competitive at M=1 |
| 2048x4096x512 | `fp8_scaled_matmul` | 17.415 | LUT decode |
| 2048x4096x512 | `mx.matmul fp16` (baseline) | 0.848 | simdgroup MMA |
| 512x8192x512 | `fp8_scaled_matmul` | 8.045 | LUT decode |
| 512x8192x512 | `mx.matmul fp16` (baseline) | 0.507 | simdgroup MMA |

Headline numbers:

- **FP8 matmul vs fp16 mx.matmul: ~23x slower** at 4096^3 (no simdgroup
  MMA path through MSL on Apple, so we lose the entire matmul tensor-core
  acceleration). This is consistent with the upstream AppMana / audiohacking
  numbers and matches the metalQwen3 reference.
- **FP8 vecmat is competitive with fp16 mx.matmul at M=1**: 0.26 ms vs
  ~0.18 ms mx.matmul. The vecmat path uses simd_sum reduction and 4-byte
  vectorized loads, so byte-level decode is amortized in memory load
  latency rather than lost to the missing MMA path.
- **Encode / decode are bandwidth-bound**: 1.1 ms / 0.6 ms for 16M
  elements, so runtime quantization is cheap relative to the matmul.
- **50% memory savings on FP8 storage** is preserved end-to-end (32 MB
  of fp16 weights becomes 16 MB of uint8 e4m3fn).

## Path C comparison

The Path C bench harness is `scripts/bench_tilelang_fp8_path_c.py`. Older
packed-dot4 and raw-MSL probe receipts showed useful correctness and scheduler
signals for `matmul_128` and M=1 vecmat, but those receipts are no longer the
production promotion gate.

The current production gate is the owner-output/tvm-ffi route that passes
existing MLX Metal buffers through DLPack and writes into caller-owned outputs.
That route is correctness-useful but still too slow for AUTO promotion:

| Surface | Current production status |
| --- | --- |
| `fp8_scaled_matmul_path_c(..., out=...)` | about 14x slower than the shipped Path B/audiohacking-style MSL route |
| `fp8_scaled_vecmat_path_c(..., out=...)` | about 1.7x slower than the shipped M=1 Path B `simd_sum` reducer |

Path B remains the production route and correctness oracle for generic FP8
matmul/vecmat. Path C must recover no-worse paired timing on the owner-output
tvm-ffi path before it can be used as an AUTO route. The B[K,N] FP8 matmul
layout remains a slow/diagnostic path; production callers should not infer that
the transpose-B or packed-dot4 probe receipts close the B[K,N] gate.

## Path C owner-output ABI

The production prepared-buffer ABI is the `out=` route:

- `fp8_scaled_matmul_path_c(..., out=existing_mx_array)`
- `fp8_scaled_vecmat_path_c(..., out=existing_mx_array)`

Those calls compile TileLang with `execution_backend="tvm_ffi"` and pass the
existing MLX buffers through DLPack. The wrapper does not allocate, copy, or
cast the output. Supported output dtypes are `mx.float32` and `mx.float16`;
unsupported shapes/dtypes fail before dispatch. `mx.bfloat16` is intentionally
rejected for now because the current TileLang Metal codegen emits invalid MSL
`bfloat` pointer/cast syntax for bf16 owner outputs. Direct-route DLPack
ownership/device failures propagate as typed TileLang DLPack errors, and the
direct route does not build `mx.fast.metal_kernel`.

The historical no-`out` APIs remain for legacy parity/bench callers and still
use MLX allocation semantics, so they are fail-closed by default. To exercise
them intentionally, set `CPPMEGA_FP8_PATH_C_LEGACY_MLX_FAST=1` in the test or
benchmark process. New production call sites must pass `out=`; the no-`out`
Path C route is non-owner-output legacy/debug only.

Data movement note: the owner-output Path C routes require prepared `mx.uint8`
e4m3 input buffers, existing `mx.float32` scale tensors, and an existing
`mx.float32`/`mx.float16` output buffer. They do not allocate, copy, or cast
large tensors at the Python wrapper boundary. The legacy no-`out` opt-in path
necessarily allocates the returned MLX output because that is the
`mx.fast.metal_kernel` API contract.

## How this complements the existing FP8 path

The repo already has a fused direct-MSL FP8 forward / backward at
`cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py` and a block-scaled MXFP8
variant at `cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py`. Those
fuse the FP8 dequant inline inside one specific attention kernel
(sparse-MLA with per-token / per-32-block scales) using a manual
bit-extraction decode.

`fp8_msl_kernels.py` is a **generic building block** layer:

- Scaled matmul / vecmat kernels for use in FP8 linear layers, MoE
  up-projections, value-only FP8 paths, etc., where the fused attention
  variant doesn't apply.
- Encode / decode helpers that match `mx.to_fp8` / `mx.from_fp8` byte
  layout but run on uint8 storage directly so callers can stage their
  own FP8 buffers without a pure-MLX roundtrip.
- LUT-based decode that is faster on Apple GPUs than the branchy
  bit-extraction decode used inside the fused attention kernels (one
  constant-memory load vs. several branches per byte).

It does **not** duplicate Agent C's TileLang FP8 codegen patch, which
targets upstream `tile-ai/tilelang` (TVM-Metal codegen for the
float8_e4m3 dtype emission). That patch fixes the *codegen* path so a
TileLang PrimFunc consuming float8 dtypes can lower to MSL. This
module bypasses TileLang entirely: the MSL is hand-vendored, so we
ship a working FP8 path today instead of waiting on upstream codegen.

## License

The vendored MSL sources retain their upstream license terms:

- LUT data and kernel bodies: Apache 2.0 (AppMana fork).
- Integer-bit encode and the underlying e4m3fn algorithm: Apache 2.0
  (AppMana) / MIT (audiohacking).

The Python wrappers and the cppmega.mlx integration around them are
covered by the cppmega.mlx repository's own LICENSE. The
``__license_notice__`` constant inside the module carries the
attribution that distributed binaries must keep visible.
