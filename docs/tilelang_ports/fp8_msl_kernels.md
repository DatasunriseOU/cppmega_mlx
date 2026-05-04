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

The Path C bench harness is `scripts/bench_tilelang_fp8_path_c.py`; its strict
gate requires Path C and Path B to run, Path C median ratio to be `<= 1.0`, and
Path C vs Path B parity to stay within `1e-5` max abs/rel for parity-enabled
shapes.

Current checked and live receipts for `matmul_128` use the compact simdgroup
MSL path, not the older scalar fallback:

| Receipt | Path B median ms | Path C median ms | Paired Path C / Path B | Parity max abs / rel | Source markers |
| --- | ---: | ---: | ---: | ---: | --- |
| `bench/tilelang_ports/fp8_path_c_vs_path_b.json` | 0.1227 | 0.1076 | 0.890x | 0.0 / 0.0 | `simdgroup_multiply_accumulate=1`, `threadgroup_half=2`, `fp8_e4m3_lut=0` |
| `bench/tilelang_ports/fp8_path_c.json` | 0.1521 | 0.1362 | 0.896x | 0.0 / 0.0 | same |
| `/tmp/fp8_path_c_matmul128_before.json` (Lane 6 live run, 3 warmups / 8 iters, `--skip-xcrun`) | 0.2068 | 0.1738 | 0.835x | 0.0 / 0.0 | same |

Path B remains valuable as the generic direct-MLX MSL fallback and correctness
oracle. Path C is the preferred measured route only for the shapes and scale
layouts covered by the strict receipts; larger or per-row/per-block shapes
still need fresh receipts before claiming the same performance relationship.

## Path C vecmat gate (Lane 7)

Path C now matches the Path B M=1 vecmat hot loop for the 4096x4096 gate:

- Runtime body: `thread_index_in_simdgroup`, `reinterpret_cast<device const uint*>`
  packed FP8 loads, LUT-backed dot4 decode, and literal `simd_sum(sum)`.
- Launch shape: four SIMD groups per threadgroup, `threadgroup=(128, 1, 1)`,
  with one SIMD group producing one output row.
- Dispatch shape: direct MLX tuple dispatch returns `(N,)`, avoiding the older
  TileLang buffer-map reshape path for the production reducer.
- Strict local receipt:
  `bench/tilelang_ports/lane7_fp8_vecmat_4096.json`, generated with
  `scripts/bench_tilelang_fp8_path_c.py --shapes vecmat_4096 --warmup 10
  --iters 50 --skip-sparse --strict --max-ratio 1.0`.

Latest strict result on Davids-Mac-Studio.local (MLX 0.31.1):

| Shape | Path B median ms | Path C median ms | Paired median ratio | Parity vs Path B |
| --- | ---: | ---: | ---: | --- |
| M=1,N=4096,K=4096 | 0.242562 | 0.238667 | 0.980702x | max_abs=0.0, max_rel=0.0 |

The blocker list is empty for this gate:
`path_b_fast_path_ready=true`, `missing=[]`, with runtime markers
`packed_uint_loads=2`, `fp8_e4m3_lut=8`, and `simd_sum=1`.

External check: current MLX custom Metal kernels support the raw MSL body/header
path used here, and TileLang's public Metal backend landed separately upstream
in `tile-ai/tilelang` PR #799. There is still no native Apple FP8 MMA path, so
this gate is about matching the scalar-LUT vecmat reducer, not turning FP8
matmul into a tensor-core-style path.

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
