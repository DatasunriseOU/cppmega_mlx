# TileLang Metal FP8 — storage-only emulation patch

Status: partial fix shipped. T.Cast between float8_e4m3 / float8_e5m2 and
float16 (or any fp/int via half) lowers cleanly on the Metal target.
T.gemm(fp8_A, fp8_B, fp32_C) does **not** work — caller must explicitly
dequantize FP8 to half/float before the gemm.

## Blocker


LOG(FATAL) << "Cannot convert type " << t << " to Metal type";


Hit at 3rdparty/tvm/src/target/source/codegen_metal.cc:271 (and the same
spot in TileLang's specialised codegen at src/target/codegen_metal.cc:279)
the moment any float8_e4m3 / float8_e5m2 / float8_e8m0fnu dtype reaches
PrintType. Sparse-MLA FP8, blockscaled, and mxfp8 lowering on Metal target
were unreachable as a result.

CUDA codegen at 3rdparty/tvm/src/target/source/codegen_cuda.cc:410-417
already handles FP8 by emitting __nv_fp8_e4m3 / __nv_fp8_e5m2 types
backed by cuda_fp8.h. Metal had no equivalent path.

## Apple FP8 reality findings (May 2026)

Researched via Apple developer documentation, MSL feature set tables, and
WWDC 2025 sessions on cooperative tensors:

| GPU family            | Native FP8 ALU            | Native FP8 simdgroup matmul |
| --------------------- | ------------------------- | --------------------------- |
| M1–M3 (Apple7–Apple8) | No                        | No                          |
| M4 / M4 Max (Apple9)  | **No**                    | No                          |
| M5 NAX (Apple10)      | **No** — FP16 / INT8 only | No                          |
| MSL 4.0 / 4.1 / 5.0   | No float8 scalar type     | n/a                         |

Apple's MetalPerformancePrimitives matmul2d and the M5 NAX cooperative
tensor primitives announced at WWDC 2025 expose **FP16 and INT8** matmul
intrinsics; they do not expose FP8. The MSL specification (current 3.2,
no 4.x bump in the FP type story) lists scalar floating types half,
bfloat, float, double only — there is no float8 type token.

**Conclusion:** any FP8 path on Metal is purely software emulation. The
right strategy is "storage-only FP8" — pack 8-bit values in uchar /
ucharN buffers, dequant to half on load, do the math in half (or
float accumulator), and quant back on store. This is exactly what
mx.quantized_matmul does for mxfp8 / nvfp4 in MLX core, and what the
TVM stock CUDA codegen does for pre-sm89 hardware.

## Patch design

Two files in two repositories, all changes are additive:

- 3rdparty/tvm/src/target/source/codegen_metal.{cc,h} — TVM's stock Metal
  codegen used by target.build.metal.
- src/target/codegen_metal.{cc,h} — TileLang's CodeGenTileLangMetal
  specialisation used by target.build.tilelang_metal.

For each codegen we add:

1. **PrintType FP8 case.** When t.is_float8() is true, emit uchar for
   lanes==1, ucharN for lanes∈[2,4], uint2 for lanes==8,
   uint4 for lanes==16. Sets enable_fp8_=true so the prelude is
   emitted. Mirrors the CUDA codegen's behaviour where FP8 vectors >4 are
   packed into wider integer storage.

2. **PrintFP8Prelude helper emission.** Inline MSL functions:
    - __tvm_fp8_e4m3_to_half(uchar) -> half
    - __tvm_fp8_e5m2_to_half(uchar) -> half
    - __tvm_half_to_fp8_e4m3(half) -> uchar
    - __tvm_half_to_fp8_e5m2(half) -> uchar

   Encodings follow the OCP "OFP8 Formats for Deep Learning" v1.0 spec.
   E4M3 uses the finite-only encoding (S.1111.111 is NaN, no Inf).
   E5M2 uses IEEE-style with NaN/Inf. Both directions implement
   round-to-nearest-even on the discarded mantissa bits.

3. **VisitExpr_(CastNode) override.** When either side of a cast is an
   FP8 dtype, scalar casts route through the helpers
   (fp8 -> half -> target or from_ty -> half -> fp8). Vector casts
   raise a clear LOG(FATAL) directing the caller to scalarise — the
   TVM lower pipeline already scalarises most user FP8 casts via the
   tir.transform.legalize_fp8 pass, so this branch is rarely hit.

4. **Finish() override.** If enable_fp8_ is set, splice the prelude
   right after using namespace metal; so the helpers see the MSL
   namespace without further qualification.

Arithmetic (binary ops, gemms, reductions) on FP8 operands is **out of
scope.** The TVM legalize_fp8 pass typically expands FP8 arithmetic
into Cast(fp8, op + op + ...) with op = Cast(half, fp8_load) chains,
which our VisitExpr_(CastNode) covers cleanly. For paths that bypass
that pass (e.g. T.gemm with FP8 buffers in metal.simdgroup storage)
the existing simdgroup-allocation check in VisitStmt_(AllocateNode)
fails fast: "Only float16, float32, and bfloat16 are supported".

## Diff stat


 3rdparty/tvm/src/target/source/codegen_metal.cc | 203 +++++++++++++++++++
 3rdparty/tvm/src/target/source/codegen_metal.h  |  10 +
 src/target/codegen_metal.cc                     | 177 ++++++++++++++++
 src/target/codegen_metal.h                      |  10 +
 4 files changed, 400 insertions(+)


Both files apply cleanly via git apply --check on TileLang/tvm fork
@ 0e15b274b.

## Test results

All four scalar cast directions lower successfully through TileLang's
lower(target=tvm.target.Target("metal")) and produce well-formed MSL
that **compiles with xcrun metal -c**:


[OK] fp8_e4m3 -> half
[OK] half -> fp8_e4m3
[OK] fp8_e5m2 -> half
[OK] half -> fp8_e5m2


Sample MSL (TileLang codegen, e4m3 round-trip kernel):

msl
#include <metal_stdlib>
using namespace metal;

// FP8 storage-only emulation helpers (MSL has no native float8 type).
inline half __tvm_fp8_e4m3_to_half(uchar x) { ... }
inline half __tvm_fp8_e5m2_to_half(uchar x) { ... }
inline uchar __tvm_half_to_fp8_e4m3(half v) { ... }
inline uchar __tvm_half_to_fp8_e5m2(half v) { ... }

kernel void fp8_round_trip_kernel(
    device uchar* A [[ buffer(0) ]],
    device uchar* B [[ buffer(1) ]],
    device half*  C [[ buffer(2) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]]) {
  half x = __tvm_fp8_e4m3_to_half(A[((int)threadIdx.x)]);
  B[((int)threadIdx.x)] = __tvm_half_to_fp8_e4m3((x + 1.000000e+00h));
  C[((int)threadIdx.x)] = x;
}


The stock TVM Metal codegen path (target.build.metal) goes through the
legalize_fp8 pass first, which expands FP8 ops into bit-shuffle code
inline. With this patch its PrintType no longer faults, so the legalised
output also compiles with xcrun metal -c.

T.gemm(fp8_A, fp8_B, fp32_C) still fails — by design — at the
metal.simdgroup allocation check (TileLang's existing assert at
src/target/codegen_metal.cc:454):
> Only float16, float32, and bfloat16 are supported, but got float8_e4m3

Caller must dequantize FP8 to half/float in shared memory before
T.gemm. This matches the FP8 gemm pattern used in
tilelang/examples/deepseek_v32/fp8_lighting_indexer.py.

mxfp8 (float8_e8m0fnu scale storage) also lowers correctly: it's a
device uchar* buffer with no helper calls (just pass-through).

## Upstream PR readiness for the TileLang/tvm fork

This patch is targeted at the TileLang/tvm fork at
https://github.com/tile-ai/tvm because the TileLang specialisation
(tilelang/src/target/codegen_metal.cc) duplicates the same PrintType
logic. Two separate PRs are appropriate:

- **Upstream apache/tvm PR:** the changes to
  3rdparty/tvm/src/target/source/codegen_metal.{cc,h} are vanilla TVM
  and should land in apache/tvm first. They mirror the existing CUDA
  FP8 storage path and are storage-only (no Apple-specific intrinsic
  use). Suggested PR title: "Metal codegen: add storage-only FP8
  emulation (e4m3 / e5m2 / e8m0fnu)".

- **Downstream tile-ai/tvm cherry-pick:** trivial; the TileLang fork
  carries an extra CodeGenTileLangMetal class with the same
  PrintType body, so the patch must be duplicated there.

- **TileLang PR:** the tilelang/src/target/codegen_metal.{cc,h}
  half can land in tile-ai/tilelang independently of the TVM
  changes, because the TileLang codegen carries its own copy.

Open questions for upstream review:

- The OFP8 e4m3 implementation uses the spec's "finite + NaN" encoding
  (S.1111.111 == NaN, no Inf). Some HuggingFace ml_dtypes libraries
  use float8_e4m3fn for this. We assume kFloat8_e4m3 and
  kFloat8_e4m3fn map to the same MSL helper; if a code path
  differentiates, route the appropriate variant in
  VisitExpr_(CastNode).

- We do not yet handle vector casts (lanes>1) inline; the TVM
  legalize_fp8 pass scalarises these for us today. If any future
  pass emits vector FP8 casts directly to codegen, we need to extend
  the lambda in VisitExpr_(CastNode) to fan out per-lane.

- Helpers are emitted unconditionally if any FP8 dtype is referenced.
  Dead-stripping by Apple's Metal compiler removes unused ones; we
  could narrow the prelude to only the helpers actually called by
  tracking calls inside VisitExpr_, but the cost is ~80 lines of
  inlined IR per kernel which is negligible.

## Files

- 0001-metal-fp8-storage-only.patch — combined patch (3rdparty/tvm
  + tilelang) ready for git apply from the tilelang repo root.
