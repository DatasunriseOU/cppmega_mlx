# TileLang Metal FP8 GEMM — software dequant-and-multiply path

Status: rebased 2026-05-03; applies cleanly on
jorgecurious/tilelang:metal-gemm-upstream-rebase (PR #2130) at HEAD
971c17b. The change makes T.gemm(fp8_A, fp8_B, fp32_C) (and the
fp16-accumulator variant) lower cleanly on the Metal target by routing
FP8 inputs to the scalar fallback so per-element T.cast(value,
accum_dtype) reads expand to the storage-only FP8 decode helpers in
MSL and compile with xcrun metal -c.

## Stacking topology

This patch sits at the top of a four-PR stack against tile-ai/tilelang:main.
All referenced PRs are publicly visible OPEN PRs:

<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>PR / artifact</th>
      <th>Branch</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>tile-ai/tilelang **#1869**</td>
      <td><https://github.com/tile-ai/tilelang/pull/1869></td>
      <td>Initial Apple Metal landing (codegen, target, simdgroup MMA<br>
      backbone)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>tile-ai/tilelang **#2118**</td>
      <td><https://github.com/tile-ai/tilelang/pull/2118></td>
      <td>Metal scalar fallback for T.gemm (GemmMetalScalar)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>tile-ai/tilelang **#2121**</td>
      <td><https://github.com/tile-ai/tilelang/pull/2121></td>
      <td>SiriusNEO's CodeGen multi-backend decoupling — refactors the<br>
      dispatcher</td>
    </tr>
    <tr>
      <td>4</td>
      <td>tile-ai/tilelang **#2130** (jorgecurious)</td>
      <td><https://github.com/jorgecurious/tilelang/tree/metal-gemm-<br>
      upstream-rebase></td>
      <td>Most-upstream-rebased Metal landing; rebases #1869 onto<br>
      post-#2121 main</td>
    </tr>
    <tr>
      <td>5</td>
      <td>this artifact (tilelang_metal_fp8_gemm)</td>
      <td>n/a (filed via this directory)</td>
      <td>Routes FP8-input T.gemm on Metal to the scalar dequant-<br>
      multiply-accumulate path</td>
    </tr>
  </tbody>
</table>

**Patch-level prereq chain:** PR #2121 (refactor) → PR #2130 (rebase) → our
tilelang_metal_fp8_gemm. The patch text is generated against jorgecurious's
post-refactor dispatcher shape and applies on a clean checkout of
metal-gemm-upstream-rebase with no other artifacts pre-applied.

**Runtime prereq:** tilelang_metal_fp8 (the storage-only FP8 codegen patch)
must be in the runtime stack — without it the FP8 dtype lowering is missing
the __tvm_fp8_e4m3_to_half / __tvm_fp8_e5m2_to_half decode helpers that
the per-element T.cast calls expand to. The mixed-dtype companion patch
(tilelang_gemm_mixed_dtype) is also a runtime prereq because it is the
artifact that maps GemmInst.Scalar to GemmMetalScalar on Metal targets;
on jorgecurious by itself, GemmInst.Scalar still maps to the CPU
GemmScalar class. The routing decision in this patch is the load-bearing
change — the runtime mapping comes from the rest of the stack.

## What changed in the rebase (vs. the apple-head version)

The original artifact was authored against the apple-head branch, which
already contained PRs #2118 + the local mixed-dtype patch. PR #2121's
dispatcher refactor in tile-ai/tilelang:main changed the shape we attach
to:

* **Dispatcher refactor (PR #2121):** the _select_gemm_instruction Metal
  branch was reduced from a multi-condition cascade
  (_has_mixed_input_dtype → scalar; _has_fp8_input_dtype + supported →
  METAL_SIMDGROUP/FP8 staging; otherwise METAL_SIMDGROUP) to a single
  return GemmInst.METAL_SIMDGROUP line, and the _has_mixed_input_dtype
  helper plus GemmMetalScalar / GemmMetalFP8 impl classes were
  removed from the post-refactor dispatcher in favour of being landed via
  later patches. We adapted by introducing _has_fp8_input_dtype as a
  new (rather than companion) helper directly on the post-refactor
  dispatcher and rewiring the Metal branch to if
  self._has_fp8_input_dtype(): return GemmInst.Scalar. The semantic
  routing decision is unchanged: FP8 inputs on Metal → scalar fallback.

* **metal_fragment_to_simdgroup simplification:** PR #2121 also
  simplified this transform (removed _extract_buffer_from_region,
  removed the simd_accum_vars / scalar_accum_vars split, switched
  from buffer-level inspection to a single accum_vars set built from
  _extract_buffer_var_from_region). We re-introduce
  _extract_buffer_from_region (the buffer-returning sibling) and
  _is_fp8_dtype, then early-return inside the visitor when an FP8
  input is detected so the C accumulator stays in local.fragment.

No other file is touched.

## Round-trip status


git apply               docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch  -> OK
git apply --reverse     docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch  -> OK


against jorgecurious/tilelang:metal-gemm-upstream-rebase at HEAD
971c17b6b2505a57f97f8a6ba385d659c0f1d051 ("fix(metal): harden simdgroup
store lowering"). The TVM submodule (3rdparty/tvm @
0e15b274bce8b46f971abf5ac390e844aa6acee5) is initialised but not
required by this patch since the change is pure Python.

## Apple FP8 reality (May 2026)

| GPU family            | Native FP8 ALU        | Native FP8 simdgroup matmul |
| --------------------- | --------------------- | --------------------------- |
| M1–M3 (Apple7–Apple8) | No                    | No                          |
| M4 / M4 Max (Apple9)  | No                    | No                          |
| M5 NAX (Apple10)      | No (FP16 / INT8 only) | No                          |
| MSL 4.0 / 4.1 / 5.0   | No float8 scalar type | n/a                         |

Therefore any FP8 GEMM on Metal must dequant in software. The correctness
fallback mirrors the audiohacking
[fp8-mps-metal](https://github.com/audiohacking/fp8-mps-metal)
fp8_scaled_matmul_kernel: per-element decode of FP8 to half / float
inside the inner loop, accumulate in float32, store. No
simdgroup_matrix intrinsics involved.

## Patch design

Two pure-Python changes; no C++ rebuild required.

### Layer 1 — codegen-level FP8 GEMM stub

**Not needed** for the simple (SS-with-fragment-C) case once the runtime
stack is in place: the TileLang scalar fallback GemmMetalScalar (PR
#2118 / tilelang_gemm_mixed_dtype) already emits per-element reads
with T.cast(value, accum_dtype) for both A and B operands. With the
FP8 storage-only patch (tilelang_metal_fp8) also applied, those
T.cast calls expand at codegen time to __tvm_fp8_e4m3_to_half(...)
/ __tvm_fp8_e5m2_to_half(...) helper calls in MSL. The resulting
kernel is the audiohacking software dequant-and-accumulate pattern.

The only missing wiring on top of that stack is **dispatcher routing**
(Layer 2) — the FP8 case must take the scalar path, not the simdgroup
path.

### Layer 2 — dispatcher routing

Two pure-Python changes:

1. **tilelang/tileop/gemm/__init__.py** — extend
   Gemm._select_gemm_instruction to route FP8 inputs through
   GemmInst.Scalar (which the runtime stack's
   _get_implementation_class then maps to GemmMetalScalar on the
   Metal target). Adds a new helper _has_fp8_input_dtype() that
   detects FP8 by string-prefix match on buffer.dtype (the only
   reliable cross-version signal; tvm.DataType.is_float8 doesn't
   exist as a Python attribute on the vendored TileLang/tvm
   0e15b274b).

2. **tilelang/transform/metal_fragment_to_simdgroup.py** — extend the
   accumulator-rewrite collection to early-return on FP8 inputs. The
   pass converts local.fragment accumulators to metal.simdgroup for
   GEMMs that will use the simdgroup_matrix intrinsic. FP8 GEMMs need
   to be excluded so the C accumulator stays in local.fragment (thread
   float[N]) instead of metal.simdgroup (which rejects FP8 — but C
   is fp32 here, so the accumulator promotion would otherwise still
   happen and trigger downstream simdgroup load/store calls that we
   don't want).

This is the natural sibling of the mixed-dtype patch
(docs/upstream/tilelang_gemm_mixed_dtype/): that patch handles
Q@Kt -> S, S@V chains; this one handles FP8 in either operand.

## Variants exercised (against the apple-head runtime stack)

| A dtype     | B dtype     | Lowering       | Helpers emitted          |
| ----------- | ----------- | -------------- | ------------------------ |
| float8_e4m3 | float8_e4m3 | OK             | __tvm_fp8_e4m3_to_half |
| float8_e5m2 | float8_e5m2 | OK             | __tvm_fp8_e5m2_to_half |
| float8_e4m3 | float8_e5m2 | OK (mixed FP8) | both e4m3 + e5m2 helpers |

Sample emitted MSL (the inner loop):

msl
for (int i_1 = 0; i_1 < 32; ++i_1) {
  for (int j_1 = 0; j_1 < 32; ++j_1) {
    for (int k = 0; k < 64; ++k) {
      float a_val = ((float)(__tvm_fp8_e4m3_to_half(A_shared[((i_1 * 64) + k)])));
      float b_val = ((float)(__tvm_fp8_e4m3_to_half(B_shared[((k * 32) + j_1)])));
      C_local[((i_1 * 32) + j_1)] = (C_local[((i_1 * 32) + j_1)] + (a_val * b_val));
    }
  }
}


This is the audiohacking pattern almost verbatim: load uchar, decode to
half, promote to float, multiply, accumulate. No
simdgroup_multiply_accumulate. No FP8 simdgroup load.

## Diff stat


 tilelang/tileop/gemm/__init__.py                  | 54 +++++++++++++++
 tilelang/transform/metal_fragment_to_simdgroup.py | 58 +++++++++++++++++
 2 files changed, 112 insertions(+)


(197 lines total including the format-patch mail header.)

## Why pure Python (no codegen change needed)

The FP8 storage-only patch's VisitExpr_(CastNode) in codegen_metal.cc
already lowers scalar T.cast(fp8 -> half) into
__tvm_fp8_e4m3_to_half(...) calls. The TileLang scalar gemm prim_func
emits exactly that scalar cast for each loaded operand. So the entire
FP8 GEMM body becomes a software dequant-multiply-accumulate loop with
zero new codegen helpers. The prelude (the helper functions) is already
injected by the storage-only patch's Finish() override when the
enable_fp8_ flag is set during PrintType / VisitExpr_(CastNode).

The only thing the pre-existing infrastructure was missing was a
**routing decision**: when do we use simdgroup MMA vs. scalar fallback?
That's the dispatcher's job and it's pure Python.

## Performance note

This scalar FP8 fallback is not performance-competitive with the
existing FP16 Metal path. On a 128x64x128 probe the fallback was about
1086x slower than the FP16 simdgroup baseline (12.78ms vs 0.0118ms).
Treat it as a correctness and lowering unblocker for tiny-K or
diagnostic paths, not as a large-GEMM optimisation. The right shape on
Metal for large FP8 GEMMs is to:

1. Pre-dequantise FP8 to FP16 in a fused load kernel (using packed
   uchar4 / uint loads and grouped conversion to half4 so each FP8
   byte is decoded once per tile), and
2. Run the actual GEMM in FP16 with the simdgroup path.

This is a possible production staging path, not what the checked audiohacking
`fp8_scaled_vecmat_kernel` does. That kernel uses byte FP8 loads, scalar
bit-extraction decode, 4-way unroll, and `simd_sum`.

## Files

- 0001-metal-fp8-gemm-software-path.patch — the dispatcher patch
- README.md — this document

## How to apply

bash
cd /tmp/upstream_pr_check
git clone --branch metal-gemm-upstream-rebase \
  https://github.com/jorgecurious/tilelang.git tilelang_fp8gemm_rebase
cd tilelang_fp8gemm_rebase
git submodule update --init --depth 1 3rdparty/tvm
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch


Expected: clean apply, exit 0. Round-trip with git apply --reverse
also passes. For end-to-end runtime validation also apply the
tilelang_metal_fp8 and tilelang_gemm_mixed_dtype artifacts.

## Attribution

The audiohacking
[fp8-mps-metal](https://github.com/audiohacking/fp8-mps-metal) project
provided the reference MSL pattern (fp8_scaled_matmul_kernel). Their
licence (MIT) and the inline IEEE-754 decode helpers from their
fp8_matmul.metal informed both the storage-only codegen patch
(docs/upstream/tilelang_metal_fp8/) and this dispatcher routing patch.
