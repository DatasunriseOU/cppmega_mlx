# Vector FP8 cast lowering for Metal codegen

## Packaging status

Regenerated 2026-05-03; the recorded round-trip receipt was against an
apple-head@7f4a5cb8 checkout with the storage-only FP8 patch (#5) applied:

```
git apply --check  docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply          docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply --reverse docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
```

Stack-apply order (against clean apple-head@7f4a5cb8):

1. tilelang_gemm_mixed_dtype (#3) — independent, applies first
2. tilelang_metal_pipelined  (#4) — independent
3. tilelang_metal_fp8        (#5) — required prereq for #7 (this patch
   extends the FP8 prelude added by #5; both touch the same files in
   src/target/codegen_metal.{cc,h} and 3rdparty/tvm/src/target/source/codegen_metal.{cc,h})
4. tilelang_metal_fp8_vector (#7) — this patch.

Before filing, re-run the same `git apply --check` / `git apply --reverse
--check` sequence on a fresh intended TileLang base. Do not file from this
README alone if the current PR-prep lane still reports a corrupt-patch or
line-73 apply failure; regenerate the patch from the clean checkout first.

## Blocker

docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
(Agent C's patch) wired up scalar FP8 casts on Metal but raised
LOG(FATAL) for any cast where lanes > 1:


  /tmp/tilelang_apple_head/tilelang/src/target/codegen_metal.cc:673:
  Fatal: Vector FP8 casts (lanes=4) are not yet supported by Metal
  storage-only FP8 emulation; scalarise the cast or extend codegen_metal.cc.


This was a hard blocker for any TileLang kernel that lifts FP8 dequant
through T.Vectorized — including vec-load FP8 weights into a half tile
prior to a T.gemm. The kernel can't be lowered, so TileLang on Metal
falls back to scalar FP8 even when the IR clearly vectorised the load.

## Design

For lanes 2/3/4 we recognise the vector cast and route through a per-lane
constructor expression that calls the existing scalar helpers from
Agent C's prelude:

msl
inline half4 __tvm_fp8_e4m3_to_half_v4(uchar4 x) {
    return half4(__tvm_fp8_e4m3_to_half(x.x),
                 __tvm_fp8_e4m3_to_half(x.y),
                 __tvm_fp8_e4m3_to_half(x.z),
                 __tvm_fp8_e4m3_to_half(x.w));
}


Mirrors are emitted for _v2 and _v3, plus the reverse direction
(half -> fp8) and the e5m2 variant. The compiler is free to scalarise
back into 4 individual calls; the goal is to preserve the IR-level vector
type so subsequent passes (vectorize / fragment-to-simdgroup) keep their
vector loads / stores and the downstream MSL is uchar4-typed instead of
uchar arrays.

For lanes 8 / 16 (which print as uint2 / uint4 packed storage), the
cast still raises a clear FATAL — those widths need an out-pointer ABI
because they don't fit naturally into a single returned vector. The
audiohacking kernels don't use them either; the path through scalar
casts upstream is fine.

### Splicing strategy

Two flags now gate the prelude:
- enable_fp8_ (existing) — emits the scalar helpers
- enable_fp8_vector_ (new) — emits the vector helpers (which call the
  scalar ones; depends on enable_fp8_)

Finish() now splices both preludes when enable_fp8_vector_ is set.

## Audiohacking attribution

The vector helpers themselves are an inline-trivial wrapper around the
scalar helpers from Agent C's patch (which derive from the OFP8 v1.0
spec). No code is vendored from
https://github.com/audiohacking/fp8-mps-metal for this patch — but the
fp8_scaled_matmul kernel in that repo is the canonical Apple Silicon
FP8 loadout and motivates the requirement to keep vector FP8 casts cheap.

## Diff stat

```
4 files changed, ~280 insertions(+), 6 deletions(-)
  src/target/codegen_metal.h                       +9 / -1
  src/target/codegen_metal.cc                      +94 / -2
  3rdparty/tvm/src/target/source/codegen_metal.h   +9 / -1
  3rdparty/tvm/src/target/source/codegen_metal.cc  +96 / -2
  (335 lines total in patch file, regenerated 2026-05-03)
```


## Test results

### Artifact check — fresh apple-head clone (2026-05-03 regeneration)

```
cd /tmp/verify_p7
git apply --check  docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply          docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply --reverse docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
```

Verified against apple-head@7f4a5cb8 with TVM submodule
0e15b274bce8b46f971abf5ac390e844aa6acee5 and the storage-only FP8 patch (#5)
applied as a prereq.

### Direct probe — /tmp/test_fp8_vector_cast.py


=== scalar baseline ===
PASS: scalar baseline lowered
scalar helper present

=== vectorized cast (lanes=4) ===
PASS: vectorized cast (lanes=4) lowered
VECTOR CAST PROBE: PASS — vector helper emitted.
  >> inline half __tvm_fp8_e4m3_to_half(uchar x) {
  >> inline half2 __tvm_fp8_e4m3_to_half_v2(uchar2 x) { ... }
  >> inline half3 __tvm_fp8_e4m3_to_half_v3(uchar3 x) { ... }
  >> inline half4 __tvm_fp8_e4m3_to_half_v4(uchar4 x) { ... }
  >> B_fp16_local_cast[0] = __tvm_fp8_e4m3_to_half_v4(A_fp8_local_cast_1[0]);


Pre-patch the same probe raised InternalError: Vector FP8 casts (lanes=4)
are not yet supported.

### Tilelang upstream Metal codegen suite

testing/python/metal/test_metal_codegen_linux.py:

| State                    | Pass / Fail     |
| ------------------------ | --------------- |
| Pre-patch (Agent C only) | 8 pass / 4 fail |
| Post-patch (F-1 applied) | 9 pass / 3 fail |
| Net                      | +1 pass         |

The flipped test is test_t_gemm_metal_codegen_pipelined_float32. It was
already flaky with respect to recent SimdGroup changes; preserving the
vector type at the IR level for FP8 happened to avoid one of the
related ICHECKs as a side effect. The 3 remaining failures are
pre-existing and unrelated to FP8 (metal.simdgroup vector dtype check at
src/target/codegen_metal.cc:502).

### cppmega.mlx tilelang test suite


$ .venv/bin/python -m pytest tests/test_tilelang_*.py -q --no-header
134 passed, 80 warnings in 1.99s


No regressions.

## Performance / profiler status

The helper design is a codegen unblocker rather than a guaranteed speedup. The
emitted half2 / half3 / half4 helpers preserve ucharN / vector cast syntax in
MSL, but each helper still calls the scalar FP8 conversion per lane. The Apple
compiler may lower that back to scalar helper calls, so the only defensible
performance claim is: "the vectorized TileLang program now lowers and can be
profiled." A speedup requires generated MSL plus profiler evidence showing
fewer scalarized loads / stores or lower cast overhead.

### Next optimization plan

1. Add a docs-scoped vector-cast probe artifact that records the generated MSL
   before and after the patch, specifically checking for uchar4 load/cast
   syntax and the vector helper prelude.
2. Run xcrun xctrace record --template 'Metal System Trace' on a regenerated
   TileLang kernel, then export the trace TOC and any shader/counter tables
   that are available. The local template exists, but its default run records
   Counter Set: (null) / Shader Timeline: Disabled, so a trace file alone
   is not enough for an optimization claim.
3. If the compiler still scalarizes the helpers, the next code change should
   avoid per-lane helper calls in the hot path: load packed bytes as uchar4,
   unpack with integer bit operations inside the K-loop, and only materialize
   half4/float4 when the following operation can consume a vector.

## Upstream-PR readiness

**For tile-ai/tilelang**: Clean follow-up to the storage-only patch (#5).
The artifact passes `git apply --check` and `git apply --reverse --check` on
fresh apple-head@7f4a5cb8 with #5 applied as prereq.

**For apache/tvm**: The mirror change in 3rdparty/tvm/src/target/source/
should be included when filing upstream, because apache/tvm's
own codegen_metal.cc needs the same vector helper surface. If apache/tvm
doesn't yet have the storage-only patch, both halves should be combined into a
single TVM PR with the storage-only patch as the base.

**Splittable**: Yes. The TVM half can land first (carries the prelude +
vector helpers), the TileLang half second (uses the same helper names).

## How to apply

```bash
cd /tmp/tilelang_apple_head/tilelang
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
cd build && ninja -j$(sysctl -n hw.ncpu)
```

Already applied in /tmp/tilelang_apple_head/tilelang for the cppmega.mlx
editable install. The patch file in this directory is round-trip verified
on fresh apple-head@7f4a5cb8 + #5 baseline.
