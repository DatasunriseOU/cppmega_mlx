# Vector FP8 cast lowering for Metal codegen

## Packaging note

The local worktree probe succeeded, but the stored patch artifact currently
needs regeneration before PR filing:

```bash
cd /Volumes/external/sources/cppmega.mlx
git apply --check docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
# error: corrupt patch at line 73
```

Treat the design and proof below as the intended contribution, not as an
applyable patch receipt, until `0001-metal-fp8-vector-cast.patch` is
re-exported from the TileLang branch and rechecked.

## Blocker

`docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch`
(Agent C's patch) wired up scalar FP8 casts on Metal but raised
`LOG(FATAL)` for any cast where lanes > 1:

```
[14:46:13] /tmp/tilelang_apple_head/tilelang/src/target/codegen_metal.cc:673:
  Fatal: Vector FP8 casts (lanes=4) are not yet supported by Metal
  storage-only FP8 emulation; scalarise the cast or extend codegen_metal.cc.
```

This was a hard blocker for any TileLang kernel that lifts FP8 dequant
through `T.Vectorized` — including vec-load FP8 weights into a half tile
prior to a `T.gemm`. The kernel can't be lowered, so TileLang on Metal
falls back to scalar FP8 even when the IR clearly vectorised the load.

## Design

For lanes 2/3/4 we recognise the vector cast and route through a per-lane
constructor expression that calls the existing scalar helpers from
Agent C's prelude:

```msl
inline half4 __tvm_fp8_e4m3_to_half_v4(uchar4 x) {
    return half4(__tvm_fp8_e4m3_to_half(x.x),
                 __tvm_fp8_e4m3_to_half(x.y),
                 __tvm_fp8_e4m3_to_half(x.z),
                 __tvm_fp8_e4m3_to_half(x.w));
}
```

Mirrors are emitted for `_v2` and `_v3`, plus the reverse direction
(`half -> fp8`) and the `e5m2` variant. The compiler is free to scalarise
back into 4 individual calls; the goal is to preserve the IR-level vector
type so subsequent passes (vectorize / fragment-to-simdgroup) keep their
vector loads / stores and the downstream MSL is `uchar4`-typed instead of
`uchar` arrays.

For lanes 8 / 16 (which print as `uint2` / `uint4` packed storage), the
cast still raises a clear FATAL — those widths need an out-pointer ABI
because they don't fit naturally into a single returned vector. The
audiohacking kernels don't use them either; the path through scalar
casts upstream is fine.

### Splicing strategy

Two flags now gate the prelude:
- `enable_fp8_` (existing) — emits the scalar helpers
- `enable_fp8_vector_` (new) — emits the vector helpers (which call the
  scalar ones; depends on `enable_fp8_`)

`Finish()` now splices both preludes when `enable_fp8_vector_` is set.

## Audiohacking attribution

The vector helpers themselves are an inline-trivial wrapper around the
scalar helpers from Agent C's patch (which derive from the OFP8 v1.0
spec). No code is vendored from
`https://github.com/audiohacking/fp8-mps-metal` for this patch — but the
`fp8_scaled_matmul` kernel in that repo is the canonical Apple Silicon
FP8 loadout and motivates the requirement to keep vector FP8 casts cheap.

## Diff stat

```
4 files changed, 158 insertions(+), 4 deletions(-)
  src/target/codegen_metal.cc                       +94 / -2
  src/target/codegen_metal.h                        +9  / -1
  3rdparty/tvm/src/target/source/codegen_metal.cc   +50 / -3
  3rdparty/tvm/src/target/source/codegen_metal.h    +9  / -1  (mirror)
```

## Test results

### Artifact check — fresh TileLang clone

Current artifact status is negative and repeatable:

```bash
git -C /tmp/cppmega_lane6_tilelang rev-parse HEAD
# 2eec5f0109125b46cd71091eeb9c2ad573d6e3d1
git -C /tmp/cppmega_lane6_tilelang apply --check \
  /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
# error: corrupt patch at line 73
```

The corruption is structural, not a base-revision mismatch: the first file
diff reaches the next `diff --git` header without a hunk header. Regenerate
the patch from the TileLang branch before running any performance claim or
PR submission against this artifact.

### Direct probe — `/tmp/test_fp8_vector_cast.py`

```
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
```

Pre-patch the same probe raised `InternalError: Vector FP8 casts (lanes=4)
are not yet supported`.

### Tilelang upstream Metal codegen suite

`testing/python/metal/test_metal_codegen_linux.py`:

| State                        | Pass / Fail |
|------------------------------|-------------|
| Pre-patch (Agent C only)     | 8 pass / 4 fail |
| Post-patch (F-1 applied)     | 9 pass / 3 fail |
| Net                          | +1 pass     |

The flipped test is `test_t_gemm_metal_codegen_pipelined_float32`. It was
already flaky with respect to recent SimdGroup changes; preserving the
vector type at the IR level for FP8 happened to avoid one of the
related ICHECKs as a side effect. The 3 remaining failures are
pre-existing and unrelated to FP8 (`metal.simdgroup` vector dtype check at
`src/target/codegen_metal.cc:502`).

### cppmega.mlx tilelang test suite

```
$ .venv/bin/python -m pytest tests/test_tilelang_*.py -q --no-header
134 passed, 80 warnings in 1.99s
```

No regressions.

## Performance / profiler status

No upstream performance claim is currently valid from the stored artifact.
Because `0001-metal-fp8-vector-cast.patch` cannot be applied, there is no
fresh TileLang-generated Metal kernel to profile from this directory.

Even after regeneration, the helper design is a codegen unblocker rather than
a guaranteed speedup. The emitted `half2` / `half3` / `half4` helpers preserve
`ucharN` / vector cast syntax in MSL, but each helper still calls the scalar
FP8 conversion per lane. The Apple compiler may lower that back to scalar
helper calls, so the only defensible performance claim after regeneration is:
"the vectorized TileLang program now lowers and can be profiled." A speedup
requires generated MSL plus profiler evidence showing fewer scalarized loads /
stores or lower cast overhead.

### Next optimization plan

1. Regenerate `0001-metal-fp8-vector-cast.patch` from a clean TileLang branch
   and add a hunk-integrity gate (`git apply --check`) before PR filing.
2. Add a docs-scoped vector-cast probe artifact that records the generated MSL
   before and after the patch, specifically checking for `uchar4` load/cast
   syntax and the vector helper prelude.
3. Run `xcrun xctrace record --template 'Metal System Trace'` on a regenerated
   TileLang kernel, then export the trace TOC and any shader/counter tables
   that are available. The local template exists, but its default run records
   `Counter Set: (null)` / `Shader Timeline: Disabled`, so a trace file alone
   is not enough for an optimization claim.
4. If the compiler still scalarizes the helpers, the next code change should
   avoid per-lane helper calls in the hot path: load packed bytes as `uchar4`,
   unpack with integer bit operations inside the K-loop, and only materialize
   `half4`/`float4` when the following operation can consume a vector.

## Upstream-PR readiness

**For `tile-ai/tilelang`**: The design is a clean follow-up to the
storage-only patch, but the stored artifact is **not PR-ready** until it is
regenerated and `git apply --check` passes. Do not submit the current
`0001-metal-fp8-vector-cast.patch`.

**For `apache/tvm`**: The mirror change in `3rdparty/tvm/src/target/source/`
should be included when regenerating from a clean branch, because apache/tvm's
own `codegen_metal.cc` needs the same vector helper surface. If apache/tvm
doesn't yet have the storage-only patch, both halves should be combined into a
single TVM PR with the storage-only patch as the base.

**Splittable after regeneration**: Yes. The TVM half can land first (carries
the prelude + vector helpers), the TileLang half second (uses the same helper
names). The current corrupt artifact should only be treated as design notes
plus local probe evidence.

## How to apply

```bash
cd /tmp/tilelang_apple_head/tilelang
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
cd build && ninja -j$(sysctl -n hw.ncpu)
```

Already applied in `/tmp/tilelang_apple_head/tilelang` for the cppmega.mlx
editable install, but regenerate the patch file first if applying from the
artifact in this directory.
