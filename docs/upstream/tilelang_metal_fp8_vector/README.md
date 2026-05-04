# Vector FP8 cast lowering for Metal codegen

## Filed upstream PRs (2026-05-04)

The combined `0001-metal-fp8-vector-cast.patch` (345 lines) was split into
two companion PRs because it touches both the TileLang supermodule and its
vendored TileLang/tvm submodule:

| Half | Repo | PR | Patch file |
|---|---|---|---|
| Supermodule (`src/target/codegen_metal.{cc,h}`) | `tile-ai/tilelang` | https://github.com/tile-ai/tilelang/pull/2145 | `0001-tilelang-metal-fp8-vector-cast.patch` (148 lines) |
| Submodule (`src/target/source/codegen_metal.{cc,h}`) | `TileLang/tvm` | https://github.com/tile-ai/tvm/pull/39 | `0002-tvm-metal-fp8-vector-cast.patch` (151 lines) |

Both branches: `apstenku123:cppmega/metal-fp8-vector-cast`.

### Dependency chain

Each PR's branch includes 2 commits stacked: the
`tilelang_metal_fp8` storage-only prereq first, then the vector-cast
patch on top. The supermodule branch additionally stacks on PR #2130
(`metal-gemm-upstream-rebase` @ `971c17b`).

```
[Apple Metal landing chain] #1869 -> #2118 -> #2121 -> #2130 (open)
                                                        |
                                                        v
                  [tilelang_metal_fp8 storage-only PR pair (parallel)]
                                                        |
                                                        v
                  [tilelang_metal_fp8_vector PR pair, this directory]
                  - tile-ai/tilelang #2145 (supermodule)
                  - TileLang/tvm #39       (submodule)
```

When the storage-only PR pair merges, the prereq commit on each
vector-cast branch should be rebased away. Until then, they are
reviewable as 2-commit stacks.

## Packaging status (pre-split)

Rebased 2026-05-03 onto jorgecurious/tilelang's
`metal-gemm-upstream-rebase` branch (PR #2130, HEAD `971c17b`). Applies
cleanly on top of the `tilelang_metal_fp8` storage-only patch:

```
# clean checkout of jorgecurious/tilelang:metal-gemm-upstream-rebase @ 971c17b
git apply  docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch         -> OK
git apply --check  docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply          docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply --reverse docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
```

### Round-trip verification of the split patches (2026-05-04)

```
# TileLang half: jorgecurious/tilelang @ 971c17b + storage-only prereq applied first
git apply --check    docs/upstream/tilelang_metal_fp8_vector/0001-tilelang-metal-fp8-vector-cast.patch  -> OK
git apply --index    docs/upstream/tilelang_metal_fp8_vector/0001-tilelang-metal-fp8-vector-cast.patch  -> OK
git apply --reverse  docs/upstream/tilelang_metal_fp8_vector/0001-tilelang-metal-fp8-vector-cast.patch  -> OK
git apply --reverse  docs/upstream/tilelang_metal_fp8/0001-tilelang-metal-fp8-storage-only.patch        -> OK (matches base)

# TVM half: TileLang/tvm @ 0e15b274 + storage-only prereq applied first
git apply --check    docs/upstream/tilelang_metal_fp8_vector/0002-tvm-metal-fp8-vector-cast.patch       -> OK
git apply --index    docs/upstream/tilelang_metal_fp8_vector/0002-tvm-metal-fp8-vector-cast.patch       -> OK
git apply --reverse  docs/upstream/tilelang_metal_fp8_vector/0002-tvm-metal-fp8-vector-cast.patch       -> OK
git apply --reverse  docs/upstream/tilelang_metal_fp8/0002-tvm-metal-fp8-storage-only.patch             -> OK (matches base)
```

## Stacking topology

This patch stacks on the following upstream PRs (all already merged into
or open against tile-ai/tilelang's `metal-gemm-upstream-rebase` lane), in
addition to our own storage-only prereq:

1. tile-ai/tilelang **#1869** — initial Apple Metal landing
2. tile-ai/tilelang **#2118** — Metal pipelined stages
3. tile-ai/tilelang **#2121** — CodeGen refactor that reformatted
   `src/target/codegen_metal.cc`
4. tile-ai/tilelang **#2130** — `metal-gemm-upstream-rebase`
   (jorgecurious branch URL:
   https://github.com/jorgecurious/tilelang/tree/metal-gemm-upstream-rebase
   @ HEAD `971c17b`)
5. our `docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch`
   — required prereq for this patch (extends the FP8 prelude added by it)

The patch keeps its two-half nature: it modifies both
`src/target/codegen_metal.{cc,h}` (Tilelang side) and
`3rdparty/tvm/src/target/source/codegen_metal.{cc,h}` (vendored TVM
mirror). When filing upstream, the TVM-mirror half belongs in
apache/tvm; the Tilelang half belongs in tile-ai/tilelang. They can be
landed independently because they only share helper names.

## Drift handled in this rebase

The `tilelang_metal_fp8` prereq adds, on the TVM-mirror side, a
4-line comment block above the `LOG(FATAL)` for the unsupported vector
case (lines like
`// Vector path: not supported by this storage-only patch; ...`).
This rebase replaces the comment block + `LOG(FATAL)` together with the
new vector path body and a tightened `LOG(FATAL)` covering only
lanes outside 2/3/4 (semantically identical to the original patch
aside from the wording tweak). The Tilelang-side context did not have
the extra comment block, so its hunk reuses the same delete shape.

Diff stat after rebase:

```
src/target/codegen_metal.h                       +7 / -0
src/target/codegen_metal.cc                      +96 / -3
3rdparty/tvm/src/target/source/codegen_metal.h   +7 / -0
3rdparty/tvm/src/target/source/codegen_metal.cc  +95 / -7
(345 lines total in patch file, regenerated 2026-05-03)
```

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

## Test results

### Artifact check — fresh jorgecurious rebase clone (2026-05-03 rebase)

```
cd /tmp/upstream_pr_check/tilelang_fp8vec_rebase  # @ 971c17b
git apply           tilelang_metal_fp8/0001-metal-fp8-storage-only.patch        -> OK
git apply --check   tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply           tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
git apply --reverse tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch  -> OK
```

Verified against jorgecurious/tilelang `metal-gemm-upstream-rebase` HEAD
`971c17b` with TVM submodule `0e15b274bce8b46f971abf5ac390e844aa6acee5`
and the `tilelang_metal_fp8` storage-only patch applied as a prereq.

The historical "corrupt patch / line 73 apply failure" against
apple-head was caused by the CodeGen reformat introduced in PR #2121;
this rebase resolves it by rewriting the hunks against the post-#2121
file layout and the post-prereq LOG(FATAL) comment block.

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

**For tile-ai/tilelang**: Clean follow-up to the `tilelang_metal_fp8`
storage-only patch. The artifact passes `git apply --check` and
`git apply --reverse --check` on fresh
`jorgecurious/tilelang:metal-gemm-upstream-rebase` (PR #2130) at HEAD
`971c17b` with `tilelang_metal_fp8` applied as prereq.

**For apache/tvm**: The mirror change in `3rdparty/tvm/src/target/source/`
should be included when filing upstream, because apache/tvm's own
`codegen_metal.cc` needs the same vector helper surface. If apache/tvm
doesn't yet have the storage-only patch, both halves should be combined
into a single TVM PR with the storage-only patch as the base.

**Splittable**: Yes. The TVM half can land first (carries the prelude +
vector helpers), the Tilelang half second (uses the same helper names).

## How to apply

```bash
# clean checkout of jorgecurious/tilelang:metal-gemm-upstream-rebase @ 971c17b
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
cd build && ninja -j$(sysctl -n hw.ncpu)
```

The patch file in this directory is round-trip verified on fresh
`jorgecurious/tilelang:metal-gemm-upstream-rebase` @ `971c17b` with the
`tilelang_metal_fp8` prereq applied first.
