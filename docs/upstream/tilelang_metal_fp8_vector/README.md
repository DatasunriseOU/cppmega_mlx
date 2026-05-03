# Vector FP8 cast lowering for Metal codegen

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

## Upstream-PR readiness

**For `tile-ai/tilelang`**: This patch is a clean follow-up to Agent C's
storage-only patch. The two together are submittable as a single
"Metal FP8 storage-only emulation (scalar + vector)" PR.

**For `apache/tvm`**: The mirror change in `3rdparty/tvm/src/target/source/`
applies to apache/tvm's own `codegen_metal.cc`. If apache/tvm doesn't yet
have Agent C's patch, both halves should be combined into a single TVM PR
with the storage-only patch as the base.

**Splittable**: Yes. The TVM half can land first (carries the prelude +
vector helpers), the TileLang half second (uses the same helper names).

## How to apply

```bash
cd /tmp/tilelang_apple_head/tilelang
git apply docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
cd build && ninja -j$(sysctl -n hw.ncpu)
```

Already applied in `/tmp/tilelang_apple_head/tilelang` for the
cppmega.mlx editable install.
