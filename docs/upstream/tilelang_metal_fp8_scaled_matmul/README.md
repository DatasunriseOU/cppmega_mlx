# T.fp8_scaled_matmul intrinsic (frontend stub)

## Blocker

The vendor agent (`a469a838bd6a1ba86`) is integrating
`audiohacking/fp8-mps-metal`'s `fp8_scaled_matmul_kernel` and
`fp8_scaled_vecmat_kernel` as `mx.fast.metal_kernel` wrappers in
`cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py` (out of scope for this
change). That gives a working FP8 GEMM on Metal, but it lives outside
the TileLang DSL — users who write TileLang prim_funcs cannot say
`T.fp8_scaled_matmul(...)` and have it compile.

This patch adds the surface: a single `T.fp8_scaled_matmul(A_fp8,
A_scale, B_fp8, B_scale, C_out)` API that mirrors the audiohacking
kernel signature.

## Design

This is a **frontend-only stub** — pure Python, no TIR op registered, no
scheduler changes, no C++ rebuild required. It dispatches at parse-time
based on the active `tvm.target.Target`:

| Target          | Behaviour                                          |
|-----------------|----------------------------------------------------|
| `metal`         | Raise `NotImplementedError` with structured redirect message pointing at `cppmega_mlx.nn._tilelang.fp8_msl_kernels`. |
| `cuda` / `rocm` | Emit a `tir.call_intrin` placeholder `tl.fp8_scaled_matmul_fallback`. Full lowering = follow-up patch. |
| `cpu` / other   | Same fallback emission. |

### Why a stub for Metal

The full Metal lowering needs:

1. **Vector FP8 cast lowering** — done in
   `docs/upstream/tilelang_metal_fp8_vector/`.
2. **Scaled-gemm scheduler pass** that fuses per-load scale
   multiplication into the K-loop. Mirrors what the audiohacking MSL
   kernel does at line 121 of `fp8_matmul.metal`:

   ```msl
   for (; k < K4; k += 4) {
       float a0 = fp8_e4m3fn_to_float(A[a_idx]);
       // ... 3 more lanes ...
       float b0 = fp8_e4m3fn_to_float(B[b_idx]);
       // ... 3 more lanes ...
       sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
   }
   sum *= sa * sb;  // per-tensor scale broadcast
   ```

3. **GemmMetalScalar extension** for FP8 operands — PR #2118 brought
   `GemmMetalScalar` for fp16/fp32 mixed; FP8 needs the dtype check
   relaxed and the dequant cast inserted at the inner-loop element
   read.

(2) and (3) are sizeable scheduler patches that exceed this PR's budget.
The stub is honest about that: users get a clear error and a precise
redirect to the production-ready MSL kernel.

### Why a stub for non-Metal too

The CUDA path is in nominally OK shape (PRs #202 / #1600 added
`tcgen05_gemm` blockscaled FP8 gemm), but the TileLang scheduler
dispatch from a generic `T.fp8_scaled_matmul` to those backend ops is
non-trivial — the existing `T.gemm` dispatcher in
`tilelang/tileop/gemm/__init__.py` selects between `gemm_cute`,
`gemm_tcgen05`, `gemm_metal_scalar` based on dtype + arch. Adding a
new entry point for "scaled FP8 matmul" duplicates the dispatcher.
Better: leave the stub here and either (a) lower into the existing
`gemm_tcgen05_blockscaled` op when the dtypes match, or (b) tell users
to call the existing op directly. Both options are documented in the
docstring.

## audiohacking attribution

The audiohacking/fp8-mps-metal MSL kernel is **not vendored** by this
patch — only the API signature is mirrored. The actual MSL ships
through `cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py` (vendor agent
work, separate PR), which carries the audiohacking source under its
Apache 2.0 license.

The redirect message in this stub references
`audiohacking/fp8-mps-metal` explicitly so users hit the correct
attribution chain when they follow the error.

## Diff stat

```
2 files changed, 181 insertions(+)
  tilelang/language/__init__.py   +1
  tilelang/language/fp8_op.py     +180  (new file)
```

## Test results

### Direct probe — `/tmp/test_fp8_scaled_matmul.py`

```
=== Phase 1: import surface ===
PASS: T.fp8_scaled_matmul resolvable
    docstring (first line): Scaled FP8 matmul intrinsic.

=== Phase 2: Metal target redirect ===
PASS: NotImplementedError raised with redirect:
     T.fp8_scaled_matmul is not yet lowered through the TileLang Metal scheduler.
     Apple Silicon (M1-M4) has no native FP8 ALU; the audiohacking pattern
     requires a scaled-gemm pass that fuses per-load scale with the K-loop,
     which is not yet wired through GemmMetalScalar.

     Use the ready-to-go MSL kernel via mx.fast.metal_kernel:
         from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (
             fp8_scaled_matmul as _fp8_scaled_matmul,
         )

=== Phase 3: fallback path (cuda target) ===
PASS: fallback returns Call

All phases OK.
```

The redirect message contains:
- `mx.fast.metal_kernel` (the MLX FFI to use)
- `cppmega_mlx.nn._tilelang.fp8_msl_kernels` (the module path)
- `audiohacking` (attribution)

### cppmega.mlx tilelang test suite

```
$ .venv/bin/python -m pytest tests/test_tilelang_*.py -q --no-header
134 passed, 80 warnings in 1.99s
```

No regressions (the import surface adds one new symbol, no existing
behaviour changes).

## Upstream-PR readiness

**For `tile-ai/tilelang`**: Submittable today as a frontend-only PR. The
follow-up to actually lower this through the scheduler can land
separately. The stub becomes a TODO marker in the codebase that's
honest about the gap.

**For `apache/tvm`**: Not applicable — this is a TileLang-side change
only. No TVM TIR op registered (yet); the placeholder
`tl.fp8_scaled_matmul_fallback` is consumed only inside TileLang's
Python dispatcher.

**Splittable**: Yes — three follow-ups to fully replace the stub with a
real lowering:

1. **TIR op registration** in `src/op/builtin.cc` (small C++ change,
   needs rebuild).
2. **Metal scheduler pass** that fuses per-load scale into the
   GemmMetalScalar inner loop. Largest part. Needs to interact with
   the existing `metal_fragment_to_simdgroup.py` rewrite.
3. **CUDA scheduler dispatch** to existing `gemm_tcgen05_blockscaled`
   on Blackwell. Smaller.

## How to apply

```bash
cd /tmp/tilelang_apple_head/tilelang
# Pre-req: Agent C's storage-only patch and the F-1 vector cast patch
git apply docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
git apply docs/upstream/tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch
```

Already applied in `/tmp/tilelang_apple_head/tilelang` for the
cppmega.mlx editable install.
