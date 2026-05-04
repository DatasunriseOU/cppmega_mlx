# Path C port: tilelang_sparse_mla/topk_selector.py

This document records the Apple Metal top-k selector ports for cppmega's DSA
sparse-MLA index selection. Path B is the hand-written MSL kernel launched via
`mx.fast.metal_kernel`. Path C is now a real TileLang DSL `@T.prim_func` lowered
to Metal and then launched through `mx.fast.metal_kernel` with the
TileLang-generated kernel body.

## Source attribution

| field            | value                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| upstream path    | cppmega/megatron/tilelang_sparse_mla/topk_selector.py (gb10 mirror)                                  |
| upstream lineage | NVIDIA Megatron-LM PR #3674 ("DSA thd" branch), in turn from tile-ai/tilelang/examples/deepseek_v32/ |
| license          | Apache 2.0 / BSD-3-Clause (matches Megatron-LM headers)                                              |
| destination      | cppmega_mlx/nn/_tilelang/topk_selector.py                                                            |
| tests            | tests/test_tilelang_topk.py and /private/tmp/tilelang_apple_head/tilelang/testing/python/metal/test_metal_topk_selector.py |
| bench            | scripts/bench_tilelang_topk.py -> bench/tilelang_ports/topk_selector.json                            |

## Source kernel summary

`topk_selector(input, starts, ends, topk)` returns, per batch row, the `topk`
indices into `input[bx, starts[bx]:ends[bx]]` of the largest values. The CUDA
implementation runs a two-stage radix-select inside one threadgroup of
`BLOCK_SIZE = 1024` threads:

1. Stage 1: 8-bit histogram over the high byte of the sign-flipped fp16/fp32
   representation, prefix-summed via Hillis-Steele over 256 threads, followed
   by tail collection.
2. Stage 2: up to 4 rounds of byte-deeper refinement on the tail candidates,
   finalizing the output once `l_new_topk == 0`.

The upstream schedule depends on `T.alloc_shared`, atomics, partial barriers,
and fp bit reinterpretation. That CUDA schedule is still not the profitable
Metal schedule, so cppmega carries a Metal-specific Path B and Path C schedule.

## Path B: direct MSL

`topk_selector_metal(...)` emits a hand-written MSL body through
`mx.fast.metal_kernel`. Each row maps to one Metal threadgroup. Threads scan a
strided slice, keep a private sorted top-K list, then merge per-thread lists
through static `threadgroup` buffers.

Path B remains the first fallback for hosts without TileLang or for unsupported
Path C dtypes/shapes.

## Path C: TileLang DSL -> Metal

`topk_selector_tilelang(...)` builds a shape-specialized TileLang PrimFunc with
the same one-threadgroup-per-row algorithm as Path B:

- private `T.alloc_local((K,), float32/int32)` top-K lists per lane
- static `T.alloc_shared((threads, K), ...)` pair buffers
- `T.sync_threads()` after local writes and after every tree-merge round
- no dynamic shared scope, no atomics, and no repeated `T.reduce_max` passes
- fp32 compare path; fp16 inputs are read directly and bf16 inputs are promoted
  to fp32 before dispatch

The MSL emitted by TileLang is split by `_msl_transform.lower_tilelang_to_msl_inline(...)`.
The kernel body is inlined into an MLX `mx.fast.metal_kernel` wrapper so the
threadgroup allocations remain legal kernel-scope MSL.

Threadgroup selection is intentionally simple and deterministic:

- `K <= 32`: prefer 32 threads.
- `K >= 64`: prefer 64 threads.
- The final value is capped by `threads * K * 8 <= 32 KiB`; for `K=256` this
  selects 16 threads.

A local thread sweep on M4 Max and the checked receipt show the important
cases:

| shape        | best Path C threads | observed C/B |
| ------------ | ------------------- | ------------ |
| B=1,T=64,K=8 | 32                  | 0.903x       |
| B=1,T=512,K=32 | 32               | 0.507x       |
| B=4,T=2048,K=64 | 64              | 0.659x       |
| B=4,T=4096,K=256 | 16             | 0.566x       |

The exact ratios move with warmup and MLX compiler cache state, so the checked
benchmark script is the source of truth.

## Runtime contract

`topk_selector_reference(scores, k, *, starts=None, ends=None)` is the pure-MLX
oracle. It uses `mx.argpartition(-scores, k, axis=-1)[..., :k]` and an optional
`[starts, ends)` mask, with `-1` sentinel fill when a row has fewer than `k`
valid columns. Output dtype is `mx.int32`.

`topk_selector_metal(...)` is Path B and returns `None` if the direct-MSL kernel
cannot dispatch.

`topk_selector_tilelang(...)` is Path C and returns `None` if TileLang/Metal
cannot dispatch. Public `topk_selector(..., backend="tilelang")` and
`backend="path_c"` fail closed if Path C is unavailable. Public
`backend="auto"` tries Path C first only for unmasked rows with strict-green
bench receipts. Masked calls and unreceipted shapes route Path B first, then the
pure-MLX fallback.

Top-k indices are non-differentiable. Tests compare set membership for the
largest `k` values, because MLX argpartition, the upstream radix kernel, Path B,
and Path C do not share a stable tie-breaking contract.

## Supported Path C shapes and dtypes

Path C is shape-specialized and supports the required 2D `(B, T)` selector
contract with static `K` where `1 <= K <= T` and `threads * K * 8 <= 32 KiB`.
Current acceptance and benchmark coverage includes:

- `(B=2, T=64, K=4)`, float32, with and without `[starts, ends)`
- `(B=1, T=64, K=8)`, float32
- `(B=1, T=512, K=32)`, float32
- `(B=1, T=2048, K=64)`, float32
- `(B=4, T=512, K=64)`, float32, float16, bfloat16
- `(B=4, T=2048, K=64)`, float32, float16, bfloat16
- `(B=1, T=4096, K=256)`, float32
- `(B=4, T=4096, K=256)`, float32
- short and empty intervals with `-1` sentinel fill

Input dtype support:

- `mx.float32`: native Path C compare path
- `mx.float16`: native load, fp32 internal compare
- `mx.bfloat16`: promoted to fp32 before dispatch

Unsupported dtypes fail closed and should use `backend="mlx"` or Path B/auto if
eligible.

## Tests

`tests/test_tilelang_topk.py` covers:

- pure-MLX reference parity vs a NumPy oracle for B in `{1, 4}`, T in
  `{64, 512, 2048}`, and k in `{1, 8, 32}`
- dtype in `{float32, float16, bfloat16}` -> output dtype is int32
- edge cases: `k == 1`, `k == seq_len`, `[starts, ends)` masking, and short or
  empty intervals
- Path B and Path C status helpers
- direct-MSL Path B parity for the main sweep and acceptance shapes
- TileLang DSL Path C parity against both the reference and Path B for the
  required acceptance shapes

The TileLang tree also carries
`/private/tmp/tilelang_apple_head/tilelang/testing/python/metal/test_metal_topk_selector.py`.
That probe lowers the standalone Path C PrimFunc, checks the emitted MSL for
static `threadgroup` buffers and `threadgroup_barrier`, and runs MPS parity
through `tilelang.compile`.

## Bench

`scripts/bench_tilelang_topk.py` compares:

- `argpartition`: the pure-MLX reference selector.
- `argsort_slice`: `mx.argsort(-scores)[..., :k]`.
- `topk_take_along`: argpartition plus value materialization.
- `path_b_msl`: hand-written MSL Path B.
- `path_c_tilelang`: TileLang DSL Path C.

Smoke output on M4 Max, `warmup=10`, `iters=50`, Python `3.13.12`,
`mlx 0.31.1`, `mlx-metal 0.31.1`, `tilelang 0.1.9+gita69d6df7`,
`apache-tvm-ffi 0.1.11rc2`, `numpy 2.4.4`:

```text
B    T       k      dtype      argpart_ms    argsort_ms    fused_ms      path_b_ms     path_c_ms     C/B
1    64      8      float32    0.1869        0.1560        0.1761        0.1704        0.1539        0.903
1    512     32     float32    0.1738        0.1551        0.1598        0.3724        0.1888        0.507
1    2048    64     float32    0.1605        0.1509        0.1715        0.5249        0.3363        0.641
4    2048    64     float32    0.1740        0.1559        0.1758        0.5300        0.3495        0.659
4    2048    64     float16    0.1780        0.1589        0.1772        0.5465        0.3531        0.646
4    2048    64     bfloat16   0.1781        0.1671        0.1875        0.6085        0.3893        0.640
4    4096    256    float32    0.1926        0.1664        0.1984        8.7999        4.9800        0.566
```

The checked receipt keeps Path C no worse than Path B on the measured required
shapes, including `K=256`, and correct for the required dtype/interval
contract. `backend="auto"` uses that receipt as a routing gate: it is Path-C
first only for the strict-green unmasked rows in
`bench/tilelang_ports/topk_selector.json`. Explicit `backend="tilelang"` remains
available and fail-closed for correctness/probe runs outside that profitable
envelope.

## Remaining blocker

This lane no longer has a correctness, feature, or measured Path-B-parity
blocker for the receipt-backed standalone top-k selector shapes. Remaining risk
is schedule generality outside the checked envelope: the TileLang DSL schedule
still lowers to scalar per-lane sorted insertions plus full `threads * K`
shared-memory list merges. It does not expose a Metal simdgroup top-k/reduce
primitive, a custom comparator network intrinsic, or scheduler glue for every
possible masked or unmeasured top-k regime, so AUTO keeps those calls Path-B
first and unsupported shapes/dtypes still fail closed to Path B or the MLX
reference.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider tests/test_tilelang_topk.py -q --tb=short
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pyright cppmega_mlx/nn/_tilelang/topk_selector.py tests/test_tilelang_topk.py scripts/bench_tilelang_topk.py
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python scripts/bench_tilelang_topk.py --warmup 3 --iters 10 --strict --no-output-file
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python scripts/bench_tilelang_topk.py --json
```

The bench writes `bench/tilelang_ports/topk_selector.json` unless
`--no-output-file` is used.
