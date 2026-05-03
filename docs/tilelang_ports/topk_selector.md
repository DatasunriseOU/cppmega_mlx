# Path B port: tilelang_sparse_mla/topk_selector.py

This document records the Apple Metal port for the smallest TileLang kernel
cppmega ships, the radix-select top-k that backs DSA sparse-MLA index
selection. TileLang's Metal lowering is still blocked, so this port uses the
Path B bypass: a hand-written MSL kernel launched through
mx.fast.metal_kernel.

## Source attribution

| field            | value                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| upstream path    | cppmega/megatron/tilelang_sparse_mla/topk_selector.py (gb10 mirror)                                  |
| upstream lineage | NVIDIA Megatron-LM PR #3674 ("DSA thd" branch), in turn from tile-ai/tilelang/examples/deepseek_v32/ |
| license          | Apache 2.0 / BSD-3-Clause (matches Megatron-LM headers)                                              |
| destination      | cppmega_mlx/nn/_tilelang/topk_selector.py                                                            |
| tests            | tests/test_tilelang_topk.py                                                                          |
| bench            | scripts/bench_tilelang_topk.py -> bench/tilelang_ports/topk_selector.json                            |

## Source kernel summary

topk_selector(input, starts, ends, topk) returns, per batch row, the
topk indices into input[bx, starts[bx]:ends[bx]] of the largest
values. The CUDA implementation runs a two-stage radix-select inside one
threadgroup of BLOCK_SIZE = 1024 threads:

1. Stage 1 -- 8-bit histogram over the high byte of the sign-flipped
   fp16 representation, prefix-summed via Hillis-Steele over 256 threads,
   followed by a tail-collection step that splits "definitely above"
   indices to the output and tail candidates to shared memory.
2. Stage 2 -- up to 4 rounds of byte-deeper refinement on the tail
   candidates, finalizing the output once l_new_topk == 0.

Resources used:

- T.alloc_shared([257], int32) -- the histogram (RADIX+1 entries to
  leave room for the cumsum prefix).
- T.alloc_shared([2, 4096], int32) -- ping-pong tail-candidate buffer.
- T.alloc_shared([2], int32) -- per-stage tail counts.
- T.alloc_shared([1], int32) -- threshold bin id.
- T.atomic_add(..., return_prev=True) -- emits output positions and tail
  positions in race-free order.
- T.sync_threads(3, RADIX) -- partial barriers covering the first 256
  threads only.
- T.reinterpret(hval, T.uint16) and T.reinterpret(x, T.uint32) --
  fp16/fp32 sign-flip bit fiddling for radix sort.

## Path B transform layer

cppmega_mlx/nn/_tilelang/_path_b_lowering.py vendors the small string-
rewriting helpers that the prototype at /tmp/path_b_msl_mlx/bench_msl_path_b.py
proved on a manual GEMM. The helpers handle three concrete pitfalls:

1. **TileLang emits kernel void <name>(...), MLX expects body only.** The
   regex r"kernel\s+void\s+(?P<name>\w+)\s*\(" plus paren-counting locates
   the signature; the body is preserved verbatim and re-emitted as
   inline void <helper>(...) injected into MLX's header=.
2. **MLX uses const device T* for inputs; TileLang's all-mutable.** The
   helper accepts a const_buffer_names set and rewrites \bdevice\b to
   const device only on those parameters.
3. **TileLang reorders buffer params alphabetically.** The MSL signature is
   parsed to recover the actual buffer order, then build_mlx_body maps
   PrimFunc names back to MLX local names so the call argument list matches
   the emitted helper's parameter order.

These helpers are still useful for kernels that produce MSL through TileLang.
They are dormant for topk_selector because TileLang refuses to lower the source
kernel before any MSL is produced; topk_selector therefore ships a direct-MSL
body instead.

## TileLang Metal lowering status -- blocked

Probed with the cppmega kernel restructured into a T.prim_func and
tilelang.engine.lower.lower(prim, target=Target("metal")), with both the
original constants and a sweep of smaller BLOCK_SIZE/RADIX to localize
the failure:

| BLOCK_SIZE | RADIX | SMEM_INPUT_SIZE | status | reason                                                                                                                                                                                 |
| ---------- | ----- | --------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 256        | 256   | 4096            | FAIL   | LowerTileOp: Loop layout is not injective: Fragment([257] -> [2], replicate: 1, thread: 256, ..., forward_thread: _i % 256, forward_index: [_i // 256], thread_range: I.Range(0, 256)) |
| 256        | 128   | 512             | FAIL   | Fatal: Unknown storage scope shared.dyn                                                                                                                                                |
| 256        | 64    | 512             | FAIL   | Fatal: Unknown storage scope shared.dyn                                                                                                                                                |
| 128        | 256   | 512             | FAIL   | layout-not-injective ([257] -> [3])                                                                                                                                                    |
| 128        | 128   | 512             | FAIL   | layout-not-injective ([129] -> [2])                                                                                                                                                    |
| 128        | 64    | 512             | FAIL   | Fatal: Unknown storage scope shared.dyn                                                                                                                                                |

There are two distinct failures and they compose:

1. **Loop layout not injective.** When the histogram size (RADIX + 1) is
   not a multiple of BLOCK_SIZE, TileLang's LowerTileOp cannot tile the
   trivial T.parallel(257) fill into 256 threads injectively. This bites
   any setting where RADIX == BLOCK_SIZE (the original cppmega config:
   BLOCK_SIZE=1024, RADIX=256 is presumably saved by the partial barrier
   pattern; on the metal target the target descriptor advertises only
   max_num_threads = 256).
2. **shared.dyn storage scope unsupported.** Once the histogram size is
   chosen to dodge the layout check, TileLang gets through legalization but
   the metal codegen rejects the storage scope. T.alloc_shared in
   tilelang 0.1.9 lowers to shared.dyn, and the TVM metal backend bundled
   with tilelang has no handler for it. A minimal probe confirms this:

       @T.prim_func
       def use_shared(A: T.Tensor((256,), 'float32'),
                      B: T.Tensor((256,), 'float32')):
           with T.Kernel(1, threads=256) as bx:
               tx = T.get_thread_binding()
               s = T.alloc_shared([256], 'float32')
               s[tx] = A[tx]; T.sync_threads(); B[tx] = s[tx] * 2.0

       target = tvm.target.Target('metal')
       with target: lower(use_shared, target=target)
       # -> Fatal: Unknown storage scope shared.dyn

This remains a hard TileLang-codegen blocker for the DSA sparse-MLA family:
every kernel in the family relies on threadgroup memory and partial barriers.
For topk_selector, the shipped path takes the second option from the original
decision tree: hand-write MSL and bypass TileLang lowering entirely.

## Runtime contract

cppmega_mlx/nn/_tilelang/topk_selector.topk_selector_reference(scores, k,
*, starts=None, ends=None) is the pure-MLX oracle. It uses
mx.argpartition(-scores, k, axis=-1)[..., :k] and an optional [starts, ends)
mask, with -1 sentinel fill when a row has fewer than k valid columns. The
output is mx.int32 to match the source kernel's index buffer dtype.

cppmega_mlx/nn/_tilelang/topk_selector.topk_selector_metal(...) is the
direct-MSL Path B runtime on Mac. It launches via mx.fast.metal_kernel, clamps
per-row starts/ends inside the kernel, emits value-descending indices, and
returns None only when the Metal path cannot dispatch. topk_selector(...,
backend="auto") prefers this Metal path and falls back to the reference;
backend="metal" fails closed if the direct-MSL kernel is unavailable.

Top-k indices are non-differentiable (the gather of the selected values
is differentiable; the indices themselves are a discrete selector). We
therefore do **not** wrap the function in mx.custom_function. Tests treat
parity as the indices' set-membership for the largest k values, because
mx.argpartition, the source radix kernel, and the direct-MSL greedy kernel do
not share a stable tie-breaking contract.

## Tests

tests/test_tilelang_topk.py covers:

- pure-MLX reference parity vs a NumPy oracle for B in {1, 4},
  T in {64, 512, 2048}, and k in {1, 8, 32} (with k <= T)
- dtype in {float32, float16, bfloat16} -> output dtype is int32
- edge cases: k == 1 returns the argmax, k == seq_len returns all
  indices, [starts, ends) masking excludes out-of-range columns, and short or
  empty intervals fill invalid slots with -1
- shape/dtype validation (rejects non-2D inputs and k <= 0,
  k > seq_len)
- Path B status helper records the direct-MSL availability and is stable across
  calls
- direct-MSL Metal-vs-reference parity for the main sweep, short/empty
  intervals, k == seq_len, and acceptance shapes B=4/T=512/k=64 plus
  B=1/T=4096/k=256 for float32, float16, and bfloat16

58 tests pass on M4 Max.

## Bench

scripts/bench_tilelang_topk.py compares three pure-MLX strategies plus the
direct-MSL Path B kernel
across a small shape matrix (10 warmup, 50 timed iterations, perf_counter
wall time, mx.get_peak_memory() for peak GiB):

- argpartition -- the reference (mx.argpartition(-scores, k)[..., :k]).
- argsort_slice -- mx.argsort(-scores)[..., :k]. Higher-work upper
  bound, kept honest for the parity sweep.
- topk_take_along -- argpartition followed by mx.take_along_axis so
  values are materialized too. Models the "fused selector" call-site
  shape downstream code uses.
- path_b_msl -- hand-written MSL via mx.fast.metal_kernel. If it cannot
  dispatch for a shape/device, the bench records ran=false instead of timing a
  fallback as if it were Metal.

Sample smoke output on M4 Max (median ms across 10 iters, warmup=3):


B    T       k      dtype      argpart_ms    argsort_ms    fused_ms      msl_ms        peak_gib
1    64      8      float32    0.1666        0.1485        0.1801        0.1784        0.0000
1    512     32     float32    0.1509        0.1391        0.1671        0.3726        0.0000
1    2048    64     float32    0.1716        0.1549        0.1849        0.8641        0.0000
4    2048    64     float32    0.1765        0.1397        0.1572        0.5147        0.0001
4    2048    64     float16    0.1627        0.1398        0.1616        0.5072        0.0001
4    2048    64     bfloat16   0.1481        0.1471        0.1590        0.4941        0.0001
4    4096    256    float32    0.1658        0.1593        0.1898        8.8177        0.0004


MLX argpartition remains the better standalone top-k implementation, especially
for B=4/T=4096/k=256. The direct-MSL kernel is kept as the smoke-test for the
Path B bypass and as a building block for future fused sparse-MLA kernels,
where selection can happen inline with the attention reduction.

Parity tolerance: bit-exact set membership for the largest k values; tie
breaking is intentionally not part of the contract.

## Reproduce


PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider tests/test_tilelang_topk.py -q --tb=short
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pyright cppmega_mlx/nn/_tilelang/topk_selector.py tests/test_tilelang_topk.py scripts/bench_tilelang_topk.py
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python scripts/bench_tilelang_topk.py --json --no-output-file


The bench writes bench/tilelang_ports/topk_selector.json and (with
--json) prints the same payload to stdout.

## TileLang Metal codegen surprises

For future ports, the failures we hit on this kernel are the
load-bearing ones:

1. T.alloc_shared -> shared.dyn, which TVM's metal backend in
   tilelang 0.1.9 does not implement. Any kernel using threadgroup
   memory hits this immediately. The bench_msl_path_b.py prototype
   sidestepped the issue by allocating into thread-local registers
   only (T.alloc_local).
2. LowerTileOp requires the parallel-loop iter space to tile
   injectively into the threadgroup size. T.fill(buffer_with_size_257,
   0) over 256 threads fails this check. Workarounds: pad the buffer
   to a multiple of BLOCK_SIZE, or hoist the fill out of the kernel
   and pass a pre-zeroed buffer in.
3. The pass_configs parameter accepted by @tilelang.jit is **not**
   accepted by tilelang.engine.lower.lower(...). The flags
   TL_DISABLE_THREAD_STORAGE_SYNC and
   TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE from the cppmega source
   are not currently reachable through the lower API; they will need
   to be set via the global PassContext once that API stabilizes.
4. T.dynamic shape variables produce TIR with unbound tir.Vars in
   T.Kernel extents, which the metal codegen does not lower. Concrete
   shapes must be baked into the PrimFunc per (B, T) instance.
5. from __future__ import annotations breaks @T.prim_func's
   get_type_hints resolution because closure variables stop being
   visible to the eager builder; the working pattern keeps types eager
   and uses string dtypes only.
