# Path B port: tilelang_sparse_mla/topk_selector.py

This document records the Apple Metal port attempt for the smallest TileLang
kernel cppmega ships, the radix-select top-k that backs DSA sparse-MLA index
selection. The port follows the Path B contract (TileLang -> MSL string ->
mx.fast.metal_kernel) used by the rest of cppmega_mlx/nn/_tilelang/.

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

These helpers are dormant for topk_selector because TileLang refuses to
lower the kernel before any MSL is produced.

## Apple Metal status -- blocked

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

This is a hard codegen blocker for the entire DSA sparse-MLA family the
top-k selector belongs to: every kernel in the family relies on
threadgroup memory and partial barriers. The alternative paths are:

- wait for a tilelang release that adds shared.dyn to its metal codegen
  (the upstream PR tile-ai/tilelang#799 has the lowering scaffold but
  does not address shared.dyn as of 0.1.9), or
- hand-write the MSL ourselves bypassing TileLang -- the same pattern the
  Mamba3 port at cppmega_mlx/nn/_tilelang/mamba3.py already follows.

The hand-written option is tracked separately. For this port, we ship
the pure-MLX reference as the runtime path and document the codegen
failure here so the next agent knows exactly which TileLang gap is in the
way.

## Pure-MLX reference contract

cppmega_mlx/nn/_tilelang/topk_selector.topk_selector_reference(scores, k,
*, starts=None, ends=None) is the runtime path on Apple. It uses
mx.argpartition(-scores, k, axis=-1)[..., :k] and an optional
[starts, ends) mask. The output is mx.int32 to match the source
kernel's index buffer dtype.

Top-k indices are non-differentiable (the gather of the selected values
is differentiable; the indices themselves are a discrete selector). We
therefore do **not** wrap the function in mx.custom_function. Tests
treat parity as the indices' set-membership for the largest k values,
because both mx.argpartition and the source kernel have
implementation-defined ordering inside the slice.

## Tests

tests/test_tilelang_topk.py covers:

- pure-MLX reference parity vs a NumPy oracle for B in {1, 4},
  T in {64, 512, 2048}, and k in {1, 8, 32} (with k <= T)
- dtype in {float32, float16, bfloat16} -> output dtype is int32
- edge cases: k == 1 returns the argmax, k == seq_len returns all
  indices, [starts, ends) masking excludes out-of-range columns
- shape/dtype validation (rejects non-2D inputs and k <= 0,
  k > seq_len)
- Path B status helper records the blocker reason and is stable across
  calls
- a placeholder test_path_b_forward_parity that skips while the codegen
  blocker is active and is ready to assert real Metal-vs-reference parity
  the moment the blocker lifts

29 tests collected; 28 pass, 1 expected skip on M4 Max.

## Bench

scripts/bench_tilelang_topk.py compares three pure-MLX strategies
across a small shape matrix (10 warmup, 50 timed iterations, perf_counter
wall time, mx.get_peak_memory() for peak GiB):

- argpartition -- the reference (mx.argpartition(-scores, k)[..., :k]).
- argsort_slice -- mx.argsort(-scores)[..., :k]. Higher-work upper
  bound, kept honest for the parity sweep.
- topk_take_along -- argpartition followed by mx.take_along_axis so
  values are materialized too. Models the "fused selector" call-site
  shape downstream code uses.

Sample output on M4 Max (median ms across 50 iters; full JSON receipts
in bench/tilelang_ports/topk_selector.json):


B    T       k      dtype      argpart_ms    argsort_ms    fused_ms      peak_gib
1    64      8      float32    0.1406        0.1382        0.1354        0.0000
1    512     32     float32    0.1337        0.1229        0.1398        0.0000
1    2048    64     float32    0.1352        0.1422        0.1408        0.0000
4    2048    64     float32    0.1399        0.1396        0.1498        0.0001
4    2048    64     float16    0.1429        0.1390        0.1461        0.0001
4    2048    64     bfloat16   0.1366        0.1357        0.1474        0.0001
4    4096    256    float32    0.1578        0.1588        0.1674        0.0004


The three strategies sit within noise of each other at these shapes
(the tile is dominated by MLX dispatch overhead). The fused
topk_take_along is the right shape for downstream callers and only
costs a take_along_axis over the indices, so we treat it as the
recommended call-site.

Parity tolerance: bit-exact set membership against a NumPy
argsort(-row)[:k] oracle for B in {1, 4}, T in {64, 512, 2048},
k in {1, 8, 32} covered in tests.

## Reproduce


.venv/bin/pytest tests/test_tilelang_topk.py
.venv/bin/python scripts/bench_tilelang_topk.py --json


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
