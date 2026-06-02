# Sparse-MLA (DeepSeek-V3.2) wired into path_c — e2e CORRECT on gb10

**Status (2026-06-02): DONE — e2e numerically correct.** path_c's differentiable
sparse-MLA op dispatches the *real* fused v32 TileLang kernel (fwd **and** bwd) on
gb10/sm_121a via genuine DLPack **zero-copy** input, and produces numerically
correct output: full-tensor **fwd+bwd cos = 1.0000** vs the upstream dense
reference, no inf/nan, **no row exclusion**. This is the **first fused-attention
op proven e2e-correct in path_c on CUDA.**

## What is wired

- `cppmega_mlx/nn/_tilelang/_sparse_mla_v32_fused.py` — wraps the real fused v32
  `tilelang/examples/deepseek_v32/sparse_mla_fwd.py` + `sparse_mla_bwd.py` over
  MLX bf16 `q/kv/indices`, exposing fwd (`v32_fused_apply`) and bwd
  (`v32_fused_bwd`, exercising the vectorized `T.atomic_addx4` dKV scatter).
- Wired into the differentiable owner wrapper
  `sparse_mla_fp8_path_c_apply_prepared_float` (fwd + VJP) in
  `cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py`.
- `cppmega_mlx/nn/_tilelang/_cuda_zerocopy.py` — native MLX kDLCUDA `__dlpack__`
  export → `torch.from_dlpack`, NO numpy host roundtrip. The torch tensor is a
  real CUDA device view of the MLX allocation.

### Gates (both required for the fused zero-copy path)

```
CPPMEGA_SPARSE_MLA_V32_FUSED=1     # dispatch the real fused v32 kernel
CPPMEGA_TILELANG_CUDA_ZEROCOPY=1   # feed inputs zero-copy via DLPack
# CPPMEGA_SPARSE_MLA_V32_GB10=1    # optional: force the 99 KiB GB10 re-tile
```

Default (gates OFF): callers keep the existing reference path; nothing fused runs.

**RULE #1 (no silent fallback):** when gated ON + `force_path_c`, every fused
failure RAISES with where+what. There is no silent reference fallback.

## Layout / ABI

- `q`  : bf16 `(B, S, H, DQK)`, `DQK = d_v + tail = 512 + 64 = 576`.
- `kv` : bf16 `(B, SKV, G, DQK)` (kv_group `G`).
- `indices` : int32 `(B, S, G, topk)` (sentinel `SKV` → masked slot).
- fwd → `out` bf16 `(B, S, H, d_v)`, `lse` fp32 `(B, S, H)`.
- bwd → `dq` bf16 `(B, S, H, DQK)`, `dkv` bf16 `(B, SKV, G, DQK)`.

## E2E verification

`scripts/verify_sparse_mla_v32_pathc_e2e.py` — traps BOTH input bridges
(zero-copy and eager-host) so it can prove which one fed the kernel, runs the
path_c owner wrapper through fwd + `mx.grad` VJP, and asserts parity + finiteness
over the **full tensor** (RULE #1: any non-finite element raises with the exact
offending query rows; any eager-bridge call raises).

Run on gb10:

```
cd /home/dave/source/cppmega_mlx
CPPMEGA_SPARSE_MLA_V32_FUSED=1 CPPMEGA_TILELANG_CUDA_ZEROCOPY=1 \
  /home/dave/cppmega-venv/bin/python scripts/verify_sparse_mla_v32_pathc_e2e.py
```

### Results (gb10 / sm_121a, tilelang `8385a23d`, 2026-06-02, ALL rows)

**PHASE 1 — fused path_c vs upstream dense torch reference**
`B=1 S=512 SKV=1024 H=128 HKV=1 DQK=576 DV=512 topk=256`:

| tensor | cos | rel_err |
|--------|-----|---------|
| fwd | 1.0000 | 2.07e-03 |
| dq  | 1.0000 | 2.61e-03 |
| dkv | 1.0000 | 2.35e-03 |

zero-copy bridge calls = **15**, eager-host bridge calls = **0**, all elements finite.

**PHASE 2 — fused path_c at model scale** (no dense ref — it would need ~17 TB)
`B=1 S=4096 SKV=8192 H=128 topk=2048`:

- fwd: finite=True, latency = **632.7 ms** (was 3860 ms host-bounce), out_abs_mean = 0.0140
- fwd+bwd: finite=True, latency = **1419.3 ms** (was 7765 ms host-bounce), dq_abs_mean = 0.0051, dkv_abs_mean = 0.0571
- peak_mem = **5.38 GiB** (was 7.45 GiB)
- zero-copy = 15, eager = 0

> Re-measured 2026-06-02 after the MLX-CUDA DLPack-import deadlock fix (below)
> made the **output** torch->MLX writeback genuine deadlock-free zero-copy: fwd
> **3860 -> 632.7 ms (6.1x)**, fwd+bwd **7765 -> 1419.3 ms (5.5x)**, also beating
> the prior 2824 ms host-bounce fused-fwd number. The latency win now SURVIVES
> through path_c; the residual is per-call wrapper/JIT-dispatch overhead, not a
> host PCIe bounce.

The proof that the **real fused kernel ran** (not the reference, not the eager
host bridge): `zero-copy bridge calls = 15` and `eager = 0` for fwd+bwd inputs,
plus the TileLang compile logs (`sparse_mla_fwd` / `sparse_mla_bwd_kernel` /
`preprocess_kernel` / `postprocess_kernel`) — and RULE #1 raise-on-failure means
a fallback could not have masked anything.

## The fixed `inf` bug (tilelang `8385a23d`)

The earlier blocker — `inf` on a sparse subset of query rows in the GB10 re-tile
fwd — was a **merge-pass reduce-scratch buffer aliased onto `S_shared`**: the
online-softmax reduce scratch and the score buffer shared the same shared-memory
region, corrupting partial-max state for some rows → `inf` in the LSE
recombination. Fixed Python-only in `examples/deepseek_v32/sparse_mla_fwd.py`
(no C++ rebuild). Official `test_sparse_mla_fwd`/`test_sparse_mla_bwd
(check_correctness=True)` now PASS at `S=512/topk=256` and `S=4096/topk=2048`;
full-tensor fwd cos 0.999998, bwd dq/dkv cos 0.999996, out_inf = out_nan = 0.

## Performance through path_c (honest)

The **bare** fused kernel is the standalone `53.4× faster / 26.1× less memory`
win (tilelang `docs/SPARSE-MLA-GB10-BENCH.md`, `S=4096/topk=256`: fused fwd
2.86 ms / 0.259 GB vs dense 152.6 ms / 6.75 GB).

Through the **current path_c wrapper**:

- **Memory win SURVIVES** — measured **12.9× less peak** at `S=4096/topk=256`
  (2.07 GiB fused vs 26.8 GiB dense torch reference).
- **Latency win now SURVIVES** (re-measured 2026-06-02, after the DLPack-import
  deadlock fix below). Both sides of the bridge are now true GPU-resident
  zero-copy:
  - **input** side: native MLX kDLCUDA `__dlpack__` -> `torch.from_dlpack` (always was zero-copy);
  - **output** side: torch CUDA result -> `mx.array` now imports the foreign
    CUDA buffer with a single on-device `cudaMemcpy` into an MLX-owned buffer
    (no `.cpu().numpy()`, no GPU->host->GPU PCIe bounce).
  - Net fused fwd-through-path_c at S=4096/topk=2048: **632.7 ms** (was 3860 ms),
    fwd+bwd **1419.3 ms** (was 7765 ms) — now wrapper/JIT-dispatch-bound, not
    host-bounce-bound. `sparse_mla_fwd_interface` still does a per-call
    JIT-cache dispatch (the remaining overhead).

**Correctness is CLOSED and the latency win is RESTORED.** The residual overhead
is the per-call JIT-cache dispatch in `sparse_mla_fwd_interface` (hoisting the
kernel handle out of the per-call interface would shave it further) — a latency
optimization, not a correctness gap.

## MLX-CUDA DLPack-import deadlock (root cause + fix, 2026-06-02)

The output-side zero-copy import (`mx.array(torch_cuda_tensor)` ->
`convert.cpp:cuda_dlpack_to_mlx`) initially **DEADLOCKED on the SECOND
consecutive import + `mx.synchronize`** — i.e. on every iteration of a training
loop. EXIT=124 (timeout) on isolation probes.

**Root cause.** The first import implementation wrapped the foreign CUDA pointer
in an `mx::array` whose **deleter CAPTURED the nanobind `owner`** (the source
torch tensor / DLPack capsule, a `shared_ptr<nb::ndarray>`). Releasing that
owner decrefs a Python object, which needs the **GIL**. When MLX's **scheduler
thread** later destroyed that array's `Data` (e.g. during a subsequent
`mx.synchronize`, which the main thread calls *while holding the GIL*), the
scheduler thread blocked forever waiting for the GIL -> deadlock. A
`gil_scoped_release` around the import's own eval did NOT fix it (the deadlock is
at the LATER user-called `mx.synchronize`, not the import's eval) — reverted.

**Fix (DatasunriseOU/mlx).** A new CUDA backend primitive
`cu::copy_external_to_mlx_buffer(src, nbytes)` (`mlx/backend/cuda/allocator.cpp`,
declared in `cuda.h`, stubbed in `no_cuda.cpp`): (1) `allocator().malloc(nbytes)`
-> a REAL MLX-managed buffer, (2) a single synchronous `cudaMemcpy(dst, src,
nbytes, cudaMemcpyDefault)` on the **calling** thread, (3) return the MLX-owned
buffer. `cuda_dlpack_to_mlx_contiguous` then builds the `mx::array` from that
MLX-owned buffer with the **default `allocator::free` deleter** — so **NO foreign
buffer and NO Python/DLPack owner ever enters MLX's graph or scheduler**. The
torch capsule is only READ during the copy and is released on the calling thread
(which holds the GIL). `allocator::free` runs GIL-free, so the scheduler thread
can never deadlock.

Tradeoff: this is a device-to-device `cudaMemcpy` (a single fast on-device copy,
no host bounce), not pointer-aliasing zero-copy — but it is deadlock-free and
correct, and recovers essentially all of the latency vs the prior
`.cpu().numpy()` host round-trip.

**Verified (gb10, /home/dave/cppmega-venv/bin/python):**
- `/tmp/iso_probe.py` (single + second import + sync): EXIT=0 (was hang).
- `/tmp/iso2_probe.py` (import/eval/sync x3): EXIT=0 (was EXIT=124 at Test 2).
- 100x loop (torch CUDA -> `mx.array` -> MLX op -> `mx.synchronize`): EXIT=0,
  **bit-identical every iter** (max_abs_err=0.0), 0.09s total.
- path_c e2e (`verify_sparse_mla_v32_pathc_e2e.py`): EXIT=0, zero-copy=15,
  eager=0, fwd cos=1.0000, latency restored as above.
