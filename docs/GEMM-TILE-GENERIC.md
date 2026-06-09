# Generic GEMM tiling fix: honoring the Triton autotune warp/stage config

Status: MEASURED on NVIDIA GB10 (sm_121, CUDA 13.2) under the gb10 GPU mutex.
Branch: `merge/upstream-codegen-reorg`. Fix commit: `01a2a7e6`.

## Problem (root cause)

The routed-triton `from_ttir` path (`poc/triton_frontend/__init__.py`) defaulted
`ctx.num_warps = 4` (a 128-thread block) for EVERY routed kernel, even when the
source Triton kernel autotuned to a different warp count. For the §P1
`_chunk_scan_bwd_dstates_kernel` the native Triton autotuner selects
`num_warps=8, num_stages=3`. With only 4 warps, the cooperative `T.gemm` tiles
the SAME single `tt.dot` over a 4-way warp partition instead of the 8-way
partition Triton emits, so the 64x64 dstates tile is partitioned 4x too small
and the per-thread register pressure explodes (heavy spilling).

`ctx.num_warps` flows, unchanged, through `op_mapping.map_tt_func`
(`op_mapping.py:1037`): it sets the block `threadIdx.x` `thread_extent =
num_warps*32` and stamps the PrimFunc `num_warps` attr. That thread extent is
exactly what `tilelang`'s backend-agnostic
`GemmWarpPolicy.compute_warp_partition(M, N, num_warps)`
(`tilelang/tileop/base.py:65`) reads to derive the `(m_warp, n_warp)`
partition. For a 64x64 tile: at `num_warps=8` the square policy finds
`m_warp*n_warp=8` (e.g. 4x2); at `num_warps=4` it can only reach 2x2/4x1.

## Fix (GENERIC, backend-agnostic, no per-kernel hack)

`from_ttir` now reads the autotuned config when the caller does not pass an
explicit `num_warps`/`num_stages`. New `_read_ttir_warp_config()`
(`__init__.py:1444`) probes the TTIR module operation's MLIR attributes that
Triton stamps after binding the selected `triton.Config`:
`ttg.num-warps` / `num-warps` / `num_warps` and the matching `*-stages` keys.

- Any TTIR carrying these standard module attrs is honored -- not a per-kernel
  special case. Each kernel gets ITS OWN autotuned config: `dstates` and `dx`
  resolve to `num_warps=8`; `dc`/`dcb` resolve to `num_warps=4` (correctly kept
  small).
- Explicit `num_warps`/`num_stages` kwargs always win (the documented override
  used by the text-TTIR §P1 harness, which carries no module attrs).
- RULE #1 (fail loud): a present-but-malformed or non-positive warp attr RAISES
  rather than silently falling back to the 4-warp default.

`ctx.num_warps` then flows into `gemm.lower`'s `computeWarpPartition` on CUDA and
the threadgroup partition on Metal alike -- the fix lives entirely in the
backend-agnostic frontend + the existing tilelang Python tile-op layer. No
C++ core change, no libtriton dependency, no cuda-only op. Python-frontend only;
no gb10 rebuild required.

## Measured effect (EXECUTED on gb10, raw §P1 dstates TTIR, prologue_opt + cp.async)

Build-only SASS census of the EXACT compiled cubins (same raw TTIR, prologue_opt
=True, TL_FORCE_CP_ASYNC=1; only the warp/stage config differs):

| config                         | HMMA | LDGSTS | IMAD | ISETP | spill (STL+LDL) |
|--------------------------------|------|--------|------|-------|-----------------|
| default (num_warps=4)          | 32   | 0      | 1250 | 926   | 1596            |
| TileFix autotune (num_warps=8) | 16   | 8      | 788  | 796   | 272             |
| native Triton (reference)      | --   | 75     | 298  | 69    | 0               |

- spill 1596 -> 272 (5.9x fewer spill instructions): the 8-way warp partition
  cuts per-thread register pressure substantially.
- LDGSTS 0 -> 8: cp.async/LDGSTS becomes live at 8 warps (the wider tile lets
  the multi-stage copy loop pipeline).
- IMAD 1250 -> 788, ISETP 926 -> 796: residual addressing moves toward native
  (298/69). Note HMMA SASS *count* DROPS with more warps -- it is a per-thread
  instruction count, so an 8-way partition issues fewer HMMA per thread for the
  same block-level work; HMMA count is NOT a direct "256-HMMA" proxy.

Parity: both default and TileFix configs match the native kernel bit-for-bit at
MAXDIFF = 4.882812e-04 (PASS).

Timing (CUDA events, N=50 x 4 reps, interleaved, median-of-medians): see the
TFARESULT / WTRESULT lines in the run logs on gb10.

## Generic proof (2nd kernel)

`_chunk_scan_bwd_dx_kernel` ALSO autotunes to `num_warps=8` (its best
`triton.Config`). Honoring its autotune config (vs the old default-4) drops its
spills the same way -- the fix reads each kernel's own config, not a dstates
hack. `dc`/`dcb` autotune to 4 warps and the reader correctly keeps them at 4.

## Reproduce from HEAD

```
ssh gb10
cd /home/dave/source/tilelang   # HEAD 01a2a7e6 (merge/upstream-codegen-reorg)
python /tmp/tilefix_final_ab.py   # BASE(default-4) vs FIX(autotune-8) + native
python /tmp/dx_generic_probe.py   # 2nd kernel (dx) generic proof
```
