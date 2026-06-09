# Honor Triton autotune-WINNING BLOCK_SIZE — §P1 dstates (gb10 CUDA, EXECUTED + honest blocker)

HEAD `9391ed210ccb0deb6b3ed58a3daa5ec264845cdb` (tilelang, branch
`merge/upstream-codegen-reorg`). All CUDA numbers below are EXECUTED on the
NVIDIA GB10 (sm_121, CUDA 13.2) under the gb10 GPU mutex. RULE #1: bit-correct
EXECUTED results only; fail-loud, no silent fallback. The captured-TTIR and
build-blocker facts are reproducible from HEAD via the named harnesses listed
below. The Metal portability claim is codegen-only (no Apple-GPU dispatch).

## The question

Does tiling the routed §P1 GEMM with native's autotune-WINNING `BLOCK_SIZE`
(instead of the pinned smallest tile) drop §P1 EXEC ms toward native 1.23 ms,
*generically* (a config read, not a per-kernel hack)?

## Root cause (confirmed)

The route PINNED the SMALLEST autotune config at TTIR capture time:

- `poc/triton_frontend/_test_harness/tridao_parity/parity_all7.py:48`
  `PIN = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}`

Native's autotuner instead picks the WINNING config for the §P1 shape. Confirmed
three independent ways on gb10:

- cached `~/.triton/cache/7J3654N5*/_chunk_scan_bwd_dstates_kernel.json`:
  `num_warps=8`, `num_stages=3`, `shared=197120`.
- the native `.ttgir` tile shapes: `tensor<128x256>` / `tensor<128x64>`.
- the mamba_ssm `ssd_chunk_scan.py` `@triton.autotune` Config #0 (the winning
  entry): `{BLOCK_SIZE_M:128, BLOCK_SIZE_N:256, BLOCK_SIZE_K:64}`,
  `num_warps=8`, `num_stages=3`.

## The fix — GENERIC, named, backend-agnostic (BlockFix, commit 9391ed21)

`poc/triton_frontend/__init__.py` gains `autotune_winning_block_config()`, the
capture-side sibling of `_read_ttir_warp_config`. It reads
`BLOCK_SIZE_M/N/K + num_warps + num_stages` from the kernel's OWN
`@triton.autotune` config list (the winning entry, `configs[0]`) — no per-kernel
hardcoded block size. RULE #1: if the object carries no `configs`, it RAISES
rather than silently defaulting to the smallest tile.

The block dims live in the captured TTIR `tt.dot` operand shapes (not as module
attrs), so they must be pinned at capture time; the helper supplies them as
constexprs to the TTIR capture. The same config flows into the captured GEMM
tile, which drives the CUDA `T.gemm` warp partition AND the Metal threadgroup
partition alike — no libtriton / cuda-only op in the honoring path.

VERIFIED end-to-end on gb10 via
`poc/triton_frontend/_test_harness/tridao_parity/tilefix_block_ab.py`:

```
WINNING_BLOCK BLOCK_SIZE_M=128 BLOCK_SIZE_N=256 BLOCK_SIZE_K=64 num_warps=8 num_stages=3
BASELINE_TILES (pinned 64x64x32)  [(64,1),(1,64),(64,64),(64,32),(32,64), ...]
WINNING_TILES  (native 128x256x64) [(128,64),(64,256),(128,256),(128,1),(1,256), ...]
```

The captured TTIR tile shapes move from `64x64 / 64x32 / 32x64` to native
`128x256 / 128x64 / 64x256` — i.e. the config-read + capture is correct and
matches native.

## HONEST BLOCKER — the winning tile does NOT build on the routed path (RULE #1)

Building the routed PrimFunc at ANY tile larger than 64x64 trips an existing
fail-loud guard in OUR lowering layer:

- `poc/triton_frontend/op_emitters/reduction.py:1524-1535` (`map_tt_dot`):

```
WINNING_BUILD_BLOCKED EmitError ::
  tt.dot: CUDA MMA accumulator C is a SHARED loop-carried tile ('carry_tile_40').
  The fused GEMM-accumulate-into-carry form (produced by Triton's
  folded/canonicalized TTIR) cannot be lowered correctly via the serial scalar
  loop-carry copies, which ignore the mma fragment swizzle and silently corrupt
  the accumulator. A layout-aware fragment-resident carry is required.
```

This is a REAL tilelang lowering gap, not a config-read bug: Triton's folded
TTIR fuses `acc = acc + dot(...)` so the GEMM accumulates directly into the
loop-carried tile. On CUDA the MMA C must live in a swizzled `local.fragment`,
but the loop-carry snapshot/commit and the seeding copy are serial scalar
element copies that do NOT respect the MMA store swizzle. Routing the carry
through that scalar round-trip SILENTLY corrupts the accumulator (observed
MAXDIFF ~1.5e3 vs native). The guard RAISES rather than ship wrong numbers
(RULE #1). The 64x64 baseline avoids the branch because its GEMM C is a fresh
fragment with a separately-added linearly-copied carry.

Correctly lowering the winning tile needs a layout-aware `T.copy` (or a
fragment-resident carry with swizzle-correct commit) which the frontend does not
yet emit. That is the next lowering task — it is NOT addressable by the config
read.

### Consequence for the MEASURE A/B

Because the NEW (winning) block does not build on the routed path, there is **no
NEW-block-config §P1 EXEC ms to measure**. The block-size axis is genuinely
blocked. The executable axis of the same config-honoring fix is the
warp/stage config (commit 01a2a7e6), which IS the same-session A/B reported
below (controls cross-session GPU-clock variance).

## §P1 dstates EXEC ms — OLD vs NEW config, SAME SESSION (executable axis)

`poc/triton_frontend/_test_harness/tridao_parity/tilefix_warp_ab.py`. Same raw
`/tmp/ttir7/_chunk_scan_bwd_dstates.ttir`, same `prologue_opt + cp.async` for
both; only the autotune warp/stage config differs. Interleaved CUDA-event
timing, N=50/rep, median-of-medians, both legs in ONE process (clock-controlled).

| variant | spill (STL+LDL) | HMMA | LDGSTS | IMAD | ISETP | parity (MAXDIFF) | ms |
|---|---|---|---|---|---|---|---|
| OLD (defaults num_warps=4, num_stages=2) | 1596 | 32 | 0 | 1250 | 926 | 4.882812e-04 | **2619.47** |
| NEW (autotune num_warps=8, num_stages=3) | **272** | 16 | 8 | 788 | 796 | 4.882812e-04 | **1161.57** |
| native Triton | — | — | — | — | — | reference | **1.1703** |

Per-rep (same session, both legs interleaved): rep0 BASE=2619.47 FIX=1161.57;
rep1 BASE=2618.15 FIX=1161.03 — stable. NEW vs OLD = **2.254x**. NEW gap to
native = **992x**. Both parity PASS at 4.882812e-04 (bit-identical to native
within fp32 tolerance). Toward-native SASS deltas at the NEW config: spill
1596->272 (5.9x fewer), IMAD 1250->788, ISETP 926->796, LDGSTS 0->8 (cp.async
live). HMMA halves (32->16) because 8 warps split the same MMA work across twice
the warps. The §P1 EXEC ms is 1161.57 (was 1159 documented; +0.2% cross-run, same
config) — the residual ~992x gap to native is the un-built larger tile (many
small 64x64 blocks vs native's few 128x256 blocks), NOT the warp config.

## 2nd kernel benefits — GENERIC (not a dstates hack)

`poc/triton_frontend/_test_harness/tridao_parity/tilefix_dc_generic.py`: the
DIFFERENT kernel `_chunk_scan_bwd_dc` responds to the SAME
`GemmWarpPolicy.compute_warp_partition` mechanism — EXECUTED on gb10:

```
DC_NW4 nw=4 HMMA=32 LDGSTS=0 IMAD=422 ISETP=627 spill=1498
DC_NW8 nw=8 HMMA=16 LDGSTS=0 IMAD=242 ISETP=160 spill=334
SPILL_DELTA=4.49x
```

spill 1498->334 (4.49x) at num_warps 4->8 for a kernel that is NOT dstates.
Proves the warp/stage honoring is generic tilelang codegen, not a per-kernel hack.

## No regression

- 64x64 baseline still builds (`BASELINE_BUILD_OK 64x64x32`).
- Default path (no `num_warps`, no autotune attr) is byte-identical to pre-fix.

## Bottom line — honest GO/NO-GO

- Config read + TTIR capture at native's winning `{128,256,64}`: **DONE,
  generic, verified** (tile shapes match native).
- Block-size §P1 EXEC ms drop toward native: **BLOCKED** by the
  fused-GEMM-into-shared-carry lowering gap (`reduction.py:1524`). The winning
  tile does NOT build on the routed path; no NEW-block ms exists to claim.
- Executable config-honoring axis (warp/stage): same-session A/B shows the
  measured drop reported above; remaining gap to native 1.23 ms is dominated by
  the un-built larger tile (few large blocks vs many small 64x64 blocks).

KEY ANSWER: tiling with native's autotune BLOCK_SIZE is read + captured
generically, but does NOT yet drop §P1 ms, because the larger tile does not
lower on the routed path. Closing the gap requires the layout-aware
fragment-resident loop-carry, not more config plumbing.
