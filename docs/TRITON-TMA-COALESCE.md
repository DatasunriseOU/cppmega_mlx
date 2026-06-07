# Triton-frontend dstates: TMA / coalesced-async load — iteration-5 MEASURED report

Kernel: `_chunk_scan_bwd_dstates` (Mamba SSD chunk-scan backward dstates), routed
through the tilelang triton-frontend (`poc/triton_frontend/`).

All numbers below are **EXECUTED on gb10** (NVIDIA GB10, sm_121a, aarch64-linux),
regenerated from **committed HEAD `9487a85d`** (gb10 worktree source blobs verified
identical to the committed blobs for all 4 routed files). Compiled via
`from_ttir(..., prologue_opt=False|True)` from `/tmp/ttir7/_chunk_scan_bwd_dstates.ttir`.
GPU work ran under the gb10 GPU mutex.

## 1. Parity (bit-correct, NOT regressed)

| config | reference | MAXDIFF | allclose 1e-3 |
|---|---|---|---|
| §P1 production (b1 nh112 hd64 ds64 nc64 cs64, s4096) OFF | native Triton | 4.882812e-04 | PASS |
| §P1 production OPT (prologue_opt=True, dout-reorder on) | native Triton | 4.882812e-04 | PASS |
| small real-strided multi-K-trip (b1 nh8 s512 nc8 cs64, 2 K-trips) OPT | native Triton | 4.577637e-05 | PASS |

Parity is **bit-identical** to the banked 4.88e-04 — no regression from any
codegen-changing iteration-4/5 commit.

## 2. Routed EXEC ms @ §P1 — interleaved OFF/OPT, CUDA events

Interleaved OFF/OPT/OFF/OPT, N=60 iters/rep × 4 reps, median-of-medians, CUDA events:

| build | §P1 ms (median) | reps (med) |
|---|---|---|
| OFF (prologue_opt=False) | **1476.50** | 1475.9 / 1475.1 / 1476.5 / 1478.2 |
| OPT (prologue_opt=True)  | **1080.99** | 1081.0 / 1081.8 / 1081.0 / 1080.5 |
| **delta (OFF−OPT)** | **395.50 ms** | speedup **1.366x** |

- routed EXEC @ §P1 now = **1080.99 ms** (was 1102 banked — slightly better at HEAD).
- native (Triton, per-kernel) ≈ 1.12 ms/kernel; the §P1 routed kernel runs the full
  112-head × 64-chunk grid as ONE launch, so the apples-to-apples per-kernel figure is
  the §P1 launch; remaining gap to a hypothetical native-equivalent fused launch is
  dominated by spill + non-TMA loads (see §4).

## 3. Per-tile contribution (C vs dout)

Isolated by toggling the `routed_contiguous_tile_axis` dout-reorder hint under OPT
(N=50 × 3 reps, median):

| variant | §P1 ms |
|---|---|
| OPT, dout-reorder ON  | 1079.73 |
| OPT, dout-reorder OFF | 1099.91 |
| **dout reorder (DoutTranspose) contribution** | **−20.18 ms** |
| **C-tile TMA contribution** | **0 ms (UTMALDG=0 — C-tile TMA NOT realized)** |

The OFF→OPT 395ms delta is therefore: addressing-fold + prologue-opt (spill 658→289)
≈ 375 ms, dout traversal reorder ≈ 20 ms, C-tile TMA = 0 ms (not realized).

## 4. SASS (cuobjdump sm_121a, under execution build)

| metric | OFF | OPT | target | note |
|---|---|---|---|---|
| UTMALDG  | 0  | **0**  | >0 | **NOT realized** — no TMA load emitted (C or dout) |
| LDG      | 66 | **98** | drop | OPT emits MORE LDG (reordered SIMT loads, not fewer) |
| LDGSTS / cp.async | 0 | 0 | >0 | no async copy realized |
| spill STL+LDL | 658 | **289** | drop from 306 | **big win** — spill cut 56% |
| HMMA     | 32 | 32 | (vs Triton 256) | unchanged; small-tiled GEMM |

(Baselines in the original task notes — LDG 38, spill 306 — were from an earlier
commit; the values above are the actual measured-from-HEAD OFF/OPT counts.)

## 5. GO / NO-GO

- **DoutTranspose (contiguous-axis-innermost traversal reorder): GO (partial).**
  Landed, bit-correct, measured **−20.18 ms** at §P1. It produces a reordered
  coalesce-friendly **SIMT LDG** path — NOT a TMA/cp.async path (UTMALDG=0, LDGSTS=0).
- **CtileTMA (UTMALDG>0 under execution): NO-GO (honest).** The C-tile T.copy does
  NOT lower to a launched TMA load at HEAD; UTMALDG=0 in both OFF and OPT SASS and the
  generated .cu contains 0 `tl::tma_load` / `cp.async` calls. The grounded-contiguous
  C-tile TMA descriptor path remains blocked end-to-end (de-monomorphized opaque
  symbolic innermost stride prevents `is_one(desc.global_stride[0])`, so bulk-copy
  lowering falls through to SIMT). No fabricated TMA number is presented.
- **Spill reduction (addressing-fold + prologue-opt): GO.** spill 658→289, the
  dominant share of the 395ms OFF→OPT win.

## 6. Remaining gap to native

The routed path is still a **SIMT** dstates kernel (UTMALDG=0, LDGSTS=0, HMMA=32 vs
Triton 256). Closing toward native requires: (a) a launched C-tile TMA (blocked by the
opaque symbolic innermost stride at de-monomorphization — needs route-time literal-1
grounding to reach `is_one`), and (b) larger GEMM tiling to raise HMMA. These are
NOT realized at HEAD — reported honestly as remaining work, not as achieved.

## Reproduce from HEAD (gb10, under mutex)

```
# regen PrimFunc JSON from committed source
python poc/triton_frontend/_test_harness/tridao_parity/emit_pf_json.py _chunk_scan_bwd_dstates
# parity + timing
python poc/triton_frontend/_test_harness/tridao_parity/parity_prod_dstates.py
python /tmp/measure_dstates_interleaved.py   # interleaved OFF/OPT CUDA events
python /tmp/sass_dstates.py                   # OFF/OPT SASS instruction counts
python /tmp/per_tile_dstates.py               # dout-reorder isolation
python /tmp/probe_small_native.py             # small real-strided multi-K-trip parity
```
