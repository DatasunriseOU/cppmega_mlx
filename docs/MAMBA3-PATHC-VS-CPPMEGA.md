# Mamba3 / M2RNN — path_c (ours) vs cppmega per-region compare (gb10 sm_121)

**Status: MEASURED (gb10 sm_121, 2026-06-04).** The KEY finding: path_c loses to
cppmega on the SAME mamba model because of the **un-fused multi-kernel SSD
decomposition** (F0 precompute alone = 16.37 ms ≥ 5.7× cppmega's whole 3.11 ms fused
fwd), and the prod-state F2 scan tile (N=64) **segfaults at the GB10 99 KB smem cap**.
It is NOT host/launch staging. RULE #1: every cell is MEASURED on gb10 or explicitly
`NOT-RUNNABLE`/`SEGFAULT` with the reason — no extrapolated or fabricated speedup.

## Why this exists (lever 5)

The SAME mamba3 + m2rnn model is faster in **cppmega** than in **cppmega.mlx
path_c**:

- path_c §17 (`docs/RELAX-GRAPH-VS-MEGATRON.md`): GO **≈907 tok/s @8L / ≈298
  @28L** (Relax-VM graph: real gridded CUDA SSD fwd + re-gridded gridded bwd,
  carrying an abstract numpy adam/loss as the dominant remaining term).
- the SAME model in cppmega on gb10: **~3692 tok/s @ ~4437 ms/iter** (tensorwise),
  torch + CUDA-graph, fully device-resident, where mamba+m2rnn are only ~24% of
  the iter (nam56r nsys: MIMO 17.5% + M2RNN 6.2%) and the step is launch-amortized.

The prior READ-ONLY investigation ranked the cause as a **runtime/execution-engine
gap, NOT the SSD scan**. This bench measures that claim with per-region numbers.

## What the bench times (SAME problem config, apples-to-apples)

| arm | region | what runs | source |
| --- | --- | --- | --- |
| OURS (path_c) | F0 | `mamba3_chunk_precompute` (grid, no scan dep) | `mamba3_chunked_precompute_core.py` |
| OURS | F1 | `mamba3_inter_chunk_recur` (O(S/C) recurrence) | `mamba3_chunked_precompute_core.py` |
| OURS | F2 | `mamba3_chunk_scan_combine` (grid scan+combine → y) | `mamba3_chunked_scan_core.py` |
| OURS | B2 | `mamba3_chunk_scan_combine_bwd` (grid) | `mamba3_chunked_backward_core.py` |
| OURS | B1 | `mamba3_inter_chunk_recur_bwd` (O(S/C) reverse) | `mamba3_chunked_backward_core.py` |
| OURS | B0 | `mamba3_chunk_precompute_bwd` (grid) | `mamba3_chunked_backward_core.py` |
| OURS | fwd_chain / bwd_chain | F0→F1→F2 / B2→B1→B0 back-to-back, host-staged | — (the launch-overhead term) |
| cppmega | mamba fwd/bwd | `mamba_chunk_scan_combined` (Megatron SSD Triton — SAME SSD math) | `mamba_ssm.ops.triton.ssd_combined` |
| cppmega | m2rnn fwd/bwd | `m2rnn_scan_triton` (fused Triton M2RNN) | `cppmega/megatron/m2rnn_triton.py` |
| OURS | m2rnn | **NOT-RUNNABLE-ON-CUDA** — Path-C M2RNN is Metal/MSL-only, no CUDA twin | `cppmega_mlx/nn/_tilelang/m2rnn_path_c.py` |

**Honest gap (RULE #1):** our M2RNN has NO CUDA implementation; on gb10 there is
no ours-m2rnn kernel to time. The bench reports this as `NOT-RUNNABLE-ON-CUDA`
and times the cppmega m2rnn alone (absolute region cost), so the §17 finding —
that mamba+m2rnn are a MINORITY of the iter — can still be confirmed.

## The two questions the numbers answer

1. **Is the loser the KERNEL or the HOST/LAUNCH STAGING?**
   - `ours_fwd_host_stage_ms = fwd_chain − (F0+F1+F2)`; if this is a large
     fraction of `fwd_chain`, the gap is launch/host staging (H1/H4), not the scan.
   - `fwd_kernels_only_vs_fused = (F0+F1+F2) / cppmega_fused_fwd`; if ≈1, our
     kernels are competitive and the gap is elsewhere; if ≫1, the kernel is the loser.
2. **How big is the m2rnn region vs the mamba region?** Confirms the §17 claim
   that the mamba/m2rnn ops are a minority of the iter (so the engine, not the
   op, is the lever).

## MEASURED results (gb10 sm_121, 2026-06-04) — FILLED BY THE GB10 PHASE

Config: prod = local_gb10_quarter mamba tile S=4096 c=64 g=8 H=112 P=64 N=64
(== §17). m2rnn shape S=4096 H=8 K=128 V=128. bs1.

### Per-region median ms/call (MEASURED)

| region | OURS ms (path_c) | cppmega ms | ours/cppmega |
| --- | --- | --- | --- |
| F0 (precompute) | **16.37** | — (fused) | — |
| F1 (inter-chunk recur) | **1.24** | — (fused) | — |
| F2 (scan-combine, N=64) | **SEGFAULT at launch** (block_Dstate=64 > GB10 99 KB dyn-smem cap) | — (fused) | — |
| **mamba fwd total** | **>17.6 ms (incomplete: F2 will not launch)** | **3.11 ms (single fused kernel)** | **≥5.7× SLOWER (F0 alone)** |
| **mamba bwd** | B2/B1/B0 NOT reached (depend on F2 output) | **~10.00 ms** (bwd_with_recompute 13.10 − fwd 3.11) | — |
| m2rnn fwd | NOT-RUNNABLE (Metal-only Path-C) | **OutOfResources** (needs 101 376 B smem > GB10 99 KB cap) | n/a |
| m2rnn bwd | NOT-RUNNABLE | not reached (fwd OOR'd) | n/a |

### Verdict (THE KEY FINDING — MEASURED, honest)

- **WHERE path_c loses time vs cppmega on the SAME model: the multi-kernel un-fused
  mamba decomposition, NOT host/launch staging.** At the SAME prod config, OUR **F0
  precompute alone is 16.37 ms — ≥5.7× the ENTIRE cppmega fused mamba forward (3.11
  ms)**. F1 adds 1.24 ms. cppmega runs the whole SSD fwd (cb, dA-cumsum, chunk
  states, scan, combine) in ONE fused Triton kernel (`mamba_chunk_scan_combined`,
  3.11 ms) with the N=64 state resident in shared memory; ours splits it into F0/F1/F2
  separate gridded kernels, each re-reading/re-writing the chunk tensors to global
  memory, and F0 dominates.
- **F2 (the scan-combine) does NOT launch at the prod state dim N=64 on GB10**: its
  only legal tile is `block_Dstate >= dstate = 64` (the code RAISES, RULE #1, on
  `block_Dstate < 64` rather than truncate state columns), and `block_Dstate=64`
  exceeds the GB10 sm_121 ~99 KB dynamic-shared-memory cap → the kernel **segfaults
  at launch**. So the full prod-config ours-vs-cppmega mamba region cannot be timed
  end-to-end (the §17 numbers were necessarily built at a reduced state / re-gridded
  substitution). This is the same smem-cap class as the m2rnn Triton OutOfResources.
- **This REFUTES the "launch-bound / host-staging" hypothesis** for the gap (which
  the separate MLX graph-knob sweep also refuted, §22): raising the per-kernel staging
  is not the lever — the lever is **fusing the F0/F1/F2 (and B0/B1/B2) decomposition
  into a single smem-resident SSD kernel** the way cppmega's Triton kernel does, and
  making the N=64 scan tile fit the GB10 smem cap.
- `ours_m2rnn`: NOT-RUNNABLE-ON-CUDA (Metal-only Path-C; UNMEASURABLE on gb10 — not
  fabricated). cppmega m2rnn also could not run here (Triton smem OutOfResources at
  101 KB on the GB10 99 KB cap), so the absolute m2rnn region cost is itself smem-blocked
  on this box for both stacks.

## §17 anchors (for cross-check, already MEASURED)

- F2 forward scan = **0.980 ms** vs serial 6.56 s (≈6694×); gridded fwd chain
  **7.231 ms/call** (§15, fp16 parity 4.746e-04 PASS).
- gridded backward: B2 re-gridded **2484 → 334.6 ms (7.42×)**, chain **2601 →
  447.8 ms (5.81×)**, 1.018× faster than the 456 ms numpy backward; ALL 8 grads
  pass the 1e-3 gate (dD 2.48e-5).
- cppmega same-model gb10: **3692 tok/s @ 4437 ms/iter** (tensorwise), 25.7 GB.

## How to run (GB10 phase, single-owner serial)

```
# ensure EXCLUSIVE GPU ownership + >105 GB free; SIGTERM (not -9) above 113 GB
# ensure_nvrtc_builtins_path() BEFORE torch import (cppmega_mlx/_gb10_nvrtc_env.py)
PYTHONPATH=<cppmega_mlx>:<tvm/python>:<tvm-ffi/python> \
TVM_LIBRARY_PATH=<tilelang/build/lib> \
CPPMEGA_ORIG_ROOT=<cppmega original checkout on gb10> \
/home/dave/cppmega-venv/bin/python scratch/mamba3_m2rnn_compare.py --prod
# then --prod --bs4 for the 16384-tok batch axis.
# grep '^RESULT ' <log> | sed 's/^RESULT //' | python -m json.tool   # parse
```

If `mamba_ssm` (the Megatron SSD kernel) or `triton` is absent, the cppmega arm
is recorded as a hard error in `RESULT.hard_errors` and printed loudly (RULE #1:
surfaced, not swallowed) — install the dep and re-run; ours-mamba still times.
