# mamba3 Path-C chunked FORWARD integration — results & remaining handoff

Branch `mamba3-pathc-integrate-fwd` (off main 2dd9dc1). Workdir
`/Volumes/external/sources/cppmega_mlx_intfwd`. All numbers measured on M4 Max,
tilelang `0.1.9+git5952468a` (has the RS-gemm Metal fix), in THIS worktree.

## What was integrated (live + tested)

Promoted the PROVEN chunked-parallel forward scan-core from the prototype branch
`mamba3-chunked-forward` (commit 351ebee) into the package as a live, tested
module:

- `cppmega_mlx/nn/_tilelang/mamba3_chunked_scan_core.py` — the Metal-compatible
  SSD chunk-scan forward `@T.prim_func` factory (`chunk_scan_fwd_metal_prim`),
  the grid/occupancy helper (`chunk_scan_fwd_grid`), and a compile entrypoint
  (`compile_chunk_scan_fwd_metal`) that uses the repo's `_as_metal_target`
  builder (the bare `"metal -thread_warp_size=32"` string is rejected by this
  tilelang's `determine_target` allowlist; the Target object bypasses it).
  RULE #1: helpers RAISE on non-divisible seqlen / compile failure — no padding
  or serial fallback.
- `tests/test_mamba3_chunked_scan_core.py` — 6 tests, ALL PASS:
  - grid/occupancy (3): full-scale grid is `(8,16,16) = 2048` threadgroups;
    grows with S; non-divisible seqlen RAISES (no fallback).
  - Metal compile+run+parity (3): the prim_func COMPILES to MSL and RUNS on
    Metal at 16 / 512 / 2048 threadgroups and matches the torch SSD reference.

The Path-C emitter replacement site
(`cppmega_mlx/runtime/path_c_fusion_schedules.py:_append_row_phased_mamba3_body`)
now carries a precise in-code CHUNKED-FORWARD INTEGRATION SITE handoff comment
pointing at the validated module and naming the exact remaining steps.

## Measured numbers (this worktree)

### MLX numerical contract (fp32) vs OUR serial `_chunked_mamba3_diagonal_scan`
`scratch/test_mamba3_chunked_parity.py` (B=2,S=256,H=4,P=16,N=16):

| chunk | out max\|d\| | h_last max\|d\| | nan | result |
|------:|-------------:|----------------:|:---:|:------:|
| 64    | 1.14e-05     | 3.34e-06        | no  | PASS   |
| 128   | 1.03e-05     | 2.86e-06        | no  | PASS   |
| 256   | 1.14e-05     | 4.29e-06        | no  | PASS   |

fp32 is the accumulation dtype the production kernel uses internally. (bf16
shows the documented input-quantization noise; both serial-bf16 and chunked-bf16
are bounded by bf16 input quantization — the parity contract is fp32.)

### TileLang Metal scan-core compile + run + parity vs torch SSD reference (fp16)
`scratch/run_chunk_scan_fwd_metal.py` (B=1, C=256, P=64, N=128):

| seq  | nheads | grid       | threadgroups | max\|abs diff\| | nan | ms/iter | result |
|-----:|-------:|:-----------|-------------:|----------------:|:---:|--------:|:------:|
| 256  | 1      | (1,16,1)   | 16           | 2.441e-04       | no  | 0.036   | PASS   |
| 1024 | 8      | (8,16,4)   | 512          | 4.883e-04       | no  | 0.164   | PASS   |
| 4096 | 8      | (8,16,16)  | **2048**     | 4.883e-04       | no  | 0.358   | PASS   |

Serial Path-C forward = **1** threadgroup. Chunked = **2048** at full scale →
multiple full waves on a 40-core M4 Max, GPU fully occupied. The prototype's
bench sweep (serial vs chunked, fp32) measured ~37.6x forward speedup at S=1024,
the Amdahl signature of removing the O(S) serial dependency.

### CUDA (gb10)
Not run this session. The same grid `(nheads, tiles, batch*nchunks)` parallelizes
identically (many blocks); the CUDA-variant prim_func is
`chunk_scan_fwd_prim` (upstream form) on the prototype branch. The Metal variant
in the new module is Metal-specific (shared-mem C accumulator + serial K-loop).

## Does the Path-C mamba3 forward run chunked (fast) with parity?

The chunked forward scan-CORE is LIVE, COMPILES + RUNS on Metal at 2048
threadgroups, and is PARITY-CORRECT (fp32 ~1e-5 vs our serial recurrence; fp16
~4.9e-4 vs SSD reference), wired into the package and covered by passing tests.

The full end-to-end Path-C emitter swap (serial scan → chunked grid for the LIVE
fused kernel, with the outer-op precompute composed around the scan-core) is
NOT yet flipped: the serial `T.Kernel(1)` forward is still what the emitter
generates. This is the validated PARTIAL the task SCOPE item 3 sanctions —
nothing unverified was shipped, and no silent serial→chunked swap was made.

## Remaining work (precise handoff) — completing the live emitter swap

The emitter wraps the entire mamba3 forward in a single `T.Kernel(1,
threads=...)` (`path_c_fusion_schedules.py:~2867/2873`) driving
`for row in T.serial(0, S)` (`_append_row_phased_hidden_body:~5053`), carrying
scan state across rows in the `state_output` buffer (the
`mamba3_scan_policy: external_state_recurrence` block, ~`6826-6831`). Flipping to
chunked means restructuring THAT launch into the grid pipeline:

1. **Precompute over the grid (reuse existing pieces):** in-proj matvec, causal
   conv (halo = `kernel-1` rows from prior chunk), dt=softplus, A=−softplus
   clamp, RoPE angle cumsum (separate associative scalar prefix), trapezoid, B/C
   RMSNorm+rope → produce `x, B, C, z, A, dt`. These stages in
   `_append_row_phased_mamba3_body` are already position-local/grid-parallel.
2. **Form scan-core inputs:** `cb = C@Bᵀ` per chunk, `dA_cumsum = cumsum(A·dt)`
   (log-space, per the underflow caveat), per-chunk `prev_states` via the
   inter-chunk recurrence (the only O(S/C) sequential part; associative combine
   `(A2,B2)·(A1,B1)=(A2·A1, A2·B1+B2)`).
3. **Launch `chunk_scan_fwd_metal_prim` grid** instead of the serial scan block.
4. **Apply `Y_off + skip + silu·z` gate** → write `mamba3_delta`, then
   out-projection.

The descriptor/ABI and carry/replay boundary-state buffers
(`_row_phased_launcher_carry_buffers_for_nodes` /
`_row_phased_replay_buffers_for_nodes`) already plumb per-chunk boundary states,
so the caller-owned-buffer plumbing stays intact. The MLX proto
(`scratch/mamba3_chunked_forward_proto.py`) is the numerical contract for the
4-step algebra; the new module is the Metal scan-core for step 2/4.

RULE #1: keep the per-target codegen choice a legitimate gate (chunked grid vs
serial launcher), never a silent fallback; RAISE on chunking/parity failure.

## UPDATE (branch mamba3-emitter-swap-fwd): production-scale scan-core proof + emitter gate

Measured on M4 Max, this worktree (`scratch/run_chunk_scan_fwd_metal_prod.py`),
at the EXACT live `local_gb10_quarter` mamba3 forward contract (hidden=3584
expand=2 -> inner=7168; head_dim=64 -> nheads=112; state_dim=64; groups=8;
chunk_size=64; S=4096):

| case        | grid          | threadgroups | max|abs diff| (fp16 vs SSD ref) | nan | compile | run ms/iter | result |
|------------:|:--------------|-------------:|---------------------------------:|:---:|--------:|------------:|:------:|
| prod-full S=4096  | (112, 4, 64) | **28672** | 9.766e-04 | no | 0.7s | 2.939 | PASS |
| prod-S1024        | (112, 4, 16) | 7168      | 9.766e-04 | no | 0.6s | 0.726 | PASS |

The serial Path-C forward = **1** threadgroup. So at production dims the chunked
scan-core uses **28,672** threadgroups (715 full waves on a 40-core M4 Max) and
COMPILES + RUNS on Metal with fp16 parity ~9.8e-4 vs the SSD reference. This is
the decisive proof that the already-landed scan-core serves the LIVE forward
contract, not just the synthetic test dims.

### Emitter wiring landed (path_c_fusion_schedules.py)
- `MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE = 64`, `mamba3_chunked_forward_scan_grid(...)`
  (RAISING gate -> `PathCSplitInfeasible` on any shape the validated scan-core
  cannot host), and `_mamba3_chunked_forward_scan_feasibility(...)` (non-raising
  classifier, the single source of truth -- NO try/except degraded path).
- `_append_row_phased_mamba3_body` now records the chunked-grid dispatch
  descriptor (`# mamba3_chunked_forward_scan: grid=(112, 4, 64)
  threadgroups=28672 chunk_size=64 ...`) at the scan-core integration site,
  immediately before the serial `external_state_recurrence` scan. The serial
  scan is STILL the only emitted compute (the swap is the documented remaining
  work below); the descriptor sits beside it, never silently replacing it, and
  non-tile-aligned test shapes are explicitly labelled `NOT FEASIBLE` rather
  than aborted.
- Tests: `tests/test_mamba3_pathc_chunked_forward_gate.py` (8 pass) +
  `tests/test_mamba3_chunked_scan_core.py` (6 pass);
  `tests/test_path_c_fusion_ir.py` (119), `test_path_c_autosplit_metal_parity.py`
  (5 golden-parity), `test_tilelang_mamba3_path_c.py`/
  `test_path_c_autosplit_caps_wiring.py`/`test_path_c_segment_estimator.py` (101)
  all stay green. Full-scale `scripts/repro_fullscale_directchain.py` compiles
  all 12 segment shaders (forward seg[0]=mamba3_mimo now emits the descriptor).

### Why the full emitter swap is NOT flipped yet (the hard ABI boundary)
The live mamba3 forward is ONE fused single-entry segment
(`T.Kernel(num_chunks, threads=1024) as chunk`) that fuses in-proj + causal conv
+ RoPE + B/C RMSNorm + the serial-row scan + out-proj, sharing lane-strided
LOCAL scratch and recurring scan state across rows in caller-owned
`state_output`. The chunked scan-core is a SEPARATE grid kernel
(`T.Kernel(nheads, tiles, batch*nchunks)`) that requires fully-materialized
GLOBAL SSD intermediates: `cb = C@Bᵀ` per chunk, `dA_cumsum = cumsum(A*dt)`, and
per-chunk `prev_states` from the inter-chunk recurrence. The segment lowerer is
`_single_entry_prim_func` (path_c_fusion.py:2776) -- it enforces EXACTLY ONE
prim_func/`T.Kernel` per segment. So flipping the live forward to the grid means
a MULTI-kernel pipeline (precompute grid -> inter-chunk recurrence ->
`chunk_scan_fwd_metal_prim` grid -> gate/out-proj) with NEW global ABI buffers,
which the single-entry segment ABI + descriptor planner + launcher do not host.
That is a multi-thousand-line restructuring across the planner/ABI/launcher --
outside what can be landed with parity verification in one session.

### Remaining work to FLIP the live forward (precise)
1. Teach the descriptor planner/segment lowerer to emit a multi-kernel mamba3
   forward segment (or split mamba3_mimo into 3 sub-segments) with global scratch
   ABI buffers `cb (b,nc,g,c,c)`, `dA_cumsum (b,h,nc,c)`, `prev_states
   (b,nc,h,p,n)` allocated + plumbed via the existing
   `_row_phased_launcher_carry_buffers_for_nodes` boundary-state machinery.
2. Emit the precompute grid (in-proj/conv/RoPE-angle-cumsum/trapezoid/RMSNorm ->
   x,B,C,z,A,dt) -> form `cb`,`dA_cumsum` -> inter-chunk associative recurrence
   `(A2,B2)·(A1,B1)=(A2·A1, A2·B1+B2)` for `prev_states` (the only O(S/C) part).
3. Dispatch `chunk_scan_fwd_metal_prim` via `mamba3_chunked_forward_scan_grid`
   (already wired to RAISE on infeasible -> no silent serial fallback).
4. Apply `Y_off + D·skip + silu(z)·` gate -> `mamba3_delta` -> out-projection.
Parity gate: max|abs diff| vs the current serial Path-C forward within fp
tolerance; threadgroups must be the grid count (28672 at full scale), not 1.
