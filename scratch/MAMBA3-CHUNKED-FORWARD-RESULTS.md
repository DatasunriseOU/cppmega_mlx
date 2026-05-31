# mamba3 Chunked Parallel-Scan FORWARD Prototype — Results & Integration Handoff

Step 3 of `docs/MAMBA3-PARALLEL-FEASIBILITY.md` (forward only). Branch
`mamba3-chunked-forward`. All numbers measured on M4 Max (40 GPU cores), MLX/Metal.

## What was built
- `scratch/mamba3_chunked_forward_proto.py` — the SSD 4-step chunked forward in
  MLX array ops, matching OUR exact recurrence (`_chunked_mamba3_diagonal_scan`):
  intra-chunk diagonal (Y_diag) → per-chunk final states → inter-chunk recurrence
  (the only sequential part, O(S/C)) → state→output (Y_off) + skip + silu gate.
  Cumulative decay is formed in **log-space** (`segsum` of A·dt, exp once) per the
  doc's underflow caveat — never a product of per-step exp.
- `scratch/mamba3_chunked_forward_tilelang.py` — the TileLang `@T.prim_func` form
  of the matmul-heavy intra/inter-chunk step, adapted from
  `tilelang/examples/linear_attention/example_mamba_chunk_scan.py`. Builds and
  reports the launch grid.
- `scratch/test_mamba3_chunked_parity.py`, `test_mamba3_chunked_bf16_truth.py`,
  `bench_mamba3_chunked_forward.py` — parity sweep, bf16 fairness, and speedup bench.

## Parity (vs OUR serial `_chunked_mamba3_diagonal_scan`)
FP32, chunk-size sweep {64,128,256}, shapes up to S=1024:
- out max|abs diff| ≈ **1–3e-5**, h_last max|abs diff| ≈ 3e-6, **no NaN**, PASS (tol 5e-3).
- Stable across all chunk sizes and at non-power-of-2 S (768) — confirms the
  reordered reductions are correct.
- bf16: chunked-vs-fp32-truth max|d| = 0.344 (rel 133) is **equal-or-better** than
  serial-bf16-vs-truth max|d| = 0.344 (rel 555). The chunked form is NOT less
  accurate than the bf16 serial reference; both are bounded by bf16 input
  quantization. The chunked-vs-serial bf16 gap is reduction-order noise — exactly
  the non-bitwise tolerance regime the doc sanctions (§2). High rel-error is from
  a few silu-gated near-zero outputs (tiny denominators), not a correctness bug.

VERDICT: **the chunked forward is parity-correct in fp32** (the accumulation dtype
the production kernel uses internally), within the documented tolerance regime.

## Measured forward speedup (Metal, M4 Max, fp32, B=1 H=8 P=64 N=16)
| seq | chunk | serial ms | chunked ms | speedup |
|----:|------:|----------:|-----------:|--------:|
| 256 | 64    | 7.03      | 0.47       | **14.9x** |
| 512 | 64    | 13.28     | 0.51       | **26.0x** |
|1024 | 64    | 26.42     | 0.70       | **37.6x** |

Speedup GROWS with S (serial scales O(S); chunked stays ~flat as the GPU
saturates) — the Amdahl signature of removing the O(S) serial dependency. At
S=1024 this is **37.6x**, approaching the ~40x occupancy ceiling in the doc (§1).

## Occupancy / threadgroup count
The TileLang chunked prim_func launches a GRID over (nheads, m/n-tiles, batch·nchunks).
For S=4096, C=256, H=8, P=64: grid = (8, 4, 16) = **512 threadgroups** vs the serial
Path-C forward's **1** (`T.Kernel(1, threads=1024)`). 512 ≫ 40 cores → multiple full
waves, GPU fully occupied. (`grid_blocks()` in the tilelang module computes this.)

## Known limitation (TileLang Metal backend)
The TileLang prim_func **builds** and produces the correct grid, but `tilelang.compile`
to the Metal target raises `Unsupported gemm combination, A: local.fragment, B:
shared.dyn` — a TileLang Metal-codegen gap for the swizzled dynamic-shared x buffer
(the upstream example targets CUDA). This is NOT an algorithm issue. Two paths:
(a) drop `scope="shared.dyn"` / swizzle on x_shared for the Metal variant, or
(b) follow the repo's existing route used by the current mamba3 kernel — lower the
prim_func and feed the extracted MSL into `_msl_transform.make_metal_kernel`
(the wave-6 plan at `cppmega_mlx/nn/_tilelang/mamba3.py:14-26`). The MLX chunked
form (proto) is the validated numerical contract for either.

## Integration handoff into Path-C emitter (replace the serial scan)
Target: `cppmega_mlx/runtime/path_c_fusion_schedules.py:_append_row_phased_mamba3_body`
(the `T.Kernel(1, threads=1024)` serial forward over S).

Mapping (the precompute is ALREADY position-local and grid-parallel; only the scan
changes):
1. **Precompute stage (unchanged, already grid-parallel):** in-proj matvec, causal
   conv, dt=softplus, A=−softplus clamp, RoPE angle increment, trapezoid, B/C
   RMSNorm+rope → produce `x, B, C, z, A, dt`. Emit via the existing multi-stage
   plumbing (`path_c_descriptor_stage_prim_funcs`, doc §3).
2. **Intra-chunk Y_diag (NEW kernel):** `T.Kernel(nheads, tiles, batch*nchunks)`,
   the prim_func in `mamba3_chunked_forward_tilelang.py`. Inputs `cb = C@Bᵀ` per
   chunk, `dA_cumsum = cumsum(A·dt)` (log-space), `x`, `dt`, `C`, `prev_states`.
   B/C RMSNorm folds intra-chunk (no extra dispatch — doc §0/§4).
3. **Inter-chunk recurrence (NEW small kernel):** scan the O(S/C) per-chunk
   summaries with associative combine `(A2,B2)·(A1,B1)=(A2·A1, A2·B1+B2)`; feeds
   `prev_states`. The RoPE angle cumsum is a SEPARATE associative scalar prefix
   here (doc §4 risk 1) — reproduce cumulative-angle magnitude exactly.
4. **State→output offset apply (grid-parallel):** the Y_off + skip + silu·z gate.

Carry/replay buffers (`_row_phased_launcher_carry_buffers_for_nodes`,
`_row_phased_replay_buffers_for_nodes`) already plumb the per-chunk boundary states.
Conv needs only a halo of `kernel-1` rows from the previous chunk (doc §4 "Conv").

RULE #1: no fallback. On chunking/parity failure RAISE with where+what; the
per-target codegen choice (grid vs launcher) stays a legitimate gating, not a
silent fallback.

## NOT done (out of scope for this forward prototype)
- Running the TileLang prim_func on Metal (blocked by the gemm-layout backend gap;
  the MLX form is the validated numerical contract).
- The backward transpose scan (step 4 of the plan — gated on this forward).
- Wiring into the live Path-C emitter (handoff above; the serial scan at
  `_append_row_phased_mamba3_body` is the replacement site).
