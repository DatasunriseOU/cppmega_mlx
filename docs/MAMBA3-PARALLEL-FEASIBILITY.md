# mamba3 / m2rnn Backward: Chunked Parallel-Scan Feasibility Verdict

Status: feasibility investigation (NOT a build commitment)
Author: lead architect
Date: 2026-05-31
Verdict: **GO-WITH-CAVEATS** (staged; gate on parity + a forward prototype before committing the backward rewrite)

---

## 0. TL;DR

The mamba3 (and m2rnn) backward today runs the reverse-time selective scan as **one
threadgroup on Metal and one CUDA block** (`T.Kernel(1, threads=1024)`), serial over
S=4096 reverse steps, ~0.35-0.47 s/step after the atomic->RMW fix. That is ~1/40 of an
M4 Max (40 GPU cores) and ~1/48 of GB10 (48 SMs, live-queried).

The recurrence **is** a diagonal selective SSM, `h_t = exp(A_t·dt_t)·h_{t-1} + B_t·x_t`,
with a **scalar-per-head decay** and **no cross-state coupling** — the textbook associative
linear recurrence that Mamba-1 (CUDA), Mamba-2 (SSD), and Flash-Linear-Attention (Triton)
all parallelize in production. The backward is its exact **transpose** (a reverse associative
scan), verified in-source. So the restructure is algorithmically solved and reference-proven
in three independent codebases.

The earlier pessimism rested on two **false premises**, both corrected here:

1. **"~12K per-step launches."** FALSE. A length-L selective scan is ONE fused kernel in
   Mamba-1 (fwd AND bwd). The chunked-parallel (SSD/FLA) form has exactly **O(S/C)
   sequential inter-chunk steps** — S=4096, C=256 → **16**; C=64 → 64. Not 4096, not 12K.

2. **"ngroups=1 B/C RMSNorm forces a cross-7168-lane per-step reduction."** FALSE. Verified
   at `path_c_fusion_schedules.py:6588-6618`: the B/C RMSNorm reduces **only over
   state_dim (~128) per (rank,group)** and is already lane-parallel across `mimo_rank*groups`
   lanes. It is an intra-block reduction that **folds into each chunk** — it does NOT force
   an extra dispatch boundary.

The realistic win is **largest and least-risky in the backward**: today it replays up to
`MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL=8` full forward recomputes (two dense matvecs + conv +
dt + B/C-RMSNorm-rope) per reverse step purely to regenerate scratch
(`:11520-11537`). Storing forward intermediates instead of replaying removes that ~8x
recompute on the replayed portion — and is a strictly smaller, lower-blast-radius change than
a full Blelloch rewrite.

---

## 1. Realistic speedup per backend (with reasoning)

**These are occupancy ceilings and literature-anchored ranges, NOT measured wall-time on our
hardware. No measured speedup for this specific op exists in-repo; any single multiplier would
be fabricated. Re-measure before promising a number.**

### Metal (M4 Max, 40 GPU cores)
- **Spatial occupancy ceiling: ~40x** (single threadgroup → full grid). This is an upper
  bound, not a wall-time prediction.
- **Realistic wall-time: materially below 40x**, bounded by:
  - Amdahl serial tail: the inter-chunk recurrence (O(S/C) ≈ 16 steps) and any group-norm
    reduction stay sequential.
  - The single resident block already hides some memory latency today.
  - Chunked scan adds per-chunk prefix-state recompute overhead.
- **Honest band: high single-digit to low double-digit x** on the recurrent-scan portion,
  with the dominant near-term win coming from removing the 8x backward replay (below).

### CUDA (GB10, 48 SMs — live `cudaDeviceGetAttribute` MultiProcessorCount=48)
- **Spatial occupancy ceiling: ~48x** (single block → 48 SMs). Confirmed single-block in
  generated device code (`__launch_bounds__(1024, 1)`, recurrent reverse-time loops carry no
  `blockIdx`).
- **Realistic wall-time: below 48x**, same Amdahl/latency-hiding caveats.
- **Chunked + tensor cores (SSD/FLA style):** steps 1/2/4 become matmuls; matmul FLOPs run
  ~16x faster than non-matmul FLOPs on modern GPUs. Published FLA/SSD kernels beat
  FlashAttention-2 by ~2-5x. On GB10 this is the path to actually exploit tensor cores that
  the single-block scan cannot touch.

### Literature anchors (re-measure before quoting on our HW)
- Fused selective scan vs naive non-fused PyTorch: **20-40x** (Mamba paper §4.5, A100,
  specific shapes — partly IO/fusion-driven, not pure scan parallelism).
- Chunked (SSD/FLA) adds **~2-5x** over Mamba-1-style scan via tensor cores.
- vs a TRUE single-block serial recurrence (our case), the headroom is plausibly **larger**
  than the 20-40x-vs-PyTorch figure (a single block underuses the GPU far more than naive
  PyTorch) — but there is **no direct citation** for that exact comparison. Treat as estimate.

### Launch-count reality (the corrected core fact)
| | Today | Chunked-parallel target |
|---|---|---|
| Recurrent scan | `T.Kernel(1, threads=1024)`, serial over S=4096 | `T.Kernel(num_chunks, ...)`, num_chunks=S/C (e.g. 4096/64=64 blocks, 1 wave) |
| Sequential dependency | S=4096 reverse steps | **O(S/C) inter-chunk steps** (16 @ C=256, 64 @ C=64) |
| Dispatch boundaries | ceil(S/8) sequential launches (LAUNCHER_CHUNKS, watchdog) | ~3: per-chunk pass + inter-chunk combine + offset-apply |
| B/C RMSNorm | intra-block, over state_dim per (rank,group) | **stays intra-chunk** — no extra dispatch |

The existing `LAUNCHER_CHUNKS` path (`rows_per_kernel_launch=8`) is a **sequential** re-launch
to dodge the macOS GPU watchdog, NOT parallelism — each launch still carries state serially.
The parallel-scan target replaces that serial chain with one concurrent wave + a tiny combine.

---

## 2. Gradient-parity risk

**Risk level: MEDIUM. Reordered reductions → within-fp-tolerance, NOT bitwise.**

The backward is **already non-bitwise-reproducible today**:
- Relaxed atomics on colliding gradient accumulators (`c_group_grad`, `b_group_grad`,
  `a_grad`, `dt_grad` at `:11667-11696`; `hidden_grad`, `d_grad`) make FP summation order
  nondeterministic run-to-run.
- Lane-disjoint non-atomic RMW for owner-exclusive grads (`in_proj_weight_grad`) is documented
  byte-identical to the atomic and ~10x cheaper.

Because FP addition is non-associative, a chunked-parallel scan (matmul + cross-chunk scan +
atomicAdd) **will NOT match the serial reference bit-for-bit**, and two chunked runs can differ
(atomic order, chunk size, reduction-tree shape). This is the **standard, accepted behavior of
every production SSM kernel** (Mamba-1 bwd uses `gpuAtomicAdd` for exactly this reason). A
chunked scan that changes only the cross-chunk combine ORDER of additive grads stays in the
**same tolerance regime the kernel already lives in** — it is not a regression in determinism.

The decay terms `exp(A·dt)` are recomputed identically per position from stored projections, so
A_t / B_t / C_t are reproduced exactly; only additive-grad combine order changes.

### Two parity hotspots that need explicit handling
1. **RoPE angle cumsum** (`:6387-6392`) is itself a running prefix-sum of `proj_angle·dt`. A
   chunked scan must reproduce the **same cumulative-angle magnitude** per position (a separate
   associative scalar prefix-scan), because cos/sin of the cumsum (`:6688-6719`) is sensitive
   to angle magnitude — recombining chunk angle offsets in the wrong order drifts the B/C
   rotation. Precedent exists (already checkpointed/reconstructed in the backward), but the
   parallel-combine order is a tolerance risk to validate.
2. **Chunk-cumulative decay underflow.** `A ≤ -0.01` (clamped at `:6384-6386`) with
   `dt=softplus`, so per-step decay < 1; a long chunk's product `∏ exp(A·dt)` can underflow to
   0. The serial path never forms this product. The parallel scan MUST form chunk-cumulative
   decay in **log-space (sum of A·dt) and exponentiate once**, never as a naive product of
   per-step exp values.

### How to validate (the gate)
- **Tolerance-based, NOT bitwise.** Max-abs-diff and relative-error gradient check vs the
  current direct-chain on EVERY grad output: `in_proj/out_proj/conv/dt_bias/B_norm/C_norm/D/h0`.
- Targets: rel err ~1e-3..1e-2 for bf16/fp16 accumulation, tighter for fp32 accumulators.
- Run on **BOTH** Metal and CUDA against the committed golden anchors (`docs/parity_anchors.md`).
- Add a chunk-size sweep (C ∈ {64,128,256}) to confirm the result is stable under reordering
  (within tolerance) — this is the de-facto contract for all SSM kernels.

---

## 3. Effort estimate

**Full chunked-parallel-scan rewrite (both ops): LARGE — multi-week, one engineer.**

The scan core is compact and provably associative (fwd `:6730-6752`, bwd `:11248-11257` /
`:11663-11696`), BUT it is embedded in:
- mamba3 bwd emitter `_append_row_phased_mamba3_bwd_body` ≈ **1445 lines** (`:10748-12193`).
- m2rnn bwd emitter `_append_row_phased_m2rnn_bwd_body` ≈ **842 lines** (`:9693-10535`).
- Containing file: 15,536 lines, with interlocking scratch buffers, lane-disjoint RMW
  invariants, forward/backward checkpoint coupling, cross-launch carry, and per-target
  (Metal/CUDA) divergence (`cuda_target` branches).

A parallel-scan rewrite touches the scan core AND the checkpoint/replay AND the cross-launch
carry simultaneously — **high blast radius**. The hard new kernel is the associative combiner
for the diagonal-SSM transition, with the RoPE angle cumsum across chunk boundaries as the
trickiest piece.

**Lower-risk high-value first step (days-to-1-week):** eliminate the backward checkpoint-replay
by persisting forward per-step intermediates (or per-chunk boundary states). This captures the
largest identified cost (the ~8x replay of two dense matvecs + conv per reverse step,
`:11520-11537`) **without rewriting the scan as Blelloch**. The comment at `:11493-11519` itself
documents this replay as the dominant body — and notes the Metal `sync_threads`-in-replay-loop
compiler crash that a stored-intermediate design also sidesteps.

### What is reusable (reduces effort)
- Multi-PrimFunc staging: `path_c_descriptor_stage_prim_funcs` (`:1028-1051`) already
  materializes multiple stage PrimFuncs with `execution_stage` / `row_dispatch_mode` /
  `rows_per_kernel_launch` — the separate-kernel decomposition the chunked scan needs.
- State handoff: `_row_phased_launcher_carry_buffers_for_nodes` (`:2445-2470`) carries
  `mamba3_angle_state` / `mamba3_conv_state` / `m2rnn_h_state` / `m2rnn_conv_state` across
  launches; `_row_phased_replay_buffers_for_nodes` (`:2473-2492`) provides
  `mamba3_h_checkpoint` / `angle_checkpoint` / `angle_grad_state`.
- TileLang primitives: `T.Kernel(*blocks, threads=...)` (1-3D grid), `T.alloc_shared` /
  `T.alloc_fragment` / `T.alloc_local` for intra-chunk scratch — all present.
- A working chunked-parallel reference already runs in this codebase (imported FLA
  `chunk_gated_delta_rule_fwd_kernel_h` in `cppmega_v4/_tilelang/linear_attention_path_d_real.py`)
  — algorithm validated; the gap is re-expressing it in the Path C TileLang emitter.

---

## 4. Recommended chunked-parallel-scan approach

### Chunk size
Start **C=64** (matches existing `DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH=8` granularity scaled up,
and `MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL=8`; literature default 64-256). S=4096 → 64 chunks
(64 concurrent blocks, ≥ one full wave on both 40-core Metal and 48-SM GB10). Sweep
{64,128,256} for the perf/parity tradeoff; larger C = fewer inter-chunk steps but more
intra-chunk work and more decay-underflow pressure (mitigated by log-space cumulative decay).

### Pipeline (4-step SSD/FLA decomposition)
1. **Precompute (grid-parallel, position-local).** All per-step coefficients depend ONLY on
   `hidden[row]` (and `hidden[row+1]` for the trapezoid term), NOT on recurrent state: in-proj
   matvec (`:6298-6322`), causal conv (`:6324-6366`), dt=softplus, A=−softplus clamp
   (`:6368-6386`), RoPE angle increment, trapezoid, B/C RMSNorm+rope. Emit these as a fully
   grid-parallel stage (no scan dependency). This is where most of the dense-matvec FLOPs live
   and they parallelize **without any scan change**.
2. **Intra-chunk parallel scan.** `T.Kernel(num_chunks, threads=...)`; each chunk computes a
   local prefix scan over its C steps + a chunk summary `(A_chunk, B_chunk)`, where
   `A_chunk = exp(Σ A·dt)` computed in **log-space** (sum the exponent, exponentiate once —
   never a product of per-step exp). The B/C RMSNorm is an **intra-chunk** reduction over
   state_dim per (rank,group) — it stays inside this kernel, no separate dispatch.
3. **Inter-chunk recurrence.** A small kernel scanning the O(S/C) per-chunk summaries
   (associative combine `(A2,B2)·(A1,B1) = (A2·A1, A2·B1+B2)`), producing per-chunk entry
   states. Cheap (16-64 steps over `groups`-wide state). The RoPE angle cumsum is scanned here
   as a **separate associative scalar prefix** and re-added to each chunk's offset to preserve
   cumulative-angle magnitude.
4. **State-to-output / offset apply.** Re-apply each chunk's entry-state offset to its local
   scan (grid-parallel again) to produce final per-position outputs.

### Conv handling
The causal depthwise conv is a **stencil**, embarrassingly parallel per position, **fully
separable from the scan**. The backward already abandons the ring-history buffer and recomputes
conv per-position from a local `hidden[t-history..t]` window (`:10942-10997`). Each chunk
computes its conv independently, needing only a **halo of `kernel-1` rows** from the previous
chunk. No carried conv ring-state along the chunk boundary.

### Backward / transpose scan
The adjoint is the **exact transpose**: `dh_{t-1} = exp(A_t·dt_t)·(dh_t + C_t·dY_t)` — verified
at `:11663-11666` (inject readout grad) and `:11693-11696` (`dh_prev = scalar3·exp(A·dt)`),
carried at `:12169`. Same diagonal decay, equally associative → a **reverse parallel scan**
(same 4-step decomposition, time-reversed). Per-position weight-grad accumulations
(`a_grad`/`dt_grad`/`b_group_grad`/`c_group_grad` + dense in/out-proj scatters) are additive and
order-independent up to fp summation order (already relaxed-atomic today).

### Group-norm reduction kernel
The B/C RMSNorm folds into the per-chunk intra-block reduction (step 2) — it does **NOT** need a
standalone kernel, because it reduces only over state_dim per (rank,group) and is position-local
(not part of the recurrence). If a separate pass is ever needed, it composes from per-chunk
partial sum-of-squares; verify the partials compose correctly (additional correctness surface).

### Caller-owned state handoff (reuse existing plumbing)
Reuse the multi-segment direct-chain plumbing: `path_c_descriptor_stage_prim_funcs` for the
3-stage decomposition, and the carry/replay buffer machinery
(`_row_phased_launcher_carry_buffers_for_nodes` / `_row_phased_replay_buffers_for_nodes`) for
per-chunk boundary states. The repo already proves cross-launch state carry is bit-faithful for
the scan state (`mamba3_state`, `angle_state`, `h0_grad`/`dh`, `angle_grad_state`). The NEW
piece is the inter-chunk PARALLEL combiner (the existing carry path is sequential).

---

## 5. Key risks

1. **RoPE angle cumsum across chunk boundaries** — must reproduce cumulative-angle magnitude
   exactly (trig sensitivity); associative scalar prefix-scan, but combine order is a tolerance
   risk to validate.
2. **Chunk-cumulative decay underflow** — must use log-space (Σ A·dt), never a product of
   per-step `exp` values. The serial path never forms this product, so this is a NEW failure
   mode the parallel form introduces.
3. **Metal compiler constraint** — `T.sync_threads()` inside a data-dependent nested loop
   crashes MTLCompilerService (`:11511-11519`). Any per-chunk combine with barriers inside
   per-chunk loops can re-trigger it; the durable design MUST be barrier-free-per-lane
   (m2rnn-style), as the comment prescribes.
4. **Parity re-validation surface** — 8 grad outputs × 2 backends × chunk-size sweep; bitwise
   parity WILL break (expected); tolerance parity must be re-anchored.
5. **High blast radius** — ~2287 lines of monolithic backward emitter, interlocking scratch /
   lane-disjoint RMW invariants / forward-backward checkpoint coupling / Metal-CUDA divergence.
6. **Profile-first uncertainty** — the dominant cost may be the position-local dense matvecs +
   the 8x replay, NOT the scan serialization per se. The dense matvecs already parallelize
   WITHOUT any scan change. **Profile to confirm where time goes before committing to the full
   Blelloch rewrite** — removing the replay may capture most of the win at a fraction of the
   risk.
7. **RULE #1 — no silent fallback.** A chunked-scan rewrite must NOT add a "fall back to
   single-block if chunking fails" path. On any chunking/parity failure it must RAISE with
   where+what. The existing target gating (grid_chunks vs launcher_chunks) is a legitimate
   per-target codegen choice, not a fallback — keep it that way.

---

## 6. Incremental de-risking plan

Each step is independently shippable and gated on parity before the next.

1. **PROFILE FIRST (1-2 days).** Instrument the current backward on both backends. Quantify the
   split between (a) the 8x checkpoint-replay of dense matvecs+conv, (b) the position-local
   dense matvecs themselves, (c) the recurrent scan serialization. This decides whether the
   scan is even the bottleneck. **No code change commits before this.**
2. **Kill the backward replay (days-to-1-week).** Persist forward per-step intermediates (or
   per-chunk boundary states) instead of re-running up to 8 full forward recomputes per reverse
   step (`:11520-11537`). Lower blast radius, no scan rewrite, also sidesteps the Metal
   `sync_threads`-in-replay-loop crash. **Gate: tolerance grad-parity on both backends.** Likely
   captures a large fraction of the available win on its own.
3. **Prototype the FORWARD chunked scan (1-2 weeks).** Implement steps 1-4 for the forward only
   (precompute → intra-chunk scan → inter-chunk combine → offset apply), log-space cumulative
   decay, separate angle prefix-scan. Validate forward outputs vs the serial path (tolerance,
   chunk-size sweep). Forward is lower-stakes (no grad accumulation reordering). **Gate: forward
   parity on both backends before touching the backward.**
4. **Port the BACKWARD transpose scan (multi-week).** Reuse the forward decomposition,
   time-reversed; reorder grad accumulation; re-anchor tolerance parity on all 8 grad outputs ×
   2 backends × chunk sizes. **Gate: full grad-parity vs golden anchors.**
5. **Tune chunk size + tensor-core matmul on CUDA.** Sweep C ∈ {64,128,256}; on GB10 push
   steps 1/2/4 onto tensor cores (the SSD ~16x matmul motivation) which the single-block scan
   cannot exploit. **Gate: re-measure wall-time; confirm the projected speedup is real.**

---

## 7. Recommendation: GO-WITH-CAVEATS

**Rationale.**
- The algorithm is **proven and reference-implemented** in three independent production
  codebases (Mamba-1 CUDA, Mamba-2 SSD, FLA Triton). The recurrence here is verified to be the
  exact diagonal selective-SSM form with a scalar-per-head decay and no cross-state coupling —
  directly associative-scan-able; the backward is its exact transpose.
- The two blockers prior agents raised are **both false**: the launch count is O(S/C) ≈ 16-64
  sequential steps (not 12K), and the ngroups=1 B/C RMSNorm is a tiny intra-block reduction over
  state_dim, not an all-lane cross-threadgroup reduction.
- Spatial headroom is real and large (~40x Metal / ~48x CUDA occupancy ceiling), though
  wall-time will land below that (Amdahl + latency hiding).
- Parity is **already non-bitwise today** (relaxed atomics), so the reordered chunked reductions
  stay in the same tolerance regime — a controllable, validatable risk, not a blocker.

**Why caveats, not unconditional GO:**
- **No measured speedup exists** for this op on our hardware — the verdict must not promise a
  multiplier. The plan is gated on re-measurement.
- The full rewrite is **multi-week, high-blast-radius**.
- A **profile (step 1) must run first** — the dominant cost may be the 8x replay and the
  position-local dense matvecs, which removing the replay (step 2) and grid-parallelizing the
  matvecs address **without** the full Blelloch rewrite. Committing the whole scan rewrite
  before profiling risks multi-week effort for marginal scan-specific gain.

**Net:** GO to start, but staged — profile, then kill the replay, then prototype the forward
scan and validate parity, and only then commit the backward transpose rewrite. Reassess after
each gate. Do not commit the full multi-week backward rewrite until the profile and forward
prototype confirm the scan serialization is actually the bottleneck and the speedup is
measured, not assumed.
