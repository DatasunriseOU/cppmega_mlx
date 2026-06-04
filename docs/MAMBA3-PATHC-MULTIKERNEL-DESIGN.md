# Mamba3 chunked-SSD inside Path-C: multi-kernel hosting design

Status: design / implementation-ready. Author handoff doc.
Date: 2026-05-31.

Scope keyword: **mamba3-as-3-PLAIN-SEGMENTS reusing the existing multi-segment caller-owned
state-handoff.** This is the recommended approach. The "new multi-kernel-segment support"
option is rejected. A hybrid is unnecessary.

---

## 0. The crux, restated

The validated chunked-SSD mamba3 (fwd 37.6x, bwd 155x; `chunk_scan_fwd_metal_prim`
compiles + runs on Metal) is **multi-kernel**: it needs at least a precompute kernel, an
inter-chunk recurrence kernel (the one O(S/C) sequential stage), and a scan+combine kernel,
each with a *different launch grid*.

Path-C's lowerer contract forces **exactly one PrimFunc / one `T.Kernel` per segment**:

- `_single_entry_prim_func` raises `ValueError("tilelang_single_entry_lowerer requires
  exactly one entry, got {n}")` when an `IRModule` has `len(funcs) != 1`.
  File: `cppmega_mlx/runtime/path_c_fusion.py:2819-2842` (raise at `:2833-2836`).
- The descriptor schedule template emits a *single* `kernel_line` (one `with T.Kernel(...)`)
  per segment, and every fused node's fragment is appended *inside that one kernel*.
  File: `cppmega_mlx/runtime/path_c_fusion_schedules.py:2866-2937` (single `kernel_line`
  `:2866-2875`; one `@T.prim_func` body `:2877-2882`; per-node fragment loop `:2919-2937`).

Therefore the multi-kernel scan **cannot** be authored as one `mamba3_mimo` node that emits
3 internal `T.Kernel` blocks — that is the forbidden "multi-kernel segment".

**The decision:** the chain planner already chains N *independently compiled* segments and
hands state between them through **caller-owned ABI buffers** — with zero planner changes,
and a production precedent (`attention_qkv_projection` writes a full-sequence KV workspace in
an EARLIER segment; the separate `sparse_mla_fp8_apply` segment READS it). So we make
mamba3's chunked scan **3 distinct op-NODES** (forward) / 3 (backward), each its own
single-entry segment, wired by caller-owned `cb / dA_cumsum / prev_states / chunk_states`
scratch buffers. Each segment satisfies the one-kernel contract trivially because each is its
own PrimFunc with its own grid.

---

## 1. Recommended approach + rationale

**mamba3-as-3-PLAIN-SEGMENTS (reuse the existing multi-segment caller-owned state-handoff).**

Why this and not new multi-kernel-segment support:

1. **The chaining + cross-segment handoff is FREE and already production-proven.** The chain
   planner (`plan_path_c_direct_fusion_chain_for_region`,
   `path_c_fusion_schedules.py:15223-15324`) is op-name-agnostic: it greedily walks
   `region.nodes`, emits one segment per maximal node window, advances `start = best.node_end`,
   and repeats. State flows segment->segment purely through shared caller-owned ABI buffers.
   The `attention_qkv_projection -> sparse_mla_fp8_apply` workspace handoff is exactly the
   `[produce-workspace][consume-workspace]` pattern a chunked split needs, and it is live at
   full scale.
2. **The "one node -> 3 segments" assumption only lives in TWO places** (region build +
   registry), both small edits — NOT in the planner, NOT in the ABI, NOT in the caps.
   Verified: `forward_max_segment_nodes` / `backward_max_segment_nodes` /
   `_effective_forward_max_segment_nodes_for_window` operate purely on node counts/op-names
   (`path_c_fusion_schedules.py:15255-15317`); a 3-node expansion just participates like any
   other ops.
3. **The allocator/validator/binder for global scratch is fully reusable as-is.** Spill ->
   ABI param, max-extent sizing, per-segment slicing, and the b330bdb single-pool flatten all
   already exist and are exercised at full scale by the reverse-scan state. No new allocator,
   no new validation, no new orchestration.
4. **New multi-kernel-segment support would require rewriting the lowerer contract and the
   descriptor template's single-`T.Kernel` emission** — large blast radius, and it
   duplicates capability the multi-segment chain already provides.

A hybrid (some scan stages inside one segment via internal `T.Kernel` blocks) is rejected for
the same reason: any second `T.Kernel` in a segment violates `_single_entry_prim_func`.

---

## 2. Forward + backward -> segment/kernel mapping

The proven SSD chunked forward is 4-5 logical kernels; they collapse to **3 single-entry
Path-C segments**. The backward is the exact analytic transpose, also **3 segments**.

### Forward (3 segments)

| Seg | New op-node name             | Grid / launch shape                                   | Reads (caller-owned)                          | Writes (caller-owned)                       |
|-----|------------------------------|-------------------------------------------------------|-----------------------------------------------|---------------------------------------------|
| F0  | `mamba3_chunk_precompute`    | grid-parallel, row-phased over S (existing K0 stages) | block ABI inputs (`_MAMBA3_REAL_ABI_INPUTS`)  | `x,B,C,z,A,dt`, **cb**, **dA_cumsum**, per-chunk **summary_states** |
| F1  | `mamba3_inter_chunk_recur`   | small O(S/C) serial, grid `(nheads,)` or `(1,...)`    | summary_states, h0, RoPE-angle-cumsum prefix  | **prev_states** (inter-chunk entry states), final_state |
| F2  | `mamba3_chunk_scan_combine`  | 3D grid `(nheads, ceil(C/bM)*ceil(P/bN), batch*nchunks)` | cb, dt, dA_cumsum, C, **prev_states**, D, x | `delta`/Output (Y_diag+Y_off+D*x, silu(z) gate) |

- F0 = K0 PRECOMPUTE (in-proj matvec, causal conv, dt=softplus, A=-softplus clamp, RoPE angle
  increment, trapezoid, B/C RMSNorm+rope) PLUS forming `cb = C@Bᵀ`, `dA_cumsum = cumsum(A·dt)`,
  and the per-chunk summary states (K1's `decay_states` einsum folds in here). Fully
  position-local / grid-parallel, NO scan dep.
- F1 = K3 INTER-CHUNK recurrence: the **only** sequential stage, O(S/C). Produces
  `prev_states` (the per-chunk entry states). Carries the RoPE-angle-cumsum scalar prefix
  across the chunk axis.
- F2 = the validated `chunk_scan_fwd_metal_prim` (K1 intra-chunk Y_diag + K4 state->output
  Y_off + skip + silu·z gate fused into ONE prim_func). This kernel already exists and runs:
  `cppmega_mlx/nn/_tilelang/mamba3_chunked_scan_core.py:101-188`. Its inputs are exactly
  `cb, x, dt, dA_cumsum, C, prev_states, D` -> `Output` (`:119-164`).

Replaces the serial scan emitted today at `_append_row_phased_mamba3_body`
(`path_c_fusion_schedules.py:6417`, serial block at `:6986-7028`,
`mamba3_scan_policy: external_state_recurrence`).

### Backward (3 segments — mirror, time-reversed)

| Seg | New op-node name                | Grid / launch shape                  | Role |
|-----|---------------------------------|--------------------------------------|------|
| B0  | `mamba3_chunk_precompute_bwd`   | grid-parallel (same as F0)           | boundary-state stash / recompute log-space cumsums; output/gate transpose (dz via silu', dy, dD, dx-skip, split dY->dY_diag/dY_off); OUR-outer-op grads (RoPE-angle VJP, B/C RMSNorm VJP, conv VJP, trapezoid VJP) |
| B1  | `mamba3_inter_chunk_recur_bwd`  | small O(S/C) **reverse** scan        | upper-tri decay_chunk contraction -> dstates, dh0, dA_cumsum via `_segsum_vjp` (the ONE genuinely new kernel) |
| B2  | `mamba3_chunk_scan_combine_bwd` | 3D grid (same as F2 intra-chunk)     | state->output + per-chunk-state transpose -> dC, dinp->dx/dB, dA_cumsum scatter; da-assembly (reverse-cumsum + segsum-VJP -> dlog_decay -> dA/ddt/softplus) |

This **eliminates the 8x checkpoint-replay** that dominates the current serial backward,
because the forward F1 already materialized the per-chunk boundary states; backward reuses
them. Replaces the ~1445-line serial reverse scan at `_append_row_phased_mamba3_bwd_body`
(`path_c_fusion_schedules.py:11081`).

**Total: 6 Path-C segments (3 fwd + 3 bwd).** Each is one single-entry PrimFunc via the
existing descriptor-stage plumbing. ESTIMATE: F0/B0 precompute may further sub-split if the
descriptor planner stages in-proj vs conv vs norm separately, pushing to 4+4; 3+3 is the
minimal scan-faithful count.

### OUR-outer-op attachment

All OUR-specific outer-ops attach to the **precompute segment** (F0 forward / B0 backward),
OUTSIDE the scan core, all position-local & grid-parallel:

- **RoPE-angle-cumsum**: a SEPARATE associative scalar prefix. It is carried/scanned in the
  inter-chunk segment (F1 / B1), re-added per chunk offset to preserve cumulative-angle
  MAGNITUDE exactly. Its grad is reverse-cumsum of `angle·dt` (same pattern as `da`).
- **B/C RMSNorm**: folds INTRA-chunk into per-position dB/dC, reduces only over `state_dim`
  per (rank,group). No separate dispatch.
- **causal conv**: a stencil, fully separable; each chunk needs only a `kernel-1` halo from
  the previous chunk. Grad is the stencil transpose.
- **trapezoidal scale**: position-local (`dt[t], dt[t+1]`); grad position-local.

None touch the scan dependency, so they live entirely in F0/B0 and never need a separate
segment.

---

## 3. Exact touch-points (file:line + change)

### 3.1 Region construction — emit 3 surfaces instead of 1 (PLANNER input)

- **File:** `cppmega_mlx/runtime/path_c_fusion.py:1152` (`build_path_c_model_regions_from_model`),
  mamba emission at `:1590-1609`.
- **Today:** one `FusionKernelSurface.path_c(op_name="mamba3_mimo", inputs=(route_hidden,
  "mamba_state", *_MAMBA3_REAL_ABI_INPUTS), outputs=(delta, state))`.
- **Change:** behind a config flag (`mamba3_chunked_scan: bool`), emit THREE surfaces:
  - `mamba3_chunk_precompute` — inputs `(route_hidden, mamba_state, *_MAMBA3_REAL_ABI_INPUTS)`;
    outputs the scratch handoff buffers (`mamba3_cb`, `mamba3_dA_cumsum`,
    `mamba3_summary_states`, plus the per-position `x/B/C/z/A/dt` it already stages).
  - `mamba3_inter_chunk_recur` — inputs `(mamba3_summary_states, mamba3_h0,
    mamba3_angle_cumsum)`; outputs `(mamba3_prev_states, scan_state)`.
  - `mamba3_chunk_scan_combine` — inputs `(mamba3_cb, mamba3_x, mamba3_dt,
    mamba3_dA_cumsum, mamba3_C, mamba3_prev_states, mamba3_D)`; outputs `(delta, ...)`.
  - `_MAMBA3_REAL_ABI_INPUTS` is at `path_c_fusion.py:110-122`; the handoff buffers are NEW
    logical buffers added to the region's buffer set (see 3.3).
- **Backward:** AOT autograd already synthesizes `_bwd` op-names; with 3 forward nodes you get
  3 `_bwd` nodes (B0/B1/B2). The registry's `descriptor_for` auto-derives `_bwd` descriptors
  (`path_c_fusion_schedules.py:1207-1232`) but for a real transpose you register explicit ones
  (3.2).

### 3.2 Registry — register 3 new descriptors with fragment_emitters

- **File:** `cppmega_mlx/runtime/path_c_fusion_schedules.py`,
  `default_path_c_brick_schedule_descriptor_registry` at `:1247`, current mamba descriptors at
  `:1329-1369` (`mamba3_mimo`) and `:1370+` (`mamba3_mimo_bwd`).
- **Why required:** `PathCFusionScheduleRegistry.select` keys targets by the op-name signature
  tuple (`:14402-14424`); `descriptors_for_signature` returns `None` for any unknown op_name
  (`:1234-1244`), and the chain then RAISES `no descriptor target for op signature`
  (`:15318-15324`). So each new op-node MUST have a registered `PathCBrickScheduleDescriptor`.
- **Change:** register 6 descriptors: `mamba3_chunk_precompute(+_bwd)`,
  `mamba3_inter_chunk_recur(+_bwd)`, `mamba3_chunk_scan_combine(+_bwd)`. Each needs its own
  `fragment_emitter`:
  - `mamba3_chunk_scan_combine` -> delegates to `chunk_scan_fwd_metal_prim`
    (`mamba3_chunked_scan_core.py:101`); it ALREADY emits the 3D `T.Kernel(nheads,
    ceildiv(chunk,bM)*ceildiv(headdim,bN), batch*nchunks)` grid.
  - `mamba3_inter_chunk_recur` -> a small O(S/C) serial kernel (NEW; architectural inference
    from the integration comment at `:6430-6459`, not yet coded).
  - `mamba3_chunk_precompute` -> reuses the precompute stages already inside
    `_append_row_phased_mamba3_body` (`:6460+`), minus the serial scan.
- Note: these descriptors emit a grid `T.Kernel` (not the row-phased `T.Kernel(1,...)`).
  `mamba3_chunk_scan_combine` is NOT a row-phased single-thread kernel; its descriptor's
  `preferred_loop_policy` must select a grid/flat policy, and `_is_row_phased_mamba3` gating
  in the template loop (`:2925`) must NOT swallow it.

### 3.3 ABI allocator — register the handoff buffers as global scratch

Two equivalent mechanisms exist; pick by the hosting segment's free-slot budget.

- **File:** `cppmega_mlx/runtime/path_c_fusion_schedules.py`, `_spill_large_shared_scratch_to_abi`
  at `:4226-4404`.
- **Mechanism A (preferred while slots allow):** let the precompute kernel declare
  `cb / dA_cumsum / prev_states / summary_states` as `T.alloc_shared`; they exceed
  `DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES = 1024` (`:129`) so they spill
  UNCONDITIONALLY to top-level `T.Tensor(...)` ABI params (`:4344-4348`, `:4394-4398`), and
  get recorded in `_cppmega_path_c_spilled_shared_scratch_shapes` (set at `:2422`). Multi-dim
  buffers each consume ONE kernel-buffer slot.
- **Mechanism B (if slot-constrained):** flatten the multi-dim buffers to 1-D float32 and route
  them through ONE coalesced pool bank — `_pool_oversized_shared_scratch_to_metal_workspace`
  (`:10924-11078`, the b330bdb pattern), reindexed via `_replace_one_dimensional_buffer_refs`
  (`:4181-4223`). Single slot total. NOTE: the coalesced/pool path is **float32/int32 only**
  (`_can_coalesce_spilled_scratch` `:4129-4133`; pool float32 guard `:11037-11043`). If the
  chunked path lands fp16 these guards RAISE — either keep the handoff buffers fp32 (preferred
  for `prev_states` precision) or add a typed fp16 pool.
- **Registration for force-spill (deterministic naming):** add the 4 handoff names to a
  force-spill frozenset (mirror `DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS` at `:236-249`)
  so the spill is forced even if a future shape shrinks below threshold, and so they land as
  stable named ABI params across the producer and consumer segments.

### 3.4 Cross-segment max-extent sizing + per-segment slicing (LAUNCHER — already done)

These exist and need NO change; they auto-discover the new spilled buffers:

- `_path_c_internal_scratch_abi_specs` collapses coalesced-bank entries to
  `max(existing_extent, offset+extent)`.
  File: `scripts/m04_train_step.py:7776-7814` (max at `:7804-7806`).
- `_path_c_merge_direct_chain_buffer_spec` keeps the LARGER shape across segments for
  `source=='scratch'` (`m04_train_step.py:7672-7693`, per research).
- Allocation once via `mx.zeros(spec['shape'])` (workspace owner `:8104-8107`).
- Validation accepts an over-sized bank (`:7847-7858`, per research): RAISES on
  missing/shape/dtype mismatch — fail-loud, no fallback.
- Launch-time slice+bind: `_path_c_exact_kernel_buffer` (`m04_train_step.py:4729-4761`)
  reshapes the max-merged bank to `(-1,)`, slices `[:expected_size]`, reshapes to the segment's
  declared shape — a zero-copy prefix view satisfying the tvm-ffi exact-element-count binder.
- The run loop wires it all: `run_path_c_direct_fusion_chain_route` at
  `m04_train_step.py:6559`; per-segment `scratch_specs = _path_c_internal_scratch_abi_specs(...)`
  (`:6635`), validate (`:6636-6647`), append scratch names to ordered kernel buffers
  (`:6648-6660`), bind positionally with `_path_c_exact_kernel_buffer` (`:6693-6705`).

**No new orchestration code** — only the kernel-side `T.alloc_shared`->spill annotation /
force-spill name registration (3.3).

### 3.5 (Optional) launcher carry/replay helpers

`_row_phased_launcher_carry_buffers_for_nodes` (`:2530-2555`) and
`_row_phased_replay_buffers_for_nodes` (`:2558-2577`) match on `op_name=='mamba3_mimo'` /
`'mamba3_mimo_bwd'`. The NEW op-names get NO auto-carries — which is **correct and cleaner**:
the chunked split plumbs `cb/dA_cumsum/prev_states/summary_states` as ordinary caller-owned ABI
scratch (3.3), exactly like the attention/mla precedent, instead of the launch-to-launch
within-one-kernel carry these helpers provide. Leave them unchanged unless F1/B1 themselves get
launcher-chunked (then extend with the new inter-chunk op-name).

---

## 4. New global scratch buffers + sizes

Shapes from `chunk_scan_fwd_metal_prim` (`mamba3_chunked_scan_core.py:119-164`). Full-scale
dims for `local_gb10_quarter`: batch=1, S=4096, nheads=112, headdim=64, dstate=16, ngroups=1.
`MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE = 64` (`path_c_fusion_schedules.py:6297`) -> nchunks=64.

| Buffer            | Shape                                         | Elems (chunk=64) | fp16   | fp32   |
|-------------------|-----------------------------------------------|------------------|--------|--------|
| `mamba3_cb`       | (batch, nchunks, ngroups, chunk, chunk)       | 262,144          | 0.50MiB| 1.00MiB|
| `mamba3_dA_cumsum`| (batch, nheads, nchunks, chunk)               | 458,752          | 0.88MiB| 1.75MiB|
| `mamba3_dt`       | (batch, nheads, nchunks, chunk)               | 458,752          | 0.88MiB| 1.75MiB|
| `mamba3_prev_states` | (batch, nchunks, nheads, headdim, dstate)  | 7,340,032        | 14.0MiB| 28.0MiB|
| `mamba3_summary_states` | same shape as prev_states (per-chunk)   | 7,340,032        | 14.0MiB| 28.0MiB|

Sensitivity: `prev_states` dominates and scales ~1/chunk_size (= nchunks). At chunk=128
(config default) it halves. **Total is trivial vs device RAM (~tens of MiB).** Memory is NOT
the gate.

**The real gate is the 31-buffer portable kernel ABI limit**
(`DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT = 31`, `:128`). The spiller computes
`available_parameters = max(0, 31 - existing_parameter_count)` (`:4274-4277`) and RAISES
`descriptor stage ABI scratch parameter budget exceeded` (`:4291-4295`) if a forced spill
cannot fit. The current fused backward already uses ~28 of ~31 slots (`:10934-10935`). The
NEW chunked segments are SMALLER per-segment than the monolithic backward, so per-segment slot
pressure is lower — but if any one segment nears the limit, route the multi-dim handoff buffers
through Mechanism B (one coalesced fp32 pool bank = 1 slot).

---

## 5. Scope / effort estimate

| Work item | Files | Effort |
|-----------|-------|--------|
| F0/B0 precompute segment (split existing precompute stages off the serial scan) | `path_c_fusion_schedules.py` (`_append_row_phased_mamba3_body`/`_bwd_body`) | LOW-MED |
| F2 scan+combine descriptor (delegate to existing `chunk_scan_fwd_metal_prim`) | `path_c_fusion_schedules.py` registry + emitter; `mamba3_chunked_scan_core.py` reused | LOW |
| F1 inter-chunk recurrence kernel (small O(S/C) serial; NEW) | new prim builder + emitter | MED |
| Region build: 1 -> 3 surfaces (flagged) | `path_c_fusion.py:1590-1609` | LOW |
| Register 6 descriptors + fragment_emitters | `path_c_fusion_schedules.py:1247-1369` | LOW-MED |
| Spill registration for 4-5 handoff buffers | `path_c_fusion_schedules.py:4226-4404`, `:236-249` | LOW |
| B1 reverse inter-chunk combiner (NEW, upper-tri) + da-assembly | new prim builders + emitters | MED-HIGH |
| OUR-outer-op grads folded into B0 (RoPE-angle/RMSNorm/conv/trapezoid VJPs) | `_append_row_phased_mamba3_bwd_body` | LOW each |

**Overall: forward ~days; backward ~1-2 weeks** (high blast radius: scan core +
checkpoint-replay elimination + cross-launch carry change land together). Allocator/validator/
binder = ZERO new code. Planner = ZERO changes.

---

## 6. Key risks

1. **Gradient parity (highest).** The backward is the exact analytic transpose; MLX parity is
   1.30e-4 worst-grad across 7 tensors and chunks {64,128,256}, no NaN — but the TileLang Metal
   backward transpose is UNVERIFIED on Metal. The B1 reverse inter-chunk combiner is genuinely
   new. Gate: per-grad-tensor `max|abs|` vs the serial backward must stay < 1e-3 before the
   serial path is retired. RULE #1: the grid-vs-serial choice is an explicit gate; on
   chunking/parity failure the helpers RAISE with where+what — never silently fall back to the
   serial scan.
2. **The one-node -> multi-segment planner change.** The split MUST be 3 distinct op-NODES at
   region-build time, NOT one node emitting 3 internal `T.Kernel` blocks (that violates
   `_single_entry_prim_func` `path_c_fusion.py:2833-2836` and the single-`kernel_line` template
   `:2866-2875`). Risk is mis-registering: any unknown op_name -> `select` returns None ->
   segment blocked (`:15318-15324`). Mitigation: register all 6 descriptors before flipping the
   region-build flag.
3. **Inter-chunk recurrence (F1/B1).** The ONLY O(S/C) sequential stage; correctness depends on
   `prev_states` being FULLY MATERIALIZED before F2 reads it — the same materialization
   invariant the `attention_qkv_projection -> sparse_mla_fp8_apply` precedent already satisfies.
   The RoPE-angle-cumsum across chunk boundaries is the trickiest associative prefix; F1 and its
   reverse B1 must reproduce the cumulative-angle MAGNITUDE exactly.
4. **31-buffer ABI limit, not memory.** `:128`, `:4291-4295`. If a hosting segment nears the
   cap, the 3-4 multi-dim handoff buffers must flatten into ONE fp32 coalesced pool (Mechanism
   B), not land as standalone params.
5. **TileLang Metal gemm-layout backend gap.** `Unsupported gemm combination A:local.fragment
   B:shared.dyn` for swizzled dynamic-shared `x`. The landed forward works around it (block_N=16
   + `shared.dyn` `x_shared`, `mamba3_chunked_scan_core.py:181`); the backward transpose grid
   hits the same path and is UNVERIFIED on Metal. This is a codegen gap, NOT an algorithm gap.
6. **fp16 coalesced-pool dtype guard.** If handoff buffers are fp16 and routed through the
   coalesced bank, `_can_coalesce_spilled_scratch` (`:4129-4133`) and the pool float32 guard
   (`:11037-11043`) RAISE. Keep `prev_states`/`summary_states` fp32 (also better precision) or
   add a typed pool.

---

## 7. Staged de-risking plan

**Stage 1 — Forward F2 only, behind shadow.** Register `mamba3_chunk_scan_combine` delegating
to the already-landed `chunk_scan_fwd_metal_prim`. Feed it `cb/dA_cumsum/prev_states`
*recomputed eagerly* (not yet from F0/F1) to isolate the scan-combine kernel from the
precompute/recurrence split. **Parity gate:** out `max|abs|` vs the serial forward < 5e-4 fp16
at S in {256, 4096}.

**Stage 2 — Forward F0+F1+F2 chained.** Emit the 3 forward surfaces (region-build flag) and
3 descriptors; let the chain compile them and hand off `cb/dA_cumsum/prev_states` through the
caller-owned scratch ABI. **Parity gate:** out `max|abs|` and `h_last` vs serial forward
< 1e-5 fp32 / < 5e-4 fp16, across chunk {64,128,256} incl non-pow2 S, no NaN. Confirm the
inter-chunk RoPE-angle-cumsum magnitude matches.

**Stage 3 — Backward, only after Stage 2 passes.** Build the B1 reverse inter-chunk combiner +
da-assembly + B0 outer-op grads; emit the 3 `_bwd` surfaces/descriptors. **Parity gate:**
per-grad-tensor `max|abs|` vs the serial backward < 1e-3 across chunk {64,128,256}, no NaN; bf16
equal-or-better than serial vs fp32 truth. Verify the 8x checkpoint-replay is gone (timing) and
the loss-cotangent bridge still sees a clean forward/backward execution-phase split (the chain
already refuses to cross it, `:15249-15254`).

**Stage 4 — Flip default + retire serial.** Only once Stages 2-3 parity gates hold at full
scale (`local_gb10_quarter`), make the chunked path the default for tile-aligned shapes; the
serial scan remains the RAISING fallback path for non-tile shapes (explicit gate, never silent).

---

## 8. Go / no-go inputs

**GO if:**
- Stage 2 forward parity < 1e-5 fp32 / < 5e-4 fp16 holds at S=4096, chunk in {64,128,256}.
- Stage 3 backward parity < 1e-3 per-grad-tensor holds at the same shapes.
- All 6 descriptors register and `select` resolves the 6 op-name signatures (no
  `no descriptor target` raise).
- The hosting segments fit the 31-buffer ABI budget (or the fp32 coalesced pool absorbs the
  handoff buffers into 1 slot).
- The B1 reverse inter-chunk combiner compiles + runs on Metal (gemm-layout gap worked around
  as in the forward).

**NO-GO / blockers:**
- Backward grad parity cannot be brought under 1e-3 on Metal (vs MLX-only 1.3e-4) — keep
  `mamba3_mimo_bwd` MONOLITHIC (serial) while forward goes chunked; the two are independent
  segments, so a chunked forward + serial backward is a valid intermediate.
- TileLang Metal gemm-layout gap blocks the B1/B2 grids and cannot be worked around — forward-
  only ships; backward stays serial.
- ABI slot budget cannot be met even with the single-pool flatten — revisit chunk_size (larger
  chunk shrinks `prev_states` and slot pressure) before considering any contract change.

**Decisive principle (RULE #1):** the chunked-vs-serial selection is always an explicit RAISING
gate keyed on shape feasibility (`_mamba3_chunked_forward_scan_feasibility`,
`path_c_fusion_schedules.py:6300`). There is NO silent fallback: if a selected chunked dispatch
fails parity or feasibility, it RAISES with where+what; the serial path is only ever chosen up
front for explicitly-classified non-tile shapes, never as a degraded catch.

## 9. Mono-fused backward port (the cppmega mono-chunk port) — flag-gated, gb10-only

**Status: BUILT + Metal-compile-verified on Apple (Mac); CUDA build+measure is the GB10 phase.**

The cppmega Mamba3 backward mono-chunk kernel
(`cppmega/megatron/cuda_ext/mamba3_mono_chunk_skeleton.cu`) is ONE kernel, grid
`(nchunks, head, batch)`, that keeps the chunk state resident in shared, runs ONE large WMMA
GEMM (`LKQ = K@Q^T`) and REUSES that tile from shared BEFORE any global write, with the
masked/3-index terms scalar-threaded in the same kernel. The path_c port of that structure is
`bwd_mono_cuda_prim` / `build_bwd_mono` in `mamba3_chunked_backward_core.py`
(`MAMBA3_BWD_MONO_OP_NAME = "mamba3_bwd_mono"`).

**What fuses (the honest maximum):** ONE CTA per `(chunk,head)` runs the FULL §27 four-GEMM B2
body (`DYX = dY@x^T`, `dC_off = dY@prev_states`, `dC_diag = M@B`,
`dchunk_states = (decay·dY)^T@C` as `T.gemm`, the `chunk_scan_combine_bwd_cuda_prim_gemm` body
verbatim) AND the dinp_diag-dependent HALF of B0 (`dx_diag = dt·sum_n dinp_diag·B`,
`dB_diag = dt·sum_p dinp_diag·x`, `ddt_diag = sum_{p,n} dinp_diag·x·B`) by keeping the
per-position `dinp_diag[l,p,n]` RESIDENT in a shared `DINP[L,P·N]` tile and consuming it before
any global re-read (the cppmega "reuse before any global write" move). This eliminates the
largest B2→B0 handoff buffer's round-trip (`dinp_diag` is fp32 `(b,S,H,P,N)`).

**What CANNOT fuse (stated up front, RULE #1):** B1 (`inter_chunk_recur_bwd`) carries a reverse
adjoint ACROSS chunks → a per-`(chunk,head)` CTA cannot express it without cooperative-grid sync;
it STAYS a separate kernel (2.41 ms, cheap). B0's dstates-COUPLED half needs `dstates` (the OUTPUT
of B1) → it STAYS in the post-B1 B0 kernel. The mono-mode caller feeds that B0 a ZEROED
`dinp_diag` (mono owns the dinp_diag→dx/dB/ddt contribution; B0 adds only the `_states` half — the
math is LINEAR in `dinp`, so the split is exact). So the maximal honest fusion is **B2 +
dinp_diag-B0** in one kernel; B1 + the dstates-coupled B0 remainder stay separate.

**SMEM WALL (MEASURED structural NO-GO at prod):** the resident `DINP[L,P·N]` fp32 tile is
`L·P·N·4`. At prod `L=P=N=64` that is **1,048,576 B**, and the total mono smem
(§27 GEMM 88.5 KB + DINP 1.0 MB) is **1.139 MB ≫ the gb10 ~99 KB budget**. `build_bwd_mono(cuda)`
RAISES with where+what at prod dims (RULE #1, no silent over-budget launch / no fallback). The
resident-dinp_diag mono fusion ONLY fits at small `L·P·N` (envelope: `L=16,P=32,N=32` = 79 KB FIT;
`L=16,P=16,N=16` = 22 KB; the `--nano` probe cfg exercises it). This is exactly the §27 prediction
that a per-CTA resident dinp tile is ~64× the GEMM operands. **Metal: `build_bwd_mono(metal)`
RAISES unconditionally** — the §27 4-GEMM fragment-staging layout alone needs 72.5 KB (the Metal
B2-GEMM gate already raises at 32 KB), and the resident DINP tile only worsens it → mono is
gb10-only; the serial 6-kernel Metal chain stays the Apple path.

**PREDICTED OUTCOME (NO-GO for throughput, three prior measurements of this kernel class):**
(1) §27 measured the four-GEMM B2 prim (which mono reuses verbatim) at **0.749×** vs v1 (1371.5 vs
1027.5 ms, same gb10 box) — single-64-tile m16n8k16 GEMMs' staging+sync exceed the serial
reductions; fusing B0's dinp_diag work does not shrink them. (2) cppmega's identical mono-chunk
WMMA kernel closed at **11.155 ms vs TileLang 3.707 ms = 3.0× SLOWER**
(`mamba3_mono_cuda_chunk_wave10_final`: "Do not merge any monolithic CUDA chunk implementation as
a production path"). (3) §18 measured the B2 kernels occupancy-SATURATED at bs1 (v2 dstate-split
0.997×), so removing the dinp_diag round-trip + one launch saves latency, not compute.

**Flag + measure plan (GB10 phase):** `build_bwd_mono` is invoked only by the probe A/B
(`scratch/probe_chunked_backward_cuda_gb10.py --bwd-mono-ab`); the §17 447.8 ms 6-kernel chain is
byte-identical when the mono path is not invoked (no caller wiring change — the mono builder is a
NEW additive entry point, the 6 builders/grids/op-names are untouched). GB10 mutex + free>105 GB +
SIGTERM-never-9 per CLAUDE.md, then:
`--bwd-mono-ab` alone (prod cfg) reports the **SMEM-NO-GO build RAISE** (the structural wall);
`--bwd-mono-ab --nano` (in-budget `L=16,P=32,N=32`) BUILDS+RUNS+MEASURES the mono kernel vs the §27
B2 GEMM + the separately-computed dinp_diag-B0 half, printing `mono_ms / b2gemm_ms / speedup /
verdict` and the per-grad `vs_ref_max_abs` (mono outputs must equal §27 B2 + the dinp_diag fold).
A mono SLOWER than the multi-kernel path is reported NO-GO, never silently used. The Mac side
verified: prim constructs at prod dims, all gates (MMA-divisibility, gb10 budget, Apple budget)
RAISE correctly, and the body lowers to a Metal `JITKernel` (race-free) at the 32 KB-fit config —
the dinp_diag-B0 fold algebra matches the B0 prim's `ddt_inp = sum_{p,n} dinp·x·B` exactly.
