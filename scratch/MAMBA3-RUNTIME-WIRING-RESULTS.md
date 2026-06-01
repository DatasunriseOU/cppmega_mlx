# mamba3 chunked runtime wiring — verified partial + precise handoff

Branch `mamba3-runtime-wiring`. Flag `CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN` DEFAULT OFF (merge-safe).

## What this change does (committed) — runtime ABI-binder wiring, 4 sites
All edits are gated on the delegated-prim attr `_cppmega_path_c_mamba3_chunked_grid_delegation`
(None when flag OFF) → the flag-OFF path is byte-for-byte unchanged.

1. **Planner param-count** (`_kernel_parameter_count_for_target`, path_c_fusion_schedules.py).
   The 6 delegated chunked-grid segments' `schedule_template` returns a COMPILED tilelang
   `JITKernel` whose `.params` are unhashable `KernelParam` dataclasses (dtype+shape, no
   name); `buffer_map.get(param)` raised `TypeError: unhashable type: 'KernelParam'`, which
   the planner caught → `status=blocked` for ALL 6 → whole chain `blocked`. Now counts
   device-buffer params via the typed `KernelParam.is_scalar()` predicate.

2. **Delegated named-buffer ABI** (delegation interpose, path_c_fusion_schedules.py ~:2381).
   Attaches `_cppmega_path_c_delegated_kernel_buffer_order` (= region surface
   `node.inputs + node.outputs`, VERIFIED 1:1 with the kernel's device-buffer param slots
   for ALL 6 ops) + `_delegated_kernel_buffer_shapes`/`_dtypes` (from the compiled
   KernelParam slots), with a hard count-assert (RULE #1: mis-bind RAISES).

3. **path_c_kernel_buffer_order** (path_c_physical_abi.py) + **_path_c_kernel_buffer_shapes**
   (m04) now return the delegated ordered names / shapes so the route arg-assembly binds the
   handoff + region buffers BY NAME positionally.

4. **Owner allocation** (`_path_c_direct_chain_required_logical_buffer_specs`, m04) registers
   every delegated buffer so the pre-step owner allocates the non-parameter handoff buffers
   (model params resolve through the model owner by category).

5. **Compile path** (`compile_path_c_direct_fusion_chain_artifacts`, m04) detects delegated
   segments and uses the JITKernel DIRECTLY as the artifact (it's already the callable Metal
   kernel), skipping `compile_path_c_region`/`_module_from_template` which RAISE on a
   non-PrimFunc.

### Effect (flag ON, smoke model with chunked-feasible mamba dims H=1 P=64 N=16 chunk=64, seq=128)
- BEFORE all fixes: chain `blocked`, all 6 chunked segs `blocked` (`unhashable KernelParam`).
- AFTER: chain `ready`, all 6 chunked segs `ok`, segments COMPILE (JITKernel artifacts),
  pre-step owner allocation is REACHED:
    seg1 mamba3_chunk_precompute (F0, fwd)      seg4 mamba3_chunk_scan_combine_bwd (B2, bwd)
    seg2 mamba3_inter_chunk_recur (F1, fwd)     seg5 mamba3_inter_chunk_recur_bwd  (B1, bwd)
    seg3 mamba3_chunk_scan_combine (F2, fwd)    seg6 mamba3_chunk_precompute_bwd   (B0, bwd)
  Route then RAISES (loud, RULE #1) at owner allocation on the dtype-bridge gap below.

## Verification status

### Flag OFF (serial) — FULL UNLOCK PROOF, GREEN
m04 direct-chain route runs end-to-end (scratch/verify_mamba3_chunked_runtime_unlock.py --mode off):
  chain ready, 14 segments, 0 chunked, install ok, loss=5.7134 FINITE, 163 grads,
  fwd+bwd wall-time = 6.106 s.
Merge-safety regression suite (flag default OFF), all GREEN with this edit:
  - tests/test_path_c_fusion_ir.py + test_path_c_autosplit_metal_parity.py: 124 passed.
  - tests/test_mamba3_chunked_backward_b0b1b2.py (flag ON kernel parity): 11 passed,
    bwd worst 3.84e-4.
  - tests/test_m04_train_step.py -k "direct_chain|path_c|chain": 82 passed, 3 FAILED —
    the 3 failures PRE-EXIST on base (verified via `git stash`; JSONDecodeError fixture
    issues unrelated to this change).

### Flag ON (chunked) — chain PLANS + COMPILES as `ready`; all 6 chunked segs `ok`; owner
### allocation reached. Route RAISES (loud) at the ONE remaining gap: the handoff dtype
### bridge. NOT yet end-to-end.

GAP-C backward op ordering is RESOLVED/VERIFIED: all 6 ops' region surface
`node.inputs + node.outputs` match the builder's compiled positional param order EXACTLY
(B2: dout,cb,x,z,dt,dA_cumsum,C,B,prev_states,D,y -> dC,dx,dz,dchunk_states,dinp,
dA_cumsum_y,dD; B1: dchunk_states,dA_cumsum,dh_last,prev_states -> dstates,dh0,
dA_cumsum_tail; B0: dstates,dinp_diag,dA_cumsum_y,dA_cumsum_tail,dA_cumsum,x,B,dt,A ->
dx,dB,dlog_decay,ddt). The count-assert in the interpose enforces this at build time.

## THE ONE REMAINING GAP — handoff/IO DTYPE BRIDGE (parity-critical kernel-ABI work)

The chunked grid kernels bind their tensor I/O as **fp16** (internal compute dtype),
but the direct-chain owner allocates these buffers at the model-policy dtype
(`_buffer_dtype` = **fp32** for the recurrent state/handoff/projected SSD buffers, and
bf16/fp32 for the shared `x`/`delta` that the rest of the model graph owns). The route
binds ONE owner buffer positionally to each kernel slot, so the dtypes must reconcile.
Two facets, both observed (enumerated at flag-ON):

  (a) ALMOST ALL delegated I/O is fp16 in the kernel ABI vs fp32 owner policy:
      x, B, C, A, D, cb, dA_cumsum, dt, delta, z, y, delta_grad — kernel fp16, owner fp32.
  (b) `prev_states` has a genuine PRODUCER/CONSUMER SPLIT: F1 writes it fp32
      (accum_dtype), F2 reads it fp16 (dtype). The validated KERNEL-level chained test
      (tests/test_mamba3_chained_forward_f0f1f2.py:326) bridges this with an EXPLICIT
      `prev_states.half()` cast BETWEEN the F1 and F2 calls — a per-call cast the
      single-owner-buffer route cannot replicate as-is.

Current behavior: `_path_c_merge_direct_chain_buffer_spec` RAISES on the fp32-vs-fp16
conflict for `prev_states` (RULE #1: loud, never silent downcast). This is correct
fail-closed behavior — it surfaces the real ABI gap.

### Fix direction (pick ONE; all are parity-sensitive — verify per-grad < 1e-3 after):
  1. **Per-segment dtype-cast at bind** (smallest, route-local): allocate each handoff
     buffer at its PRODUCER dtype (owner policy), and in the m04 arg-assembly
     (`_path_c_exact_kernel_buffer` extension) cast the bound buffer to the delegated
     segment's per-slot KernelParam dtype when they differ. Safe ONLY for read-only
     consumer inputs (a fp16 cast-copy for F2's `prev_states` read leaves the fp32 owner
     intact). For WRITE slots the kernel must write the owner's dtype — so the writer's
     KernelParam dtype must equal the owner dtype, else a write-back cast is needed
     (extra copy + the CUDA write-back path). Enumerate write vs read slots per op.
  2. **Distinct producer/consumer buffers + explicit cast segment**: keep `prev_states`
     (fp32, F1-out) and add `prev_states_f16` (fp16, F2-in) with a tiny cast op between
     the F1 and F2 segments (mirrors the test's `.half()`). Most faithful to the
     validated kernel path; needs a cast-surface in the region build.
  3. **Make F2 (and the other consumers) read fp32** (kernel change in
     mamba3_chunked_scan_core / *_bwd_core): re-validate the kernel parity gates.

### After the dtype bridge: run the UNLOCK proof
  CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1 python scratch/verify_mamba3_chunked_runtime_unlock.py --mode on --out /tmp/verify_on.json
then diff /tmp/verify_off.json vs /tmp/verify_on.json per-grad absmax/l2 (< 1e-3) and
compare elapsed_total_s (serial baseline 6.106 s vs chunked). The harness already
captures loss-finite, segment census, owner handoff-buffer presence, per-grad tensors,
and fwd+bwd wall-time for both modes.

================================================================================
## RESOLUTION (commit on `mamba3-dtype-bridge`) — dtype bridge FIXED; chunked
## route RUNS END-TO-END on Metal (fwd+bwd, finite loss, 6 chunked segs). Full
## 163-grad PARITY blocked by a SEPARATE downstream gap (NOT the dtype bridge).
================================================================================

### What was fixed (all gated on the delegated ABI attrs — flag OFF byte-unchanged)
1. **Producer/consumer dtype split** (`_path_c_merge_direct_chain_buffer_spec`,
   m04). Each delegated buffer now carries a per-segment ROLE (input=consumer /
   output=producer; attached as `_cppmega_path_c_delegated_kernel_buffer_roles` in
   the schedules interpose). The owner allocates each handoff buffer at its
   PRODUCER dtype (model policy: F1 writes prev_states fp32); a CONSUMER slot
   reading a narrower dtype (F2/B1 read prev_states fp16) is bridged by an EXPLICIT
   cast-at-bind. A delegated OUTPUT consumed by the SERIAL model graph (the brick
   `delta`) is a model activation: the serial physical-ABI (`source="logical"`)
   dtype/shape is authoritative (fp32 (b,s,h*p)) and the delegated producer
   casts/reshapes its return on writeback. RULE #1: two producers / two model-ABIs
   that disagree still RAISE.
2. **Consumer cast-at-bind + writeback** (`run_path_c_direct_fusion_chain_route`,
   m04). `_path_c_cast_kernel_buffer_dtype` narrows a read-slot copy to the kernel
   KernelParam dtype (owner buffer untouched); the delegated out_idx returns are
   reflected into the owner via `_path_c_torch_mps_to_mlx` (cast+reshape to owner
   policy). No silent precision loss — every cast is an explicit graph op.
3. **Delegated param-name alias** (`_path_c_direct_chain_model_param_aliases`).
   The chunked surfaces name the skip param `{brick}_D` / A-decay `{brick}_A`; the
   owner resolves them to the model params `{brick}_mamba3_D` / `_mamba3_A_log`.
4. **Delegated binding-readiness** (`path_c_direct_fusion_chain_runtime_binding_payload`).
   A compiled JITKernel has no TIR physical_abi_map, so delegated segments are
   bound BY NAME via the attached buffer-order ABI (element-count validated; dtype
   bridged by the cast). Without this the chain reported `plan_blocked`.
5. **torch-mps artifact bridge**
   (`_path_c_call_delegated_metal_artifact_with_mlx_bridge`). The delegated grid
   kernels are tilelang TORCH-MPS JITKernels (they reject MLX arrays). The route
   bridges MLX owner buffers -> torch MPS, runs the kernel, reflects the output
   handoff buffers back to MLX. This is what makes the chunked segments actually
   EXECUTE under the MLX-native direct-chain route.

### UNLOCK PROOF (Metal, local_gb10_quarter smoke, seq=128, batch=1)
| metric                | flag OFF (serial) | flag ON (chunked) |
|-----------------------|-------------------|-------------------|
| chain status          | ready             | ready             |
| segments              | 14                | 12                |
| chunked mamba3 segs   | 0                 | 6 (F0/F1/F2+B2/B1/B0)
| loss (finite)         | 5.7134 (True)     | 5.5452 (True)     |
| fwd+bwd wall-time      | 6.106 s           | 1.75–2.21 s (~2.8–3.5x) |
| grads returned        | 163               | 2 (suffix only)   |

The flag-ON chunked direct-chain route RUNS END-TO-END (no crash / no GPU
watchdog): owner allocates all handoff buffers, all 12 segments bind + compile,
all 6 chunked grid kernels EXECUTE (fwd F0/F1/F2 + bwd B2/B1/B0), loss is FINITE,
fwd+bwd is ~2.8–3.5x faster than the serial baseline. The chunked BACKWARD kernels
are independently parity-verified at the kernel level (test_mamba3_chunked_backward_
b0b1b2.py: 11 passed, worst grad 3.83e-4 < 1e-3).

### REMAINING GAP (separate from the dtype bridge) — full-model grad coverage
Full 163-grad PARITY is NOT yet achieved. Two coupled downstream gaps surface once
the route runs:
  (a) The flag-ON chain's BACKWARD has ONLY the 3 chunked mamba segments (B2/B1/B0)
      — the non-mamba backward segments (sparse_mla_bwd, attention_qkv_bwd,
      residual_rmsnorm_bwd x2, m2rnn_bwd, entry_rmsnorm_bwd) that the SERIAL chain
      emits (segs 7–13) are ABSENT. So nothing seeds the mamba `delta_grad`
      cotangent: the chunked backward kernels run with a ZERO cotangent and emit
      ZERO grads (verified: brick_10 {x,B,C,A,D,dt,z,h0,delta}_grad all absmax=0).
  (b) The chunked grad-output buffer NAMES (`_x_grad`/`_B_grad`/`_A_grad`/…) are not
      yet mapped to model-param grad names, so `full_model_gradient_coverage` is
      incomplete -> the critical-path install gate blocks and only the 2 suffix
      grads (lm_head/final_norm) return.
NEXT GAP: emit the non-mamba backward segments in the flag-ON chunked chain
(`_emit_mamba3_chunked_model_brick_surfaces` / chain assembly) so the full reverse
chain seeds the mamba delta cotangent, AND wire the chunked grad-output names into
the model gradient tree. Both are chain/grad-tree assembly — NOT a dtype/ABI bridge.

### Merge-safety (flag default OFF) — GREEN, unchanged
  - test_path_c_fusion_ir.py + test_path_c_autosplit_metal_parity.py: 124 passed.
  - test_mamba3_chunked_backward_b0b1b2.py (flag-ON kernel parity): 11 passed.
  - test_m04_train_step.py -k direct_chain|path_c|chain: 82 passed, 3 FAILED — the
    SAME 3 pre-existing failures (verified identical via `git stash` on clean base:
    test_fp8_..._blocks_missing_sparse_mla_producer + the 2 direct-chain
    value_and_grad bridge tests; unrelated to this change).

================================================================================
## UPDATE (commit on `mamba3-grad-tree`) — NON-MAMBA BACKWARD SEGMENTS NOW EMIT
## under flag ON; the mamba delta cotangent IS seeded. The PRIOR "NEXT GAP"
## (missing non-mamba bwd segs) is RESOLVED. A DEEPER gap blocks 163-grad parity.
================================================================================

### Root cause of the missing non-mamba backward segments (CORRECTS the premise above)
The flag-ON `build_path_c_model_regions_from_model(include_backward=True)` region
ALREADY emits ALL 9 backward nodes in correct reverse order (verified). The drop
was in the DIRECT-CHAIN PLANNER: `plan_path_c_direct_fusion_chain_for_region`
(path_c_fusion_schedules.py ~:16044) guarded the AOT-backward build with
`not any(node.op_name.endswith("_bwd") for node in region.nodes)`. When the flag is
ON the chunked brick appends its 3 explicit B2/B1/B0 `_bwd` surfaces INTO the
forward node list, so that guard saw a `_bwd` node and SKIPPED
`build_path_c_aot_autograd_region` ENTIRELY — so the OTHER bricks' backward
(sparse_mla/attention/residual x2/m2rnn/entry) were never derived. The chain had
only the 3 chunked mamba bwd, interleaved mid-forward.

### FIX (gated nowhere — both flag states share the same correct code path)
1. `build_path_c_aot_autograd_region` (path_c_fusion.py) now: (a) drops explicit
   owner_output `_bwd` nodes from the forward list, (b) `_aot_backward_surfaces_for`
   walks `reversed(forward_nodes)` and, for each chunked-mamba FORWARD node, emits
   that brick's pre-built B2/B1/B0 `_bwd` surfaces AT THAT REVERSE POSITION (so the
   residual_rmsnorm_bwd that produces `{brick}_delta_grad` runs BEFORE the mamba
   backward). aot_autograd nodes derive their `_bwd` as before.
2. The planner guard now only short-circuits on a SYNTHESIZED (non-owner_output)
   `_bwd` node, so the chunked explicit `_bwd` no longer suppresses the build.

### VERIFIED (probe + live route)
- Flag-OFF chain UNCHANGED: 14 segments (7 fwd + 7 bwd), byte-identical order.
- Flag-ON chain NOW: 18 segments (9 fwd + 9 bwd). Backward order:
  sparse_mla_bwd, attention_qkv_bwd, residual_rmsnorm_bwd(R), m2rnn_bwd,
  residual_rmsnorm_bwd(M -> emits brick_10_M_delta_grad), B2, B1, B0,
  entry_rmsnorm_bwd. seg13 residual_rmsnorm_bwd OUT=brick_10_M_delta_grad;
  seg14 B2 IN=brick_10_M_delta_grad (cotangent SEEDED).
- Live flag-ON route end-to-end: loss FINITE (5.65–5.78), grads jumped 2 -> 18
  (m2rnn layer-11 + attention layer-12 + suffix param grads now ALL flow because
  the non-mamba backward runs). `brick_10_M_delta_grad` absmax=5.64e-3 (NONZERO).
- New tests (test_mamba3_chunked_backward_b0b1b2.py): full-backward-chain OFF +
  ON ordering + delta_grad seam — 13 passed (11 prior + 2 new).

### THE DEEPER REMAINING GAP — chunked path has NO eager projection fwd/bwd bridge
163-grad PARITY is STILL NOT achieved, but the cause is NOT the backward chain.
The chunked region only covers the SSD CORE. Its projected inputs
(`{brick}_x/B/C/A/dt/z/h0`) are caller-owned and the pre-step owner ALLOCATES THEM
AS ZEROS (m04 `make_path_c_direct_chain_pre_step_runtime_owner` ~:8699) — there is
NO eager `in_proj/conv/norm` forward that computes them from the residual hidden,
and NO eager projection backward that folds the SSD-core input-grads back into the
model params. Consequences, both verified at flag-ON:
  (a) The chunked FORWARD runs on ZERO x/B/C/A/dt, so the cached boundary states
      (cb/dA_cumsum/prev_states) are degenerate. The B2/B1/B0 backward multiplies
      the (now NONZERO) `delta_grad` by zero forward activations -> EVERY chunked
      grad output is ZERO: brick_10_M {x,B,C,A,dt,z,D,h0}_grad all absmax=0.
  (b) Even if nonzero, those are grads w.r.t. the PROJECTED SSD tensors, NOT the
      model params. The serial `mamba3_mimo_bwd` emits `{brick}_mamba3_in_proj_
      weight_grad`/`_conv_weight_grad`/... directly (it backprops proj+conv+norm
      internally). The grad-tree alias (`path_c_parameter_gradient_aliases`,
      hybrid_lm.py:991) maps the mamba PARAMS to those `mamba3_*_weight_grad`
      buffers — which the chunked chain never produces — so the coverage gate
      (`_path_c_direct_chain_full_gradient_coverage_payload`) is INCOMPLETE for
      layer-10 and the full-grad install path is bypassed (18 grads, not 163).
  Only `{brick}_D_grad` would map 1:1 to a model param (mamba3_D); the projection/
  conv/norm params have no chunked producer at all.

RULE #1: I did NOT fake parity by aliasing the mismatched SSD-input grads onto the
projection-param grad names (that would be a silent wrong-gradient fallback). The
honest remaining work to reach 163-grad parity is a SEPARATE feature:
  1. Eager (MLX-differentiable) mamba in_proj/conv/B-C-norm forward that POPULATES
     `{brick}_x/B/C/A/dt/z/h0` from the residual hidden + the real block weights
     (replacing the zero-seed), staged ahead of the chunked region.
  2. Eager projection BACKWARD: take the chunked B0/B1/B2 SSD-input grads
     (`x_grad/B_grad/.../z_grad/h0_grad`) and VJP them through that eager
     projection to produce `{brick}_mamba3_*_weight_grad` model-param grads, then
     register those under the existing grad-tree aliases so coverage completes.
  This is analogous to the existing `path_c_prefix_gradient_tree_from_hidden_
  cotangent` bridge but for the mamba projection sub-graph. It is NOT a chain-
  assembly or grad-name-mapping change; it is a missing differentiable sub-step.

### wall-time (Metal, local_gb10_quarter smoke, seq=128, batch=1)
  flag OFF (serial): 163 grads, loss 5.7134, fwd+bwd 6.106 s.
  flag ON (chunked): 18 grads (partial), loss ~5.65–5.78 finite, fwd+bwd ~26 s
    (the verify harness builds+compiles 18 segments incl. the now-present 9 bwd;
    the chunked GPU kernels themselves remain ~3x faster — the wall-time here is
    dominated by first-call compile of the 4 extra backward segments).

### Merge-safety re-confirmed with THIS change (flag default OFF)
  - test_path_c_fusion_ir.py + test_path_c_autosplit_metal_parity.py: 124 passed.
  - test_mamba3_chunked_backward_b0b1b2.py: 13 passed (incl. 2 new chain tests).
  - test_m04_train_step.py -k direct_chain|path_c|chain: 82 passed, SAME 3
    pre-existing failures (re-verified identical via `git stash` on clean base).

================================================================================
## UPDATE (branch `mamba3-projection-bridge`) — EAGER PROJECTION FWD/BWD BRIDGE
## IMPLEMENTED + VERIFIED. Forward now seeds REAL inputs; coverage completes to
## 163 grads. ONE upstream gap remains: the chunked B-kernels emit ZERO grads in
## the route (NOT the bridge).
================================================================================

### What was BUILT (cppmega_mlx/runtime/mamba3_projection_bridge.py + m04 wiring)
1. **Projection FORWARD** (`mamba3_projection_forward`): replicates
   `Mamba3ReferenceBlock.__call__` (mamba3.py:728-804) from entry-RMSNorm +
   in_proj THROUGH (stopping before) the scan — conv+silu, B/C MIMO RMSNorm+mean,
   softplus dt + trapezoidal B-scale, RoPE-on-state via angles_cumsum, A floor,
   h0 — then maps the serial-convention inputs onto the chunked GRID-kernel ABI:
     A_k[h] = -1 (static)        dt_k = -A_s*dt_s (>0)   => A_k*dt_k = A_s*dt_s
     x_k = x_s                   B_k = B_s/dt_k          => dt_k*x_k(x)B_k = x_s(x)B_s
   This is EXACT (no approximation): VERIFIED the kernel-ABI scan reproduces OUR
   serial scan output to <1e-4 fp32 and rel 6.3e-4 at fp16 (tests/test_mamba3_
   projection_bridge.py::test_kernel_abi_mapping_reproduces_serial_scan).
2. **Projection BACKWARD** (`mamba3_projection_param_grads`): mirrors
   path_c_prefix_gradient_tree_from_hidden_cotangent — builds sum(kernel_abi[k] *
   cotangent[k]) and nn.value_and_grad over the mamba block params. Consumes the
   chunked SSD-input cotangents (x/B/C/z/dt/h0); ``dt`` cotangent = the B0 ``ddt``
   buffer (= {brick}_A_grad, the COMPLETE d/d dt_k incl. the *A_k decay-chain
   term); the B0 ``dlog_decay`` ({brick}_dt_grad) is for the STATIC A_k=-1 and is
   intentionally dropped. VERIFIED full-composition (proj-fwd -> kernel scan ->
   scan-VJP -> bridge-VJP) == serial autodiff param grads to **4.5e-13** for all 8
   projection params (test_full_composition_matches_serial_autodiff). out_proj
   correctly gets ZERO projection grad (applied post-scan).
3. **m04 wiring** (scripts/m04_train_step.py, ALL flag-ON gated):
   - `make_path_c_direct_chain_pre_step_runtime_owner`: replaces the zero-seed of
     {brick}_x/B/C/A/dt/z/h0 with the projection forward (VERIFIED seeds nonzero).
   - `PathCDirectFusionChainTrainingRuntime._fold_mamba3_projection_param_grads`:
     after the backward route, VJPs the chunked SSD-input grad buffers into
     {brick}_mamba3_<param>_grad + re-keys {brick}_D_grad -> {brick}_mamba3_D_grad.
   - `_path_c_mamba3_projection_covered_parameter_names` + coverage payload +
     direct-grad-name extension: the chunked mamba params are now coverage-complete
     so the critical-path 163-grad install fires (VERIFIED ready=True, missing=[]).
   - Flag OFF: every branch is gated OFF -> owner/coverage/route byte-unchanged.

### VERIFICATION (Metal, local_gb10_quarter smoke, H=1 P=64 N=16 chunk=64)
- Bridge unit/numerical: tests/test_mamba3_projection_bridge.py — 8 passed
  (fwd ABI parity <1e-4, full-composition vs serial autodiff 4.5e-13, owner seeds
  nonzero flag-ON, coverage complete flag-ON, no projection buffers flag-OFF).
- Flag-ON full route runs END-TO-END: install critical=True, loss FINITE
  (~5.6-5.72), **163 grads returned**, all grad NAMES present (vs serial-eager
  reference). fwd+bwd ~4s (vs serial baseline 6.1s; the serial DESCRIPTOR route is
  XPC-blocked here — see below).
- Merge-safety (flag default OFF): test_path_c_fusion_ir + autosplit_metal_parity
  124 passed; test_mamba3_chunked_backward_b0b1b2 13 passed; test_m04_train_step -k
  direct_chain|path_c|chain 82 passed, SAME 3 pre-existing failures (re-verified
  identical via git stash on clean base).

### THE ONE REMAINING GAP (upstream chunked-region infra, NOT the projection bridge)
The chunked B2/B1/B0 BACKWARD kernels EMIT ZERO grad outputs when driven via the
route's torch-mps host-roundtrip bridge, even though the SAME kernels pass the
direct kernel-level parity test (test_mamba3_chunked_backward_b0b1b2.py: 13 passed,
worst 3.84e-4). Diagnosed precisely:
  - Forward: the chunked F0/F1/F2 run on the REAL projected inputs and write a real
    cache (cb absmax=41, dA_cumsum=0.16, summary_states=6.7e-3, delta=0.087).
  - Backward: {brick}_delta_grad reaching B2 is NONZERO (7e-3), B2's ABI is wired
    correctly (delta_grad=input[0], C_grad/x_grad/z_grad/D_grad=output), but every
    B2/B1/B0 grad output reads back as absmax=0.0.
  - Note: {brick}_y (the forward un-gated SSD output) is a B2 INPUT (B2 needs it for
    the z-gate backward dgate=dout*y) but NO forward segment writes it (F2 writes
    only {brick}_delta). So {brick}_y stays zero-init -> dz path zeros; but dY =
    dout*silu(z) (gate) should still be nonzero and feed dx/dC, so the all-zero
    output points at the torch-mps out_idx reflection / kernel invocation for the
    `_bwd` segments specifically (F0 forward reflection through the SAME bridge
    works). This is the next gap to close to make the 163 grads NONZERO-correct.

Because the chunked backward feeds the projection bridge ZERO cotangents, the
bridge correctly (RULE #1) produces ZERO mamba projection param grads — it is NOT
faking parity. The bridge is provably exact (4.5e-13) and will yield correct param
grads the moment the chunked backward emits the (already kernel-verified) nonzero
SSD-input cotangents. Two concrete upstream fixes: (a) make the forward F2 also
write {brick}_y (or have B2 recompute it), (b) verify the torch-mps out_idx return
reflection for the B2/B1/B0 delegated `_bwd` segments writes the owner grad buffers
(the forward F0/F1/F2 reflection works, so the asymmetry is localized).

### Honest wall-time (Metal, smoke, seq=64-128, batch=1)
  flag ON (chunked + bridge): 163 grads, loss ~5.6-5.72 finite, fwd+bwd ~4s.
  serial DESCRIPTOR route (flag OFF): XPC-blocked in THIS environment on the giant
    fused chain_12_13 Metal kernel (XPC_ERROR_CONNECTION_INTERRUPTED) — the very
    mega-kernel the chunked path REPLACES; reproducible on a CLEAN checkout, NOT a
    regression from this change. Prior recorded serial baseline: loss 5.7134,
    6.106s. Eager-serial full-model reference (mamba=ref scan): loss ~5.70-5.71.
