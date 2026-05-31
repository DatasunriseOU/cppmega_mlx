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
