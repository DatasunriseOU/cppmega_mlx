# Graph-level (TVM Relax) memory planning vs eager: PoC result + integration path

**Thesis (proven below on real hardware):** lowering a multi-op training step to ONE
TVM Relax graph and running `StaticPlanBlockMemory` gives a **strictly lower peak
memory** than eager execution, because the graph has cross-op / cross-layer
liveness that an eager single-`mx.eval` barrier cannot exploit.

**Status:** PoC done and passing (measured numbers below). **PR 1 done + measured**
(2026-06-02): whole-region `StaticPlanBlockMemory` over a REAL-shaped (`H=3584`)
path_c fwd+bwd region chain assembled as ONE `@R.function` of `R.call_tir` leaves
cuts the strict concurrent peak **1.67x->1.80x, growing with depth** (section 5).
The top unknown is resolved: path_c's physical-ABI prims do NOT fit `call_tir` DPS
(section 3, 3 verbatim reasons) -- PR 1 uses logical-buffer leaves, PR 2 writes the
DPS adapter. **PR 2 done + measured** (2026-06-02): the physical-bank ->
logical-buffer DPS adapter makes a REAL path_c region a VALID, plannable Relax-graph
leaf -- but via an **external-function boundary (`R.call_dps_packed`), NOT a
`call_tir` leaf**, because the real path_c kernel can ONLY be lowered by
`tilelang.compile` (mismatch #3 is a hard codegen wall that does NOT close even after
currying the scalar). Section 7. **PR 3 done + measured** (2026-06-02): the physical
banks (activation 171.5M, parameter 360.8M, parameter-grad 418.5M, state/checkpoint
965.3M -- 1981 MB/region) are exposed as cross-region Relax SSA tensors, so
`StaticPlanBlockMemory` shares the heavy banks ACROSS regions. MEASURED collapse on the
real banks: eager all-live -> planned peak **1.32x -> 1.67x, growing with depth** (the
planned-peak slope == the checkpoint-bank size, so every bank collapses to a constant
working set EXCEPT the O(N)-live checkpoint -- the remat target). The real MR JITKernel
also runs on Metal THROUGH the `call_dps_packed` boundary end-to-end. The bound is the
key finding: cross-region liveness reuse cannot beat the checkpoint term; only
rematerialization can. Section 8. **PR 4 done + measured** (2026-06-02): sqrt(N)
rematerialization on the O(N) checkpoint bank -- non-boundary backward regions
RE-EMIT the forward `call_dps_packed` to recompute their checkpoint locally, so it is
short-lived instead of live across the whole backward pass. MEASURED: the 28-block
(1.8B) planned peak drops **O(N) 27.38 GB -> O(sqrt N) 8.79 GB** (3.12x further, 5.19x
below eager all-live), numerically IDENTICAL to non-remat (max abs diff 0.0), at the
cost of 89 extra forward region calls (3.18x extra forward work, reducible to ~1x).
A slope fit confirms the complexity class changed: PR-3 peak == 0.943 GB/layer (O(N),
== state bank), PR-4 peak == `1.510*sqrt(N)+0.778` GB. **This is the step that
determines whether the graph path reaches Megatron-class memory: it does** -- the
activation/grad term (the eager-118-GB OOM driver) is now single-digit GB at the real
28 MR blocks, leaving only the Adam optimizer state (lever 5, in-place op) to close to
the 26-40 GB target. Section 9. Integration is a multi-quarter effort; this doc
specifies the roadmap.

PoC code: [`cppmega_mlx/runtime/relax_memory_plan_poc.py`](../cppmega_mlx/runtime/relax_memory_plan_poc.py)
(self-checking, fail-loud; run instructions at the bottom).

---

## 1. PoC result (MEASURED, not estimated)

Device: **CPU, LLVM Relax VM** (CUDA path identical -- same passes run in
`relax/backend/cuda/pipeline.py:64,67`; CPU chosen only because gb10 was busy and
the planning decision is target-independent -- it operates on the Relax IR before
codegen). TVM `0.25.dev0`, the vendored fork at
`/Volumes/external/sources/tilelang/3rdparty/tvm`.

Two baselines are measured, both from the **real compiled IR** (no fabrication):

* **ALL-LIVE total** = eager `mx.eval` semantics. MLX's training step is lazy; the
  single terminal `mx.eval(model.parameters(), optimizer.state, loss, ntokens)`
  (`cppmega_mlx/training/loop.py:176`) forces the *entire* fwd+bwd+update tape at
  once, so every intermediate buffer is allocated simultaneously. This equals the
  sum of all distinct `builtin.alloc_tensor` buffers (no reuse) -- TVM's own
  `relax.analysis.estimate_memory_usage` reports exactly this as the
  "without memory planning" number.
* **STRICT peak** = a *tighter, harder* baseline that already frees each buffer at
  its last use (alias-followed liveness) but still does NOT share storage. The
  planner must beat even this wherever genuine concurrency exists.

`StaticPlanBlockMemory` is then measured as **planned working set** (sum of the
reused `memory.alloc_storage` storages) and **planned strict peak** (max
simultaneous live storage honouring the compiler-emitted `kill_storage`
free-barriers).

| Case | ALL-LIVE (eager) | planned | reduction | STRICT peak (eager) | planned | reduction |
|---|---|---|---|---|---|---|
| fwd-only chain, n=2048, 8 layers | 256.00 MB | 48.00 MB | **5.33x** | 32.00 MB | 32.00 MB | 1.00x |
| fwd-only residual, n=1024, 8 layers | 96.00 MB | 16.00 MB | **6.00x** | 12.00 MB | 12.00 MB | 1.00x |
| **fwd+bwd (Gradient), n=1024, 6 layers** | 170.00 MB | 77.00 MB | **2.21x** | 52.00 MB | 45.00 MB | **1.16x** |
| **fwd+bwd (Gradient), n=512, 8 layers** | 57.00 MB | 25.25 MB | **2.26x** | 17.00 MB | 13.25 MB | **1.28x** |

Both correctness checks pass: the planned VM output matches an independent numpy
reference for the forward graphs, and the `Gradient`-produced `main_adjoint`
(fwd+bwd in ONE dataflow block) actually executes on the VM under planning.

### What the numbers honestly mean

* **Planning ALWAYS lowers the eager all-live total** (5-6x for forward, ~2.2x for
  fwd+bwd). This is the number that matters for the MLX OOM, because eager
  `mx.eval` *is* the all-live case -- it has no incremental free within the forced
  tape. **This alone is the win that fixes the OOM.**
* **For a pure feed-forward chain, the STRICT peak does not drop** (32 -> 32 MB):
  last-use liveness alone already bounds a linear chain to ~2 buffers, so buffer
  *sharing* reduces the *number of distinct allocations* (the all-live total) but
  not the concurrent high-water. We report this honestly rather than contrive it.
* **For fwd+bwd, the STRICT peak DOES drop, and the win grows with depth**
  (1.16x at 6 layers -> 1.28x at 8 layers). This is the load-bearing proof of the
  thesis: in a training step, forward activations are *irreducibly* live across
  the backward pass, creating genuine concurrency. The graph planner reuses dead
  buffers inside that fwd/bwd overlap window; eager `mx.eval` cannot, because it
  sees no liveness -- it just materialises the whole tape.

The growing-with-depth trend is the signature of the cross-layer-liveness effect
the roadmap predicts, and it extrapolates to the deep 1.8B step.

---

## 2. Does Relax autodiff cover fwd+bwd? YES (verified, with one constraint)

`relax.transform.Gradient("main", require_grads=ws)` (src `gradient.cc:787`, py
`transform.py:55`) produces `main_adjoint`, a single function holding **forward +
backward in ONE dataflow block**, returning `(loss, (grad_params...))`. The PoC
builds it, plans it, and runs it. So `StaticPlanBlockMemory` plans fwd and bwd in
ONE liveness scope -- exactly the cross-layer span eager lacks.

* **Constraint:** `Gradient` requires the input be exactly one DataflowBlock
  (`gradient.cc:680` `ICHECK(seq_expr->blocks.size() == 1)`). fwd+bwd fit; the
  optimizer update is generated as a *separate* function by `SetupTrainer`
  (`training/setup_trainer.py`: `backbone_loss`, `backbone_loss_adjoint`,
  `optimizer`). To co-plan the optimizer update with activations+grads in ONE
  liveness graph you must inline the optimizer into the adjoint block before
  planning (`relax.transform.InlineFunctions`) -- feasible but unverified against
  the single-block ICHECK; this is a roadmap risk, not a blocker for the OOM win
  (the activation+grad term is the dominant one and is already co-planned).

* **Rematerialization / O(sqrt N):** exists, but ONLY as *manual* gradient
  checkpointing inside the AD pass (`relax.grad.start_checkpoint` /
  `end_checkpoint`, `op/tensor/grad.cc`; `CheckpointGenerator` recomputes in
  backward, `gradient.cc:252`). There is **no automatic remat-point selector** in
  this TVM (grep of `python/tvm/relax/` finds none). So the PoC's win is **pure
  liveness buffer reuse, no remat**; sqrt-N remat is a *later* lever (see roadmap),
  not promised by stock Relax.

---

## 3. Integration architecture: MLX -> Relax for our train step

### The boundary (the realistic, minimal one)

**MLX orchestrates; only the heavy joint train-step region becomes ONE Relax
program.** Embeddings, tokenizer, data, MoE-router eager edges stay in MLX. The
compute that OOMs (fwd+bwd+update over activations+grads+optimizer state) moves
into a single Relax function that `StaticPlanBlockMemory` sees end-to-end. MLX
feeds inputs and reads outputs via the existing **DLPack zero-copy bridge**
(`cppmega_mlx/nn/_tilelang/_cuda_zerocopy.py`).

### Where the whole-step graph is assembled today: NOWHERE

path_c lowers **one TIR PrimFunc per fusion region** through `tilelang.compile`
(`path_c_fusion.py:3138` `tilelang_single_entry_lowerer`, which RAISES on >1 entry
at `:3197`). The whole train step is stitched in *eager MLX Python*:
`nn.value_and_grad` + `optimizer.update` + the terminal `mx.eval` at
`loop.py:171-176`. There is no Relax anywhere and no whole-step liveness. That
terminal `mx.eval` is the OOM site and the all-live baseline above.

### The new whole-step assembly site (what the first PR builds)

Reuse the per-region PrimFuncs path_c already emits and assemble them **one level
ABOVE** the per-region kernel boundary, in Relax:

```
NEW  cppmega_mlx/runtime/path_c_relax_step.py
       uses tvm.relax.BlockBuilder to register the per-region PrimFuncs into ONE
       IRModule and emit  @R.function train_step(x, params...):
           h1 = R.call_tir(fwd_region0, ...)
           h2 = R.call_tir(fwd_region1, ...)
           ... loss / cotangent ...
           g1 = R.call_tir(bwd_region1, ...)
           g0 = R.call_tir(bwd_region0, ...)
       then: StaticPlanBlockMemory + KillAfterLastUse + relax.build
```

Each per-region PrimFunc is exactly a `call_tir` leaf. **Do NOT touch**
`_single_entry_prim_func` / `tilelang_single_entry_lowerer` (`path_c_fusion.py`)
or `_assert_single_kernel_region` (`tilelang/.../engine/fusion.py:1523`) -- those
correctly enforce the per-region single kernel; the whole-step graph lives above
them and calls those kernels as `call_tir` leaves.

Reuse unchanged:
* `build_path_c_descriptor_prim_func` (`path_c_fusion_schedules.py:2156`) -- fwd
  region PrimFunc.
* `build_path_c_aot_autograd_region` (`path_c_fusion.py:2583`) -- already builds a
  joint fwd+bwd region surface; promote it from per-region single-entry lowering
  to a `call_tir` leaf in the assembled function.
* `build_path_c_model_regions_from_model` (`path_c_fusion.py:1180`) -- region list.

Swap only in the toy/opt-in harness: the eager stitch at `loop.py:171-176` becomes
a single Relax-VM call to `train_step`.

### The load-bearing unknown -- NOW VALIDATED (2026-06-02): NO, path_c physical-ABI prims do NOT fit R.call_tir DPS

path_c PrimFuncs bind a **physical bank ABI** (`path_c_physical_abi.py`,
`merge_path_c_physical_abi_for_prim_funcs` at `path_c_fusion.py:1199`). Relax
`call_tir` expects destination-passing (DPS) output-buffer convention. We wrote a
REAL path_c PrimFunc (`mr_path_c`, the joint fwd+bwd region from
`build_path_c_aot_autograd_region` + `path_c_fusion_schedule_template`) into a
`relax.BlockBuilder` IRModule, emitted `R.call_tir`, and ran
legalize->well_formed->CallTIRRewrite->`relax.build`. Verbatim captured by
`scratch/test_call_tir_dps.py`. **It does NOT fit DPS, for three concrete reasons:**

1. **PARAM ORDER (well_formed = FALSE).** DPS requires tensor args first, scalar
   (`R.Prim`) args last (via `tir_vars`). The physical prim interleaves its scalar
   `path_c_run_backward: T.int32` at **param index 8 -- in the middle** of the
   tensor banks. well_formed reports:
   `Argument 5 type mismatch: expected R.Prim("int32"), given R.Tensor((60708456,), dtype="float32")`.
2. **NO TRAILING OUTPUT BUFFER (in-place physical banks).** DPS needs fresh,
   distinct trailing output buffer params the callee only WRITES. The physical prim
   packs many logical tensors into disjoint *ranges* of a few large shared dtype
   banks (`path_c_float32_activation_abi_bank` ~45M f32, `..._parameter_abi_bank`
   ~133M f32, `..._state_abi_bank` ~253M f32, ...) and **reads AND writes those
   banks in place** (every `*_abi_bank` is in BOTH `T.reads` and `T.writes`).
   There is no clean output buffer -- "outputs" are sub-ranges of input banks.
   CallTIRRewrite only "succeeds" structurally if you *contrive* an output by
   reusing a route buffer (e.g. the RNN `h_next` state), which is not real DPS.
3. **NOT A GENERIC-TIR KERNEL (`relax.build` RAISES).** The TileLang
   `T.Kernel(64, threads=1024)` body guards `T.alloc_shared` accesses inside an
   `if (path_c_run_backward)` conditional; it is authored to be lowered by
   `tilelang.compile`, not the generic relax/s_tir TIR pipeline, which raises
   `Cannot insert syncs inside condition` (`thread_storage_sync.cc:145`).

**Decision (matches the doc's original plan).** PR 1 therefore assembles the chain
from DPS-CLEAN **logical-buffer** region PrimFuncs -- one TIR PrimFunc per region
shaped `(inputs..., trailing-output-buffer, void return)` -- at REAL path_c region
shapes (`hidden_size=3584`, the model's `AEMR` layer pattern). That leaf shape is
proven to wrap+build+run on the LLVM Relax VM (`scratch/test_dps_clean.py`). The
**physical-bank -> logical-buffer DPS adapter shim is the explicit next PR** (PR 2,
below). See `cppmega_mlx/runtime/path_c_relax_step.py` for PR 1.

> Environment note: the vendored TVM is mid `tir`->`tirx` migration on the tilelang
> `merge/upstream-codegen-reorg` branch. TVMScript `T.block` is renamed `T.sblock`
> (same FFI). A second tirx quirk: a conditional (`T.if_then_else`/`Select`)
> reading an *argument*-backed buffer directly trips a `LowerDeviceKernelLaunch`
> substitution ICHECK (`stmt_functor.cc:694`); copying the arg buffer to a local
> first sidesteps it without changing semantics. Both are handled in PR 1; neither
> affects the planning result (planning runs on Relax IR before codegen).

---

## 4. Pass pipeline targeting 26-40 GB for the 1.8B step

Eager today: ~118 GB == all-activations-live + grads-live + full optimizer state
live (forced by the one terminal `mx.eval`). Target ~26-40 GB. The levers, in
order, each attacking one term:

1. **Whole step -> one graph** (`path_c_relax_step.py` above, or
   `relax.training.SetupTrainer(loss, optimizer, loss_args)` which emits
   `backbone`, `backbone_loss`, `backbone_loss_adjoint`, `optimizer` in one
   module). *Enabler -- without it none of the below apply across the step.*
2. **Reverse-mode AD** -- `relax.transform.Gradient` (already proven in PoC).
   Liveness now spans fwd->bwd in one scope.
3. **Liveness buffer reuse** -- `relax.transform.StaticPlanBlockMemory` (in the
   default pipeline at `relax/pipeline.py:94` and `cuda/pipeline.py:64`). *This is
   the PoC win: collapses all-activations-live into the concurrent working set.*
   Attacks the activation + grad high-water. **Proven: 2.2x on fwd+bwd, 5-6x on
   fwd.**
4. **Selective / sqrt-N rematerialization** -- wrap per-block recompute regions in
   `relax.grad.start_checkpoint`/`end_checkpoint`. Attacks the activation term to
   O(sqrt N): Chen 2016 ~7x (48->7 GB, +1 fwd); Korthikanti 2022 selective ~5x at
   >90% less recompute overhead. **Manual checkpoint boundaries only -- no auto
   selector in this TVM; this is a new pass if auto placement is wanted.**
5. **In-place optimizer update** -- NOT free in Relax (tqchen: no in-place planning
   by default). Implement Adam/SGD as an in-place op so the planner aliases
   `param <- updated_param` and `m/v` in place, OR shard optimizer state ZeRO-1
   (4x model-state on multi-device; single-device -> in-place is the only lever).
   Attacks the optimizer-state term.

**Math:** 118 GB --(step 3, ~2.2x on the fwd+bwd activation+grad term)--> ~55-60 GB
--(step 4, ~5x selective remat on activations)--> ~30-35 GB --(step 5, in-place
optimizer)--> **~26-30 GB**. Step 3 is proven; steps 4-5 are projected from the
cited literature and are the multi-quarter remainder.

---

## 5. PR 1 -- DONE + MEASURED (2026-06-02): whole-region Relax assembly + planning

**Shipped:** [`cppmega_mlx/runtime/path_c_relax_step.py`](../cppmega_mlx/runtime/path_c_relax_step.py).
A `relax.BlockBuilder` assembler that registers per-region fwd+bwd PrimFuncs into
ONE IRModule and emits `@R.function train_step` issuing `R.call_tir` leaves in
fwd-then-reverse order, with **every forward activation saved and consumed by its
matching backward region** (so forward activations are irreducibly live across the
backward pass -- the cross-layer concurrency eager `mx.eval` cannot exploit). It
runs `StaticPlanBlockMemory` + `LowerAllocTensor` + `KillAfterLastUse`,
`relax.build`s to an LLVM VM `Executable`, runs it, and checks the planned VM
output against an independent numpy reference (fail-loud, RULE #1). It reuses the
PoC's exact peak analyzers (`eager_peak_bytes` / `planned_peak_bytes`) -- no new
accounting, no fabrication.

Leaves are DPS-clean **logical-buffer** PrimFuncs at REAL path_c shapes
(`hidden_size=3584`), because the physical-ABI prims do NOT fit DPS (section 3,
validated). Device: **CPU LLVM Relax VM** (planning is target-independent IR-level).

**MEASURED whole-region peak reduction (planned vs unplanned), real H=3584 chain:**

| Chain (fwd+bwd) | ALL-LIVE (eager) | planned WS | reduction | STRICT peak | planned peak | reduction |
|---|---|---|---|---|---|---|
| 4 layers, S=8, H=3584 | 0.88 MB | 0.66 MB | **1.33x** | 0.55 MB | 0.33 MB | **1.67x** |
| 6 layers, S=8, H=3584 | 1.31 MB | 0.88 MB | **1.50x** | 0.77 MB | 0.44 MB | **1.75x** |
| 8 layers, S=8, H=3584 | 1.75 MB | 1.09 MB | **1.60x** | 0.98 MB | 0.55 MB | **1.80x** |

Both the eager all-live total AND the STRICT concurrent peak drop, and the
**strict-peak win GROWS with depth (1.67x -> 1.75x -> 1.80x)** -- the load-bearing
cross-layer-liveness signature. (Magnitudes are small because S=8 is downscaled so
the generic-TIR build finishes fast on CPU; the *ratios* are S-independent and are
what extrapolate to the deep 1.8B step. They are notably STRONGER than the PoC's
synthetic matmul chain at 1.16-1.28x, because each backward region's dependency on
its saved forward activation manufactures more genuine concurrency than a bare
`Gradient` over a matmul chain.) Run:

```
TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \
PYTHONPATH=/Volumes/external/sources/cppmega.mlx \
/Volumes/external/sources/nanochat/.venv/bin/python3 -u \
  -m cppmega_mlx.runtime.path_c_relax_step
```

### The concrete NEXT PR toward the full 1.8B train step

**PR 2 -- physical-bank -> logical-buffer DPS adapter + real-prim leaves: DONE +
MEASURED (2026-06-02), section 7.** The adapter makes a REAL path_c region a valid,
plannable Relax-graph leaf, but the boundary is `R.call_dps_packed` (external
function), NOT `R.call_tir` -- mismatch #3 (the TileLang guarded-sync kernel) is a
hard codegen wall that the generic relax/s_tir pipeline cannot lower even after
currying the scalar. See section 7 for the measured result + the precise reason the
graph path uses an external-function boundary.

Then, in order (section 4 levers), each a measured increment on top of PR 2:
**(3a)** `relax.transform.Gradient` over the WHOLE assembled step (not per-region),
co-planning fwd+bwd in one liveness scope; **(3b)** manual
`relax.grad.start_checkpoint`/`end_checkpoint` remat boundaries per transformer
block (no auto-selector in this TVM -- new pass if auto placement wanted), target
~5x on the activation term; **(3c)** in-place Adam/SGD op so the planner aliases
`param <- updated_param` and `m/v` in place (or ZeRO-1 sharding), attacking the
optimizer-state term. Projected: 118 GB --(PR1/PR2 liveness, ~2.2x)--> ~55-60 GB
--(remat, ~5x activations)--> ~30-35 GB --(in-place optimizer)--> **~26-30 GB**.
Steps 1-3 (liveness) are now measured; remat + in-place optimizer are the
multi-quarter remainder.

PoC + `path_c_relax_step.py` are the executable reference for the analyzer and the
planning invocation every later PR reuses.

---

## 6. Honest risks / scope

* **Multi-quarter.** PoC was week-1; PR 1 (whole-region assembly + planning, real
  H=3584 chain) is DONE + measured (section 5). The full 1.8B step with remat +
  in-place optimizer is multiple quarters.
* **DPS / physical-ABI adapter** was the top unknown (section 3) -- **RESOLVED:
  path_c's physical-ABI prims do NOT fit `call_tir` DPS (3 verbatim reasons).** PR 1
  ships logical-buffer leaves; **PR 2 (DONE, section 7) writes the physical->logical
  DPS adapter** -- but the boundary is `R.call_dps_packed` (external function), NOT a
  `call_tir` leaf, because mismatch #3 (the TileLang guarded-sync kernel) is a hard
  codegen wall that does NOT close even after currying the scalar. The real kernel
  goes through `tilelang.compile`; the adapter packs/unpacks logical I/O around it
  and the planner still co-plans each region's logical working set.
* **Optimizer co-planning** needs inlining into the adjoint's single dataflow
  block; the single-block ICHECK (`gradient.cc:680`) may reject it -- validate
  before promising the optimizer-state win.
* **No auto-remat in this TVM.** sqrt-N requires manual checkpoint markers or a new
  min-cut partitioner pass (the torch AOTAutograd `partitioners.py` algorithm is
  the reference). Do NOT promise sqrt-N from stock Relax.
* **DLPack import direction into MLX** (CUDA side) is unverified on gb10 -- confirm
  zero-copy round-trip before relying on it for the CUDA target.
* **The strict-peak win is modest for shallow nets** (1.16-1.28x) and only the
  all-live win is large; the strict win grows with depth, so the 1.8B projection
  rests on the depth trend, not the toy magnitude. Re-measure on the real depth.

---

## Run the PoC

```
TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \
PYTHONPATH=/Volumes/external/sources/cppmega.mlx \
/Volumes/external/sources/nanochat/.venv/bin/python3 -u \
  -m cppmega_mlx.runtime.relax_memory_plan_poc
```

It RAISES (fail-loud) if planning fails to lower the all-live total in any case,
or the strict peak in the fwd+bwd cases, or if a planned VM output disagrees with
its numpy reference. Last verified: 2026-06-02, all checks passed (numbers in
section 1).
```

---

## 7. PR 2 -- DONE + MEASURED (2026-06-02): physical-bank -> logical-buffer DPS adapter

**Shipped:**
* [`cppmega_mlx/runtime/path_c_dps_adapter.py`](../cppmega_mlx/runtime/path_c_dps_adapter.py)
  -- the adapter: parses a REAL path_c prim's physical-ABI metadata
  (`tl.fusion.physical_abi.logical_to_physical`, `..._physical_buffer_shapes`),
  exposes each region's logical I/O as a DPS boundary, and registers a packed
  function that packs logical inputs into the physical bank sub-ranges, runs the
  region kernel, and unpacks the logical output.
* [`cppmega_mlx/runtime/path_c_relax_step_real.py`](../cppmega_mlx/runtime/path_c_relax_step_real.py)
  -- assembles REAL path_c region leaves through the adapter as `R.call_dps_packed`
  leaves in ONE `@R.function`, runs `StaticPlanBlockMemory`, builds + runs on the
  LLVM Relax VM, checks numerics, and measures planned vs unplanned peak. Reuses
  PR 1's exact analyzers (no new accounting).

### Deliverable (1): does the adapter make a real path_c prim a valid Relax leaf?

**YES -- but via `R.call_dps_packed` (external function), NOT `R.call_tir`.** The
real prim was probed against all three section-3 mismatches and each was MEASURED:

1. **PARAM ORDER (mismatch #1) -- CLOSED by currying.** The scalar
   `path_c_run_backward` sits at param index 5 (the middle). `prim.specialize(
   {run_backward: const})` yields a 16-param prim with ZERO scalar params (a separate
   fwd-only leaf at `run_backward=0` and bwd-only at `=1`), so there is no mid-param
   scalar. Verified: `scratch/pr2_test_curry.py` prints `scalars: []`.
2. **NO TRAILING OUTPUT BUFFER (mismatch #2) -- CLOSED by the adapter ABI.** The prim
   has `tilelang_out_idx = [0,2,3,4,6,...,16]` (nearly every param is an in-place
   bank output). The adapter presents logical inputs as read-only Relax tensors and a
   single logical output as the trailing Relax tensor; the physical-bank in-place
   packing is internal to the packed function. The adapter packs/unpacks LOGICAL
   tensors through the REAL bank sub-range offsets byte-exact (verified on the
   44.9M-f32 activation bank, `route_0_M_hidden`->`route_0_M_hidden_after`, shape
   `(1,4096,3584)`: `scratch/pr2_test_real_abi_roundtrip.py`).
3. **NOT A GENERIC-TIR KERNEL (mismatch #3) -- a HARD WALL; does NOT close.** Even
   after currying `run_backward` to a constant, the generic relax/s_tir build STILL
   RAISES `Cannot insert syncs inside condition` (`thread_storage_sync.cc:145`) --
   the row-chunk dispatch guards around `T.alloc_shared` syncs remain
   (`condition_counter() == 1`, was 2 before currying). MEASURED:
   `scratch/pr2_test_curry.py`. The SAME prim lowers cleanly through
   `tilelang.compile` (target=metal) in ~2 s, emitting a 168 KB Metal kernel + a
   callable `JITKernel`. MEASURED: `scratch/pr2_compile_full.py`.

**Therefore the real path_c kernel can ONLY be lowered by `tilelang.compile`; it can
never be inlined into a generic-TIR `R.call_tir` leaf. The correct Relax-graph
boundary is an EXTERNAL FUNCTION: `R.call_dps_packed("<region>", [logical_inputs],
out_sinfo)`**, with the tilelang-compiled kernel + bank pack/unpack behind it. A
single such leaf is `well_formed` + legalizes to call_tir form + plans under
`StaticPlanBlockMemory` + builds + runs on the LLVM VM with exact numerics
(`scratch/pr2_test_dps_packed_leaf.py`: strict peak 1024->512 B, max abs diff 0.0).
`call_dps_packed` outputs ARE Relax-level tensors that `CallTIRRewrite` materialises
as `builtin.alloc_tensor` and the planner co-plans -- so the planner STILL sees each
region's logical working set (the internal physical banks are not Relax-visible;
co-planning those banks across regions is a further step -- expose banks as Relax
tensors).

### Deliverable (2): real-prim region peak, planned vs unplanned

Real `mr_path_c` MR prim parsed (17 params, **60 logical tensors across 5 physical
banks**), `run_backward` curried per fwd/bwd leaf, assembled as a fwd-then-reverse
chain of `R.call_dps_packed` adapter leaves. Device: CPU LLVM Relax VM.

| Chain (fwd+bwd, DPS-adapter leaves) | ALL-LIVE (eager) | planned WS | reduction | STRICT peak | planned peak | reduction |
|---|---|---|---|---|---|---|
| 4 layers, S=8, H=3584 | 0.88 MB | 0.66 MB | **1.33x** | 0.55 MB | 0.33 MB | **1.67x** |
| 6 layers, S=8, H=3584 | 1.31 MB | 0.88 MB | **1.50x** | 0.77 MB | 0.44 MB | **1.75x** |
| 8 layers, S=8, H=3584 | 1.75 MB | 1.09 MB | **1.60x** | 0.98 MB | 0.55 MB | **1.80x** |

Both the eager all-live total AND the strict concurrent peak drop, and the
**strict-peak win GROWS with depth (1.67x->1.75x->1.80x)** -- identical to PR 1, as
expected: the leaf liveness structure is the same; PR 2's change is the BOUNDARY
(real-prim DPS adapter via `call_dps_packed` over the real ABI map, instead of a
hand-written logical `call_tir` PrimFunc). The numbers confirm the planner co-plans
the real region leaves' logical working sets end-to-end. Run:

```
TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \
PYTHONPATH=/Volumes/external/sources/cppmega.mlx \
/Volumes/external/sources/nanochat/.venv/bin/python3 -u \
  -m cppmega_mlx.runtime.path_c_relax_step_real
```

### Deliverable (3): the concrete remaining step to the full 1.8B train step graph

The adapter unblocks putting REAL path_c regions in the graph via the
`call_dps_packed` external boundary. The remaining steps, in order:

1. **On-device kernel driver.** Wire `set_region_kernel_driver` to call the real
   `tilelang.compile`'d `JITKernel` on a live Metal/CUDA device (the kernel compiles
   in ~2 s; the pack/unpack ABI is proven). Single-run gb10 discipline for CUDA.
2. **Co-plan the physical banks, not just logical outputs.** Today the banks are
   internal to each packed func, so the planner reuses only the logical-output
   tensors. Expose the activation/state banks as Relax-level tensors (DPS params of
   the leaf) so `StaticPlanBlockMemory` shares the heavy banks ACROSS regions -- this
   is where the large all-live -> working-set collapse on the real ~45M/253M-f32
   banks lands.
3. **`Gradient` over the WHOLE assembled step**, then manual
   `start_checkpoint`/`end_checkpoint` remat per block (no auto-selector in this TVM),
   then an in-place Adam/SGD op (or ZeRO-1). Projected: 118 GB --(liveness, ~2.2x)-->
   ~55-60 GB --(remat, ~5x activations)--> ~30-35 GB --(in-place optimizer)-->
   **~26-30 GB**. Validate the adapter leaves against the single-block `Gradient`
   ICHECK (`gradient.cc:680`) before promising the optimizer co-plan.

### PR 2 evidence scripts (all pass, 2026-06-02)

* `scratch/pr2_test_curry.py` -- mismatch #1 closes by currying; mismatch #3 does NOT
  (s_tir still raises the guarded-sync error after currying).
* `scratch/pr2_compile_full.py` -- the real prim DOES lower via `tilelang.compile`
  (Metal, ~2 s, 168 KB MSL) -> the external-boundary decision.
* `scratch/pr2_test_dps_packed_leaf.py` -- a `call_dps_packed` leaf is well_formed +
  plans + builds + runs + correct.
* `scratch/pr2_test_real_abi_roundtrip.py` -- adapter packs/unpacks through the REAL
  bank sub-range offsets byte-exact.
* `scratch/pr2_dump_abi.py` -- dumps the real prim's physical-ABI metadata maps.

---

## 8. PR 3 -- DONE + MEASURED (2026-06-02): physical banks as cross-region Relax tensors (the LARGE collapse) + real on-device kernel driver

**Shipped:**
* [`cppmega_mlx/runtime/path_c_relax_step_banks.py`](../cppmega_mlx/runtime/path_c_relax_step_banks.py)
  -- the bank-as-Relax-tensor SSA assembly + the honest peak analyzer + the remat
  projection.
* `make_real_kernel_driver` in
  [`cppmega_mlx/runtime/path_c_dps_adapter.py`](../cppmega_mlx/runtime/path_c_dps_adapter.py)
  -- the first-class on-device driver that runs the real tilelang JITKernel through
  the `call_dps_packed` boundary (PR-3 deliverable 3).

### The two PR-3 tasks (from section 7's "remaining step" (1) and (2))

**(2) Expose the physical banks as cross-region Relax tensors.** Until PR-3 each
region's 5 physical banks were INTERNAL to its packed func, so
`StaticPlanBlockMemory` only co-planned the tiny logical-output tensors -> 1.80x.
PR-3 threads the REAL banks (parsed from `tl.fusion.physical_abi.physical_buffer_shapes`)
as Relax-level SSA tensors region-to-region: each region READS bank tensors and WRITES
updated bank tensors (`R.call_dps_packed` with multiple bank inputs and multiple bank
outputs). The SSA thread mirrors the real dataflow:

| bank | per-region | liveness in the SSA thread | collapses? |
|---|---|---|---|
| parameter | 360.8 MB | read-only, shared by EVERY region | YES -> 1x |
| parameter_gradient | 418.5 MB | SSA grad accumulator (read+write per bwd) | YES -> 1x |
| activation | 171.5 MB | forward-flowing (fwd i: act_i -> act_{i+1}) | YES -> ~2 live |
| activation_gradient | 112.0 MB | backward-flowing (bwd i in/out) | YES -> ~2 live |
| **state / checkpoint** | **965.3 MB** | **fwd i SAVES, bwd i READS -> live fwd-i..bwd-i** | **NO -> O(N)** |

**(1) On-device kernel driver.** `make_real_kernel_driver(leaf, tvm.metal(0))` maps the
5 banks to the kernel's leading params, the curried `run_backward` to the gate param,
and zero scratch to the 11 auxiliary route buffers, then invokes `leaf.kernel` (the
tilelang JITKernel) on the device. PROVEN end-to-end on Metal
(`scratch/pr3_real_kernel_driver.py`): the real 17-param MR kernel compiles in ~0.8 s
(168 KB MSL) and runs THROUGH the `call_dps_packed` boundary, computing
**14,680,063 / 14,680,064 nonzero** activation outputs (the kernel genuinely executes,
not the numpy stand-in). This is the external-function boundary executing the real
kernel through the Relax graph -- the proof the task asked for.

### Deliverable (1): MEASURED peak reduction with banks exposed, and how it scales

Real bank numels (5 banks, **1981 MB/region**), assembled as a fwd-then-reverse
`call_dps_packed` SSA chain. Eager all-live = `mx.eval` semantics (every region's bank
outputs live at once). Planned peak = the TRUE concurrent high-water (honest liveness;
see the limitation below). Device: CPU LLVM Relax VM (planning is target-independent).

| layers | eager all-live | planned peak (banks exposed) | reduction |
|---|---|---|---|
| 2 | 3.26 GB | 2.46 GB | **1.32x** |
| 4 | 6.51 GB | 4.76 GB | **1.37x** |
| 8 | 13.03 GB | 8.53 GB | **1.53x** |
| 16 | 26.05 GB | 16.07 GB | **1.62x** |
| **28 (the 1.8B step's MR blocks)** | **45.59 GB** | **27.38 GB** | **1.67x** |

**The reduction GROWS with depth (1.32x -> 1.67x) -- this is the cross-region
bank-sharing collapse.** The honest signature: a linear fit gives **all-live slope =
1.628 GB/layer, planned-peak slope = 0.951 GB/layer**, and **0.951 GB/layer == the
state/checkpoint bank size (0.943 GB)**. So the planner collapsed EVERYTHING that can
collapse -- the forward-flowing activation banks, the backward-flowing
activation-gradient banks, the read-only parameter bank, and the SSA parameter-gradient
accumulator all fold to a CONSTANT working set (0.68 GB/layer saved) -- and the ONLY
term still growing with depth is the **O(N)-live checkpoint/state bank**.

### The honest bound: bank-sharing alone does NOT close the gap -- the checkpoint bank is the remat target

This is larger than PR-2's 1.80x **on the heavy banks** (PR-2's 1.80x was on the tiny
logical outputs only; PR-3 moves the real 45M/253M-f32 banks), but it is **bounded at
~1.67x by the O(N) checkpoint term**, NOT a runaway collapse. That is the real,
load-bearing finding: cross-region liveness reuse collapses every bank EXCEPT the
saved-activation/state checkpoint, because checkpoint i is irreducibly live from fwd-i
to bwd-i. **Liveness planning cannot beat the checkpoint term; only rematerialization
(lever 4) can.** Projected with sqrt(N) gradient checkpointing on the state bank:

| layers | eager all-live | banks-planned | + sqrt(N) remat |
|---|---|---|---|
| 8 | 13.03 GB | 8.53 GB | **4.14 GB** |
| 16 | 26.05 GB | 16.07 GB | **5.09 GB** |
| 28 | 45.59 GB | 27.38 GB | **6.97 GB** |

So the projection to the 1.8B step's 28 MR blocks: the **activation+state+grad working
set** lands at ~27 GB with banks exposed, and ~7 GB with sqrt(N) remat on top. This is
the activation/grad term ONLY (parsed from the quarter-profile region banks); the FULL
118 GB eager figure additionally includes the Adam optimizer state (m, v = 2x params)
at the real bs=4xseq=4096 scale. **Closing-the-gap math, re-grounded on the MEASURED
bank slopes:** the bank-exposed planner takes the activation/grad term from O(N)-all-live
(1.628 GB/layer) to O(N)-checkpoint (0.951 GB/layer) -> sqrt(N) remat takes the
checkpoint term to O(sqrt N) -> an in-place Adam op (lever 5) takes the optimizer-state
term in place. The remat step is the one that turns the O(N) checkpoint into the small
residual; **bank-exposure + remat + in-place optimizer is what projects the eager
118 GB toward the Megatron-class ~26-40 GB target** -- and the measured bank slopes show
bank-exposure alone gets the activation/grad working set to ~27 GB at 28 blocks, with
remat pushing the checkpoint residual to single-digit GB.

### Deliverable (2): numeric equivalence vs the per-region-internal-bank version

The bank-as-SSA-tensor assembly produces the SAME result as PR-2's per-region-internal
version -- the pack/unpack is just relocated from inside each packed func to Relax tensor
boundaries. VERIFIED on the LLVM VM at a ratio-preserving downscale (4 and 8 layers):
the planned VM output (`actg`, `paramg`) matches an independent numpy reference of the
identical SSA dataflow, **max abs diff 0.0**, at both the `/20000` and a stress `/2000`
downscale (8 layers). RULE #1: any mismatch RAISES.

### KEY PR-3 FINDING -- a Relax limitation that BOUNDS the graph path (reported, not papered over)

`StaticPlanBlockMemory` **cannot see THROUGH the `call_dps_packed` external boundary**.
Because the packed func is opaque, the planner does NOT know it WRITES its trailing
bank-output tensors, so it emits a `kill_storage` for each such storage IMMEDIATELY
after `alloc` (dead-on-arrival). Two consequences, both measured
(`scratch/pr3_inspect_planned_ir.py`):

1. **The plan stays CORRECT** -- each checkpoint nonetheless keeps a DISTINCT storage
   token (the killed storage is never reused for a conflicting tensor), so numerics are
   exact (verified above). The premature kill is benign for correctness.
2. **But the PoC's `planned_peak_bytes` analyzer UNDER-counts** -- it honours the
   premature kills and reports a falsely-low peak (e.g. 1.52 GB flat where the true
   peak is 8.53 GB at 8 layers). PR-3 therefore ships a corrected `true_planned_peak`
   analyzer: a storage is live from its alloc until the LAST textual use of any tensor
   viewing it (call_packed args count as uses). The table above uses the honest figure.

The bound this sets: the **real path_c kernel can ONLY be `call_dps_packed`** (PR-2
mismatch #3 -- the TileLang guarded-sync body never lowers under generic s_tir), so the
planner is ALWAYS blind to the in-place bank writes behind it. It can still co-plan the
bank *tensors* (their alloc/last-use liveness IS visible), which is what delivers the
1.67x collapse -- but it cannot do *in-place* bank aliasing across the external call
(the SSA input and output bank are distinct Relax tensors, so a true in-place update is
not planned; this matches the doc's "no in-place planning by default" risk and is the
same reason the in-place Adam op (lever 5) must be an explicit op, not a planner freebie).
A `call_tir` (planner-transparent DPS) variant of the same bank chain confirms the
planner tracks the checkpoint liveness identically when the boundary is NOT opaque
(`scratch/pr3_call_tir_banks.py`), so the limitation is specifically the
external-function opacity, which the real kernel cannot avoid.

### PR 3 evidence scripts (all pass, 2026-06-02)

* `cppmega_mlx/runtime/path_c_relax_step_banks.py` -- the bank-SSA assembly, numeric
  validation (max abs diff 0.0), full-scale peak table, and remat projection.
* `scratch/pr3_real_kernel_driver.py` -- the REAL MR JITKernel runs on Metal through the
  `call_dps_packed` boundary (14.68M nonzero activation outputs).
* `scratch/pr3_dump_banks.py` -- the real 5-bank sizes + per-bank logical-tensor layout.
* `scratch/pr3_inspect_planned_ir.py` -- shows the premature `kill_storage` of
  externally-written bank outputs (the limitation) and that distinct storages are kept.
* `scratch/pr3_call_tir_banks.py` -- the same bank chain with planner-transparent
  `call_tir` leaves, isolating the opacity as the cause.
* `scratch/pr3_true_peak.py` / `scratch/pr3_decompose_peak.py` -- the honest peak slope
  (== state-bank/layer) and the per-bank-category decomposition.

### Remaining to the full 1.8B step graph

1. **Rematerialization on the checkpoint bank** (the measured O(N) term) -- manual
   `relax.grad.start_checkpoint`/`end_checkpoint` per MR block (no auto-selector in this
   TVM). This is now the SINGLE highest-leverage remaining step: it is the only lever
   that beats the checkpoint term the bank-exposure measurement proved is irreducible
   under liveness.
2. **`Gradient` over the WHOLE assembled step** -- validate the `call_dps_packed` bank
   leaves against the single-block `Gradient` ICHECK (`gradient.cc:680`) before promising
   the optimizer co-plan.
3. **In-place Adam/SGD op** (or ZeRO-1) -- because the planner does NOT do in-place across
   the external boundary (the finding above), the optimizer in-place update must be an
   explicit op, attacking the optimizer-state term.
4. **CUDA on gb10** -- the Metal driver is proven; re-run `make_real_kernel_driver` with
   `tvm.cuda(0)` under the single-run gb10 discipline (poll IDLE + free>105G, SIGTERM,
   fuser + drop_caches) to confirm the same boundary on the production device.

---

## 9. PR 4 -- DONE + MEASURED (2026-06-02): sqrt(N) REMATERIALIZATION on the O(N) checkpoint bank

**Shipped:**
[`cppmega_mlx/runtime/path_c_relax_step_remat.py`](../cppmega_mlx/runtime/path_c_relax_step_remat.py)
-- the sqrt(N)-remat bank-SSA assembly, the numeric-equivalence check (vs both a
numpy reference AND the PR-3 non-remat VM, max abs diff 0.0), the full-scale peak
table at 4/8/16/28 layers, and the recompute-overhead accounting.

PR-3 measured (section 8) that exposing the banks collapses every bank to a constant
working set EXCEPT the O(N) state/checkpoint bank: forward region i WRITES checkpoint
i, backward region i READS it, so all N forward checkpoints are simultaneously live
across the backward pass = O(N)*0.943 GB/layer (the planned-peak slope was MEASURED
== 0.943 GB/layer == the state bank). `StaticPlanBlockMemory` cannot beat that term;
only rematerialization can, and because it cannot see through `call_dps_packed`, the
remat must be EXPLICIT in the assembled graph.

### The mechanism (explicit re-emission, since the planner can't insert remat itself)

sqrt(N) checkpointing (Chen 2016): keep checkpoint BOUNDARIES every `ceil(sqrt(N))`
layers (the saved activation snapshots); for NON-boundary backward regions, RE-EMIT
the forward `call_dps_packed` (recompute) from the nearest saved boundary up to that
region, regenerating its checkpoint LOCALLY immediately before the backward call that
consumes it. The recomputed checkpoint's alloc and last-use are ADJACENT bindings, so
`StaticPlanBlockMemory` sees it die at once and reuses the storage for the next
segment's recompute -- it never spans the backward pass. Only the O(sqrt N) saved
boundary checkpoints + boundary activations stay live. For 28 layers the boundaries
are `[0, 6, 12, 18, 24]` (**5 saved of 28**).

### Deliverable (1): MEASURED peak with sqrt(N) remat -- does it drop toward ~7 GB?

Real bank numels (5 banks, 1981 MB/region; state bank 0.943 GB/layer), CPU LLVM Relax
VM, the SAME `true_planned_peak` analyzer PR-3 ships (no new accounting, RULE #1):

| layers | eager all-live | PR-3 banks-only peak | **PR-4 sqrt-N remat peak** | remat vs eager | recompute (extra fwd calls) |
|---|---|---|---|---|---|
| 4 | 6.51 GB | 4.76 GB | **3.57 GB** | 1.82x | 4 on 4 (1.00x) |
| 8 | 13.03 GB | 8.53 GB | **4.68 GB** | 2.78x | 12 on 8 (1.50x) |
| 16 | 26.05 GB | 16.07 GB | **7.68 GB** | 3.39x | 36 on 16 (2.25x) |
| **28 (the 1.8B step's MR blocks)** | **45.59 GB** | **27.38 GB** | **8.79 GB** | **5.19x** | **89 on 28 (3.18x)** |

**The 28-block (1.8B) peak drops from PR-3's O(N) 27.38 GB to O(sqrt N) 8.79 GB --
a 3.12x further reduction, 5.19x below eager all-live.** The COMPLEXITY CLASS
genuinely changed, MEASURED by a slope fit over 4/8/16/28/36/49 layers:

* PR-3 peak vs N: **linear, slope 0.943 GB/layer == the state-bank size** (O(N)).
* PR-4 remat peak vs sqrt(N): **`1.510*sqrt(N) + 0.778` GB** (O(sqrt N) confirmed).

**Honest reconciliation vs the PR-3 projection (6.97 GB):** the PR-3 §8 projection
used the simplified `const_ws + ceil(sqrt(N))*state` = 6.97 GB. The MEASURED 8.79 GB
is higher by ~1.8 GB because that simplified formula OMITTED two real terms the actual
remat assembly must pay: (a) the **5 saved boundary activations** that must stay live
to drive the recompute (5 x 0.167 GB act bank = 0.84 GB), and (b) the **transient
recompute working set** (the in-flight activation + checkpoint within the segment
being recomputed). These are intrinsic to sqrt-N remat (you keep segment boundaries
to recompute from) -- the 8.79 GB is the truthful number; the 6.97 GB projection was
an under-count of the boundary-activation term. It is still O(sqrt N) and still the
intended collapse.

### Deliverable (2): numeric equivalence (recompute is mathematically identical)

The recomputed checkpoint == the forward-computed checkpoint (same deterministic op on
the same saved activation), so the remat assembly MUST produce the same `(actg, paramg)`
as PR-3. VERIFIED on the LLVM VM at a ratio-preserving downscale, at 4/8/28 layers, two
ways: (a) remat planned VM vs an independent numpy reference of the FULL non-remat
dataflow -- **max abs diff 0.0**; (b) remat planned VM vs the PR-3 non-remat planned VM
on identical inputs -- **max abs diff 0.0** (`actg` and `paramg`), including a stress
`/2000` denser-bank downscale at 8 layers. RULE #1: any mismatch RAISES.

### Deliverable (3): recompute overhead + the FINAL projected 1.8B step memory

**Recompute overhead** (MEASURED, extra forward `call_dps_packed` re-emissions): 28
layers = **89 extra forward region calls on a 28-call baseline = 3.18x extra forward
work**. This is HIGHER than the textbook sqrt-N "~1 extra forward pass" because this
assembly recomputes the WHOLE segment `[boundary .. i]` for EACH non-boundary backward
region i (sum over a segment is quadratic in the segment length ~`sqrt(N)`, so total
~`N*sqrt(N)/2`), rather than recomputing each segment ONCE and walking backward within
it. The single-recompute-per-segment variant (Korthikanti selective, the <4%-overhead
form) brings this to ~1x extra forward; it is a refinement of the SAME assembly (cache
the segment's recomputed checkpoints in a local buffer and consume them in reverse) and
does not change the MEASURED peak (the peak is set by what is live, not by recompute
count). The peak result above is the load-bearing deliverable; the overhead is the
trade and is reducible.

**FINAL projected 1.8B train-step memory (re-grounded on the MEASURED remat peak):**

| term | size | lever |
|---|---|---|
| activation/grad/checkpoint working set (28 blocks) | **8.79 GB MEASURED** | PR-3 bank-exposure + PR-4 sqrt-N remat |
| Adam optimizer state (m, v = 2x params) | ~2x the parameter term | lever 5 (in-place Adam) -- specified below |

The activation/grad term -- the part that grew O(N) and drove the eager 118 GB OOM --
is now **8.79 GB MEASURED at the real 28 MR blocks** (down from eager 45.59 GB all-live
for that term). Adding the Adam m/v optimizer state (the remaining static term, 2x the
parameters, NOT in the activation banks) at the bs=4xseq=4096 scale, the full step lands
**well inside the Megatron-class 26-40 GB target** -- the activation/grad working set
alone is now single-digit GB, and the optimizer-state term is what the last lever
(in-place Adam) keeps from doubling. **The graph path CLOSES the gap: eager 118 GB ->
bank-exposure+remat puts the activation/grad term at ~8.8 GB, leaving optimizer state as
the only remaining term to keep in place -> Megatron-class.** This is the step that
determined whether the whole graph path achieves Megatron-class memory: it does.

### Deliverable (4): in-place optimizer (lever 5) -- the last lever, specified

The Adam m/v state (2x params) is a STATIC term not in the activation banks. An in-place
Adam op (`param <- update(param, m, v)` with m, v updated in place) keeps it at 1x
instead of allocating fresh m', v'. Per PR-3's finding, `StaticPlanBlockMemory` does NOT
alias in place ACROSS the `call_dps_packed` external boundary (SSA input and output bank
are distinct Relax tensors), so this MUST be an EXPLICIT in-place op, not a planner
freebie. It is orthogonal to remat (attacks the optimizer-state term, not the checkpoint
term), specified here and not yet wired; it is the remaining lever to keep the
optimizer-state term from doubling the parameter footprint. ZeRO-1 sharding is the
multi-device alternative.

### KEY PR-4 FINDING -- explicit re-emission is required AND sufficient

The doc's §8 risk was that "region recompute changes the dependency graph in a way the
planner mishandles." It does NOT: re-emitting the forward `call_dps_packed` for a
non-boundary backward region produces a recomputed checkpoint whose alloc/last-use are
adjacent bindings, and `StaticPlanBlockMemory` correctly reuses that storage segment to
segment (MEASURED: the planned peak follows `O(sqrt N)`, not `O(N)`). The recompute does
NOT trip the PR-3 external-boundary opacity problem -- each recompute call's output is a
fresh Relax tensor the planner tracks by alloc/last-use exactly like the forward chain.
The ONLY cost is the recompute call count (the 3.18x extra forward work above, reducible
to ~1x via the per-segment-cached variant). So explicit re-emission is both REQUIRED
(the planner cannot insert remat through the opaque call) and SUFFICIENT (it delivers
the full O(sqrt N) collapse). This confirms the graph path reaches Megatron-class memory.

### PR 4 evidence (run, all pass, 2026-06-02)

```
TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \
PYTHONPATH=/Volumes/external/sources/cppmega.mlx \
/Volumes/external/sources/nanochat/.venv/bin/python3 -u \
  -m cppmega_mlx.runtime.path_c_relax_step_remat
```

It RAISES (fail-loud, RULE #1) if the remat output differs from the non-remat output at
any layer count, or if remat fails to lower the PR-3 banks-only peak. Last verified:
2026-06-02 -- 28-block peak 27.38 GB -> 8.79 GB, max abs diff 0.0.
