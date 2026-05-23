# Path C full-graph runtime — concrete blockers as of 2026-05-23

Status: **does not beat path_b yet.** This document is the lowest-level
contract for the next iteration so any further hand-waving stops here.

## Goal restated

User-stated objective: "полный graph path c через доработку
tilelang -> tvm -> tvm-ffi и чтоб у нас была структура в которую мы
добавляем блоки, с ней tilelang оптимизирует fusion, schedulers,
parallel execution... наш path c должен начать работать быстрее и
экономичнее чем path b. Предпочтение нижнему уровню доработкам на
c/c++/mm/llvm и python в tilelang/tvm/tvm-ffi, и только в крайнем
случае через nanobind и минимальные изменения cppmega.mlx."

Achievement criterion: `scripts/bench_1b_training_matrix.py --paths
path_b,path_c_warm`  shows path_c_warm tok/s >= path_b on >= 6/7
optimizers in bf16, with peak GB <= path_b + 10%.

## What is actually wired right now

* `cppmega_mlx/runtime/path_c_fusion.py` builds a `PathCFusionRegion`
  from selected route bricks 10 (M) / 11 (R) / 12 (A).
* `cppmega_mlx/runtime/path_c_fusion_schedules.py` lowers that region
  to one TileLang/TVM PrimFunc via `plan_path_c_fusion_schedule_for_region`
  with `include_backward=True`, plus a suffix loss block.
* `HybridTinyLM.path_c_fused_train_block_prim_func` materialises that
  PrimFunc and exposes the physical ABI map (`_cppmega_path_c_physical_buffer_abi_map`)
  + bank shapes. Banks are 3 dtype-uniform arrays (float32 / uint8 / int32),
  total ~273 KB for `local_gb10_quarter` tiny smoke at seq_len 127.
* `HybridTinyLM.make_path_c_physical_abi_bank_owner` zero-inits banks
  with the right dtypes/shapes.
* `HybridTinyLM.bind_path_c_in_region_parameter_views_into_bank` writes
  each in-region parameter into its bank slot (one slice-assign each,
  no large staging tensor) and replaces the model attribute with a
  bank-view snapshot (`logical_bank_view`).
* `HybridTinyLM.sync_path_c_in_region_parameters_into_bank` re-writes
  optimizer-replaced parameter tensors into the bank slots once per
  training step.
* `cppmega_mlx/training/path_c_fused_suffix.py` builds an
  `mx.custom_function` that on forward writes
  (`hidden_entry`, `target_ids`, `target_mask`, *params*) into bank
  slots and runs `artifact.forward(bank_owner=...)`, and on backward
  returns bank-view cotangents for every primal.
* `PathCFusedPlusEagerTrainingRuntime` honors three modes:
  suffix-bypass (only when its install gate accepts), an explicit
  fail-closed warmup+eager mode (today's default), and the older
  warmup-only path when no aliases are bound.
* `scripts/m04_train_step.py::install_path_c_fused_train_block_runtime_for_model`
  now refuses to attach the suffix-bypass loss function when the
  generated suffix loss ABI says `"full loss codegen is pending"` or
  when scalar outputs are not marked computed.
* `cppmega_mlx/runtime/path_c_physical_abi.py::write_into_bank_slot`
  fails closed on dtype mismatch instead of silently coercing.

## What blocks `path_c_warm > path_b`

These are all in the lowest layer of the stack
(`cppmega_mlx/runtime/path_c_fusion_schedules.py` brick descriptors +
TileLang/TVM lowering on the Metal target). cppmega_mlx app code is
intentionally not the place to fix any of them.

### Block 1: brick descriptors do not write all forward outputs

Verified with a direct kernel call against pre-populated bank inputs
(see `tmp/parity_natural.py`):

* `local_gb10_quarter_brick_11_R_hidden_after` slot does receive
  data after the kernel runs (sum != 0).
* `local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out` stays at
  zero even when we supply a non-trivial `hidden_entry` plus
  `sparse_mla_sm_scale = 1/sqrt(head_dim)` and `has_sinks = 0`.

That means the row-phased sparse MLA apply descriptor
(`_append_row_phased_sparse_mla_fp8_apply_body` ~line 4888 of
`path_c_fusion_schedules.py`) either skips writing
`attention_out` for this region wiring, or its required FP8/index
inputs (`q_fp8`, `q_scale`, `kv_fp8`, `kv_scale`, sparse `indices`)
are not produced by the upstream qkv_projection descriptor and remain
zero. Until apply writes a non-zero output, suffix loss runs against
half the residual stream and final_norm/lm_head grads diverge from
eager by several percent.

Fix layer: TileLang descriptor for `sparse_mla_fp8_apply` in the
brick-graph schedule, and the qkv_projection descriptor that feeds it.

### Block 2: gradient codegen covers only `final_norm_weight` and
`lm_head_weight`

`_train_step_suffix_loss_parameter_grad_buffers` returns just
`("final_norm_weight_grad", "lm_head_weight_grad")` and
`_append_train_step_suffix_loss_parameter_grads` only writes those
two grad slots. The ABI map advertises 27 grad slots (all in-region
brick params plus norm/head), but 25 of them are left as
uninitialised bank memory by the kernel. Overlaying them onto the
eager grad tree (the original "merged-grad" mode) is therefore unsafe
and currently fail-closed in the runtime.

Fix layer: TileLang descriptors for mamba3 / m2rnn / sparse_mla_fp8_apply
backward, plus matching emit logic in `_append_train_step_suffix_loss_parameter_grads`
so every brick weight grad is generated alongside `final_norm_weight_grad`
and `lm_head_weight_grad`.

### Block 3: suffix loss ABI flag still advertises pending

`_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_REASON` literally contains
`"full loss codegen is pending"`. The m04 installer reads this and
declines to attach the suffix-bypass loss function. This is the
correct behaviour: until Block 1 and Block 2 are closed, suffix-bypass
silently trains a different loss. Once Blocks 1 and 2 are closed, the
reason string in
`cppmega_mlx/runtime/path_c_fusion_schedules.py` line ~133 must be
flipped to a non-"pending" value.

### Block 4: `path_c_training_sequence_length(args)` returns
`seq_len - 1`, the bank/PrimFunc are specialised on that, but the
public `model.path_c_fused_in_region_parameter_bank_aliases()` (no
arg) returns offsets for the default 512-length PrimFunc. External
inspectors who call the public helper get offsets that do not match
the installed bank.

Fix layer: thread the install-time sequence_length through
`HybridTinyLM` so the public alias getter returns the bound offsets
(or cache the alias map on the runtime's contract surface and
deprecate the no-arg form).

### Block 5: per-call MLX `value_and_grad` cotangent assumption

`fused_suffix.vjp` discards the upstream loss cotangent and returns
unscaled bank-view cotangents. That is correct only when the trainer
runs `nn.value_and_grad(model, lambda model, batch: model.path_c_fused_suffix_loss(batch))`
with no outer scaling. `CompiledPretrainingStep._accumulate_or_update`
later scales the merged grads by `1 / grad_accum_steps`, so the
single-step assumption holds today. But the VJP should still scale by
the runtime cotangent before returning bank views to be future-proof
under e.g. mixed-precision loss scaling. This is a small fix inside
`cppmega_mlx/training/path_c_fused_suffix.py::fused_suffix_vjp`.

## Lower-level work items in order

These items are ordered so each unblocks the next.

1. Finish the sparse_mla_fp8_apply descriptor so it writes
   `attention_out` deterministically for the `brick_12_A` region.
   That includes wiring qkv_projection_kv_fp8 / kv_scale /
   row-phased indices into apply (today they are zeros). Numerical
   gate: `local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out` is
   bit-equal (or float32 within 1e-4) to an eager reference attention
   on the same `hidden_after_11`.

2. Add backward emitters for every in-region brick weight, mirroring
   `_append_train_step_suffix_loss_parameter_grads` for mamba3,
   m2rnn, sparse-MLA-apply, and residual_norm parameters. Numerical
   gate: bank-resident grad slot of each in-region parameter is bit-equal
   (or float32 within 1e-4) to the eager grad on the same prefix
   hidden_entry.

3. Flip the suffix loss ABI reason away from `"pending"` once
   (1) and (2) pass the numerical gate. The m04 installer will then
   attach the suffix-bypass loss function automatically.

4. Cotangent scaling in `fused_suffix.vjp`.

5. Sequence-length-aware public alias helper on `HybridTinyLM`.

6. Re-run `scripts/bench_1b_training_matrix.py --paths path_b,path_c_warm`
   and confirm path_c_warm tok/s >= path_b on >= 6/7 optimizers in
   bf16 with peak GB <= path_b + 10%. Update
   `docs/production_kernel_routing.md`.

## Current honest end-state

* Route stays green: `m04_path_c_training_route_available`,
  `run_path_c_fused_train_block_route`.
* `parameter_bank_residency_active = True`, but
  `bank_grad_overlay_active = False` and
  `suffix_bypass_available = False`. Eager remains the source of every
  gradient so training is correct; Path C does not yet replace any
  eager work.
* Tests: `tests/test_path_c_fused_plus_eager_runtime.py` (20),
  `tests/test_path_c_fused_suffix_custom_function.py` (3),
  `tests/test_hybrid_lm_path_c_physical_abi_bank_owner.py` (8),
  `tests/test_path_c_physical_abi.py` (35 incl. dtype-mismatch
  guard), `tests/test_m04_train_step.py -k "path_c or
  fused_train_block"` (66), `tests/v4/test_fusion_stage_{a..f}.py`
  + `test_fusion_roadmap_gaps.py` (401). All pass on commit `b04fbbb`.

## Why this matters

The earlier Codex turn happily reported
`returns_full_model_grads = True`, `merged_parameter_count = 27`,
and a `b54c348..222bd9f` chain that "flips the gate to ok". A
critical review showed those grads were the kernel's output against
all-zero hidden_entry / target_ids / target_mask, then overlaid onto
the eager grad tree — a silent training corruption that the unit
tests could not catch (they used a fake artifact). Commit `b04fbbb`
removes the unsafe overlay and replaces it with the fail-closed
warmup+eager path. The work above is what is left to actually beat
path_b, and it lives at the TileLang descriptor + Metal codegen
layer, exactly where the user asked the work to happen.

## Update 2026-05-23: B2 partial advance + structural Block A discovered

### Fix B-1 (committed, `26575cd`): block bwd no longer clobbers residual chain

The row-phased block backward emitters (`_append_row_phased_mamba3_bwd_body`,
`_append_row_phased_m2rnn_bwd_body`,
`_append_row_phased_attention_qkv_projection_bwd_body`) used to
zero-init the input's `hidden_grad` slot at the start of every row
iteration, then accumulate the block's in-projection contribution
into the same slot with `+=`. When the input's `hidden_grad`
output mapped to a **full-sequence bank slot** that the upstream
`residual_rmsnorm_bwd` had already written with `=` in the same
row iteration (specifically for the FIRST brick of an in-region
chain), the zero-init clobbered the residual chain-rule contribution,
so the only term that reached the input residual stream was the
block's in-projection grad. The downstream eager prefix then saw a
hidden_entry cotangent that was missing the chain through the
fused region's residual bridges.

New helper `_is_full_sequence_bank_slot(buffer_name, access_by_buffer)`
detects this case via the access string pattern
`path_c_..._abi_bank[OFFSET + i]` (no `% H` modulo). The three
block bwd emitters now skip the zero-init when this helper returns
True, so the existing `hidden_grad_ref += block_contribution`
accumulator adds onto whatever the residual_rmsnorm_bwd wrote with
`=`. Per-row scratch `*_hidden_grad` slots (used by the LATER
bricks in the same chain, e.g. R and A consuming bridge-normalized
hidden) keep their per-row zero-init because no earlier bwd writes
to them in the same row iteration.

This change is mathematically correct and bit-stable (no extra
shared-memory sync, no FP order changes for `attention_out`); all
tested suites stay green (94 + 111 + 93 + 66 + 401).

### Block A (NEW): first brick of the fused region is missing its pre-block norm

`_path_c_model_surfaces_from_bricks` initialises
`context.route_hidden = initial_hidden` for the first brick. Brick
N+1 in the chain reads `context.route_hidden = norm_{N+1}(hidden +
delta_N)` from the inter-brick `residual_rmsnorm` bridge (which uses
`layers.{N+1}.norm.weight` per the alias map). For brick **0** of
the region, the M-block consumes raw `hidden` (the entry residual
stream from the eager prefix) directly, **without** applying its
own `layers.{first_in_region}.norm`.

The eager `HybridTinyBlock.route_delta` always does
`x = self.norm(hidden); delta = block(x); return delta`. The fused
region therefore feeds `mamba3(hidden)` instead of
`mamba3(norm_first(hidden))` into its first brick, so:

* `local_gb10_quarter_brick_10_M_delta` is computed against a
  ~16x-larger-magnitude input (`||hidden||` vs
  `||norm(hidden)||`).
* `layers.10.block.D` and `layers.10.block.in_proj.weight`
  gradients pick up the inflated input as a multiplicative factor
  in the bwd chain rule, yielding relative-error
  ~10^3 — 10^4 against eager (see
  `/tmp/path_c_blocker2_probe.py` receipts).
* `layers.10.norm.weight` is never bound to any logical name in the
  fused region's ABI map; the alias loop in
  `path_c_parameter_logical_aliases` binds
  `layers.{i+1}.norm.weight` to brick `i`'s bridge weight for
  `i in [0, N-2]`, so `layers.{first_in_region}.norm.weight` lands
  on the bridge BEFORE the first brick — which is in the eager
  prefix, not in the fused region.

Confirming probe (`/tmp/probe_entry_norm.py`): with deterministic
weights and a synthetic `hidden_entry`, the eager M-block produces
`delta_M` with `sumabs=8.18`, while the fused-equivalent
`mamba3(hidden_entry)` (no entry norm) produces `delta_M` with
`sumabs=0.021`. The ratio matches `1 / inv_rms ~= 19` (and propagates
multiplicatively through subsequent layers).

#### Fix layer (Block A)

The right place is the TileLang surface layer (lower than the
cppmega_mlx app code, per the constraints in this file). Two
ladder approaches:

1. **(Minimal-invasive) Add an inline entry-norm path inside the
   first brick's fwd codegen and a matching bwd accumulator into the
   bank `hidden_grad` slot.** Add a new real-ABI input
   `f"{first_brick.name}_entry_rmsnorm_weight"` and have
   `_emit_mamba3_model_brick_surfaces` consume it. The bwd accumulates
   `inv_rms * (entry_normed_hidden_grad * weight - hidden * dot *
   inv_rms^2 / D)` into `hidden_grad` (the bank slot we now fixed in
   Fix B-1 to accumulate). Pros: no new op signature, no
   `_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE` churn, no acceptance
   profile invalidation. Cons: ad-hoc M-block fwd/bwd changes.

2. **(Structurally cleanest) Add a new `entry_rmsnorm` op-node** that
   the surface builder prepends to the brick chain, with its own
   fwd/bwd codegen and descriptor, and update the canonical op
   signature `_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE` to
   `("entry_rmsnorm", "mamba3_mimo", "residual_rmsnorm", ...,
   "entry_rmsnorm_bwd")`. Wire `layers.{first_in_region}.norm.weight`
   into the alias loop as an additional candidate alias mapping to
   `f"{first_brick.path_c_brick_name}_entry_rmsnorm_weight"`. Pros:
   self-contained op, reuses the bridge codegen patterns. Cons:
   requires a new descriptor, an acceptance-profile bump, and the
   aliases mapping update.

Either fix is the next-step after `26575cd`. The B6 bench gate stays
blocked until Block A lands plus the rest of Block 2 numeric parity
passes on `local_gb10_quarter` tiny smoke.
