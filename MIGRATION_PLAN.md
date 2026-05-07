# cppmega.mlx → unified TileLang pipeline migration plan

This plan migrates the kernels under `cppmega_mlx/nn/_tilelang/` from
hand-written Apple-Metal-only TileLang into the new unified fused-kernel
compiler at `/private/tmp/tl_poc_review/poc/{triton_frontend,torch_dynamo,
extern_intrinsic_examples}/` (branch `poc-integrations-review`).

The new pipeline gives us:

* portable codegen (CUDA / HIP / Apple Metal) instead of Path B (raw MSL)
  + Path C (TileLang→MSL) duplication;
* `tl.extern_intrinsic` (#08) with simdgroup_a/b/c factories for the MSL
  fragments we currently inline;
* `triton_frontend` (#01) with vendored microsoft/triton-shared
  `PtrAnalysis` for arbitrary Triton-style pointer kernels;
* `torch_dynamo` backend (#02) with multi-region launcher, joint-graph
  autograd (#09), aot_autograd glue, and an autotune shortlist;
* `LowerTMAToPtrArith` (#07) so Hopper-only TMA copies fall back to plain
  pointer arithmetic on Metal/HIP.

## 1 Inventory

Style key: **TL** = `@T.prim_func` TileLang DSL (Path C). **MSL** = raw
Metal Shading Language strings (Path B). **TRITON** = `@triton.jit` /
`tl.load`/`tl.store`. **HELPER** = pure python utility / dispatcher.
"call sites (mlx)" counts intra-`cppmega.mlx` imports; "call sites (cm)"
counts upstream cppmega/megatron imports.

| File | LOC | Style | call sites (mlx / cm) |
|---|---:|---|---:|
| `sparse_mla_path_c.py` | 2611 | TL Path-C | 6 / 0 |
| `sparse_mla_fp8_path_c.py` | 1760 | TL Path-C + FP8-MSL | 3 / 0 |
| `topk_selector.py` | 1245 | TL Path-C | 4 / 0 |
| `dsa_splitk_indexer_loss.py` | 1192 | TL Path-C + Triton ref | 0 / 2 |
| `sparse_mla_blockscaled_path_c.py` | 1117 | TL Path-C + MSL helpers | 6 / 0 |
| `fp8_vecmat_path_c.py` | 1089 | TL Path-C + FP8-MSL | 5 / 0 |
| `sparse_mla_fp8.py` | 1074 | MSL Path-B | 4 / 0 |
| `m2rnn.py` | 1028 | MSL Path-B | 4 / 0 |
| `sparse_mla_blockscaled.py` | 992 | MSL Path-B | 7 / 0 |
| `sparse_mla.py` | 905 | MSL Path-B | 14 / 0 |
| `_msl_transform.py` | 862 | HELPER (TL→MSL lowering) | 17 / 0 |
| `mamba3.py` | 826 | MSL Path-B | 12 / 2 |
| `fp8_msl_kernels.py` | 748 | MSL Path-B vendored fp8 | 8 / 0 |
| `fp8_amax.py` | 638 | TL Path-C | 0 / 2 |
| `_mamba3_helpers_tilelang.py` | 627 | TL helpers | 2 / 0 |
| `mamba3_path_c.py` | 627 | TL Path-C | 4 / 0 |
| `_path_b_lowering.py` | 282 | HELPER (MSL emission) | 1 / 0 |
| `_mamba3_helpers.py` | 236 | HELPER | 6 / 0 |
| `__init__.py` | 232 | HELPER (re-exports) | 2 / 0 |
| `topk_selector` *(see above)* | 1245 | TL Path-C | 4 / 0 |
| `_experimental.py` | 83 | HELPER | 1 / 0 |
| `fp8_msl_kernels` *(see above)* | 748 | MSL Path-B | 8 / 0 |

Style observation: ~half the bytes are duplicated between Path B (raw MSL
shaders, no portability) and Path C (TileLang DSL routed through
`_msl_transform.lower_tilelang_to_msl_inline`). The new pipeline collapses
both into a single TL source that can target Metal / CUDA / HIP from one
file, deprecating Path B and `_msl_transform.py`.

## 2 Top-5 migration targets

Ranked by (LOC × call-site count) and novelty of fused pattern.

### 2.1 `sparse_mla_path_c.py` (2611 LOC, 6 mlx call sites)

* Entry path: **#08 `tl.extern_intrinsic`** + **#01 `triton_frontend`**
  PtrAnalysis-backed `T.copy`. The kernel already lives in TileLang DSL,
  so the migration is "swap the lowering tail": replace
  `lower_tilelang_to_msl_inline` with `tilelang.engine.lower(prim_func,
  target=Target("cuda"|"hip"|"metal"))` once #07 (LowerTMAToPtrArith) and
  #08 (simdgroup_a/b/c factories) are wired into the engine pass list.
* Pipeline gaps exposed:
  * #08 — needs the SIMDgroup-matrix MMA factories landed by wave-3
    (commit `81552cb8` and follow-ups). Currently `Layout()` placeholder
    is documented as the "correct opaque answer", but the per-thread
    `simdgroup_matrix<half,8,8>` register tile must be reachable from
    `layout_inference.cc`.
  * #07 — Sparse-MLA fwd uses `T.alloc_shared` + atomic-free dKV partial
    contract. No TMA used directly, but the bwd kernel emits
    `tl_ptr_copy_elem` once #07's `kEmitOpaque=false` toggle lands in
    runtime defaults.
* Effort: **large** (16-stage scheduling, 4 dialects of test parity vs Path B).

### 2.2 `topk_selector.py` (1245 LOC, 4 mlx call sites)

* Entry path: **#01 `triton_frontend`**. The radix-select kernel uses
  `T.alloc_shared([257], int32)`, `T.atomic_add(..., return_prev=True)`,
  and partial-warp barriers `T.sync_threads(3, RADIX)`. PtrAnalysis can
  already lower the histogram + scatter pattern.
* Pipeline gaps exposed:
  * #01 — partial-thread barrier (`T.sync_threads(mask, count)`) is not
    yet in the OP_TABLE (`poc/triton_frontend/op_mapping.py`). Add as new
    primitive `tl.sync_threads_partial(mask, n)`.
  * #01 — `atomic_add(..., return_prev=True)` needs a fast-path emitter
    that lowers to `T.atomic_add` in PrimFunc form on Metal (no
    `simd_active_threads_mask` translation yet).
* Effort: **moderate** (single kernel, well-tested upstream).

### 2.3 `dsa_splitk_indexer_loss.py` (1192 LOC, 2 cppmega call sites)

* Entry path: **#01 `triton_frontend`**. Has both Triton-style markers
  (`tl.load`/`tl.store`) AND `@T.prim_func` Path-C — a perfect first
  conformance kernel to test the dual-source path. Wave-3 already added
  `_active_sk_tiles` clamps and stage-2 Q hoist (commit `a2ffcc1`).
* Pipeline gaps exposed:
  * #01 — Triton `tl.dot(.., trans_b=True)` not yet in OP_TABLE
    (only `tt.dot` direct). Trivial extension.
  * #02 — `aot_autograd_glue.compile_symbolic` (commit `c1bf9bc9`) must
    accept the SymInt `BLOCK_SQ` parameterization from this kernel.
* Effort: **moderate** (fwd + bwd, but each ~600 LOC).

### 2.4 `fp8_vecmat_path_c.py` (1089 LOC, 5 mlx call sites)

* Entry path: **#08 `tl.extern_intrinsic`** for the FP8 MSL kernels +
  **#01 `triton_frontend`** for the TileLang surrounds.
* Pipeline gaps exposed:
  * #08 — needs an FP8 `Frag` factory (`simdgroup_a_fp8`,
    `simdgroup_b_fp8`) once Apple ships a hardware path; until then,
    keep using the vendored MSL kernels via `extern_intrinsic` and the
    existing simdgroup_a/b/c factories for fp16 carriers.
  * #05 — wave-3 commit `76e187a6` added `FloatingPointError` on NaN
    amax; that contract must be exposed via the new pipeline's
    `tl.extern_intrinsic` body validator.
* Effort: **large** (FP8 register layouts + 2-byte-LUT vendored shaders).

### 2.5 `mamba3.py` (826 LOC) + `mamba3_path_c.py` (627 LOC) + `_mamba3_helpers_tilelang.py` (627 LOC)

* Entry path: **#02 `torch_dynamo`** with **#09 autograd glue**. The
  state-space recurrence is a perfect `aot_autograd` joint-graph case;
  the `_mamba3_helpers_tilelang` already exposes the building blocks.
* Pipeline gaps exposed:
  * #02 — `aten.cumsum`, `aten.scan`, and Mamba's fused selective-scan
    are not in the wave-3 ATEN coverage list. Add SequentialEmitter
    coverage (or hand kernel under `_kernels/mamba_selective_scan.py`).
  * #09 — selective-scan double-backward is non-trivial. `register_double_backward`'s
    atomic-accumulator analytical-zero shortcut (commit `c1bf9bc9`) does
    NOT apply; needs a real custom backward.
* Effort: **large**.

## 3 Pipeline gaps (sorted by blocker severity)

1. **#01 `tl.sync_threads_partial(mask, n)`** primitive — blocks topk_selector. **HIGH**.
2. **#02 `aten.cumsum` / fused selective-scan emitter** — blocks Mamba family (≈2.5kLOC). **HIGH**.
3. **#01 `tl.dot(..., trans_b=True)` op** — blocks dsa_splitk + sparse_mla family. **HIGH**.
4. **#08 simdgroup_matrix per-thread fragment layout** — currently `Layout()`
   placeholder is "correct opaque", but `layout_inference.cc` cannot reason
   about the per-thread tiles → forces conservative shared-memory staging in
   bwd kernels. **MEDIUM**.
5. **#07 `BufferStore(BufferLoad(...))` non-opaque fallback** as the runtime
   default (currently a `kEmitOpaque=false` constexpr toggle in wave-2
   commit `9d2bb653`) — needed to recover `cp.async` on Ampere when these
   kernels run on CUDA. **MEDIUM**.
6. **#02 dynamic-shape symbolic tiles in `lower_node`** — wave-3 commit
   `c1bf9bc9` ships compile-time scaffolding; runtime `T.var("M")`
   substitution in the launcher is still TODO(wave-4). **MEDIUM**.
7. **#08 FP8 register layouts (`simdgroup_a_fp8`)** — blocks
   `fp8_vecmat_path_c.py` from going hardware-portable. Awaits Apple
   silicon FP8 path; until then keep `extern_intrinsic` MSL bodies.
   **LOW**.
8. **#09 selective-scan double-backward** — only relevant if downstream
   uses `torch.func.grad ∘ torch.func.grad` on Mamba. **LOW**.

## 4 Phased rollout

### Phase 1 — trivial (≤ 1 fix-agent each, no API changes)

* Add `tl.sync_threads_partial` to `op_mapping.py` (#01 gap 1).
* Add `tl.dot(trans_b=True)` lowering (#01 gap 3).
* Migrate `fp8_amax.py`: already in TileLang DSL; flip lowering to
  `tilelang.engine.lower(target=...)` once gaps 1+3 above land. Keeps
  call-site `tilelang_supports_with_reason` (commit `44f4f88`).
* Migrate `dsa_splitk_indexer_loss.py`: same flip; needs `compile_symbolic`
  exposure which c1bf9bc9 already wired.

### Phase 2 — moderate (per-kernel fix-agents, contract additions)

* `topk_selector.py` migration (after Phase 1 gap 1 lands).
* `sparse_mla_blockscaled.py` (Path B → Path C → unified) — moderate
  because it shares the `_msl_transform` codepath we want to retire.
* `_msl_transform.py` deprecation — once 4 of the 5 Path-C kernels
  switch to the unified pipeline, remove this 862-line shim.

### Phase 3 — large (multi-kernel families, real backward refactor)

* `sparse_mla_path_c.py` + `sparse_mla_fp8_path_c.py` +
  `sparse_mla_blockscaled_path_c.py` family.
* `mamba3` family (mamba3.py + mamba3_path_c.py + helpers).
* `fp8_vecmat_path_c.py` + `fp8_msl_kernels.py` consolidation.
* `m2rnn.py` (RNN cell with MSL Path B path).
* Full retirement of Path B (raw MSL shaders) once all backends
  reachable via TileLang DSL.

### Phase 4 — strict equivalence + deprecation

* Add cppmega-side parity tests that import each kernel via the new
  pipeline and the legacy Path B/Path C and assert numerical equality
  (rtol/atol per-dtype) for the fwd path. Keep Path B as a fallback for
  one release; remove in the next.
