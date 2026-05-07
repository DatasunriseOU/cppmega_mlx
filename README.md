# cppmega.mlx

MLX-native local training port for the cppmega model family on Apple Silicon.

This repository is intentionally not a direct Megatron/CUDA port. It keeps the
cppmega model contracts, feature semantics, and test ladder while replacing the
runtime with MLX and MLX-LM-derived local patterns. Apple Metal kernels remain
optional research/prototype seams until differentiated training support is
proven behind pure-MLX fallbacks.

Current status:

- compiled/eager tiny MLX pretraining step with fixed-key side-channel batches
- NPZ fixed-shape token dataset and tiny train smoke
- optional local-only GB10 Parquet sample smoke path under ignored
  data/parquet_samples/
- MLX safetensors checkpoint/resume helper
- tiny A/M/E/R hybrid smoke model for route coverage, not full NAM56R
- benchmark harness for local Apple GPU regression baselines
- optional prototype Metal kernel seam with pure MLX fallback
- package-root exports for local MLX subpackages and helper surfaces, not
  foreign trainer/runtime aliases

## Unified TileLang fused-kernel pipeline integration (waves 1-5)

The `_tilelang/` kernel directory is being migrated from hand-written Apple
Metal-only TileLang into the unified fused-kernel compiler maintained at
`DatasunriseOU/tilelang` (branch `poc-integrations-review`). The new pipeline
ingests Triton TTIR / torch.fx / cute-dsl / raw CUDA, normalises into TileLang
TIR, fuses register/shared-resident across former source boundaries, and
lowers to a single CUDA / HIP / Apple Metal SIMDgroup kernel. See
[`MIGRATION_PLAN.md`](./MIGRATION_PLAN.md) for the full inventory and phasing.

What landed (5 waves of agent-driven review-fix loops with grok-4):

- **Phase 1 unblockers**: `tl.sync_threads_partial`, `tt.dot trans_a/trans_b`,
  and a feature-flagged `_engine_dispatch.dispatch_lower(prim, target)` shim
  (env: `CPPMEGA_MLX_TILELANG_ENGINE=auto|engine|shim|engine_with_msl_extraction`).
- **Phase 2 migrations**: `topk_selector.py`, `sparse_mla_blockscaled_path_c.py`
  routed through the engine path; `_msl_transform.lower_tilelang_to_msl_inline`
  marked deprecated with a one-shot `DeprecationWarning`.
- **Phase 3 migrations**: `sparse_mla_path_c.py` (2611 LOC fwd+bwd),
  `mamba3_path_c.py` and `_mamba3_helpers_tilelang.py`, `fp8_vecmat_path_c.py`
  consolidation. `mamba3.py` and `fp8_msl_kernels.py` (Path-B raw MSL) keep
  legacy emission until FP8 SIMDgroup factories land.
- **MSL-extraction adapter** (Phase 3 keystone): `_msl_extraction.py` reads
  `artifact.kernel_source` from a `tilelang.engine.lower(...)` artifact and
  reconstructs a `TileLangMSLLowering`-compatible object via the existing
  `_msl_transform` text helpers. Unblocks the remaining 16 callers; flip them
  one-by-one with `dispatch_lower(..., return_msl=True)`.
- **DSA splitK perf hardening (waves 1-5)**: NaN/Inf guard, JIT bucket cache,
  dynamic `BLOCK_SIZE` picker, partial-block invariant, sparse-mask sign
  convention coverage, debug-gated GPU↔CPU sync (`CPPMEGA_MLX_DSA_DEBUG`),
  stage-1 head-0 fragment alloc gate, **wave-5** budget-gated full Q-cache
  (`use_q_cache_v5=True`, auto on Metal ≤16 KB / CUDA ≤64 KB / HIP ≤32 KB).
- **FP8 amax perf hardening**: pow-of-two shape buckets, target-aware block
  picker, partial-block guard, `tilelang_supports_with_reason` 2-tuple API.

Use `from cppmega_mlx.nn._tilelang import dispatch_lower, tilelang_engine_mode`
to opt-in via env or programmatically. The shim path is preserved verbatim for
correctness fallback.

Repo hygiene:

- Keep .venv, __pycache__, pytest caches, .beads, agent logs, and
  data/parquet_samples/ out of commits. The Parquet samples are useful for
  local real-data smoke tests, but they are large local artifacts, not repo
  fixtures.

Non-goals and limits:

- A local M4 Max benchmark is not a GB10 parity claim until the same workload is
  run on GB10 with matched model shape, dtype, data contract, warmup, measured
  steps, and metric definitions.
- This repo does not yet prove full NAM56R readiness, distributed training,
  production-scale Parquet input, production-scale Megatron .bin/.idx input,
  or training-path custom Metal kernels.
- Hugging Face Apple M4 kernels are research references only unless a future
  lane proves local parity, backward behavior, dtype coverage, and a measured
  cppmega hotspot.
- ../nanochat is a Torch reference only: useful for behavior checks, but too
  slow and not Metal-native enough to be the local Mac training substrate.

Useful commands:

bash
./.venv/bin/python -m pytest --collect-only -q
./.venv/bin/python -m pytest


Base package dependencies are mlx, mlx-lm, numpy, and safetensors.
Parquet loading stays optional: install the parquet extra or provide an
environment with pyarrow or pandas before using TokenParquetDataset.

bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --json


bash
TMP_DIR="$(mktemp -d)"
TMP_NPZ="$TMP_DIR/tiny_tokens.npz"
./.venv/bin/python - "$TMP_NPZ" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
tokens = (np.arange(2 * 128, dtype=np.int32) % 64).reshape(2, 128)
np.savez(path, tokens=tokens, vocab_size=np.array(64, dtype=np.int32))
PY

./.venv/bin/python scripts/train_tiny_npz.py "$TMP_NPZ" \
  --batch-size 2 \
  --seq-len 64 \
  --steps 2 \
  --dtype bfloat16 \
  --json

rm -rf "$TMP_DIR"


See docs/porting_plan.md for the implemented / wave-next / blocked roadmap
and docs/perf_baseline.md for the M4 Max vs GB10 comparison protocol.

## TileLang Z3 roadmap integration (2026-05-07)

Wired Z3-driven optimization passes (12 ideas total) from
`DatasunriseOU/tilelang` branch `z3-final` into MLX-side TileLang kernels.
All passes default OFF; opt-in via PassConfig. Conservative-by-default —
every Z3 query runs under try/catch with fallback to the slow path.

**Mac Metal performance (apples-to-apples, default-on):**

| Kernel                          | Path B   | Path C   | Speedup |
|---------------------------------|----------|----------|---------|
| `mamba3` fwd+bwd                | 7.577 ms | 5.950 ms | -21.5%  |
| `mamba3` bwd                    | 6.557 ms | 4.945 ms | -24.6%  |
| `topk_selector` B4·T4096·K256   | 9.009 ms | 5.087 ms | -43.5%  |
| `topk_selector` 6/7 shapes      | —        | —        | -34%..-44% |
| `sparse_mla` fp16 B4 S1024      | 0.610 ms | 0.616 ms | tie (gate passed) |

**Per-kernel wiring (this branch, `mlx-z3-wiring` merged into `main`):**

- `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py` — opt-in
  `tl.drop_provable_bound_checks` (#4) + filtered `tl.simd_lift_reductions`
  (#9, runtime probe). `K > 0` precondition + narrow exception in
  `_filter_supported_pass_configs`. `threading.Lock` on the PassConfig cache.
- `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` — hoist-aware
  `_canonicalize_*_lane_indexing` + `_strip_z3_hoisted_address_decls` so
  z3-final's algebraic rewrites compile cleanly to MSL.
- `cppmega_mlx/nn/_tilelang/topk_selector.py` — dual-gated Path B / Path C
  with auto-routing receipts (`_PATH_C_AUTO_PROFITABLE_RECEIPTS`); fixed
  insertion-sort early-break correctness regression on K=16/32; hard
  assertion in `_path_c_rewrite_merge_round` so MSL emission drift fails
  loud.
- `cppmega_mlx/nn/_tilelang/_msl_transform.py` — libz3 dev preload on
  Darwin gated by `CPPMEGA_ALLOW_UNSAFE_LIBZ3=1` (security: `/tmp` is
  world-writable; opt-in only). `OSError` vs `FileNotFoundError`
  distinguished; `_failed_attempts` retry cap = 3.
- `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py` — 32 KB shared-mem
  budget gate on `M_pre/D_pre` fragments (avoids CUDA register spill);
  `topk_indices` bounds-check before `scatter_`; NaN guard for
  fully-masked rows.

**Safety env vars (forwarded from TileLang fork):**

- `TILELANG_DISABLE_Z3=1` — global Z3 kill-switch (CUDA / gb10 workaround).
- `TILELANG_DISABLE_Z3_<NAME>=1` — per-pass kill (VECTORIZE,
  PREDICATE_FUSION, DROP_BOUND_CHECKS, TMA_LEGALITY, BARRIER_ELISION,
  INT24, DOT4_LEGALITY, SIMDGROUP).
- `CPPMEGA_ALLOW_UNSAFE_LIBZ3=1` — opt-in libz3 preload from
  `/tmp/tl_apache_tvm_swap/build/lib`.

Quality: 7 fix-rounds + 5 review-waves with grok+meta cross-LLM
corroboration. Bench receipts at `bench/tilelang_ports/*.json` (gate
threshold `paired_ratio ≤ 1.0`; auto-routing falls back to Path B for
non-profitable shapes). See the upstream
[TileLang README](https://github.com/DatasunriseOU/tilelang/blob/z3-final/README.md)
for the full pass index and design notes.

## Wave-7/8 — empirical test results & remaining bugs (2026-05-07)

The 5 waves of static review-fix loops with grok-4 produced multiple
"GREEN ship-ready" verdicts that did NOT survive empirical runtime
testing on Apple M4 Max (macOS 26.4.1, Metal 3.2, MLX 0.31.1). Test
matrix at
[`tilelang/docs/research/runtime_test_matrix.md`](https://github.com/DatasunriseOU/tilelang/blob/main/docs/research/runtime_test_matrix.md)
+ `numerical_parity_metal.md` + `engine_vs_shim_parity.md` +
`torch_compile_e2e.md` + `wave5_q_hoist_bench.md`.

**Empirical verdict — what really works on Metal**:

| Integration | Static review claim | Runtime reality |
|-------------|---------------------|-----------------|
| #01 Triton mapper | GREEN | Wave-3 ok; reduce_prod blocked by C++ pass bug (xfail) |
| #02 torch.compile | GREEN | **3/4 e2e cases work** after wave-7 #4 redo (`-> List[Tensor]`) |
| #03 PtrAnalysis | GREEN | 8/8 tests pass on Metal |
| #04 TritonStructured | GREEN | 2/2 vendor-drift pass |
| #05 fp8_amax | wave-3 ok | **Broken**: closure cells invisible to `get_type_hints` (wave-8 fix queued) |
| #06 dsa_splitk wave-5 | "~2× speedup" | Claim cannot fire on production shapes (AH≥8); budget gate too tight; `denom` IR leak fixed wave-8 |
| #07 TMA fallback | GREEN | Wave-7 #3 `tilelang.Allocate` dispatcher landed |
| #08 extern_intrinsic | GREEN | factories ok |
| #09 autograd | GREEN | 12/12 `test_autograd_compose.py` pass |

**Wave-7/8 fix commits** in this repo (mlx-z3-wiring → main):

- `a439df0` (wave-7 #1) — `_amax_kernel_for` closure capture for `in_dtype`
- `cac10a0` (wave-7 #2) — wave-5 stage-2 `s` IR scoping
- `bbe9334` (wave-8 #2) — wave-5 stage-2 `denom`/`denom1` IR scoping
- `fb73493` (wave-8 topk) — MSL extraction fallback chain in `_path_c_kernel_for`

**Wave-8 backlog (queued, agent-rate-limited)**:

1. fp8_amax `get_type_hints(__closure__)` fix — inject closure cells into `_kernel.__globals__` before `@T.prim_func`
2. dsa_splitk budget gate — tiled Q-cache for `AH≥8` production shapes
3. ATEN_DISPATCH `_scaled_dot_product_flash_attention_for_cpu` wiring (FA TileLang kernel exists at `_kernels/flash_attention.py`)
4. reduce_prod `vectorize_loop.cc` / `storage_rewrite.cc` mul-kind handling
5. `scripts/check_mlx_abi.sh` to catch venv-vs-brew dylib mismatch (host venv `mlx.core.so` was built against older `libmlx.dylib` — silent test skips)

NVFP4 on MLX research is committed at
[`tilelang/docs/research/nvfp4_mlx_metal.md`](https://github.com/DatasunriseOU/tilelang/blob/main/docs/research/nvfp4_mlx_metal.md):
`mlx.core.quantize(mode="nvfp4")`, group_size=16, packed `uint32` + signed
E4M3 scale buffer; **no FP4 tensor core on M3/M4/M5** — implemented as
Metal compute shaders (PR ml-explore/mlx#2946, v0.30.3). Wave-8 candidates
for NVFP4 adoption: `linear_qmm`, `embedding_qlookup`, `gemm_quantized`,
`gather_qmm_rhs`.
