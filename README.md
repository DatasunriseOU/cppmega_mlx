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
