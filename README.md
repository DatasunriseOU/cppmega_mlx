# cppmega.mlx

MLX-native local training port for the cppmega model family on Apple Silicon.

This repository is intentionally not a direct Megatron/CUDA port. It keeps the
cppmega model contracts, feature semantics, and test ladder while replacing the
runtime with MLX, MLX-LM patterns, and later Apple Metal kernels.

Current status:

- compiled/eager tiny MLX pretraining step with fixed-key side-channel batches
- NPZ fixed-shape token dataset and tiny train smoke
- optional local-only GB10 Parquet sample smoke path under ignored
  `data/parquet_samples/`
- MLX safetensors checkpoint/resume helper
- tiny A/M/E/R hybrid smoke model for route coverage, not full NAM56R
- benchmark harness for local Apple GPU regression baselines
- optional prototype Metal kernel seam with pure MLX fallback

Repo hygiene:

- Keep `.venv`, `__pycache__`, pytest caches, `.beads`, agent logs, and
  `data/parquet_samples/` out of commits. The Parquet samples are useful for
  local real-data smoke tests, but they are large local artifacts, not repo
  fixtures.

Non-goals and limits:

- A local M4 Max benchmark is not a GB10 parity claim until the same workload is
  run on GB10 with matched model shape, dtype, data contract, warmup, measured
  steps, and metric definitions.
- This repo does not yet prove full NAM56R readiness, distributed training,
  production-scale Parquet input, production-scale Megatron `.bin/.idx` input,
  or training-path custom Metal kernels.
- Hugging Face Apple M4 kernels are research references only unless a future
  lane proves local parity, backward behavior, dtype coverage, and a measured
  cppmega hotspot.
- `../nanochat` is a Torch reference only: useful for behavior checks, but too
  slow and not Metal-native enough to be the local Mac training substrate.

Useful commands:

```bash
./.venv/bin/python -m pytest --collect-only -q
./.venv/bin/python -m pytest
```

Base package dependencies are `mlx`, `mlx-lm`, `numpy`, and `safetensors`.
Parquet loading stays optional: install the `parquet` extra or provide an
environment with `pyarrow` or `pandas` before using `TokenParquetDataset`.

```bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --json
```

```bash
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
```

See `docs/porting_plan.md` for the implemented / wave-next / blocked roadmap
and `docs/perf_baseline.md` for the M4 Max vs GB10 comparison protocol.
