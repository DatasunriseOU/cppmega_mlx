#!/usr/bin/env bash
set -euo pipefail

: "${VBGUI_E2E_PYTHON:?VBGUI_E2E_PYTHON must name the isolated runner Python}"
: "${CPPMEGA_MLX_LM_CHECKOUT:?CPPMEGA_MLX_LM_CHECKOUT must name the patched mlx-lm checkout}"
: "${CPPMEGA_MLX_LM_COMMIT:?CPPMEGA_MLX_LM_COMMIT must pin the patched mlx-lm checkout}"

repo_root="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
actual_commit="$(git -C "$CPPMEGA_MLX_LM_CHECKOUT" rev-parse HEAD)"
if [[ "$actual_commit" != "$CPPMEGA_MLX_LM_COMMIT" ]]; then
  printf 'patched mlx-lm checkout mismatch: expected=%s actual=%s\n' \
    "$CPPMEGA_MLX_LM_COMMIT" "$actual_commit" >&2
  exit 1
fi
git -C "$CPPMEGA_MLX_LM_CHECKOUT" diff --quiet
git -C "$CPPMEGA_MLX_LM_CHECKOUT" diff --cached --quiet

"$VBGUI_E2E_PYTHON" -m pip install --disable-pip-version-check \
  -e "$CPPMEGA_MLX_LM_CHECKOUT"
"$VBGUI_E2E_PYTHON" -m pip install --disable-pip-version-check \
  -e "$repo_root[gui,parquet,widget]"

"$VBGUI_E2E_PYTHON" - "$CPPMEGA_MLX_LM_CHECKOUT" <<'PY'
from pathlib import Path
import sys

from mlx_lm.models import (
    bailing_hybrid,
    deepseek_v4,
    gemma4_assistant,
    mistral4,
    nemotron_h,
    turbo_cache,
)

from cppmega_mlx.training.native_optim import status


checkout = Path(sys.argv[1]).resolve()
for module in (
    bailing_hybrid,
    deepseek_v4,
    gemma4_assistant,
    mistral4,
    nemotron_h,
    turbo_cache,
):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(checkout):
        raise RuntimeError(
            f"mlx-lm module {module.__name__} came from {module_path}, "
            f"not pinned checkout {checkout}"
        )

native_status = status()
if not native_status.get("available", False):
    raise RuntimeError(
        native_status.get("reason", "native optimizer extension unavailable")
    )
PY
