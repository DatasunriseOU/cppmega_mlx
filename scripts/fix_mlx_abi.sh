#!/usr/bin/env bash
# fix_mlx_abi.sh — repair MLX venv-vs-brew dylib version mismatch.
#
# Strategy:
#   1. Pin the venv's mlx package to the brew-installed version so the
#      bundled .dylib matches what DYLD will resolve.
#   2. Fall back to plain pin (no force-reinstall) if step 1 fails.
#   3. Print manual instructions if both fail.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
BREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

if [[ ! -x "$VENV_PIP" ]]; then
  echo "ERROR: $VENV_PIP not found. Activate or create the venv first."
  exit 2
fi

BREW_VER=$(ls "$BREW_PREFIX/Cellar/mlx/" 2>/dev/null | head -1)
if [[ -z "$BREW_VER" ]]; then
  echo "INFO: no brew mlx detected. Pinning venv to latest pip-released mlx."
  TARGET=""
else
  TARGET="==$BREW_VER"
fi

echo "Reinstalling mlx$TARGET into $VENV_PY"
if "$VENV_PIP" install --force-reinstall --no-cache-dir "mlx$TARGET" 2>&1 | tail -5; then
  echo "OK: reinstall succeeded"
else
  echo "WARN: --force-reinstall failed; trying plain install"
  if "$VENV_PIP" install "mlx$TARGET" 2>&1 | tail -5; then
    echo "OK: plain install succeeded"
  else
    cat <<MANUAL
ERROR: pip install failed.

Manual recovery options:
  1. Recreate the venv from scratch:
       python3 -m venv .venv
       ./.venv/bin/pip install -e .
  2. Unlink the brew mlx so DYLD only finds the pip-installed dylib:
       brew unlink mlx
  3. Pin DYLD_LIBRARY_PATH to brew before running tests:
       export DYLD_LIBRARY_PATH=$BREW_PREFIX/lib

See docs/mlx_abi_troubleshooting.md for the full triage tree.
MANUAL
    exit 1
  fi
fi

# Verify
NEW_VER=$("$VENV_PY" -c "import mlx.core as mx; print(mx.__version__)" 2>&1 | head -1)
echo "post-fix venv mlx version: $NEW_VER"
exit 0
