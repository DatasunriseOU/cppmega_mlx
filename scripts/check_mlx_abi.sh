#!/usr/bin/env bash
# check_mlx_abi.sh — diagnose MLX venv-vs-brew dylib version mismatch.
#
# Symptom: cppmega_mlx engine-path tests skip silently because the venv's
# mlx.core.so was built against one libmlx.dylib but DYLD picks up a
# different version from /opt/homebrew/Cellar/mlx/.
#
# Exit 0 on PASS (no mismatch), 1 on FAIL (mismatch detected).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
BREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

probe_python() {
  local py=$1
  if [[ ! -x "$py" ]]; then
    echo "MISSING"; return
  fi
  "$py" -c "import mlx.core as mx; print(mx.__version__)" 2>/dev/null || echo "IMPORT_ERROR"
}

probe_brew() {
  ls "$BREW_PREFIX/Cellar/mlx/" 2>/dev/null | head -1 || echo "NOT_INSTALLED"
}

VENV_VER=$(probe_python "$VENV_PY")
BREW_VER=$(probe_brew)

echo "venv ($VENV_PY): $VENV_VER"
echo "brew ($BREW_PREFIX/Cellar/mlx): $BREW_VER"

if [[ "$VENV_VER" == "MISSING" || "$VENV_VER" == "IMPORT_ERROR" ]]; then
  echo "FAIL: venv python cannot import mlx.core"
  echo "FIX: run scripts/fix_mlx_abi.sh"
  exit 1
fi

if [[ "$BREW_VER" == "NOT_INSTALLED" ]]; then
  echo "PASS: no brew mlx, no mismatch possible"
  exit 0
fi

if [[ "$VENV_VER" == "$BREW_VER" ]]; then
  echo "PASS: venv and brew agree on mlx==$VENV_VER"
  exit 0
fi

echo "FAIL: venv mlx==$VENV_VER vs brew mlx==$BREW_VER (ABI may diverge)"
echo "FIX: run scripts/fix_mlx_abi.sh"
exit 1
