#!/usr/bin/env bash
# Reconcile one explicitly selected isolated environment with uv.lock.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_ENV_ROOT="$(cd "$REPO_ROOT/.." && pwd)/.venvs/cppmega.mlx"
EXPECTED_ENV_ROOT="${CPPMEGA_MLX_ENV_ROOT:-$DEFAULT_ENV_ROOT}"
PYTHON_BIN="${CPPMEGA_MLX_PYTHON:-$EXPECTED_ENV_ROOT/bin/python}"

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--apply" ) ]]; then
  echo "ERROR: unexpected argument; use no arguments for a probe or exactly --apply" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  echo "No packages changed. Running the fail-closed runtime/ABI probe."
  exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
    -u CPPMEGA_MLX_SOURCE_ROOT -u CPPMEGA_TILELANG_SOURCE_ROOT \
    CPPMEGA_MLX_ENV_ROOT="$EXPECTED_ENV_ROOT" \
    CPPMEGA_MLX_PYTHON="$PYTHON_BIN" \
    "$REPO_ROOT/scripts/check_mlx_abi.sh"
fi

EXPECTED_REAL="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$EXPECTED_ENV_ROOT")"
EXPECTED_LEXICAL="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$EXPECTED_ENV_ROOT")"
PYTHON_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PYTHON_BIN")"
case "$PYTHON_ABS" in
  "$EXPECTED_LEXICAL/bin/"*)
    ;;
  *)
    echo "ERROR: refusing to sync unowned environment: Python $PYTHON_ABS is outside $EXPECTED_REAL/bin" >&2
    exit 2
    ;;
esac

if ! ENV_INFO="$(env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  -u CPPMEGA_MLX_SOURCE_ROOT -u CPPMEGA_TILELANG_SOURCE_ROOT \
  "$PYTHON_BIN" -c 'import os,sys; print(os.path.realpath(sys.prefix)); print(os.path.realpath(sys.base_prefix))')"; then
  echo "ERROR: cannot inspect Python environment prefix: $PYTHON_BIN" >&2
  exit 2
fi
ENV_PREFIX="${ENV_INFO%%$'\n'*}"
BASE_PREFIX="${ENV_INFO#*$'\n'}"
if [[ -z "$ENV_PREFIX" || -z "$BASE_PREFIX" || "$ENV_PREFIX" == "$BASE_PREFIX" ]]; then
  echo "ERROR: refusing to modify a system/base Python installation: prefix=$ENV_PREFIX base=$BASE_PREFIX" >&2
  exit 2
fi
if [[ "$ENV_PREFIX" != "$EXPECTED_REAL" ]]; then
  echo "ERROR: refusing to sync unowned environment $ENV_PREFIX" >&2
  echo "Expected $EXPECTED_REAL (override only with CPPMEGA_MLX_ENV_ROOT)." >&2
  exit 2
fi
if [[ ! -f "$ENV_PREFIX/pyvenv.cfg" ]]; then
  echo "ERROR: selected Python is not a verified virtual environment (missing $ENV_PREFIX/pyvenv.cfg)" >&2
  exit 2
fi

if ! GIT_ROOT="$("$PYTHON_BIN" - "$ENV_PREFIX" "$EXPECTED_LEXICAL" <<'PY'
from pathlib import Path
import os
import sys

# Check both the resolved and the lexical spelling of each candidate:
# when the selected env is reached through a symlink (e.g. a checkout
# .venv that points at a shared env outside the repo), the resolved
# walk alone sees no .git and the guard would be bypassed.
seen = set()
for arg in sys.argv[1:]:
    for path in (Path(arg).resolve(), Path(os.path.abspath(arg))):
        if path in seen:
            continue
        seen.add(path)
        for candidate in (path, *path.parents):
            if (candidate / ".git").exists():
                print(candidate)
                sys.exit(0)
PY
)"; then
  echo "ERROR: cannot inspect whether the selected environment is inside a Git checkout" >&2
  exit 2
fi
if [[ -n "$GIT_ROOT" ]]; then
  echo "ERROR: refusing to sync an environment inside git checkout $GIT_ROOT" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is unavailable; refusing to apply a package sync" >&2
  exit 2
fi

PURELIB="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
case "$PURELIB" in
  "$ENV_PREFIX/"*)
    ;;
  *)
    echo "ERROR: environment site-packages is outside $ENV_PREFIX: $PURELIB" >&2
    exit 2
    ;;
esac
CPPMEGA_MANIFEST="$ENV_PREFIX/cppmega-environment.json"
CPPMEGA_SOURCE_PATHS="$PURELIB/00_cppmega_sources.pth"
if [[ -e "$CPPMEGA_MANIFEST" || -e "$CPPMEGA_SOURCE_PATHS" ]]; then
  if [[ ! -f "$CPPMEGA_MANIFEST" || -L "$CPPMEGA_MANIFEST" \
      || ! -f "$CPPMEGA_SOURCE_PATHS" || -L "$CPPMEGA_SOURCE_PATHS" ]]; then
    echo "ERROR: refusing to preserve an incomplete or symlinked cppmega receipt" >&2
    exit 2
  fi
  RECEIPT_BACKUP="$(mktemp -d "${TMPDIR:-/tmp}/cppmega-mlx-receipt.XXXXXX")"
  cp -p "$CPPMEGA_MANIFEST" "$RECEIPT_BACKUP/cppmega-environment.json"
  cp -p "$CPPMEGA_SOURCE_PATHS" "$RECEIPT_BACKUP/00_cppmega_sources.pth"

  restore_cppmega_receipt() {
    mkdir -p "$(dirname "$CPPMEGA_SOURCE_PATHS")"
    cp -p "$RECEIPT_BACKUP/cppmega-environment.json" "$CPPMEGA_MANIFEST"
    cp -p "$RECEIPT_BACKUP/00_cppmega_sources.pth" "$CPPMEGA_SOURCE_PATHS"
    cmp -s "$RECEIPT_BACKUP/cppmega-environment.json" "$CPPMEGA_MANIFEST"
    cmp -s "$RECEIPT_BACKUP/00_cppmega_sources.pth" "$CPPMEGA_SOURCE_PATHS"
  }
  cleanup_receipt_backup() {
    rm -f "$RECEIPT_BACKUP/cppmega-environment.json"
    rm -f "$RECEIPT_BACKUP/00_cppmega_sources.pth"
    rmdir "$RECEIPT_BACKUP"
  }
  restore_receipt_on_exit() {
    status="$?"
    trap - EXIT
    restore_cppmega_receipt || status=1
    cleanup_receipt_backup || status=1
    exit "$status"
  }
  trap restore_receipt_on_exit EXIT
fi

env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  -u CPPMEGA_MLX_SOURCE_ROOT -u CPPMEGA_TILELANG_SOURCE_ROOT \
  -u TVM_HOME -u TVM_ROOT -u TVM_LIBRARY_PATH -u TVM_IMPORT_PYTHON_PATH \
  -u TVM_FFI_INCLUDE_PATH -u TVM_FFI_DLPACK_INCLUDE_PATH \
  -u TL_APACHE_TVM_SOURCE_HOME -u TL_APACHE_TVM_SWAP_HOME \
  -u TL_EXTERNAL_TVM_HOME -u TL_TILELANG_SITE -u TL_TILELANG_VENVS \
  -u TL_TVM_IMPORT_PYTHON_PATH -u DYLD_LIBRARY_PATH \
  uv lock --check --no-sources --project "$REPO_ROOT"

echo "Reconciling isolated environment from uv.lock: $ENV_PREFIX"
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  -u CPPMEGA_MLX_SOURCE_ROOT -u CPPMEGA_TILELANG_SOURCE_ROOT \
  -u TVM_HOME -u TVM_ROOT -u TVM_LIBRARY_PATH -u TVM_IMPORT_PYTHON_PATH \
  -u TVM_FFI_INCLUDE_PATH -u TVM_FFI_DLPACK_INCLUDE_PATH \
  -u TL_APACHE_TVM_SOURCE_HOME -u TL_APACHE_TVM_SWAP_HOME \
  -u TL_EXTERNAL_TVM_HOME -u TL_TILELANG_SITE -u TL_TILELANG_VENVS \
  -u TL_TVM_IMPORT_PYTHON_PATH -u DYLD_LIBRARY_PATH \
  UV_PROJECT_ENVIRONMENT="$ENV_PREFIX" \
  uv sync --project "$REPO_ROOT" --locked --no-sources \
    --python "$PYTHON_BIN" --extra parquet --extra gui --extra widget \
    --extra path-c --group dev

if [[ -n "${RECEIPT_BACKUP:-}" ]]; then
  restore_cppmega_receipt
  cleanup_receipt_backup
  trap - EXIT
fi

exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  -u CPPMEGA_MLX_SOURCE_ROOT -u CPPMEGA_TILELANG_SOURCE_ROOT \
  CPPMEGA_MLX_ENV_ROOT="$EXPECTED_REAL" \
  CPPMEGA_MLX_PYTHON="$PYTHON_BIN" \
  "$REPO_ROOT/scripts/check_mlx_abi.sh"
