#!/usr/bin/env bash
# ===========================================================================
# Build the standalone CUTLASS >= 4.5.1 SM120/SM121 MXFP8 GEMM .so on gb10.
#
# LEVER 3 (lever="cutlass-mxfp8"). HOST/CPU-ONLY (does NOT touch the GPU; the
# only GPU touch is the later bench). Side-checkout header-only build that does
# NOT bump the live tilelang 3rdparty/cutlass (== 4.1.0).
#
# RULE #1: every step that fails RAISES (set -e + explicit checks). NO silent
# fall-through to a stale .so or a different arch.
# ===========================================================================
set -euo pipefail

# --- Config (override via env) --------------------------------------------
CUTLASS_DIR="${CPPMEGA_CUTLASS_DIR:-/home/dave/source/cutlass-451}"
CUTLASS_TAG="${CPPMEGA_CUTLASS_TAG:-v4.5.1}"
NVCC="${CPPMEGA_NVCC:-/usr/local/cuda-13.3/bin/nvcc}"
# GB10/DGX-Spark is sm_121 (cc 12.1). The 'a' suffix is MANDATORY (CUTLASS
# #2820 / #3227): plain sm_121/sm_120 emits an arch-conditional MMA without the
# arch-specific target and ptxas/the kernel reject it.
ARCH="${CPPMEGA_CUTLASS_ARCH:-compute_121a,code=sm_121a}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/_cutlass_mxfp8_sm120.cu"
OUT="${HERE}/_cutlass_mxfp8_sm120.so"

echo "[build] CUTLASS_DIR=${CUTLASS_DIR} (tag ${CUTLASS_TAG})"
echo "[build] NVCC=${NVCC}"
echo "[build] ARCH=${ARCH}"
echo "[build] SRC=${SRC}"
echo "[build] OUT=${OUT}"

# --- STEP 1: side-checkout (header-only) ----------------------------------
if [ ! -d "${CUTLASS_DIR}/include/cutlass" ]; then
  echo "[build] cloning CUTLASS ${CUTLASS_TAG} side-checkout (header-only)..."
  git clone --depth 1 --branch "${CUTLASS_TAG}" \
      https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
else
  echo "[build] reusing existing CUTLASS side-checkout at ${CUTLASS_DIR}"
fi

VERSION_H="${CUTLASS_DIR}/include/cutlass/version.h"
if [ ! -f "${VERSION_H}" ]; then
  echo "[build][FATAL] missing ${VERSION_H}; side-checkout is broken." >&2
  exit 11
fi
echo "[build] CUTLASS version.h:"
grep -E 'CUTLASS_(MAJOR|MINOR|PATCH)' "${VERSION_H}" || true

# Guard: refuse to build against < 4.5.1 (the SM120 block-scaled MMAOP did not
# exist before 4.5.0/4.5.1). RULE #1: fail loud rather than emit a wrong kernel.
MAJ=$(grep -E 'define CUTLASS_MAJOR' "${VERSION_H}" | grep -oE '[0-9]+' | head -1)
MIN=$(grep -E 'define CUTLASS_MINOR' "${VERSION_H}" | grep -oE '[0-9]+' | head -1)
PAT=$(grep -E 'define CUTLASS_PATCH' "${VERSION_H}" | grep -oE '[0-9]+' | head -1)
echo "[build] parsed CUTLASS ${MAJ}.${MIN}.${PAT}"
if [ "${MAJ}" -lt 4 ] || { [ "${MAJ}" -eq 4 ] && [ "${MIN}" -lt 5 ]; }; then
  echo "[build][FATAL] CUTLASS ${MAJ}.${MIN}.${PAT} < 4.5.x: SM120 block-scaled" \
       "MXFP8 MMAOP is absent. Re-checkout v4.5.1." >&2
  exit 12
fi

# --- STEP 2: verify toolchain ---------------------------------------------
if [ ! -x "${NVCC}" ]; then
  echo "[build][FATAL] nvcc not found/executable at ${NVCC}" >&2
  exit 13
fi
"${NVCC}" --version | head -5

# --- STEP 3: compile (HOST/CPU-only) --------------------------------------
echo "[build] compiling MXFP8 GEMM .so (this is CPU-only; no GPU touch)..."
"${NVCC}" -std=c++17 -O3 --shared -Xcompiler -fPIC \
  -gencode "arch=${ARCH}" \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  --expt-relaxed-constexpr --expt-extended-lambda -lineinfo \
  "${SRC}" -o "${OUT}"

if [ ! -f "${OUT}" ]; then
  echo "[build][FATAL] nvcc reported success but ${OUT} is missing." >&2
  exit 14
fi
echo "[build] OK -> ${OUT}"
ls -la "${OUT}"

# --- STEP 4: symbol sanity (the extern C launcher must be present) --------
if command -v nm >/dev/null 2>&1; then
  echo "[build] exported launcher symbols:"
  nm -D "${OUT}" | grep -E 'cppmega_mxfp8' || {
    echo "[build][FATAL] launcher symbols cppmega_mxfp8_* missing from ${OUT}" >&2
    exit 15
  }
fi
echo "[build] DONE."
