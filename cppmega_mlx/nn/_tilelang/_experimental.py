"""Experimental Path C kernels — PROBE-ONLY / REDUCERS-ONLY surface.

These are not stable user APIs. See ``docs/production_kernel_routing.md`` for
routing status.

What lives here:

- ``fp8_vecmat_path_c`` status / feature / lowering helpers. The full apply
  ``fp8_scaled_vecmat_path_c`` is intentionally *not* re-exported here; it
  remains reachable via the submodule path. Runtime dispatch is currently
  broken pending ``tirx.metal.fp8_e4m3_dot4`` landing in the in-tree
  TileLang/TVM. See ``reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/``.
- ``sparse_mla_blockscaled_path_c`` E8M0 QK probe surfaces and the real-shape
  QK reducer apply (``..._reduce_path_c`` is a reducer apply, NOT a full
  end-to-end attention apply — there is no ``sparse_mla_blockscaled_path_c_apply``).

Everything re-exported below is also surfaced from
``cppmega_mlx.nn._tilelang`` via ``from ._experimental import *`` so existing
call sites (e.g. ``from cppmega_mlx.nn._tilelang import
blockscaled_sparse_mla_qk_reduce_path_c``) continue to work unchanged. This
module exists purely to keep the package ``__init__.py`` legible as more
partial Path C kernels land.
"""

# FP8 vecmat Path C — only status / lowering helpers re-exported.
# The full apply `fp8_scaled_vecmat_path_c` lives in the submodule and is
# NOT re-exported, partly because runtime dispatch is currently broken
# (`tirx.metal.fp8_e4m3_dot4` not registered in the in-tree TileLang/TVM).
from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import (
    FP8VecmatPathCStatus,
    fp8_vecmat_msl_features,
    fp8_vecmat_path_c_status,
    lower_fp8_vecmat_msl,
    make_fp8_vecmat_reduce_kernel,
)

# Blockscaled (E8M0) Sparse-MLA Path C is PROBE-ONLY — there is no full
# `sparse_mla_blockscaled_path_c_apply`. The exported `..._reduce_path_c` is
# a real-shape QK reducer apply, not an end-to-end attention apply. See the
# module docstring of `sparse_mla_blockscaled_path_c.py` and the routing doc
# (`docs/production_kernel_routing.md`) for the honest status.
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
    E8M0_BLOCK_SIZE,
    E8M0_LAYOUT,
    E8M0_SCALE_FORMAT,
    SparseMLABlockScaledPathCStatus,
    SparseMLABlockScaledQKReducePathCStatus,
    blockscaled_sparse_mla_qk_msl_features,
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_msl_features,
    blockscaled_sparse_mla_qk_reduce_path_c,  # reducer apply — NOT full Sparse-MLA
    blockscaled_sparse_mla_qk_reduce_path_c_status,
    blockscaled_sparse_mla_qk_scaled_matmul_probe_status,
    lower_blockscaled_sparse_mla_qk_msl,
    lower_blockscaled_sparse_mla_qk_reduce_msl,
    make_blockscaled_sparse_mla_qk_kernel,
    make_blockscaled_sparse_mla_qk_reduce_kernel,
)

__all__ = [
    # fp8_vecmat_path_c (REDUCERS-ONLY surface)
    "FP8VecmatPathCStatus",
    "fp8_vecmat_msl_features",
    "fp8_vecmat_path_c_status",
    "lower_fp8_vecmat_msl",
    "make_fp8_vecmat_reduce_kernel",
    # sparse_mla_blockscaled_path_c (PROBE-ONLY + reducer apply)
    "E8M0_BLOCK_SIZE",
    "E8M0_LAYOUT",
    "E8M0_SCALE_FORMAT",
    "SparseMLABlockScaledPathCStatus",
    "SparseMLABlockScaledQKReducePathCStatus",
    "blockscaled_sparse_mla_qk_msl_features",
    "blockscaled_sparse_mla_qk_path_c_status",
    "blockscaled_sparse_mla_qk_reduce_msl_features",
    "blockscaled_sparse_mla_qk_reduce_path_c",
    "blockscaled_sparse_mla_qk_reduce_path_c_status",
    "blockscaled_sparse_mla_qk_scaled_matmul_probe_status",
    "lower_blockscaled_sparse_mla_qk_msl",
    "lower_blockscaled_sparse_mla_qk_reduce_msl",
    "make_blockscaled_sparse_mla_qk_kernel",
    "make_blockscaled_sparse_mla_qk_reduce_kernel",
]
