"""Path B/C TileLang kernel ports for cppmega MLX.

This subpackage hosts Apple Metal kernel ports of cppmega's TileLang sources.
Each port keeps a pure-MLX fallback and pairs the Metal forward with a manual
VJP via mx.custom_function so it remains differentiable.

Membership:

- _msl_transform.py: legacy TileLang->MSL inline lowering shim. **DEPRECATED.**
  New callers should route through ``dispatch_lower(prim, target)`` which
  prefers ``tilelang.engine.lower(target=...)`` (engine path) and only falls
  back to this shim when the engine is unavailable. See
  ``_engine_dispatch.py`` and ``MIGRATION_PLAN.md``. Existing callers that
  consume ``TileLangMSLLowering.body``/``.header`` strings for
  ``mx.fast.metal_kernel`` need an adapter layer (Phase 3) before they can
  flip — the engine artifact is a runtime callable, not an MSL string.
- _engine_dispatch.py: ``dispatch_lower(prim, target)`` — phase-1 dispatcher
  that flips between engine and shim based on ``$CPPMEGA_MLX_TILELANG_ENGINE``.
- _path_b_lowering.py: vendored TileLang->MSL string-rewrite helpers used
  when a TileLang PrimFunc actually lowers to Metal (does not apply to
  direct-MSL Path B kernels).
- _mamba3_helpers.py: pure-MLX rewrites of three Triton helpers
  (compute_dacs_segsum, bwd_dadt_fused, bwd_dtrap_ddt) that have no Metal
  backend in upstream Triton.
- mamba3.py: Path B port of mamba_ssm.ops.tilelang.mamba3.{fwd,bwd} plus the
  mx.custom_function wrapper that ties forward to backward.
- topk_selector.py: Path B/C port for cppmega's tilelang_sparse_mla
  topk-selector kernel. AUTO prefers Path C where the checked-in bench receipt
  keeps Path C no worse than Path B, then falls back to Path B and pure MLX.

Path B emits MSL directly via mx.fast.metal_kernel. Path C lowers TileLang DSL
to Metal when the in-tree lowering bridge supports the shape/kernel.

Public Path C surface — what is *exported here* vs what only lives in submodules:

- ``sparse_mla_path_c.py`` (full apply): exports the ``_fwd``/``_bwd`` raw
  kernels, ``_status``, and lowering dumps. The ``sparse_mla_path_c_apply``
  user wrapper is **intentionally not re-exported** — callers go through the
  AUTO gate in ``sparse_mla_apply``. ``test_package_exports`` enforces this.
- ``sparse_mla_blockscaled_path_c.py`` (prepared-buffer apply): the submodule
  exposes E8M0 QK probes/reducers and ``sparse_mla_blockscaled_path_c_apply``
  for existing FP8/scales buffers. It is **not** a high-level float-tensor
  wrapper and is intentionally not re-exported here.
- ``sparse_mla_fp8_path_c.py`` (prepared-buffer apply): not imported here.
  Tests reach QK reducers and ``sparse_mla_fp8_path_c_apply`` via the
  submodule path. The apply consumes prepared FP8/scales buffers; high-level
  float carriers still route through Path B or the graph planner.
- ``fp8_vecmat_path_c.py`` (full apply lives in submodule):
  ``fp8_scaled_vecmat_path_c`` exists in code but is **not** exported here.
  Only the status/feature/lowering helpers are re-exported. Callers must
  ``from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import fp8_scaled_vecmat_path_c``
  if they want the apply, and accept that runtime dispatch is currently broken
  pending the FP8 dot4 intrinsic landing in TileLang/TVM.
- ``mamba3_path_c.py`` (full apply): not imported here. The Path B
  ``mamba3_mimo_apply`` is the production entrypoint; Mamba3 Path C is a
  proof/override surface reached via the submodule import.
"""

from cppmega_mlx.nn._tilelang import (
    _engine_dispatch,
    _mamba3_helpers,
    _mamba3_helpers_tilelang,
    _msl_transform,
    _path_b_lowering,
    fp8_msl_kernels,
    m2rnn,
    mamba3,
    sparse_mla,
    sparse_mla_blockscaled,
    sparse_mla_fp8,
    topk_selector,
)
from cppmega_mlx.nn._tilelang._engine_dispatch import (
    dispatch_lower,
    tilelang_engine_mode,
)
from cppmega_mlx.nn._tilelang._mamba3_helpers import (
    bwd_dadt_fused,
    bwd_dtrap_ddt,
    compute_dacs_segsum,
)
from cppmega_mlx.nn._tilelang._path_b_lowering import (
    TransformedKernel,
    build_mlx_body,
    transform_tilelang_kernel,
)
from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (
    FP8MSLKernelStatus,
    fp8_msl_status,
    fp8_scaled_matmul,
    fp8_scaled_matmul_raw,
    fp8_scaled_vecmat,
    fp8_to_half,
    half_to_fp8,
)
from cppmega_mlx.nn._tilelang.m2rnn import (
    M2RNNMetalStatus,
    m2rnn_apply,
    m2rnn_apply_with_state,
    m2rnn_bwd_metal,
    m2rnn_fwd_metal,
    m2rnn_metal_status,
    m2rnn_reference,
)
from cppmega_mlx.nn._tilelang.mamba3 import (
    Mamba3MetalStatus,
    mamba3_mimo_apply,
    mamba3_mimo_bwd_metal,
    mamba3_mimo_fwd_metal,
    mamba3_mimo_metal_status,
    mamba3_mimo_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla import (
    SparseMLAMetalStatus,
    sparse_mla_apply,
    sparse_mla_bwd_metal,
    sparse_mla_fwd_metal,
    sparse_mla_metal_status,
)
from cppmega_mlx.nn._tilelang.sparse_mla_path_c import (
    SparseMLAPathCStatus,
    dump_lowered_bwd_msl,
    dump_lowered_fwd_msl,
    sparse_mla_bwd_path_c,
    sparse_mla_fwd_path_c,
    sparse_mla_path_c_status,
)
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled import (
    MXFP8_BLOCK_SIZE,
    SparseMLABlockScaledMetalStatus,
    sparse_mla_blockscaled_apply,
    sparse_mla_blockscaled_bwd_metal,
    sparse_mla_blockscaled_fwd_metal,
    sparse_mla_blockscaled_metal_status,
    sparse_mla_blockscaled_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
    SparseMLAFp8MetalStatus,
    sparse_mla_fp8_apply,
    sparse_mla_fp8_bwd_metal,
    sparse_mla_fp8_fwd_metal,
    sparse_mla_fp8_metal_status,
    sparse_mla_fp8_reference,
    sparse_mla_quantized_matmul_reference,
)
from cppmega_mlx.nn._tilelang.topk_selector import (
    PathBStatus,
    topk_selector_path_b_status,
    topk_selector_reference,
)
from cppmega_mlx.nn._tilelang.topk_selector import topk_selector as topk_selector_fn

# Experimental status/lowering helpers — re-exported via _experimental for organization.
from cppmega_mlx.nn._tilelang._experimental import (
    E8M0_BLOCK_SIZE,
    E8M0_LAYOUT,
    E8M0_SCALE_FORMAT,
    FP8VecmatPathCStatus,
    SparseMLABlockScaledPathCStatus,
    SparseMLABlockScaledQKReducePathCStatus,
    blockscaled_sparse_mla_qk_msl_features,
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_msl_features,
    blockscaled_sparse_mla_qk_reduce_path_c,
    blockscaled_sparse_mla_qk_reduce_path_c_status,
    blockscaled_sparse_mla_qk_scaled_matmul_probe_status,
    fp8_vecmat_msl_features,
    fp8_vecmat_path_c_status,
    lower_blockscaled_sparse_mla_qk_msl,
    lower_blockscaled_sparse_mla_qk_reduce_msl,
    lower_fp8_vecmat_msl,
    make_blockscaled_sparse_mla_qk_kernel,
    make_blockscaled_sparse_mla_qk_reduce_kernel,
    make_fp8_vecmat_reduce_kernel,
)

__all__ = [
    "dispatch_lower",
    "tilelang_engine_mode",
    "_engine_dispatch",
    "FP8MSLKernelStatus",
    "FP8VecmatPathCStatus",
    "E8M0_BLOCK_SIZE",
    "E8M0_LAYOUT",
    "E8M0_SCALE_FORMAT",
    "M2RNNMetalStatus",
    "Mamba3MetalStatus",
    "MXFP8_BLOCK_SIZE",
    "PathBStatus",
    "SparseMLABlockScaledMetalStatus",
    "SparseMLABlockScaledQKReducePathCStatus",
    "SparseMLABlockScaledPathCStatus",
    "SparseMLAFp8MetalStatus",
    "SparseMLAMetalStatus",
    "SparseMLAPathCStatus",
    "TransformedKernel",
    "_mamba3_helpers",
    "_mamba3_helpers_tilelang",
    "_msl_transform",
    "_path_b_lowering",
    "build_mlx_body",
    "bwd_dadt_fused",
    "bwd_dtrap_ddt",
    "blockscaled_sparse_mla_qk_msl_features",
    "blockscaled_sparse_mla_qk_path_c_status",
    "blockscaled_sparse_mla_qk_reduce_msl_features",
    "blockscaled_sparse_mla_qk_reduce_path_c",
    "blockscaled_sparse_mla_qk_reduce_path_c_status",
    "blockscaled_sparse_mla_qk_scaled_matmul_probe_status",
    "compute_dacs_segsum",
    "fp8_msl_kernels",
    "fp8_msl_status",
    "fp8_scaled_matmul",
    "fp8_scaled_matmul_raw",
    "fp8_scaled_vecmat",
    "fp8_to_half",
    "fp8_vecmat_msl_features",
    "fp8_vecmat_path_c_status",
    "half_to_fp8",
    "lower_fp8_vecmat_msl",
    "lower_blockscaled_sparse_mla_qk_msl",
    "lower_blockscaled_sparse_mla_qk_reduce_msl",
    "make_blockscaled_sparse_mla_qk_reduce_kernel",
    "make_blockscaled_sparse_mla_qk_kernel",
    "make_fp8_vecmat_reduce_kernel",
    "m2rnn",
    "m2rnn_apply",
    "m2rnn_apply_with_state",
    "m2rnn_bwd_metal",
    "m2rnn_fwd_metal",
    "m2rnn_metal_status",
    "m2rnn_reference",
    "mamba3",
    "mamba3_mimo_apply",
    "mamba3_mimo_bwd_metal",
    "mamba3_mimo_fwd_metal",
    "mamba3_mimo_metal_status",
    "mamba3_mimo_reference",
    "sparse_mla",
    "sparse_mla_apply",
    "sparse_mla_blockscaled",
    "sparse_mla_blockscaled_apply",
    "sparse_mla_blockscaled_bwd_metal",
    "sparse_mla_blockscaled_fwd_metal",
    "sparse_mla_blockscaled_metal_status",
    "sparse_mla_blockscaled_reference",
    "sparse_mla_bwd_metal",
    "sparse_mla_bwd_path_c",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "sparse_mla_fp8",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_bwd_metal",
    "sparse_mla_fp8_fwd_metal",
    "sparse_mla_fp8_metal_status",
    "sparse_mla_fp8_reference",
    "sparse_mla_fwd_metal",
    "sparse_mla_fwd_path_c",
    "sparse_mla_metal_status",
    "sparse_mla_path_c_status",
    "sparse_mla_quantized_matmul_reference",
    "topk_selector",
    "topk_selector_fn",
    "topk_selector_path_b_status",
    "topk_selector_reference",
    "transform_tilelang_kernel",
]
