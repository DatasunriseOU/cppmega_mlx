"""Path B TileLang kernel ports for cppmega MLX.

This subpackage hosts hand-written Apple Metal kernel ports of cppmega's
TileLang sources. Each port keeps a pure-MLX fallback and pairs the Metal
forward with a manual VJP via mx.custom_function so it remains differentiable.

Membership:

- _msl_transform.py: tiny MSL string assembly + dispatch helper.
- _path_b_lowering.py: vendored TileLang->MSL string-rewrite helpers used
  when a TileLang PrimFunc actually lowers to Metal (does not apply to
  topk_selector or sparse_mla, which fail at lower(); see their docs).
- _mamba3_helpers.py: pure-MLX rewrites of three Triton helpers
  (compute_dacs_segsum, bwd_dadt_fused, bwd_dtrap_ddt) that have no Metal
  backend in upstream Triton.
- mamba3.py: Path B port of mamba_ssm.ops.tilelang.mamba3.{fwd,bwd} plus the
  mx.custom_function wrapper that ties forward to backward.
- topk_selector.py: Path B port attempt for cppmega's tilelang_sparse_mla
  topk-selector kernel. Documents the TileLang metal codegen blockers and
  ships a pure-MLX runtime path.

The upstream TileLang TVM-Metal lowering (PR tile-ai/tilelang#799) is *not*
required at runtime: this module emits MSL directly via mx.fast.metal_kernel.
That is the Path B contract noted in docs/kernel_coverage_matrix.md.
"""

from cppmega_mlx.nn._tilelang import (
    _mamba3_helpers,
    _mamba3_helpers_tilelang,
    _msl_transform,
    _path_b_lowering,
    fp8_msl_kernels,
    mamba3,
    sparse_mla,
    sparse_mla_blockscaled,
    sparse_mla_fp8,
    topk_selector,
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

__all__ = [
    "FP8MSLKernelStatus",
    "Mamba3MetalStatus",
    "MXFP8_BLOCK_SIZE",
    "PathBStatus",
    "SparseMLABlockScaledMetalStatus",
    "SparseMLAFp8MetalStatus",
    "SparseMLAMetalStatus",
    "TransformedKernel",
    "_mamba3_helpers",
    "_mamba3_helpers_tilelang",
    "_msl_transform",
    "_path_b_lowering",
    "build_mlx_body",
    "bwd_dadt_fused",
    "bwd_dtrap_ddt",
    "compute_dacs_segsum",
    "fp8_msl_kernels",
    "fp8_msl_status",
    "fp8_scaled_matmul",
    "fp8_scaled_matmul_raw",
    "fp8_scaled_vecmat",
    "fp8_to_half",
    "half_to_fp8",
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
    "sparse_mla_fp8",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_bwd_metal",
    "sparse_mla_fp8_fwd_metal",
    "sparse_mla_fp8_metal_status",
    "sparse_mla_fp8_reference",
    "sparse_mla_fwd_metal",
    "sparse_mla_metal_status",
    "sparse_mla_quantized_matmul_reference",
    "topk_selector",
    "topk_selector_fn",
    "topk_selector_path_b_status",
    "topk_selector_reference",
    "transform_tilelang_kernel",
]
