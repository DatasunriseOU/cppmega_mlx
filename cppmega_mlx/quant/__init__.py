"""V8-R05 MXFP4 quantization package.

Standalone module hosting the e2m1 block-scaled codec for Apple Metal.
Re-exported from :mod:`cppmega_mlx.training._quantize_8bit` via the
``QUANT_SCHEME_MXFP4`` route.
"""

from cppmega_mlx.quant.mxfp4_metal import (
    QUANT_SCHEME_MXFP4,
    MXFP4_BLOCK_SIZE,
    MXFP4_LUT,
    quantize_mxfp4_blockwise,
    dequantize_mxfp4_blockwise,
    mxfp4_round_trip,
    quantize_round_trip_rmse,
)


__all__ = [
    "QUANT_SCHEME_MXFP4",
    "MXFP4_BLOCK_SIZE",
    "MXFP4_LUT",
    "quantize_mxfp4_blockwise",
    "dequantize_mxfp4_blockwise",
    "mxfp4_round_trip",
    "quantize_round_trip_rmse",
]
