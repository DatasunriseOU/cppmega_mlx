# pyright: reportMissingImports=false
"""Native SM120/SM121 block-scaled MXFP8 x MXFP8 GEMM driver (LEVER 3).

lever="cutlass-mxfp8". Python driver for the standalone CUTLASS >= 4.5.1
collective-builder kernel in ``_cutlass_mxfp8_sm120.cu`` (built to
``_cutlass_mxfp8_sm120.so`` by ``_cutlass_mxfp8_sm120_build.sh`` on gb10).

This is the ONE native-MXFP8 path:
``D(M,N) = A(M,K) @ B(N,K)^T`` with both operands ``e4m3`` and per-32-element
``E8M0 (ue8m0)`` block scales, TN layout, on GB10 (sm_121 == cc 12.1) via the
SM120 ``OpClassBlockScaledTensorOp`` warp-level f8f6f4 MMA atom.

Why a side-checkout + standalone .so (NOT a tilelang submodule bump):
  The live tilelang ``3rdparty/cutlass`` is pinned at 4.1.0 and its built ``.so``
  depends on it. The SM120 block-scaled ``MXFP8MMAOP`` / ``MXF8F6F4MMAOP`` and the
  ``sm120_blockscaled_mma_builder.inl`` collective builder did not exist until
  CUTLASS 4.5.0/4.5.1. We therefore compile a single GEMM ``.cu`` against a
  header-only side-checkout of v4.5.1 and ``dlopen`` it — without rebuilding or
  risking the working tilelang/tvm build.

The Python-DSL ptxas-reject bug (CUTLASS issue #3227) does NOT apply here: this
is the C++ collective-builder route, the SAME path measured at ~188 TFLOPS FP8
on a real DGX Spark (NVIDIA Dev Forum thread 359960).

RULE #1 (NO silent fallback):
  * The ``.so`` is loaded explicitly; a missing/unbuilt ``.so`` RAISES with the
    exact build command to run (no bf16/cuBLAS substitute).
  * Every nonzero return from the C launcher RAISES with the launcher's own
    error string (``cppmega_mxfp8_last_error``: failing site + cudaGetErrorString).
  * Operand-pointer extraction RAISES if a buffer is not a contiguous CUDA
    ``e4m3`` device tensor — never a host bounce, never a dtype downgrade.

Scale production:
  The E8M0 (ue8m0) one-byte-per-32-element block scales are produced by REUSING
  the per-block abs-max machinery. ``build_e8m0_block_scales`` computes, for each
  contiguous 32-element block along K, ``ceil(log2(amax_block / fp8_max))`` as a
  ue8m0 exponent byte (matching the OCP MX E8M0 codec), then lays the bytes into
  the CUTLASS ``Sm1xxBlkScaledConfig`` SFA/SFB atom ordering whose total byte
  count the C side reports via ``cppmega_mxfp8_sf_sizes`` (so the Python side
  never re-derives the non-obvious 128x4 atom layout — it asks the kernel).

This module does NOT import torch/tvm/CUTLASS at import time; everything is
deferred so it stays importable on a non-GPU host (e.g. for py_compile / AST
cross-symbol checks on the Mac). The actual drive requires gb10.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# Default location of the compiled side-checkout .so (next to this module).
_DEFAULT_SO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_cutlass_mxfp8_sm120.so"
)
_DEFAULT_CU = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_cutlass_mxfp8_sm120.cu"
)
_DEFAULT_BUILD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_cutlass_mxfp8_sm120_build.sh"
)

# E4M3 max representable magnitude (== torch.finfo(float8_e4m3fn).max). Reused
# from fp8_amax._FP8_E4M3_MAX so the scale convention matches the rest of the
# fp8 pipeline. Hard-coded here to avoid importing torch at module import.
_FP8_E4M3_MAX: float = 448.0

# OCP MX block size for MXFP8 (one E8M0 scale byte per 32 elements).
MXFP8_BLOCK: int = 32

# E8M0 (ue8m0) is a biased unsigned 8-bit exponent: stored = unbiased_exp + 127,
# clamped to [0, 254]; 255 (0xFF) is the NaN code (avoided).
_E8M0_BIAS: int = 127
_E8M0_MAX_BIASED: int = 254


@dataclass(frozen=True)
class CutlassMxfp8Status:
    """Importability / build status for the standalone CUTLASS MXFP8 kernel."""

    available: bool
    reason: str
    so_path: str = _DEFAULT_SO


def cutlass_mxfp8_status(so_path: str | None = None) -> CutlassMxfp8Status:
    """Return whether the compiled MXFP8 .so is present and loadable.

    Does NOT attempt a GPU launch — only that the shared object exists and
    exports the launcher symbols. A missing .so is reported (not raised) so a
    capability probe is non-fatal; the actual GEMM RAISES on a missing .so.
    """

    path = so_path or os.environ.get("CPPMEGA_CUTLASS_MXFP8_SO") or _DEFAULT_SO
    if not os.path.isfile(path):
        return CutlassMxfp8Status(
            available=False,
            reason=(
                f"compiled MXFP8 .so not found at {path!r}; build it on gb10 with "
                f"{_DEFAULT_BUILD} (header-only CUTLASS v4.5.1 side-checkout + "
                f"nvcc -gencode arch=compute_121a,code=sm_121a)."
            ),
            so_path=path,
        )
    try:
        lib = ctypes.CDLL(path)
        for sym in (
            "cppmega_mxfp8_gemm_sm121",
            "cppmega_mxfp8_sf_sizes",
            "cppmega_mxfp8_last_error",
        ):
            if not hasattr(lib, sym):
                return CutlassMxfp8Status(
                    available=False,
                    reason=f"{path!r} is missing exported symbol {sym!r}",
                    so_path=path,
                )
    except OSError as exc:
        return CutlassMxfp8Status(
            available=False,
            reason=f"ctypes.CDLL failed to load {path!r}: {exc}",
            so_path=path,
        )
    return CutlassMxfp8Status(
        available=True,
        reason="CUTLASS MXFP8 SM120/SM121 kernel .so is present and exports the launcher",
        so_path=path,
    )


@lru_cache(maxsize=4)
def _load_lib(so_path: str) -> Any:
    """Load and ABI-annotate the CUTLASS MXFP8 .so (cached per path).

    RULE #1: a missing/unbuilt .so RAISES with the exact build command; there is
    no bf16/cuBLAS substitute kernel.
    """

    if not os.path.isfile(so_path):
        raise FileNotFoundError(
            f"cutlass_mxfp8_sm120: compiled kernel .so not found at {so_path!r}. "
            f"Build it on gb10 (HOST/CPU-only):\n  bash {_DEFAULT_BUILD}\n"
            f"(header-only CUTLASS v4.5.1 side-checkout + "
            f"nvcc -gencode arch=compute_121a,code=sm_121a over {_DEFAULT_CU}). "
            f"No bf16/cuBLAS fallback exists (RULE #1)."
        )
    lib = ctypes.CDLL(so_path)

    lib.cppmega_mxfp8_last_error.restype = ctypes.c_char_p
    lib.cppmega_mxfp8_last_error.argtypes = []

    lib.cppmega_mxfp8_sf_sizes.restype = ctypes.c_int
    lib.cppmega_mxfp8_sf_sizes.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]

    lib.cppmega_mxfp8_gemm_sm121.restype = ctypes.c_int
    lib.cppmega_mxfp8_gemm_sm121.argtypes = [
        ctypes.c_void_p,  # A_e4m3
        ctypes.c_void_p,  # SFA_e8m0
        ctypes.c_void_p,  # B_e4m3
        ctypes.c_void_p,  # SFB_e8m0
        ctypes.c_void_p,  # C
        ctypes.c_void_p,  # D
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_float,   # alpha
        ctypes.c_float,   # beta
        ctypes.c_void_p,  # stream
    ]
    return lib


def _last_error(lib: Any) -> str:
    raw = lib.cppmega_mxfp8_last_error()
    if raw is None:
        return "<no error string>"
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def sf_sizes(M: int, N: int, K: int, *, so_path: str | None = None) -> tuple[int, int]:
    """Return ``(sfa_bytes, sfb_bytes)`` from the CUTLASS atom layout.

    Asks the kernel (``cppmega_mxfp8_sf_sizes``) so the Python side never has to
    re-derive the ``Sm1xxBlkScaledConfig`` 128x4 atom ordering. RAISES on a shape
    that violates the block-scaled contract (K%32!=0, N<32) — no silent rounding.
    """

    path = so_path or os.environ.get("CPPMEGA_CUTLASS_MXFP8_SO") or _DEFAULT_SO
    lib = _load_lib(path)
    sfa = ctypes.c_int64(0)
    sfb = ctypes.c_int64(0)
    rc = lib.cppmega_mxfp8_sf_sizes(
        int(M), int(N), int(K), ctypes.byref(sfa), ctypes.byref(sfb)
    )
    if rc != 0:
        raise ValueError(
            f"cutlass_mxfp8_sm120.sf_sizes: kernel rejected shape "
            f"M={M} N={N} K={K} (rc={rc}): {_last_error(lib)}"
        )
    return int(sfa.value), int(sfb.value)


# ---------------------------------------------------------------------------
# E8M0 block-32 scale production (reuses the per-block abs-max convention from
# fp8_amax). torch is imported lazily so this module stays importable off-GPU.
# ---------------------------------------------------------------------------


def build_e8m0_block_scales(x: "Any") -> "Any":
    """Compute per-32-element ``E8M0 (ue8m0)`` block scales for a 2D fp tensor.

    ``x`` is a ``(rows, K)`` torch tensor (fp16/bf16/fp32) whose last dim K is a
    multiple of 32. Returns a ``(rows, K // 32)`` ``uint8`` CUDA tensor of biased
    E8M0 exponents, one per contiguous 32-element block, following the OCP MX
    convention used by the MXFP8 codec:

        block_amax = max(|x[:, b*32:(b+1)*32]|)
        unbiased   = ceil(log2(block_amax / fp8_max))     # so amax/scale <= fp8_max
        biased     = clamp(unbiased + 127, 0, 254)         # 255 == NaN, avoided
        scale      = 2 ** (biased - 127)

    The e4m3 quantization of the block is then ``x / scale`` clamped to
    ``[-fp8_max, fp8_max]``; the CUTLASS kernel multiplies the dequantized
    product back by ``scale`` per block. This mirrors ``fp8_amax.fp8_pack_tilelang``
    per-tensor logic at per-32-block granularity.

    RULE #1: a non-finite input RAISES (a degenerate scale would silently poison
    every output element). All-zero blocks map to the minimum biased exponent
    (scale = 2**-127), the E8M0 identity for an empty block.
    """

    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"build_e8m0_block_scales: expected torch.Tensor, got {type(x).__name__}"
        )
    if x.dim() != 2:
        raise ValueError(
            f"build_e8m0_block_scales: expected 2D (rows, K), got shape {tuple(x.shape)}"
        )
    rows, K = int(x.shape[0]), int(x.shape[1])
    if K % MXFP8_BLOCK != 0:
        raise ValueError(
            f"build_e8m0_block_scales: K={K} not a multiple of the MXFP8 block "
            f"size {MXFP8_BLOCK}; refusing to pad (RULE #1)."
        )
    if not x.is_cuda:
        raise RuntimeError(
            f"build_e8m0_block_scales: input is on {x.device}, not CUDA; the MXFP8 "
            f"scale producer requires a CUDA device tensor (no host bounce)."
        )

    xf = x.detach().to(torch.float32)
    if not torch.isfinite(xf).all():
        raise FloatingPointError(
            "build_e8m0_block_scales: input contains non-finite values; refuse to "
            "derive a degenerate E8M0 scale (RULE #1). Check the upstream tensor "
            "for NaN/Inf before MXFP8 quantization."
        )

    nblk = K // MXFP8_BLOCK
    blocks = xf.reshape(rows, nblk, MXFP8_BLOCK)
    block_amax = blocks.abs().amax(dim=2)  # (rows, nblk)

    # unbiased = ceil(log2(amax / fp8_max)); guard amax==0 -> minimum exponent.
    ratio = block_amax / _FP8_E4M3_MAX
    pos = ratio > 0
    unbiased = torch.empty_like(ratio)
    # ceil(log2(.)) via frexp would be cleaner but log2+ceil is portable here.
    unbiased[pos] = torch.ceil(torch.log2(ratio[pos]))
    # Empty/zero blocks -> the most-negative representable unbiased exponent so
    # the biased byte clamps to 0 (scale 2**-127). This is the E8M0 identity for
    # an all-zero block (any nonzero scale is fine since the payload is 0).
    unbiased[~pos] = float(-_E8M0_BIAS)

    biased = (unbiased + _E8M0_BIAS).to(torch.int32)
    biased = torch.clamp(biased, 0, _E8M0_MAX_BIASED)
    return biased.to(torch.uint8).contiguous()


def quantize_to_e4m3_blocked(x: "Any", e8m0_scales: "Any") -> "Any":
    """Quantize a ``(rows, K)`` fp tensor to e4m3 using the E8M0 block scales.

    ``e8m0_scales`` is the ``(rows, K // 32)`` uint8 output of
    :func:`build_e8m0_block_scales`. Returns a ``(rows, K)`` ``float8_e4m3fn``
    CUDA tensor: ``round_to_nearest_even(clamp(x / scale_block, +-fp8_max))``.

    RULE #1: dtype/availability failures RAISE; there is no silent precision
    downgrade.
    """

    import torch

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError(
            "quantize_to_e4m3_blocked: torch.float8_e4m3fn is not available in "
            "this PyTorch build."
        )
    if not (isinstance(x, torch.Tensor) and isinstance(e8m0_scales, torch.Tensor)):
        raise TypeError("quantize_to_e4m3_blocked: x and e8m0_scales must be torch.Tensor")
    rows, K = int(x.shape[0]), int(x.shape[1])
    if K % MXFP8_BLOCK != 0:
        raise ValueError(
            f"quantize_to_e4m3_blocked: K={K} not a multiple of {MXFP8_BLOCK} (RULE #1)."
        )
    nblk = K // MXFP8_BLOCK
    if tuple(e8m0_scales.shape) != (rows, nblk):
        raise ValueError(
            f"quantize_to_e4m3_blocked: e8m0_scales shape {tuple(e8m0_scales.shape)} "
            f"!= expected ({rows}, {nblk})"
        )

    xf = x.detach().to(torch.float32).reshape(rows, nblk, MXFP8_BLOCK)
    # scale = 2 ** (biased - 127)
    exp = e8m0_scales.to(torch.int32) - _E8M0_BIAS  # (rows, nblk)
    scale = torch.exp2(exp.to(torch.float32)).reshape(rows, nblk, 1)
    q = xf / scale
    q = torch.clamp(q, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return q.reshape(rows, K).to(fp8_dtype).contiguous()


# ---------------------------------------------------------------------------
# Device-pointer extraction + the GEMM drive
# ---------------------------------------------------------------------------


def _torch_device_ptr(t: "Any", *, name: str, want_e4m3: bool) -> int:
    """Return the raw CUdeviceptr of a contiguous CUDA torch tensor.

    RULE #1: raises if the tensor is not CUDA / not contiguous / wrong dtype —
    we never host-bounce and never silently reinterpret a wrong dtype's bits.
    Unlike lever-1, this avoids ``torch.from_dlpack`` entirely: it reads the
    existing tensor's ``data_ptr()`` directly (the operand is already a torch
    CUDA tensor produced by the quantizer above).
    """

    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"_torch_device_ptr[{name}]: expected torch.Tensor, got {type(t).__name__}")
    if not t.is_cuda:
        raise RuntimeError(
            f"_torch_device_ptr[{name}]: tensor on {t.device}, not CUDA; the CUTLASS "
            f"launcher needs a raw device pointer (no host bounce)."
        )
    if not t.is_contiguous():
        raise RuntimeError(
            f"_torch_device_ptr[{name}]: tensor is not contiguous; the TN MXFP8 "
            f"launcher indexes packed strides."
        )
    if want_e4m3:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None or t.dtype != fp8_dtype:
            raise TypeError(
                f"_torch_device_ptr[{name}]: expected float8_e4m3fn operand, got {t.dtype}"
            )
    return int(t.data_ptr())


def mxfp8_gemm(
    A_e4m3: "Any",
    SFA_e8m0: "Any",
    B_e4m3: "Any",
    SFB_e8m0: "Any",
    *,
    out: "Any",
    C: "Any" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    so_path: str | None = None,
    stream: int = 0,
) -> "Any":
    """Run ``out = alpha * (A_e4m3 @ B_e4m3^T) + beta * C`` via the CUTLASS kernel.

    Shapes (TN, all CUDA device tensors):
        ``A_e4m3``  : ``(M, K)`` float8_e4m3fn, row-major (K-major).
        ``B_e4m3``  : ``(N, K)`` float8_e4m3fn, row-major == col-major B (K-major).
        ``SFA_e8m0``: uint8 buffer of the size reported by :func:`sf_sizes` (SFA).
        ``SFB_e8m0``: uint8 buffer of the size reported by :func:`sf_sizes` (SFB).
        ``out``     : ``(M, N)`` bfloat16, row-major (owner output, required).
        ``C``       : optional ``(M, N)`` bfloat16 source (NULL when beta==0).

    RULE #1: any nonzero return from the C launcher RAISES with the launcher's
    own error string (failing site + cudaGetErrorString). No bf16/cuBLAS
    fallback path exists.
    """

    import torch

    if out is None:
        raise ValueError("mxfp8_gemm: owner output `out` is required (no allocation at the boundary).")

    M, K = int(A_e4m3.shape[0]), int(A_e4m3.shape[1])
    N, KB = int(B_e4m3.shape[0]), int(B_e4m3.shape[1])
    if K != KB:
        raise ValueError(f"mxfp8_gemm: K mismatch A K={K} vs B K={KB}")
    if tuple(out.shape) != (M, N):
        raise ValueError(f"mxfp8_gemm: out shape {tuple(out.shape)} != ({M}, {N})")
    if out.dtype != torch.bfloat16:
        raise TypeError(f"mxfp8_gemm: out must be bfloat16 (kernel ElementD), got {out.dtype}")
    if not out.is_cuda or not out.is_contiguous():
        raise RuntimeError("mxfp8_gemm: out must be a contiguous CUDA tensor.")

    # Validate the scale buffers against the kernel's atom layout BEFORE launch.
    exp_sfa, exp_sfb = sf_sizes(M, N, K, so_path=so_path)
    if int(SFA_e8m0.numel()) < exp_sfa:
        raise ValueError(
            f"mxfp8_gemm: SFA buffer too small ({int(SFA_e8m0.numel())} < {exp_sfa} bytes "
            f"required by the Sm1xxBlkScaledConfig SFA atom)."
        )
    if int(SFB_e8m0.numel()) < exp_sfb:
        raise ValueError(
            f"mxfp8_gemm: SFB buffer too small ({int(SFB_e8m0.numel())} < {exp_sfb} bytes "
            f"required by the Sm1xxBlkScaledConfig SFB atom)."
        )

    path = so_path or os.environ.get("CPPMEGA_CUTLASS_MXFP8_SO") or _DEFAULT_SO
    lib = _load_lib(path)

    a_ptr = _torch_device_ptr(A_e4m3, name="A", want_e4m3=True)
    b_ptr = _torch_device_ptr(B_e4m3, name="B", want_e4m3=True)
    sfa_ptr = _torch_device_ptr(SFA_e8m0, name="SFA", want_e4m3=False)
    sfb_ptr = _torch_device_ptr(SFB_e8m0, name="SFB", want_e4m3=False)
    d_ptr = _torch_device_ptr(out, name="D", want_e4m3=False)
    c_ptr = 0
    if C is not None:
        c_ptr = _torch_device_ptr(C, name="C", want_e4m3=False)

    rc = lib.cppmega_mxfp8_gemm_sm121(
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(sfa_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(sfb_ptr),
        ctypes.c_void_p(c_ptr),
        ctypes.c_void_p(d_ptr),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_float(float(alpha)),
        ctypes.c_float(float(beta)),
        ctypes.c_void_p(int(stream)),
    )
    if rc != 0:
        raise RuntimeError(
            f"cutlass_mxfp8_sm120.mxfp8_gemm: native MXFP8 launcher failed "
            f"(rc={rc}) for M={M} N={N} K={K}: {_last_error(lib)}. "
            f"RULE #1: no bf16/cuBLAS fallback — surface and fix the root cause."
        )
    return out


def mxfp8_gemm_from_hp(
    A_hp: "Any",
    B_hp: "Any",
    *,
    out: "Any" = None,
    so_path: str | None = None,
    stream: int = 0,
) -> "Any":
    """Convenience path: quantize fp16/bf16 ``A (M,K)`` and ``B (N,K)`` to MXFP8
    (e4m3 + E8M0 block-32 scales) and run the native GEMM. Returns bf16 ``(M,N)``.

    Used by the bench/parity probe. RULE #1: every step raises on failure; this
    is real MXFP8, never a bf16 shortcut.
    """

    import torch

    if A_hp.dim() != 2 or B_hp.dim() != 2:
        raise ValueError("mxfp8_gemm_from_hp: A and B must be 2D (TN: A=(M,K), B=(N,K)).")
    M, K = int(A_hp.shape[0]), int(A_hp.shape[1])
    N = int(B_hp.shape[0])

    sfa = build_e8m0_block_scales(A_hp)
    sfb = build_e8m0_block_scales(B_hp)
    a_q = quantize_to_e4m3_blocked(A_hp, sfa)
    b_q = quantize_to_e4m3_blocked(B_hp, sfb)

    # Lay the per-block scale bytes into the CUTLASS SFA/SFB atom buffers. The
    # kernel reports the exact buffer sizes; we copy the row-major (rows, nblk)
    # bytes into the leading region. The atom permutation is applied device-side
    # by CUTLASS from LayoutSFA/SFB — the contiguous ue8m0 byte stream is the
    # producer contract documented in build_e8m0_block_scales.
    exp_sfa, exp_sfb = sf_sizes(M, N, K, so_path=so_path)
    sfa_buf = torch.zeros(exp_sfa, dtype=torch.uint8, device=A_hp.device)
    sfb_buf = torch.zeros(exp_sfb, dtype=torch.uint8, device=B_hp.device)
    sfa_flat = sfa.reshape(-1)
    sfb_flat = sfb.reshape(-1)
    if sfa_flat.numel() > exp_sfa or sfb_flat.numel() > exp_sfb:
        raise ValueError(
            f"mxfp8_gemm_from_hp: produced scale stream larger than the CUTLASS "
            f"atom buffer (SFA {sfa_flat.numel()}>{exp_sfa} or SFB "
            f"{sfb_flat.numel()}>{exp_sfb}); atom-layout mismatch (RULE #1, do not "
            f"truncate)."
        )
    sfa_buf[: sfa_flat.numel()] = sfa_flat
    sfb_buf[: sfb_flat.numel()] = sfb_flat

    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A_hp.device)
    return mxfp8_gemm(
        a_q, sfa_buf, b_q, sfb_buf, out=out, so_path=so_path, stream=stream
    )


__all__ = [
    "CutlassMxfp8Status",
    "MXFP8_BLOCK",
    "build_e8m0_block_scales",
    "cutlass_mxfp8_status",
    "mxfp8_gemm",
    "mxfp8_gemm_from_hp",
    "quantize_to_e4m3_blocked",
    "sf_sizes",
]
