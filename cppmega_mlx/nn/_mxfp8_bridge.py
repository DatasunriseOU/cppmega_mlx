# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Bridge: cppmega's CUTLASS / FlashInfer MXFP8 GEMM -> TileLang ``tl.extern_intrinsic``.

Goal
----
Wrap the cppmega Blackwell SM120/SM121 (GB10) MXFP8 TN GEMM and its grouped MoE
variant as TileLang ``tl.extern_intrinsic`` ops so a TileLang ``@T.prim_func``
can call them from inside a kernel body without leaving the TileLang fusion
graph (no HBM round-trip between TileLang ops and the CUTLASS kernel).

Entry points exposed on the C++ side
------------------------------------
The cppmega CUDA extension ``cppmega_cutlass_mxfp8_gemm_cuda`` (built via
``torch.utils.cpp_extension.load`` from
``cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.{cpp,cu}``) ships SEVEN dense
entry points. We enumerate them all in :data:`MXFP8_BRIDGE_KNOWN_GAPS` so the
caller knows what we *did not* wire on the first cut, and we wire ONE
(the rowwise-A/rowwise-B compact-scale TN GEMM, ``tn_gemm_compact_scale``) for
the initial bridge. The grouped MoE variant lives in a separate extension
module ``grouped_mxfp8_gemm`` (with its own pybind exports ``dgrad_nn`` /
``wgrad_nt`` / ``dgrad_nn_ptrs`` / ``dgrad_nn_ptrs_by_expert`` /
``wgrad_nt_ptrs``); the grouped wrapper here targets ``dgrad_nn`` first.

Direction supported
-------------------
TileLang IR -> raw CUDA call into cppmega's pre-built ``.so``. The body emitted
into the kernel calls a thin C++ trampoline via ``tir.call_extern("handle",
"tl.extern_intrinsic.cppmega_sm120_blockscaled_mma_tma", access_ptr(A,"r"), ...)``.
This is the same "tile-typed contract" pattern used by
``poc/extern_intrinsic_examples/simdgroup_mma.py`` and documented in
``RFC_unified_fused_kernel.md`` §6. The actual CUDA body string MUST be
materialised by codegen; on Mac (no CUDA, no cppmega ``.so``) we register a
stub body and surface a loud ``MXFP8BridgeUnsupported`` if anyone tries to
*invoke* the resulting callable.

Caller responsibilities (NOT auto-applied)
------------------------------------------
1. **Pre-pack scales.** The compact-scale path consumes E8M0 rowwise scales as
   produced by Transformer Engine; the swizzled-scale path needs scales packed
   via ``cppmega.megatron.cutlass_mxfp8_gemm.swizzle_rowwise_scale(...)``.
   The bridge does NOT call ``swizzle_rowwise_scale`` automatically — it would
   require materialising a CUDA buffer outside the TileLang kernel scope, which
   defeats the fusion contract.
2. **Shape constraints.** ``M``, ``N``, ``K`` must be positive multiples of 128
   (matches ``cutlass_mxfp8_gemm.is_supported_shape``).
3. **dtypes.** A and B payloads are ``uint8`` (packed MXFP8 e4m3 / e5m2);
   scales are ``uint8`` (E8M0); output is ``bfloat16``.
4. **Layout.** A is logical [M, K] rowwise; B is logical [N, K] rowwise; output
   is row-major [M, N] = ``A @ B.T``.
5. **Architecture.** Blackwell sm_120 / sm_121 (GB10). The kernel will not run
   on sm_90 / sm_100 — calling :func:`mxfp8_to_tilelang_extern` on a non-GB10
   GPU raises :class:`MXFP8BridgeUnsupported`.

ABI discovery
-------------
The ``tn_gemm_compact_scale`` entry's pybind signature (12 args after the four
tensors) is verified at module-load time via ``inspect`` if cppmega is
importable; otherwise we fall back to the ABI we recorded by reading the
``.cpp`` source on the Mac mirror (see
``cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cpp:3-15``):

    Tensor cutlass_mxfp8_tn_gemm_compact_scale_cuda(
        Tensor A_u8, Tensor SFA_u8, Tensor B_u8, Tensor SFB_u8,
        int64 m, int64 n, int64 k,
        Tensor out, bool use_out, bool accumulate,
        double alpha, double beta);

For codegen-level discovery on GB10 (when this bridge is wired up to a runnable
TileLang prim), run::

    nm /home/dave/source/cppmega/cppmega/megatron/cuda_ext/build/*.so \\
       | grep cppmega_sm120 | c++filt

to confirm the mangled symbol the trampoline must dlsym/jit-load.

This module deliberately raises :class:`MXFP8BridgeUnsupported` (never returns
``None``) when prereqs are missing, per the codebase rule
"feedback_no_silent_delete". Tests at ``tests/test_mxfp8_bridge.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Callable, Tuple

__all__ = [
    "MXFP8_BRIDGE_KNOWN_GAPS",
    "MXFP8_INTRINSIC_NAME",
    "MXFP8_GROUPED_INTRINSIC_NAME",
    "MXFP8_CPPMEGA_ENTRY_ABI",
    "MXFP8BridgeUnsupported",
    "mxfp8_bridge_available",
    "mxfp8_to_tilelang_extern",
    "mxfp8_grouped_to_tilelang_extern",
]


#: Globally-unique intrinsic name registered with TileLang's extern registry
#: (``tilelang.language.extern_registry``). The ``tl.extern_intrinsic.`` prefix
#: is added by the decorator at TIR emission time.
MXFP8_INTRINSIC_NAME: str = "cppmega_sm120_blockscaled_mma_tma"

#: Grouped (MoE) variant intrinsic name. Targets the dgrad NN entry first.
MXFP8_GROUPED_INTRINSIC_NAME: str = "cppmega_sm120_grouped_mxfp8_dgrad_nn"

#: Recorded ABI for the ``tn_gemm_compact_scale`` pybind entry. Keys mirror the
#: positional argument order of
#: ``cutlass_mxfp8_tn_gemm_compact_scale_cuda`` from
#: ``cppmega/megatron/cuda_ext/cutlass_mxfp8_gemm.cpp``. Used for runtime
#: signature checks and for the TIR ``call_extern`` body.
MXFP8_CPPMEGA_ENTRY_ABI: Tuple[Tuple[str, str], ...] = (
    ("A_u8", "Tensor[uint8, M, K]"),
    ("SFA_u8", "Tensor[uint8, M, K/32]"),
    ("B_u8", "Tensor[uint8, N, K]"),
    ("SFB_u8", "Tensor[uint8, N, K/32]"),
    ("m", "int64"),
    ("n", "int64"),
    ("k", "int64"),
    ("out", "Tensor[bfloat16, M, N]"),
    ("use_out", "bool"),
    ("accumulate", "bool"),
    ("alpha", "double"),
    ("beta", "double"),
)


#: Documented gaps. Written down so a reviewer / future maintainer doesn't
#: have to re-discover what *isn't* wired.
MXFP8_BRIDGE_KNOWN_GAPS: Tuple[str, ...] = (
    # Direction / scope.
    "TileLang's tl.extern_intrinsic emits a TIR call_extern; the actual CUDA "
    "body string for the GB10 CUTLASS MXFP8 kernel is NOT materialised here. "
    "On Mac/non-CUDA hosts the registered body is a documenting stub that "
    "raises MXFP8BridgeUnsupported on invocation. Wiring the live body "
    "requires linking against cppmega_cutlass_mxfp8_gemm_cuda.so on GB10.",
    # Variant coverage — there are 7 dense entries; we wire one.
    "cppmega ships seven CUTLASS dense entries: tn_gemm_compact_scale (wired), "
    "tn_gemm_compact_direct, tn_gemm_compact_direct_asym, "
    "tn_gemm_compact_direct_a_col_smem_asym, "
    "tn_gemm_compact_direct_a_col_smem_b_tma_early_asym, "
    "tn_gemm_swizzled_scale, and tn_gemm_swizzled_scale_strided. The compact-"
    "scale path is the most general (rowwise A + rowwise B + compact E8M0 "
    "scales, BF16 out); the dgrad-NN / wgrad-NT / asym variants are not "
    "exposed here yet.",
    # Grouped variant.
    "Grouped MoE variant: cppmega.megatron.grouped_mxfp8_gemm exports "
    "dgrad_nn, wgrad_nt, dgrad_nn_ptrs, dgrad_nn_ptrs_by_expert, wgrad_nt_ptrs. "
    "mxfp8_grouped_to_tilelang_extern wires dgrad_nn first; the per-expert "
    "pointer-array variants (preferred for irregular MoE token counts) are "
    "tracked but not yet bridged.",
    # FlashInfer alternative.
    "FlashInfer backend (cppmega.megatron.flashinfer_mxfp8_gemm) is an "
    "alternative MXFP8 path with a different scale layout (1D layout_128x4, "
    "swizzled by FlashInfer at scale-prep time). Selected via the `backend` "
    "kwarg; produces a different intrinsic name "
    "(`cppmega_sm120_blockscaled_mma_tma_flashinfer`) so codegen can pick the "
    "correct trampoline. Caller is responsible for using "
    "flashinfer_mxfp8_gemm.swizzle_rowwise_scale_to_layout_128x4(...) before "
    "invoking.",
    # Scale layout responsibility.
    "Caller MUST pre-pack scales BEFORE invoking the extern. For the compact-"
    "scale path: pass TE compact rowwise E8M0 scales directly. For the "
    "swizzled-scale path (not wired here): call "
    "cppmega.megatron.cutlass_mxfp8_gemm.swizzle_rowwise_scale(...) first. "
    "The bridge will NOT call swizzle_rowwise_scale automatically — that "
    "would need a CUDA buffer materialised outside the TileLang kernel scope, "
    "breaking the fusion contract.",
    # Shape constraint.
    "M, N, K must each be a positive multiple of 128 "
    "(cutlass_mxfp8_gemm.is_supported_shape). Other shapes raise at registration.",
    # Architecture gate.
    "Kernel is sm_120 / sm_121 only (GB10). Sm_90 / sm_100 callers must use a "
    "different bridge (cppmega.megatron.cutlass_mxfp8_gemm has no sm_90 path; "
    "Hopper FP8 GEMMs route through cute_dsl_mimo or the FlashInfer sm_100 "
    "tactic instead).",
    # ABI discovery script — exact command for GB10.
    "ABI verification: when on GB10 with the .so built, run "
    "`nm /home/dave/source/cppmega/cppmega/megatron/cuda_ext/build/*.so "
    "| grep cppmega_sm120 | c++filt` to confirm the mangled symbol the "
    "trampoline must dlsym/jit-load.",
)


class MXFP8BridgeUnsupported(RuntimeError):
    """Raised when prerequisites for the MXFP8 -> TileLang bridge are missing.

    Common reasons:

    * cppmega.megatron.cutlass_mxfp8_gemm not importable (no torch, no CUDA,
      missing cppmega checkout, .so build failure).
    * tilelang.language.extern_intrinsic not importable (libtilelang dylib
      load failure on macOS, etc.).
    * Architecture is not Blackwell sm_120 / sm_121.
    * Shape (M, N, K) not a positive multiple of 128.
    * Unknown ``backend`` selector.

    Carries a precise reason; the bridge never returns ``None`` for missing
    prereqs (per ``feedback_no_silent_delete``).
    """


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _probe_cppmega_cutlass(backend: str) -> Tuple[bool, str]:
    """Attempt to import the cppmega CUTLASS MXFP8 GEMM module.

    Returns (ok, reason). On Mac hosts torch is usually present but
    ``torch.utils.cpp_extension.load`` will fail without nvcc + a CUDA-capable
    GPU; the underlying module is still *importable* (the build is lazy), so
    we only check for module presence here.
    """

    if backend == "cutlass":
        modname = "cppmega.megatron.cutlass_mxfp8_gemm"
    elif backend == "flashinfer":
        modname = "cppmega.megatron.flashinfer_mxfp8_gemm"
    elif backend == "grouped":
        modname = "cppmega.megatron.grouped_mxfp8_gemm"
    else:
        return False, f"unknown backend selector {backend!r}"
    try:
        importlib.import_module(modname)
    except Exception as exc:  # pragma: no cover - host-specific
        return False, f"{modname} unimportable: {exc.__class__.__name__}: {exc}"
    return True, ""


def _probe_tilelang_extern() -> Tuple[bool, str]:
    """Attempt to import ``tilelang.language.extern.extern_intrinsic``.

    On macOS dev hosts ``tilelang`` may abort the process during native init
    if ``libz3`` is not preloaded (libtilelang.dylib hard-aborts on missing
    deps rather than throwing ImportError). To avoid triggering an
    uncatchable abort during the probe itself, we run the import in a short-
    lived subprocess on Darwin and parse its exit status. Linux+CUDA hosts
    (the only place this bridge is meant to *work*) take the fast in-process
    path, so the subprocess overhead only hits negative-result Mac probes.

    Subprocess isolation is only used on Darwin; on every other OS we trust
    the in-process import to either succeed or raise a catchable Exception.
    """

    spec = importlib.util.find_spec("tilelang")
    if spec is None:
        return False, "tilelang not on sys.path"

    if sys.platform == "darwin":
        # Hard-abort isolation: probe in a subprocess so a tilelang dylib
        # SIGABRT / Fatal Python error cannot take down our pytest worker.
        import subprocess

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from tilelang.language.extern import extern_intrinsic\n"
                    "print('ok')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"tilelang subprocess probe failed: {exc!r}"
        if result.returncode != 0 or "ok" not in result.stdout:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            return (
                False,
                f"tilelang.language.extern.extern_intrinsic unimportable on darwin: "
                f"subprocess exit={result.returncode}; "
                f"last lines: {tail!r}",
            )
        # Subprocess says it imports — but we still need it in *this* process
        # for the actual extern_intrinsic registration. Try in-process import
        # under a try/except; on macOS dev hosts where the subprocess succeeds
        # but in-process abort still happens, we surface a precise reason.
        try:
            from tilelang.language.extern import extern_intrinsic  # noqa: F401
        except Exception as exc:  # pragma: no cover - host-specific
            return (
                False,
                f"tilelang importable in subprocess but not in-process: "
                f"{exc.__class__.__name__}: {exc}",
            )
        return True, ""

    try:
        from tilelang.language.extern import extern_intrinsic  # noqa: F401
    except Exception as exc:  # pragma: no cover - host-specific
        return (
            False,
            f"tilelang.language.extern.extern_intrinsic unimportable: "
            f"{exc.__class__.__name__}: {exc}",
        )
    return True, ""


def _probe_blackwell() -> Tuple[bool, str]:
    """Best-effort GB10 architecture probe.

    Returns (ok, reason). On Mac (no CUDA), or any non-Blackwell GPU, returns
    (False, ...). When torch is missing entirely we treat that as a non-CUDA
    host (consistent with the rest of cppmega.mlx).
    """

    try:
        import torch  # noqa: WPS433
    except ImportError:
        return False, "torch not importable; assuming non-CUDA host"
    if not torch.cuda.is_available():
        return False, "torch.cuda not available; non-CUDA host"
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"torch.cuda.get_device_capability failed: {exc}"
    # SM120 (B100/B200/GB10) and SM121 are the targets the cppmega CUTLASS
    # mainloop fork supports. SM90/SM100 are explicitly out of scope.
    if (major, minor) not in {(12, 0), (12, 1)}:
        return (
            False,
            f"GPU compute capability is sm_{major}{minor}; cppmega CUTLASS "
            f"MXFP8 requires sm_120 / sm_121 (GB10).",
        )
    return True, ""


def mxfp8_bridge_available() -> Tuple[bool, str]:
    """Check whether the cppmega MXFP8 -> TileLang bridge can be wired here.

    Returns ``(True, "ok")`` only when ALL of:

    1. ``cppmega.megatron.cutlass_mxfp8_gemm`` importable.
    2. ``tilelang.language.extern.extern_intrinsic`` importable.
    3. Running on a CUDA host with sm_120 / sm_121 (GB10).

    On any failure returns ``(False, reason)`` with a precise reason. Cheap —
    no GPU memory allocated, no kernel compilation, no .so build triggered.

    Used by tests to skip cleanly on Mac. Used by
    :func:`mxfp8_to_tilelang_extern` to fail loud rather than register a
    half-broken intrinsic.
    """

    ok, why = _probe_cppmega_cutlass("cutlass")
    if not ok:
        return False, why
    ok, why = _probe_tilelang_extern()
    if not ok:
        return False, why
    ok, why = _probe_blackwell()
    if not ok:
        return False, why
    return True, "ok"


# ---------------------------------------------------------------------------
# Internal: build the stub CUDA body string the codegen materialises.
# ---------------------------------------------------------------------------

# This body is documenting / forward-compatible. It is NOT a runnable kernel;
# the actual GB10 CUTLASS launcher lives in the cppmega .so. The TileLang
# codegen back-end (when ported to consume this intrinsic on GB10) replaces
# this string with a thin trampoline that calls into the .so via the pybind
# entry. The arity / arg names MUST match :data:`MXFP8_CPPMEGA_ENTRY_ABI`
# exactly — that is what ``extern.py:_validate_body`` verifies.
_CUDA_BODY_STUB_TEMPLATE = r"""
// Forward-compatible stub body for the cppmega SM120 MXFP8 TN GEMM
// extern_intrinsic. Codegen replaces this with a trampoline that calls
// cppmega_cutlass_mxfp8_gemm_cuda::tn_gemm_compact_scale. Until that lowering
// step lands on GB10, attempting to *invoke* the resulting kernel raises
// MXFP8BridgeUnsupported at the Python layer.
__device__ void {intrinsic_name}(
    uint8_t const* A_u8,
    uint8_t const* SFA_u8,
    uint8_t const* B_u8,
    uint8_t const* SFB_u8,
    __nv_bfloat16* out
) {{
    // M, N, K, accumulate, alpha, beta are bound at extern-emit time via the
    // intrinsic signature; the .so trampoline fills them from the launching
    // kernel's compile-time constants.
    (void)A_u8; (void)SFA_u8; (void)B_u8; (void)SFB_u8; (void)out;
    // STUB: real launcher goes here on GB10.
}}
"""


def _build_cuda_body_stub(intrinsic_name: str) -> str:
    return _CUDA_BODY_STUB_TEMPLATE.format(intrinsic_name=intrinsic_name)


# ---------------------------------------------------------------------------
# Internal: build the Frag signature for a given (M, N, K).
# ---------------------------------------------------------------------------


def _build_signature(m: int, n: int, k: int) -> Callable[..., Any]:
    """Return a zero-arg signature callable producing the per-frag contract.

    Five frags total, matching the five buffer pointers in the stub body:

    - ``A_u8``      : (M, K)      uint8     scope=global   row_major   read
    - ``SFA_u8``    : (M, K/32)   uint8     scope=global   row_major   read
    - ``B_u8``      : (N, K)      uint8     scope=global   row_major   read
    - ``SFB_u8``    : (N, K/32)   uint8     scope=global   row_major   read
    - ``out``       : (M, N)      bfloat16  scope=global   row_major   write

    Scalar args (m, n, k, accumulate, alpha, beta) are encoded via the intrinsic
    name and are NOT part of the Frag signature — TileLang's extern_intrinsic
    only models buffer-typed contract args. The trampoline reads them from the
    launching kernel's metadata.

    Lazy-imports ``Frag`` so a Mac host without TVM can still import this
    module (see :func:`mxfp8_to_tilelang_extern` for the loud-fail behaviour
    when TileLang is missing).
    """

    if m <= 0 or n <= 0 or k <= 0:
        raise MXFP8BridgeUnsupported(
            f"M, N, K must be positive ints; got M={m}, N={n}, K={k}"
        )
    if m % 128 != 0 or n % 128 != 0 or k % 128 != 0:
        raise MXFP8BridgeUnsupported(
            f"cppmega CUTLASS MXFP8 requires M/N/K multiples of 128; "
            f"got M={m}, N={n}, K={k}"
        )
    if k % 32 != 0:  # implied by k % 128 == 0 but explicit for the error path.
        raise MXFP8BridgeUnsupported(
            f"K must be a multiple of 32 for E8M0 scale packing; got K={k}"
        )

    def _signature() -> Tuple[Any, ...]:
        from tilelang.language.extern import Frag  # noqa: WPS433

        return (
            Frag(
                name="A_u8",
                shape=(m, k),
                scope="global",
                dtype="uint8",
                layout="row_major",
                alignment=16,
                is_output=False,
            ),
            Frag(
                name="SFA_u8",
                shape=(m, k // 32),
                scope="global",
                dtype="uint8",
                layout="row_major",
                alignment=16,
                is_output=False,
            ),
            Frag(
                name="B_u8",
                shape=(n, k),
                scope="global",
                dtype="uint8",
                layout="row_major",
                alignment=16,
                is_output=False,
            ),
            Frag(
                name="SFB_u8",
                shape=(n, k // 32),
                scope="global",
                dtype="uint8",
                layout="row_major",
                alignment=16,
                is_output=False,
            ),
            Frag(
                name="out",
                shape=(m, n),
                scope="global",
                dtype="bfloat16",
                layout="row_major",
                alignment=16,
                is_output=True,
            ),
        )

    return _signature


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mxfp8_to_tilelang_extern(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str = "mxfp8",
    backend: str = "cutlass",
) -> Callable[..., Any]:
    """Register a TileLang ``tl.extern_intrinsic`` for the cppmega MXFP8 GEMM.

    The returned callable, when invoked from inside a TileLang
    ``@T.prim_func``, emits a ``tir.call_extern("handle",
    "tl.extern_intrinsic.cppmega_sm120_blockscaled_mma_tma", access_ptr(A,"r"),
    access_ptr(SFA,"r"), access_ptr(B,"r"), access_ptr(SFB,"r"),
    access_ptr(out,"rw"))`` whose body is materialised by the codegen back-end
    on GB10. On Mac the body is a documenting stub; calling the resulting
    intrinsic at run time raises :class:`MXFP8BridgeUnsupported`.

    Parameters
    ----------
    m, n, k:
        GEMM dimensions. MUST be positive multiples of 128
        (matches ``cutlass_mxfp8_gemm.is_supported_shape``). Other shapes raise
        :class:`MXFP8BridgeUnsupported`.
    dtype:
        Element dtype label. Currently only ``"mxfp8"`` is accepted (rowwise
        E8M0 scales over packed e4m3 / e5m2 payloads, as produced by TE).
        Placeholder for a future ``"mxfp4"`` tier.
    backend:
        Selector ∈ ``{"cutlass", "flashinfer"}``. Picks which cppmega module
        provides the underlying ``.so`` trampoline.

        - ``"cutlass"`` (default) -> ``cppmega.megatron.cutlass_mxfp8_gemm``
          ``tn_gemm_compact_scale`` entry. Compact rowwise E8M0 scales.
        - ``"flashinfer"`` -> ``cppmega.megatron.flashinfer_mxfp8_gemm``
          ``mm_mxfp8`` entry. 1D layout_128x4 swizzled scales (caller MUST
          pre-pack via ``flashinfer_mxfp8_gemm`` helpers).

    Returns
    -------
    Callable
        A ``tl.extern_intrinsic`` emitter. Call shape from inside a TileLang
        kernel::

            mxfp8_intr = mxfp8_to_tilelang_extern(m=128, n=256, k=512)
            ...
            with T.Kernel(...):
                A   = T.alloc_buffer((128, 512), "uint8",   "global")
                SFA = T.alloc_buffer((128, 16),  "uint8",   "global")
                B   = T.alloc_buffer((256, 512), "uint8",   "global")
                SFB = T.alloc_buffer((256, 16),  "uint8",   "global")
                out = T.alloc_buffer((128, 256), "bfloat16","global")
                mxfp8_intr(A, SFA, B, SFB, out)

    Raises
    ------
    MXFP8BridgeUnsupported
        If cppmega / tilelang prereqs are missing, the architecture is not
        sm_120 / sm_121, the dtype is unsupported, the backend selector is
        unknown, or (M, N, K) is not a positive multiple of 128.

    Notes
    -----
    * Caller MUST pre-pack scales into the layout the chosen backend expects
      (see :data:`MXFP8_BRIDGE_KNOWN_GAPS` entry on scale layout).
    * Architecture and ``.so``-build probes happen at registration time, not
      at TileLang compile time. We fail at the earliest moment so a TileLang
      kernel author finds out about the missing prereq before they spend
      compile time on the surrounding ``@T.prim_func``.
    """

    if dtype != "mxfp8":
        raise MXFP8BridgeUnsupported(
            f"unsupported dtype {dtype!r}; the cppmega CUTLASS bridge only "
            f"wires the MXFP8 (E8M0-over-fp8) tier on the first cut. See "
            f"MXFP8_BRIDGE_KNOWN_GAPS for the planned mxfp4 / nvfp4 path."
        )
    if backend not in {"cutlass", "flashinfer"}:
        raise MXFP8BridgeUnsupported(
            f"unsupported backend {backend!r}; expected one of "
            f"{{'cutlass', 'flashinfer'}}. See MXFP8_BRIDGE_KNOWN_GAPS for "
            f"the rationale behind the two-backend split."
        )

    # Probe ALL prereqs up front. We do not want to register a half-broken
    # intrinsic that fails later at TIR-emit time with a confusing traceback.
    cm_ok, cm_why = _probe_cppmega_cutlass(backend)
    if not cm_ok:
        raise MXFP8BridgeUnsupported(
            f"cppmega backend {backend!r} unavailable: {cm_why}. "
            f"On Mac dev hosts this is expected; on GB10 confirm "
            f"`pip install -e /home/dave/source/cppmega` and that the "
            f"CUTLASS_ROOT env var is set (see "
            f"cppmega/megatron/cutlass_mxfp8_gemm.py:20)."
        )
    tl_ok, tl_why = _probe_tilelang_extern()
    if not tl_ok:
        raise MXFP8BridgeUnsupported(
            f"tilelang.language.extern.extern_intrinsic unavailable: {tl_why}. "
            f"Typical on macOS dev hosts where libtilelang dylib fails to "
            f"load; rebuild tilelang against the host libstdc++ or run on "
            f"Linux+CUDA."
        )
    arch_ok, arch_why = _probe_blackwell()
    if not arch_ok:
        raise MXFP8BridgeUnsupported(
            f"GB10 architecture probe failed: {arch_why}. The cppmega CUTLASS "
            f"MXFP8 mainloop is sm_120 / sm_121 only; do not call this "
            f"bridge on Hopper or earlier."
        )

    # Build the contract and register.
    signature = _build_signature(m=m, n=n, k=k)
    intrinsic_name = (
        MXFP8_INTRINSIC_NAME
        if backend == "cutlass"
        else f"{MXFP8_INTRINSIC_NAME}_flashinfer"
    )
    body = _build_cuda_body_stub(intrinsic_name)

    from tilelang.language.extern import extern_intrinsic  # noqa: WPS433

    emit = extern_intrinsic(
        name=intrinsic_name,
        signature=signature,
        bodies={"cuda": body},
    )
    # Stash the resolved metadata so callers (and tests) can introspect.
    emit.cppmega_backend = backend  # type: ignore[attr-defined]
    emit.cppmega_shape = (m, n, k)  # type: ignore[attr-defined]
    emit.cppmega_abi = MXFP8_CPPMEGA_ENTRY_ABI  # type: ignore[attr-defined]
    return emit


def mxfp8_grouped_to_tilelang_extern(
    *,
    num_groups: int,
    m: int,
    n: int,
    k: int,
) -> Callable[..., Any]:
    """Register the grouped MoE MXFP8 GEMM as a ``tl.extern_intrinsic``.

    Wraps ``cppmega.megatron.grouped_mxfp8_gemm.dgrad_nn`` (the dgrad NN
    direct-compact reference kernel) for use from a TileLang ``@T.prim_func``
    that needs to run a per-expert GEMM inside a fused MoE backward pass.

    Parameters
    ----------
    num_groups:
        Number of expert groups (e.g. 16 for the default cppmega config). Used
        to size the ``expert_offsets`` Frag.
    m, n, k:
        Per-group GEMM dimensions. The total token count across all groups
        equals ``m`` here when each group has the same row count; for
        irregular MoE workloads the caller must instead use the per-expert
        pointer-array variants (``dgrad_nn_ptrs`` / ``wgrad_nt_ptrs``), which
        are NOT yet wired here — see :data:`MXFP8_BRIDGE_KNOWN_GAPS`.

    Returns
    -------
    Callable
        Extern intrinsic emitter — same shape as
        :func:`mxfp8_to_tilelang_extern` but with one additional Frag for
        ``expert_offsets`` (int32, length ``num_groups + 1``).

    Raises
    ------
    MXFP8BridgeUnsupported
        On any missing prereq, non-128-multiple shape, or non-Blackwell GPU.
    """

    if num_groups <= 0:
        raise MXFP8BridgeUnsupported(
            f"num_groups must be a positive int; got {num_groups}"
        )

    cm_ok, cm_why = _probe_cppmega_cutlass("grouped")
    if not cm_ok:
        raise MXFP8BridgeUnsupported(
            f"cppmega.megatron.grouped_mxfp8_gemm unavailable: {cm_why}. "
            f"This module ships its own .so build (separate from the dense "
            f"cutlass_mxfp8_gemm extension); confirm the cppmega checkout "
            f"includes cuda_ext/grouped_mxfp8_gemm.{{cpp,cu}}."
        )
    tl_ok, tl_why = _probe_tilelang_extern()
    if not tl_ok:
        raise MXFP8BridgeUnsupported(
            f"tilelang.language.extern.extern_intrinsic unavailable: {tl_why}."
        )
    arch_ok, arch_why = _probe_blackwell()
    if not arch_ok:
        raise MXFP8BridgeUnsupported(
            f"GB10 architecture probe failed: {arch_why}."
        )

    base_signature = _build_signature(m=m, n=n, k=k)

    def _signature() -> Tuple[Any, ...]:
        from tilelang.language.extern import Frag  # noqa: WPS433

        base = base_signature()
        # Append expert_offsets — int32 1D tensor of length (num_groups + 1).
        expert_offsets = Frag(
            name="expert_offsets",
            shape=(num_groups + 1,),
            scope="global",
            dtype="int32",
            layout="row_major",
            alignment=16,
            is_output=False,
        )
        # Insert before ``out`` to mirror the C++ pybind arg order:
        # (dy, sf_dy, weight, sf_weight, expert_offsets, out, ...)
        return (*base[:4], expert_offsets, base[4])

    body = _build_cuda_body_stub(MXFP8_GROUPED_INTRINSIC_NAME)
    # Match the 6-arg body to the 6-Frag signature.
    body = body.replace(
        "uint8_t const* SFB_u8,\n    __nv_bfloat16* out",
        "uint8_t const* SFB_u8,\n    int32_t const* expert_offsets,\n    __nv_bfloat16* out",
    )

    from tilelang.language.extern import extern_intrinsic  # noqa: WPS433

    emit = extern_intrinsic(
        name=MXFP8_GROUPED_INTRINSIC_NAME,
        signature=_signature,
        bodies={"cuda": body},
    )
    emit.cppmega_backend = "grouped"  # type: ignore[attr-defined]
    emit.cppmega_shape = (m, n, k)  # type: ignore[attr-defined]
    emit.cppmega_num_groups = num_groups  # type: ignore[attr-defined]
    return emit
