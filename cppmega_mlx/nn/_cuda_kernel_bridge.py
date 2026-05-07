# pyright: reportMissingImports=false
"""Arch-aware dispatcher for CUDA kernel bridges.

Routes by ``torch.cuda.get_device_capability()`` to the right per-arch
bridge (Triton, CuTe DSL, MXFP8). Mac/non-CUDA hosts get a clean
:class:`CUDABridgeUnsupported` error instead of import-time failures.

Public API
----------
* :func:`detect_cuda_arch` -> ``"sm_89" / "sm_90a" / "sm_100" / "sm_120" / "sm_121" / "cpu"``
* :func:`dispatch_kernel_bridge` -- ``kind in {"triton", "cute_dsl", "mxfp8", "mxfp8_grouped"}``;
  kwargs forwarded to the chosen sub-bridge.
* :data:`ARCH_BRIDGE_TABLE` -- mapping ``arch -> frozenset[bridge_kind]`` documenting
  the support matrix exhaustively.
* :class:`CUDABridgeUnsupported` -- raised for any arch/kind combo that has no
  supported route. Per the codebase ``feedback_no_silent_delete`` rule, every
  unsupported combination raises *loudly* with a precise reason; we never
  silently return ``None``.

Support matrix (verified 2026-05-07)
------------------------------------

==================  =======  ========  =======  =============
Arch                Triton   CuteDSL   MXFP8    MXFP8 grouped
==================  =======  ========  =======  =============
sm_89  (Ada)        yes      no        no       no
sm_90/sm_90a (H*)   yes      partial   no       no
sm_100 (BlackwellDC)yes      no        yes      yes
sm_120 (BlackwellC) yes      no        yes      yes
sm_121 (GB10)       yes      no        yes      yes
cpu / no CUDA       no       no        no       no
==================  =======  ========  =======  =============

* sm_90 CuTe-DSL is *partial*: only the supported direction
  (``_cute_bridge.compile_prim_to_cutedsl``) routes here; the reverse
  direction (external ``@cute.kernel`` -> TileLang IR import) is loud-fail
  via ``CuteBridgeUnsupported`` from ``_cute_bridge`` itself.
* Unknown SM (e.g. future ``sm_130``) emits a ``RuntimeWarning`` and
  *falls back to sm_120 dispatch*. This keeps newer client GPUs usable for
  Triton/MXFP8 paths until we cut a release with explicit support.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, FrozenSet, Mapping

__all__ = [
    "ARCH_BRIDGE_TABLE",
    "CUDABridgeUnsupported",
    "FALLBACK_ARCH",
    "KNOWN_ARCHES",
    "SUPPORTED_KINDS",
    "detect_cuda_arch",
    "dispatch_kernel_bridge",
]


SUPPORTED_KINDS: FrozenSet[str] = frozenset(
    {"triton", "cute_dsl", "mxfp8", "mxfp8_grouped"}
)


# Exhaustive arch -> set-of-supported-kinds map. Keep this in lockstep with
# the docstring table above. ``"cpu"`` is intentionally an empty set so the
# "no CUDA device" branch hits the standard not-in-set error path.
ARCH_BRIDGE_TABLE: Mapping[str, FrozenSet[str]] = {
    "sm_89": frozenset({"triton"}),
    "sm_90": frozenset({"triton", "cute_dsl"}),
    "sm_90a": frozenset({"triton", "cute_dsl"}),
    "sm_100": frozenset({"triton", "mxfp8", "mxfp8_grouped"}),
    "sm_120": frozenset({"triton", "mxfp8", "mxfp8_grouped"}),
    "sm_121": frozenset({"triton", "mxfp8", "mxfp8_grouped"}),
    "cpu": frozenset(),
}

KNOWN_ARCHES: FrozenSet[str] = frozenset(ARCH_BRIDGE_TABLE)

#: Arch used when ``detect_cuda_arch()`` returns something we don't recognise
#: (e.g. a future ``sm_130``). Chosen as sm_120 because client-Blackwell is
#: the most likely forward-compatible target for new consumer cards.
FALLBACK_ARCH = "sm_120"


class CUDABridgeUnsupported(RuntimeError):
    """Raised when an (arch, kind) combo has no supported route.

    Carries a precise reason string mentioning both the detected arch and
    the requested bridge kind so the failure is debuggable from a single
    log line. Never raised silently -- per the codebase
    ``feedback_no_silent_delete`` rule, callers should treat this as a
    hard failure rather than an "optional path missing".
    """


@functools.lru_cache(maxsize=1)
def detect_cuda_arch() -> str:
    """Detect the current process's primary CUDA arch tag.

    Returns one of :data:`KNOWN_ARCHES` -- including ``"cpu"`` when no
    CUDA device is visible. The result is cached for the lifetime of the
    process via ``functools.lru_cache``; tests that monkeypatch the
    detection should patch the *symbol* (e.g.
    ``monkeypatch.setattr(_cuda_kernel_bridge, 'detect_cuda_arch',
    lambda: 'sm_130')``) rather than relying on the cache being cleared.

    Hopper sub-variant note: ``torch.cuda.get_device_capability()`` cannot
    distinguish sm_90 from sm_90a (the latter being the arch-flag form
    used by H100/H200 for WGMMA). We default capability ``9.0`` to
    ``sm_90a`` because every shipping Hopper SKU we care about uses the
    'a' variant for kernel codegen.
    """

    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        # Any failure -- torch missing, driver mismatch, MPS-only host --
        # collapses to the "no CUDA" branch.
        return "cpu"

    arch = f"sm_{major}{minor}"
    if arch == "sm_90":
        return "sm_90a"
    return arch


def _resolve_arch(arch: str) -> str:
    """Map a possibly-unknown arch to a known one, warning on fallback."""

    if arch in ARCH_BRIDGE_TABLE:
        return arch
    warnings.warn(
        f"unrecognized arch {arch!r}; defaulting to {FALLBACK_ARCH} dispatch",
        RuntimeWarning,
        stacklevel=3,
    )
    return FALLBACK_ARCH


def _no_cuda_error(kind: str) -> CUDABridgeUnsupported:
    return CUDABridgeUnsupported(
        f"cannot dispatch CUDA bridge kind={kind!r}: no CUDA device "
        f"detected on this host (detect_cuda_arch() == 'cpu'). On Mac/MLX "
        f"hosts use the MLX/Metal kernels in cppmega_mlx instead."
    )


def _unsupported_combo_error(arch: str, kind: str) -> CUDABridgeUnsupported:
    supported = sorted(ARCH_BRIDGE_TABLE.get(arch, frozenset()))
    return CUDABridgeUnsupported(
        f"CUDA bridge kind={kind!r} is not supported on arch {arch!r}; "
        f"supported kinds for this arch are {supported!r}. See "
        f"cppmega_mlx.nn._cuda_kernel_bridge.ARCH_BRIDGE_TABLE for the full "
        f"matrix."
    )


def dispatch_kernel_bridge(kind: str, /, **kwargs: Any) -> Any:
    """Dispatch ``kind`` to the appropriate per-arch CUDA bridge.

    ``kind`` must be one of :data:`SUPPORTED_KINDS`. ``kwargs`` are
    forwarded verbatim to the selected sub-bridge entry point:

    * ``"triton"`` -> :func:`cppmega_mlx.nn._triton_bridge.triton_to_tilelang_prim`
    * ``"cute_dsl"`` -> :func:`cppmega_mlx.nn._cute_bridge.compile_prim_to_cutedsl`
      (only the *supported* direction; reverse direction must be obtained
      by calling ``_cute_bridge.cute_dsl_to_tilelang_prim`` directly, which
      raises :class:`~cppmega_mlx.nn._cute_bridge.CuteBridgeUnsupported`).
    * ``"mxfp8"`` -> ``_mxfp8_bridge.mxfp8_to_tilelang_extern`` (lazy import;
      file may not exist on every checkout -- raised as
      :class:`CUDABridgeUnsupported` with a precise reason if missing).
    * ``"mxfp8_grouped"`` -> ``_mxfp8_bridge.mxfp8_grouped_to_tilelang_extern``.

    Raises
    ------
    CUDABridgeUnsupported
        If ``kind`` is unknown, the host has no CUDA, the (arch, kind) pair
        is not in :data:`ARCH_BRIDGE_TABLE`, or the lazy-imported MXFP8
        bridge module is unavailable.
    """

    if kind not in SUPPORTED_KINDS:
        raise CUDABridgeUnsupported(
            f"unknown CUDA bridge kind={kind!r}; expected one of "
            f"{sorted(SUPPORTED_KINDS)!r}"
        )

    raw_arch = detect_cuda_arch()
    if raw_arch == "cpu":
        raise _no_cuda_error(kind)

    arch = _resolve_arch(raw_arch)
    supported = ARCH_BRIDGE_TABLE[arch]
    if kind not in supported:
        raise _unsupported_combo_error(arch, kind)

    if kind == "triton":
        # Lazy import: triton frontend is heavy and pulls in poc.triton_frontend
        # via sys.path mutation. We only want that side-effect on demand.
        from cppmega_mlx.nn._triton_bridge import triton_to_tilelang_prim

        return triton_to_tilelang_prim(**kwargs)

    if kind == "cute_dsl":
        from cppmega_mlx.nn._cute_bridge import compile_prim_to_cutedsl

        return compile_prim_to_cutedsl(**kwargs)

    # MXFP8 paths -- lazy import because the bridge module may be authored
    # in parallel and not yet present in this checkout. Per the
    # ``feedback_verify_review_agent_claims`` rule we don't pretend symbols
    # exist that we haven't grepped; instead we surface a precise error.
    if kind in {"mxfp8", "mxfp8_grouped"}:
        try:
            from cppmega_mlx.nn import _mxfp8_bridge  # type: ignore[attr-defined]
        except ImportError as exc:
            raise CUDABridgeUnsupported(
                f"MXFP8 bridge ({kind!r}) requested on arch {arch!r} but "
                f"cppmega_mlx.nn._mxfp8_bridge is not importable: {exc}. "
                f"This bridge is authored separately; ensure the module is "
                f"present before dispatching MXFP8 kinds."
            ) from exc

        entry_name = (
            "mxfp8_to_tilelang_extern"
            if kind == "mxfp8"
            else "mxfp8_grouped_to_tilelang_extern"
        )
        entry = getattr(_mxfp8_bridge, entry_name, None)
        if entry is None:
            raise CUDABridgeUnsupported(
                f"_mxfp8_bridge is importable but does not expose "
                f"{entry_name!r}; cannot dispatch kind={kind!r} on arch "
                f"{arch!r}."
            )
        return entry(**kwargs)

    # Defensive: SUPPORTED_KINDS guard above should make this unreachable.
    raise CUDABridgeUnsupported(  # pragma: no cover - defensive
        f"internal error: kind={kind!r} passed SUPPORTED_KINDS check but "
        f"has no dispatch branch."
    )
