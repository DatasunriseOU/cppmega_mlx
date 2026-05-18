"""GDN Path D (bumped-to-real) — FLA chunk_delta_h Triton kernel through
the ``tilelang.poc.triton_frontend`` reducer end-to-end.

Distinct from ``linear_attention_path_d.py``, which was the *gated*
seam that returned the actionable "not yet runnable" message while
op_mapping was Tier-1. This module is the **real** seam: it captures
TTIR from FLA's ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` via
``triton_jit_to_ttir`` and threads it through
``from_triton_kernel()`` (or, equivalently, ``from_ttir()`` on the
captured text) so the reducer actually walks every dialect op.

What "real" means here
----------------------
* **TTIR capture is live**. ``triton_jit_to_ttir`` calls Triton 3.6's
  ``ASTSource.make_ir(target, options, codegen, module_map, ctx)``
  against the ``apple/mps`` backend (triton-pr9701) — never invokes
  Metal codegen so the Apple metal-as / metal-ll quirks stay out of
  the way.
* **Reducer is live**. ``from_triton_kernel()`` parses the captured
  TTIR with ``mlir.ir`` (when bindings present) or the text walker
  (fallback) and dispatches each op through the OP_TABLE emitters.
* **Status is degraded, not full**. The Mac dev box has
  ``MLIR_WALKER_AVAILABLE == False`` (mlir.ir python bindings absent),
  so we land at ``LOWERED_DEGRADED``: every op routes through OP_TABLE
  without raising, but the walker never populates ``ctx.value_map`` /
  ``ctx.buffers`` because the regex walker is coverage-only by design.
  That's the documented Tier-2 contract, not a silent failure.

When the MLIR bindings become available (e.g. ``MLIR_PYTHON_PACKAGE_PREFIX``
points at jaxlib's bundled ``mlir.ir`` or brew-llvm), this same call
path produces a real ``TileLangPrimFunc`` and lands at
``LOWERED_FULL``. No code changes needed here — the seam is identical
across degraded / full status.

Constraints
-----------
* Caller (the cppmega_v4 dispatcher) must arrange for
  ``/Volumes/external/sources/tilelang`` to be on ``sys.path`` (or the
  installed ``tilelang`` wheel must contain ``poc.triton_frontend``).
  We do not modify ``sys.path`` from this module.
* We do not modify the FLA source.
* We never invoke Metal codegen — the lowering stops at TileLang
  PrimFunc (or degraded coverage report); the caller decides whether
  to compile further with ``tilelang.compile(...)``.

Public surface
--------------
* :func:`lower_fla_chunk_h` -> :class:`LowerResult` — full attempt with
  ``constexprs={K:64,...}``.
* :func:`gdn_fwd_path_d_real_call` — entry shaped like the existing
  ``_gdn_fwd_path_d_call`` so the dispatcher can swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


# Default constexpr set used when the caller doesn't pin its own. K=64
# picks the single-block recurrence (the smallest tractable FLA chunk-h
# config); the gate / varlen flags are off because Path A handles those
# cases natively and we want the smoothest reducer exercise here.
DEFAULT_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32, "BT": 16, "BV": 32,
    "USE_G": False, "USE_GK": False,
    "USE_INITIAL_STATE": False, "STORE_FINAL_STATE": False,
    "SAVE_NEW_VALUE": False, "TRANSPOSE_STATE": False,
    "IS_VARLEN": False,
}

# Explicit Triton signature: pointer params get their real element type
# (k/v/w/v_new are fp16, gates/state are fp32). The reducer's
# ``_infer_signature`` helper only handles ``_ptr``-suffixed names; FLA
# names its pointers ``k``, ``v``, ``w`` … so we override.
DEFAULT_SIGNATURE: Dict[str, str] = {
    "k": "*fp16", "v": "*fp16", "w": "*fp16", "v_new": "*fp16",
    "g": "*fp32", "gk": "*fp32",
    "h": "*fp32", "h0": "*fp32", "ht": "*fp32",
    "cu_seqlens": "*fp32", "chunk_offsets": "*fp32",
    "T": "i32",
}


@dataclass
class LowerResult:
    """Outcome of one Path D real-lowering attempt.

    ``status`` mirrors the reducer's taxonomy
    (:class:`poc.triton_frontend._test_harness.run_corpus.Status`):

    * ``LOWERED_FULL``     — MLIR walker + TVM available; ``prim_func`` set.
    * ``LOWERED_DEGRADED`` — text walker only; ``prim_func`` is None,
      ``visited_ops`` populated.
    * ``FAILED_OPS``       — at least one op missing from OP_TABLE;
      ``missing_ops`` lists which.
    * ``FAILED_PARSE``     — couldn't capture TTIR (Triton compile failed).
    * ``FAILED_OTHER``     — unexpected exception (``error_type`` /
      ``error_message`` populated).
    """

    status: str
    visited_ops: list
    missing_ops: list
    prim_func: Optional[Any]
    ttir_text_len: int
    error_type: Optional[str]
    error_message: Optional[str]
    constexprs: Dict[str, Any]


def _unwrap_to_jit_function(kfn: Any) -> Any:
    """Peel Heuristics / Autotuner wrappers off until we have a JITFunction.

    FLA decorates the inner kernel with ``@triton.heuristics`` +
    ``@fla_cache_autotune`` + ``@triton.jit``. The wrapper chain is
    ``Heuristics -> CachedAutotuner -> JITFunction`` and each layer
    exposes ``.fn`` for the next layer. We walk ``.fn`` until we hit
    the real JITFunction (which is what ``triton_jit_to_ttir`` needs).
    """
    from triton.runtime.jit import JITFunction

    cur = kfn
    seen = set()
    while not isinstance(cur, JITFunction):
        if id(cur) in seen:
            raise RuntimeError(
                f"unwrap loop: stuck at {type(cur).__name__}; "
                "no .fn / .base_fn attribute"
            )
        seen.add(id(cur))
        nxt = getattr(cur, "fn", None) or getattr(cur, "base_fn", None)
        if nxt is None or nxt is cur:
            raise RuntimeError(
                f"unwrap dead-end at {type(cur).__name__}"
            )
        cur = nxt
    return cur


@lru_cache(maxsize=4)
def _cached_lower(constexprs_key: Tuple[Tuple[str, Any], ...]) -> LowerResult:
    """Internal: cache lowering by frozen constexpr signature.

    The key is a sorted tuple of constexpr items (dicts are unhashable);
    the result is memoised so repeated dispatcher calls don't re-run
    Triton's frontend.
    """
    return _lower_uncached(dict(constexprs_key))


def _lower_uncached(constexprs: Dict[str, Any]) -> LowerResult:
    """Real lowering driver, no cache. Catches every step's exception so
    the dispatcher gets a structured ``LowerResult`` instead of a raise."""
    # Step 0: import the FLA kernel + triton_frontend pieces.
    try:
        from fla.ops.common.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as kfn,
        )
    except Exception as exc:
        return LowerResult(
            status="FAILED_PARSE",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=f"FLA import failed: {exc}",
            constexprs=constexprs,
        )
    try:
        from poc.triton_frontend._test_harness.jit_to_ttir import (
            TTIRCaptureError,
            TritonUnavailable,
        )
    except Exception as exc:
        return LowerResult(
            status="FAILED_OTHER",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=f"poc.triton_frontend import failed: {exc}",
            constexprs=constexprs,
        )

    # Step 1: unwrap heuristics/autotuner to the JITFunction.
    try:
        inner = _unwrap_to_jit_function(kfn)
    except Exception as exc:
        return LowerResult(
            status="FAILED_OTHER",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=str(exc),
            constexprs=constexprs,
        )

    # Step 2: capture TTIR via Triton 3.6 make_ir with explicit signature.
    # We bypass the harness's auto-inferred signature because FLA's
    # pointer params (k, v, w, …) don't carry the ``_ptr`` suffix the
    # default inferer keys off, so they'd be typed i32 and the kernel
    # would fail at the first ``tt.addptr`` with "Expected base to be a
    # scalar pointer type".
    try:
        ttir_text = _capture_ttir_with_explicit_signature(
            inner, constexprs, DEFAULT_SIGNATURE,
        )
    except TTIRCaptureError as exc:
        return LowerResult(
            status="FAILED_PARSE",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=str(exc),
            constexprs=constexprs,
        )
    except TritonUnavailable as exc:
        return LowerResult(
            status="FAILED_PARSE",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=str(exc),
            constexprs=constexprs,
        )
    except Exception as exc:
        return LowerResult(
            status="FAILED_OTHER",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=0,
            error_type=type(exc).__name__,
            error_message=str(exc),
            constexprs=constexprs,
        )

    # Step 3: thread through the reducer. Prefer ``from_ttir`` directly
    # so we control the text-vs-mlir routing; ``from_triton_kernel``
    # would re-capture and double the work.
    try:
        from poc.triton_frontend import from_ttir, _walk_text_ttir
        from poc.triton_frontend import OP_TABLE
        # Try the MLIR walker first; fall back to text walker.
        try:
            from poc.triton_frontend import mlir_walker as _mw  # noqa: F401
            mlir_ok = bool(getattr(_mw, "MLIR_WALKER_AVAILABLE", False))
        except Exception:
            mlir_ok = False
        if mlir_ok:
            try:
                prim = from_ttir(ttir_text, name="fla_chunk_delta_h")
                return LowerResult(
                    status="LOWERED_FULL",
                    visited_ops=[],  # mlir walker doesn't return op list here
                    missing_ops=[],
                    prim_func=prim,
                    ttir_text_len=len(ttir_text),
                    error_type=None,
                    error_message=None,
                    constexprs=constexprs,
                )
            except NotImplementedError as exc:
                # Specific op missing from OP_TABLE -> FAILED_OPS.
                return LowerResult(
                    status="FAILED_OPS",
                    visited_ops=[],
                    missing_ops=[str(exc)],
                    prim_func=None,
                    ttir_text_len=len(ttir_text),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    constexprs=constexprs,
                )
            except Exception as exc:
                # Fall through to text walker for at least op coverage.
                pass

        # Degraded path: text walker. Returns visited op list; raises
        # NotImplementedError if an op isn't in OP_TABLE.
        try:
            visited = _walk_text_ttir(ttir_text)
        except NotImplementedError as exc:
            return LowerResult(
                status="FAILED_OPS",
                visited_ops=[],
                missing_ops=[str(exc)],
                prim_func=None,
                ttir_text_len=len(ttir_text),
                error_type=type(exc).__name__,
                error_message=str(exc),
                constexprs=constexprs,
            )
        return LowerResult(
            status="LOWERED_DEGRADED",
            visited_ops=visited,
            missing_ops=[],
            prim_func=None,
            ttir_text_len=len(ttir_text),
            error_type=None,
            error_message=None,
            constexprs=constexprs,
        )
    except Exception as exc:
        return LowerResult(
            status="FAILED_OTHER",
            visited_ops=[],
            missing_ops=[],
            prim_func=None,
            ttir_text_len=len(ttir_text),
            error_type=type(exc).__name__,
            error_message=str(exc),
            constexprs=constexprs,
        )


def _capture_ttir_with_explicit_signature(
    inner: Any,
    constexprs: Dict[str, Any],
    signature: Dict[str, str],
) -> str:
    """Drive Triton 3.6 ``ASTSource.make_ir`` directly with an explicit
    signature.

    Mirrors :func:`poc.triton_frontend._test_harness.jit_to_ttir._try_triton_3_6`
    but lets us pass a signature that knows ``k``/``v``/etc. are pointer
    types (the auto-inferer in the harness only handles ``_ptr``-suffixed
    names). Stops at the TTIR stage exactly like the harness — never
    invokes Metal codegen.
    """
    from triton.compiler.compiler import ASTSource
    from triton.backends import backends as _backends
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir as _libir

    backend_to_gputarget = [
        ("apple", GPUTarget("mps", "apple_m2", 32)),
        ("nvidia", GPUTarget("cuda", 80, 32)),
        ("amd", GPUTarget("hip", "gfx942", 64)),
    ]
    last_err: Optional[Exception] = None
    for be_name, gpu_target in backend_to_gputarget:
        pkg = _backends.get(be_name)
        if pkg is None:
            continue
        try:
            binst = pkg.compiler(gpu_target)
            opts = binst.parse_options({})
            codegen = binst.get_codegen_implementation(opts)
            mmap = binst.get_module_map()
            ctx = _libir.context()
            binst.load_dialects(ctx)
            src = ASTSource(fn=inner, signature=signature, constexprs=constexprs)
            mod = src.make_ir(gpu_target, opts, codegen, mmap, ctx)
            return str(mod)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    from poc.triton_frontend._test_harness.jit_to_ttir import TTIRCaptureError
    raise TTIRCaptureError(
        f"3.6+ make_ir failed across {[b for b,_ in backend_to_gputarget]}: {last_err}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower_fla_chunk_h(
    constexprs: Optional[Dict[str, Any]] = None,
) -> LowerResult:
    """Real Path D entry — capture FLA chunk-h TTIR and walk via OP_TABLE.

    Parameters
    ----------
    constexprs:
        Optional override of the default config
        (:data:`DEFAULT_CONSTEXPRS`). Pass a dict if you want a different
        ``K``/``V``/gate combination; the cache key includes every item.

    Returns
    -------
    LowerResult
        Always returned — the only way this function raises is on a
        truly catastrophic interpreter error.
    """
    cfg = dict(DEFAULT_CONSTEXPRS) if constexprs is None else dict(constexprs)
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower(key)


def gdn_fwd_path_d_real_call(*args, **kwargs):
    """Dispatcher-shaped entry. Returns the lowered PrimFunc when
    ``LOWERED_FULL``; raises with the structured reason otherwise.

    Symmetric with ``linear_attention_path_d._gdn_fwd_path_d_call``.
    """
    res = lower_fla_chunk_h()
    if res.status == "LOWERED_FULL" and res.prim_func is not None:
        return res.prim_func
    raise RuntimeError(
        f"GDN Path D (real) status={res.status}; "
        f"visited={len(res.visited_ops)} ops; missing={res.missing_ops!r}; "
        f"error={res.error_type}: {res.error_message}. "
        "Install mlir.ir python bindings (brew llvm or "
        "MLIR_PYTHON_PACKAGE_PREFIX) to escalate from LOWERED_DEGRADED "
        "to LOWERED_FULL."
    )


__all__ = [
    "DEFAULT_CONSTEXPRS",
    "DEFAULT_SIGNATURE",
    "LowerResult",
    "lower_fla_chunk_h",
    "gdn_fwd_path_d_real_call",
]
