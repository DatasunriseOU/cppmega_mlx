"""GDN Path D (bumped-to-real) — FLA chunk_delta_h Triton kernel through
the ``poc.triton_frontend`` reducer end-to-end.

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
* **Reducer is live**. ``from_ttir()`` parses the captured TTIR through
  the POC's C++ generic-form round trip plus jaxlib ``mlir.ir`` fallback
  and dispatches each op through the OP_TABLE emitters.
* **Status is full for the real FLA chunk-h capture**. The reducer now
  returns a real ``TileLangPrimFunc`` and lands at ``LOWERED_FULL`` on
  this Mac. If those dependencies disappear, the function still reports
  ``LOWERED_DEGRADED`` rather than pretending the runtime path is ready.

Constraints
-----------
* Local dev checkouts are discovered via ``_path_d_deps`` so the cppmega
  runtime adapter can call this module directly.
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

from cppmega_v4._tilelang._path_d_deps import (
    ensure_fla_root,
    ensure_triton_frontend_root,
)


def _ensure_path_d_import_roots() -> None:
    """Make sibling FLA and Triton frontend checkouts importable."""

    ensure_fla_root()
    ensure_triton_frontend_root()


# Default constexpr set used when the caller doesn't pin its own. K=64
# picks the single-block recurrence (the smallest tractable FLA chunk-h
# config); varlen stays off because the runtime adapter binds the fixed
# recurrent prefill signature first.
DEFAULT_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32, "BT": 64, "BV": 32,
    "USE_G": True, "USE_GK": False,
    "USE_INITIAL_STATE": False, "STORE_FINAL_STATE": False,
    "SAVE_NEW_VALUE": True, "TRANSPOSE_STATE": False,
    "IS_VARLEN": False,
}

# Explicit Triton signature: pointer params get their real element type
# (k/v/w/v_new/h are fp16, gates/final state are fp32). The reducer's
# ``_infer_signature`` helper only handles ``_ptr``-suffixed names; FLA
# names its pointers ``k``, ``v``, ``w`` … so we override.
DEFAULT_SIGNATURE: Dict[str, str] = {
    "k": "*fp16", "v": "*fp16", "w": "*fp16", "v_new": "*fp16",
    "g": "*fp32", "gk": "*fp32",
    "h": "*fp16", "h0": "*fp32", "ht": "*fp32",
    "cu_seqlens": "*i64", "chunk_offsets": "*i64",
    "T": "i32",
}

DEFAULT_CHUNK_O_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32,
    "BT": 64, "BK": 64, "BV": 32,
    "USE_G": True, "USE_G_GAMMA": False,
    "TRANSPOSE_STATE": False, "IS_VARLEN": False,
}

DEFAULT_CHUNK_O_SIGNATURE: Dict[str, str] = {
    "q": "*fp16", "k": "*fp16", "v": "*fp16", "h": "*fp16",
    "g": "*fp32", "g_gamma": "*fp32", "o": "*fp16",
    "cu_seqlens": "*i64", "chunk_indices": "*i64",
    "scale": "fp32", "T": "i32",
}

DEFAULT_GDN_KKT_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64,
    "BT": 64, "BC": 16, "BK": 64,
    "USE_G": True, "IS_VARLEN": False,
}

DEFAULT_GDN_KKT_SIGNATURE: Dict[str, str] = {
    "k": "*fp16", "g": "*fp32", "beta": "*fp32", "A": "*fp16",
    "cu_seqlens": "*i64", "chunk_indices": "*i64",
    "T": "i32",
}

DEFAULT_GDN_RECOMPUTE_W_U_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32,
    "BT": 64, "BK": 64, "BV": 64,
    "USE_G": True, "IS_VARLEN": False,
}

DEFAULT_GDN_RECOMPUTE_W_U_SIGNATURE: Dict[str, str] = {
    "k": "*fp16", "v": "*fp16", "beta": "*fp32",
    "w": "*fp16", "u": "*fp16", "A": "*fp16", "g": "*fp32",
    "cu_seqlens": "*i64", "chunk_indices": "*i64",
    "T": "i32",
}

DEFAULT_KDA_INTRA_TOKEN_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64,
    "BT": 32, "BC": 16, "BH": 1,
    "IS_VARLEN": False,
}

DEFAULT_KDA_INTRA_TOKEN_SIGNATURE: Dict[str, str] = {
    "q": "*fp16", "k": "*fp16", "g": "*fp32", "beta": "*fp32",
    "Aqk": "*fp16", "Akk": "*fp32",
    "scale": "fp32", "cu_seqlens": "*i64",
    "N": "i32", "T": "i32",
}

DEFAULT_KDA_INTER_SOLVE_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64,
    "BT": 32, "BC": 16, "NC": 2, "BK": 32,
    "IS_VARLEN": False, "USE_SAFE_GATE": False,
}

DEFAULT_KDA_INTER_SOLVE_SIGNATURE: Dict[str, str] = {
    "q": "*fp16", "k": "*fp16", "g": "*fp32", "beta": "*fp32",
    "Aqk": "*fp16", "Akkd": "*fp32", "Akk": "*fp16",
    "scale": "fp32", "cu_seqlens": "*i64",
    "chunk_indices": "*i64", "T": "i32",
}

DEFAULT_KDA_RECOMPUTE_W_U_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32,
    "BT": 32, "BK": 64, "BV": 64,
    "STORE_QG": True, "STORE_KG": True, "IS_VARLEN": False,
}

DEFAULT_KDA_RECOMPUTE_W_U_SIGNATURE: Dict[str, str] = {
    "q": "*fp16", "k": "*fp16", "qg": "*fp16", "kg": "*fp16",
    "v": "*fp16", "beta": "*fp32",
    "w": "*fp16", "u": "*fp16", "A": "*fp16", "gk": "*fp32",
    "cu_seqlens": "*i64", "chunk_indices": "*i64",
    "T": "i32",
}

DEFAULT_KDA_CHUNK_O_CONSTEXPRS: Dict[str, Any] = {
    "H": 1, "HV": 1, "K": 64, "V": 32,
    "BT": 32, "BK": 64, "BV": 16,
    "TRANSPOSE_STATE": False, "IS_VARLEN": False,
}

DEFAULT_KDA_CHUNK_O_SIGNATURE: Dict[str, str] = {
    "q": "*fp16", "v": "*fp16", "g": "*fp32", "h": "*fp16",
    "o": "*fp16", "A": "*fp16",
    "cu_seqlens": "*i64", "chunk_indices": "*i64",
    "scale": "fp32", "T": "i32",
}

DEFAULT_RUNTIME_T = 64


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


@lru_cache(maxsize=4)
def _cached_lower_chunk_o(constexprs_key: Tuple[Tuple[str, Any], ...]) -> LowerResult:
    return _lower_chunk_o_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_gdn_kkt(constexprs_key: Tuple[Tuple[str, Any], ...]) -> LowerResult:
    return _lower_gdn_kkt_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_gdn_recompute_w_u(
    constexprs_key: Tuple[Tuple[str, Any], ...],
) -> LowerResult:
    return _lower_gdn_recompute_w_u_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_kda_intra_token(
    constexprs_key: Tuple[Tuple[str, Any], ...],
) -> LowerResult:
    return _lower_kda_intra_token_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_kda_inter_solve(
    constexprs_key: Tuple[Tuple[str, Any], ...],
) -> LowerResult:
    return _lower_kda_inter_solve_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_kda_recompute_w_u(
    constexprs_key: Tuple[Tuple[str, Any], ...],
) -> LowerResult:
    return _lower_kda_recompute_w_u_uncached(dict(constexprs_key))


@lru_cache(maxsize=4)
def _cached_lower_kda_chunk_o(
    constexprs_key: Tuple[Tuple[str, Any], ...],
) -> LowerResult:
    return _lower_kda_chunk_o_uncached(dict(constexprs_key))


def _lower_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.common.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as kfn,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_SIGNATURE,
        primfunc_name="fla_chunk_delta_h",
        grid=grid,
        arg_buffer_shapes=_chunk_h_arg_buffer_shapes(constexprs, grid),
    )


def _lower_chunk_o_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.common.chunk_o import chunk_fwd_kernel_o as kfn
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA chunk_o import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_CHUNK_O_SIGNATURE,
        primfunc_name="fla_chunk_o",
        grid=grid,
        arg_buffer_shapes=_chunk_o_arg_buffer_shapes(constexprs, grid),
    )


def _lower_gdn_kkt_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.gated_delta_rule.chunk_fwd import (
            chunk_gated_delta_rule_fwd_kkt_solve_kernel as kfn,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA GDN KKT import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_GDN_KKT_SIGNATURE,
        primfunc_name="fla_gdn_kkt_solve",
        grid=grid,
        arg_buffer_shapes=_gdn_kkt_arg_buffer_shapes(constexprs, grid),
    )


def _lower_gdn_recompute_w_u_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd_kernel as kfn
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA GDN recompute_w_u import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_GDN_RECOMPUTE_W_U_SIGNATURE,
        primfunc_name="fla_gdn_recompute_w_u",
        grid=grid,
        arg_buffer_shapes=_gdn_recompute_arg_buffer_shapes(constexprs, grid),
    )


def _lower_kda_intra_token_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.kda.chunk_intra_token_parallel import (
            chunk_kda_fwd_kernel_intra_token_parallel as kfn,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA KDA intra-token import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_KDA_INTRA_TOKEN_SIGNATURE,
        primfunc_name="fla_kda_intra_token_parallel",
        grid=grid,
        arg_buffer_shapes=_kda_intra_token_arg_buffer_shapes(constexprs, grid),
    )


def _lower_kda_inter_solve_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.kda.chunk_intra import (
            chunk_kda_fwd_kernel_inter_solve_fused as kfn,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA KDA inter-solve import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_KDA_INTER_SOLVE_SIGNATURE,
        primfunc_name="fla_kda_inter_solve",
        grid=grid,
        arg_buffer_shapes=_kda_inter_solve_arg_buffer_shapes(constexprs, grid),
    )


def _lower_kda_recompute_w_u_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.kda.wy_fast import recompute_w_u_fwd_kda_kernel as kfn
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA KDA recompute_w_u import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_KDA_RECOMPUTE_W_U_SIGNATURE,
        primfunc_name="fla_kda_recompute_w_u",
        grid=grid,
        arg_buffer_shapes=_kda_recompute_arg_buffer_shapes(constexprs, grid),
    )


def _lower_kda_chunk_o_uncached(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    _ensure_path_d_import_roots()
    try:
        from fla.ops.gla.chunk import chunk_gla_fwd_kernel_o as kfn
    except Exception as exc:
        return _failed_lower(
            "FAILED_PARSE",
            type(exc).__name__,
            f"FLA KDA/GLA chunk_o import failed: {exc}",
            constexprs,
        )
    return _lower_fla_kernel_uncached(
        kfn=kfn,
        constexprs=constexprs,
        signature=DEFAULT_KDA_CHUNK_O_SIGNATURE,
        primfunc_name="fla_kda_chunk_o_gk",
        grid=grid,
        arg_buffer_shapes=_kda_chunk_o_arg_buffer_shapes(constexprs, grid),
    )


def _runtime_t(_constexprs: Dict[str, Any]) -> int:
    """Static T paired with the cppmega Path D scalar specialization."""

    return int(_constexprs.get("_RUNTIME_T", DEFAULT_RUNTIME_T))


def _runtime_n(constexprs: Dict[str, Any], fallback: int) -> int:
    return int(constexprs.get("_RUNTIME_N", fallback))


def _runtime_batch(constexprs: Dict[str, Any], fallback: int) -> int:
    return int(constexprs.get("_RUNTIME_BATCH", fallback))


def _runtime_num_chunks(constexprs: Dict[str, Any], fallback: int) -> int:
    return int(constexprs.get("_RUNTIME_NT", fallback))


def _runtime_cu_len(constexprs: Dict[str, Any], fallback: int) -> int:
    return int(constexprs.get("_RUNTIME_CU_LEN", fallback))


def _num_chunks(t: int, bt: int) -> int:
    return max((int(t) + int(bt) - 1) // int(bt), 1)


def _batch_from_grid(grid: Optional[Tuple[int, ...]], hv: int) -> int:
    if grid is None or len(grid) < 2:
        return 1
    return max(int(grid[1]) // max(int(hv), 1), 1)


def _batch_from_token_grid(grid: Optional[Tuple[int, ...]], t: int) -> int:
    if grid is None or len(grid) < 1:
        return 1
    return max(int(grid[0]) // max(int(t), 1), 1)


def _full_arg(extent: int) -> Tuple[int]:
    return (max(int(extent), 1),)


def _gdn_varlen_meta_lengths(
    constexprs: Dict[str, Any],
    b: int,
) -> Tuple[int, int]:
    """Return ``(cu_seqlens_len, chunk_indices_len)`` honoring IS_VARLEN.

    When packed varlen is active the kkt/recompute/chunk_o kernels receive the
    real cu_seqlens (length N+1) and chunk_indices (2 ints per chunk); under
    fixed-length they are unused single-element placeholders.
    """

    t = _runtime_t(constexprs)
    bt = int(constexprs["BT"])
    if bool(constexprs.get("IS_VARLEN", False)):
        n = _runtime_n(constexprs, b)
        nt = _runtime_num_chunks(constexprs, _num_chunks(t, bt))
        return _runtime_cu_len(constexprs, n + 1), nt * 2
    return 1, 1


def _gdn_kkt_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    bt = int(constexprs["BT"])
    b = _batch_from_grid(grid, hv)
    cu_len, chunk_indices_len = _gdn_varlen_meta_lengths(constexprs, b)
    return {
        0: _full_arg(b * t * h * k),      # k
        1: _full_arg(b * t * hv),         # g
        2: _full_arg(b * t * hv),         # beta
        3: _full_arg(b * t * hv * bt),    # A
        4: _full_arg(cu_len),             # cu_seqlens
        5: _full_arg(chunk_indices_len),  # chunk_indices
    }


def _gdn_recompute_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    v = int(constexprs["V"])
    bt = int(constexprs["BT"])
    b = _batch_from_grid(grid, hv)
    cu_len, chunk_indices_len = _gdn_varlen_meta_lengths(constexprs, b)
    return {
        0: _full_arg(b * t * h * k),      # k
        1: _full_arg(b * t * hv * v),     # v
        2: _full_arg(b * t * hv),         # beta
        3: _full_arg(b * t * hv * k),     # w
        4: _full_arg(b * t * hv * v),     # u
        5: _full_arg(b * t * hv * bt),    # A
        6: _full_arg(b * t * hv),         # g
        7: _full_arg(cu_len),             # cu_seqlens
        8: _full_arg(chunk_indices_len),  # chunk_indices
    }


def _chunk_h_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    v = int(constexprs["V"])
    bt = int(constexprs["BT"])
    b = _batch_from_grid(grid, hv)
    if bool(constexprs.get("IS_VARLEN", False)):
        b_data = _runtime_batch(constexprs, 1)
        n = _runtime_n(constexprs, b)
        nt = _runtime_num_chunks(constexprs, _num_chunks(t, bt))
        cu_len = _runtime_cu_len(constexprs, n + 1)
    else:
        b_data = b
        n = b_data
        nt = b * _num_chunks(t, bt)
        cu_len = 1
    return {
        0: _full_arg(b_data * t * h * k),  # k
        1: _full_arg(b_data * t * hv * v), # v/u
        2: _full_arg(b_data * t * hv * k), # w
        3: _full_arg(b_data * t * hv * v), # v_new
        4: _full_arg(b_data * t * hv),     # g
        5: _full_arg(b_data * t * hv * k), # gk
        6: _full_arg(nt * hv * k * v),     # h
        7: _full_arg(n * hv * k * v),      # h0
        8: _full_arg(n * hv * k * v),      # ht
        9: _full_arg(cu_len),              # cu_seqlens
        10: _full_arg(cu_len),             # chunk_offsets
    }


def _chunk_o_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    v = int(constexprs["V"])
    bt = int(constexprs["BT"])
    b = _batch_from_grid(grid[1:] if grid is not None and len(grid) == 3 else grid, hv)
    if grid is not None and len(grid) >= 3:
        b = max(int(grid[2]) // max(hv, 1), 1)
    if bool(constexprs.get("IS_VARLEN", False)):
        b = _runtime_batch(constexprs, b)
        n = _runtime_n(constexprs, b)
        nt = _runtime_num_chunks(constexprs, _num_chunks(t, bt))
        cu_len = _runtime_cu_len(constexprs, n + 1)
        chunk_indices_len = nt * 2
    else:
        nt = b * _num_chunks(t, bt)
        cu_len = 1
        chunk_indices_len = 1
    return {
        0: _full_arg(b * t * h * k),       # q
        1: _full_arg(b * t * h * k),       # k
        2: _full_arg(b * t * hv * v),      # v_new
        3: _full_arg(nt * hv * k * v),     # h
        4: _full_arg(b * t * hv),          # g
        5: _full_arg(hv),                  # g_gamma
        6: _full_arg(b * t * hv * v),      # o
        7: _full_arg(cu_len),              # cu_seqlens
        8: _full_arg(chunk_indices_len),   # chunk_indices
    }


def _kda_intra_token_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    bt = int(constexprs["BT"])
    bc = int(constexprs["BC"])
    b = _batch_from_token_grid(grid, t)
    cu_len = 1
    if bool(constexprs.get("IS_VARLEN", False)):
        n = _runtime_n(constexprs, b)
        cu_len = _runtime_cu_len(constexprs, n + 1)
    return {
        0: _full_arg(b * t * h * k),       # q
        1: _full_arg(b * t * h * k),       # k
        2: _full_arg(b * t * hv * k),      # g cumulative
        3: _full_arg(b * t * hv),          # beta
        4: _full_arg(b * t * hv * bt),     # Aqk
        5: _full_arg(b * t * hv * bc),     # Akk diagonal blocks
        7: _full_arg(cu_len),              # cu_seqlens
    }


def _kda_inter_solve_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    bt = int(constexprs["BT"])
    bc = int(constexprs["BC"])
    b = _batch_from_grid(grid, hv)
    if bool(constexprs.get("IS_VARLEN", False)):
        n = _runtime_n(constexprs, b)
        cu_len = _runtime_cu_len(constexprs, n + 1)
        chunk_indices_len = _runtime_num_chunks(
            constexprs, _num_chunks(t, bt)
        ) * 2
    else:
        cu_len = 1
        chunk_indices_len = 1
    return {
        0: _full_arg(b * t * h * k),       # q
        1: _full_arg(b * t * h * k),       # k
        2: _full_arg(b * t * hv * k),      # g cumulative
        3: _full_arg(b * t * hv),          # beta
        4: _full_arg(b * t * hv * bt),     # Aqk
        5: _full_arg(b * t * hv * bc),     # Akkd
        6: _full_arg(b * t * hv * bt),     # Akk
        8: _full_arg(cu_len),              # cu_seqlens
        9: _full_arg(chunk_indices_len),   # chunk_indices
    }


def _kda_recompute_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    v = int(constexprs["V"])
    bt = int(constexprs["BT"])
    b = _batch_from_grid(grid, hv)
    if bool(constexprs.get("IS_VARLEN", False)):
        n = _runtime_n(constexprs, b)
        cu_len = _runtime_cu_len(constexprs, n + 1)
        chunk_indices_len = _runtime_num_chunks(
            constexprs, _num_chunks(t, bt)
        ) * 2
    else:
        cu_len = 1
        chunk_indices_len = 1
    return {
        0: _full_arg(b * t * h * k),       # q
        1: _full_arg(b * t * h * k),       # k
        2: _full_arg(b * t * hv * k),      # qg
        3: _full_arg(b * t * hv * k),      # kg
        4: _full_arg(b * t * hv * v),      # v
        5: _full_arg(b * t * hv),          # beta
        6: _full_arg(b * t * hv * k),      # w
        7: _full_arg(b * t * hv * v),      # u
        8: _full_arg(b * t * hv * bt),     # Akk
        9: _full_arg(b * t * hv * k),      # gk cumulative
        10: _full_arg(cu_len),             # cu_seqlens
        11: _full_arg(chunk_indices_len),  # chunk_indices
    }


def _kda_chunk_o_arg_buffer_shapes(
    constexprs: Dict[str, Any],
    grid: Optional[Tuple[int, ...]],
) -> Dict[int, Tuple[int]]:
    t = _runtime_t(constexprs)
    h = int(constexprs["H"])
    hv = int(constexprs["HV"])
    k = int(constexprs["K"])
    v = int(constexprs["V"])
    bt = int(constexprs["BT"])
    b = 1
    if grid is not None and len(grid) >= 3:
        b = max(int(grid[2]) // max(hv, 1), 1)
    if bool(constexprs.get("IS_VARLEN", False)):
        b = _runtime_batch(constexprs, b)
        n = _runtime_n(constexprs, b)
        nt = _runtime_num_chunks(constexprs, _num_chunks(t, bt))
        cu_len = _runtime_cu_len(constexprs, n + 1)
        chunk_indices_len = nt * 2
    else:
        nt = b * _num_chunks(t, bt)
        cu_len = 1
        chunk_indices_len = 1
    return {
        0: _full_arg(b * t * h * k),       # q
        1: _full_arg(b * t * hv * v),      # v_new
        2: _full_arg(b * t * hv * k),      # g cumulative
        3: _full_arg(nt * hv * k * v),     # h
        4: _full_arg(b * t * hv * v),      # o
        5: _full_arg(b * t * hv * bt),     # Aqk
        6: _full_arg(cu_len),              # cu_seqlens
        7: _full_arg(chunk_indices_len),   # chunk_indices
    }


def _failed_lower(
    status: str,
    error_type: str,
    error_message: str,
    constexprs: Dict[str, Any],
    *,
    ttir_text_len: int = 0,
    missing_ops: Optional[list] = None,
) -> LowerResult:
    return LowerResult(
        status=status,
        visited_ops=[],
        missing_ops=missing_ops or [],
        prim_func=None,
        ttir_text_len=ttir_text_len,
        error_type=error_type,
        error_message=error_message,
        constexprs=constexprs,
    )


def _lower_fla_kernel_uncached(
    *,
    kfn: Any,
    constexprs: Dict[str, Any],
    signature: Dict[str, str],
    primfunc_name: str,
    grid: Optional[Tuple[int, ...]] = None,
    arg_buffer_shapes: Optional[Dict[int, Tuple[int]]] = None,
) -> LowerResult:
    """Real lowering driver, no cache.

    Catches every step's exception so the dispatcher gets a structured
    ``LowerResult`` instead of a raise. The explicit signature avoids the
    harness's ``_ptr``-suffix heuristic, which would otherwise type FLA pointer
    params as i32 and fail at ``tt.addptr``.
    """
    _ensure_path_d_import_roots()
    try:
        from poc.triton_frontend._test_harness.jit_to_ttir import (
            TTIRCaptureError,
            TritonUnavailable,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_OTHER",
            type(exc).__name__,
            f"poc.triton_frontend import failed: {exc}",
            constexprs,
        )

    try:
        inner = _unwrap_to_jit_function(kfn)
    except Exception as exc:
        return _failed_lower(
            "FAILED_OTHER", type(exc).__name__, str(exc), constexprs,
        )

    try:
        ttir_text = _capture_ttir_with_explicit_signature(
            inner,
            {k: v for k, v in constexprs.items() if not str(k).startswith("_")},
            signature,
        )
    except (TTIRCaptureError, TritonUnavailable) as exc:
        return _failed_lower(
            "FAILED_PARSE", type(exc).__name__, str(exc), constexprs,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_OTHER", type(exc).__name__, str(exc), constexprs,
        )

    # Prefer ``from_ttir`` directly so we control the text-vs-mlir routing;
    # ``from_triton_kernel`` would re-capture and double the work.
    try:
        from poc.triton_frontend import from_ttir, _walk_text_ttir

        full_error_type: Optional[str] = None
        full_error_message: Optional[str] = None
        try:
            prim = from_ttir(
                ttir_text,
                name=primfunc_name,
                grid=grid,
                arg_buffer_shapes=arg_buffer_shapes,
            )
            return LowerResult(
                status="LOWERED_FULL",
                visited_ops=_walk_text_ttir(ttir_text),
                missing_ops=[],
                prim_func=prim,
                ttir_text_len=len(ttir_text),
                error_type=None,
                error_message=None,
                constexprs=constexprs,
            )
        except NotImplementedError as exc:
            return _failed_lower(
                "FAILED_OPS",
                type(exc).__name__,
                str(exc),
                constexprs,
                ttir_text_len=len(ttir_text),
                missing_ops=[str(exc)],
            )
        except Exception as exc:  # noqa: BLE001
            full_error_type = type(exc).__name__
            full_error_message = str(exc)

        try:
            visited = _walk_text_ttir(ttir_text)
        except NotImplementedError as exc:
            return _failed_lower(
                "FAILED_OPS",
                type(exc).__name__,
                str(exc),
                constexprs,
                ttir_text_len=len(ttir_text),
                missing_ops=[str(exc)],
            )
        return LowerResult(
            status="LOWERED_DEGRADED",
            visited_ops=visited,
            missing_ops=[],
            prim_func=None,
            ttir_text_len=len(ttir_text),
            error_type=full_error_type,
            error_message=full_error_message,
            constexprs=constexprs,
        )
    except Exception as exc:
        return _failed_lower(
            "FAILED_OTHER",
            type(exc).__name__,
            str(exc),
            constexprs,
            ttir_text_len=len(ttir_text),
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
    grid: Optional[Tuple[int, ...]] = None,
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
    cfg = dict(DEFAULT_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower(key)


def lower_fla_chunk_o(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA common chunk_o through the Triton frontend."""
    cfg = dict(DEFAULT_CHUNK_O_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_chunk_o_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_chunk_o(key)


def lower_fla_gdn_kkt_solve(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA GDN KKT solve through the Triton frontend."""
    cfg = dict(DEFAULT_GDN_KKT_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_gdn_kkt_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_gdn_kkt(key)


def lower_fla_gdn_recompute_w_u(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA GDN recompute_w_u through the Triton frontend."""
    cfg = dict(DEFAULT_GDN_RECOMPUTE_W_U_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_gdn_recompute_w_u_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_gdn_recompute_w_u(key)


def lower_fla_kda_intra_token_parallel(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA KDA token-parallel diagonal blocks."""
    cfg = dict(DEFAULT_KDA_INTRA_TOKEN_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_kda_intra_token_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_kda_intra_token(key)


def lower_fla_kda_inter_solve(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA KDA fused inter-subchunk solve."""
    cfg = dict(DEFAULT_KDA_INTER_SOLVE_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_kda_inter_solve_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_kda_inter_solve(key)


def lower_fla_kda_recompute_w_u(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower FLA KDA recompute_w_u through the Triton frontend."""
    cfg = dict(DEFAULT_KDA_RECOMPUTE_W_U_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_kda_recompute_w_u_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_kda_recompute_w_u(key)


def lower_fla_kda_chunk_o(
    constexprs: Optional[Dict[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
) -> LowerResult:
    """Capture and lower the GLA output kernel used by KDA Path D."""
    cfg = dict(DEFAULT_KDA_CHUNK_O_CONSTEXPRS)
    if constexprs is not None:
        cfg.update(constexprs)
    if grid is not None:
        return _lower_kda_chunk_o_uncached(cfg, tuple(int(x) for x in grid))
    key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
    return _cached_lower_kda_chunk_o(key)


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
        "Path D runtime still needs the cppmega_v4 compile/launch adapter "
        "that maps this PrimFunc back to the recurrent call signature."
    )


__all__ = [
    "DEFAULT_CHUNK_O_CONSTEXPRS",
    "DEFAULT_CHUNK_O_SIGNATURE",
    "DEFAULT_CONSTEXPRS",
    "DEFAULT_GDN_KKT_CONSTEXPRS",
    "DEFAULT_GDN_KKT_SIGNATURE",
    "DEFAULT_GDN_RECOMPUTE_W_U_CONSTEXPRS",
    "DEFAULT_GDN_RECOMPUTE_W_U_SIGNATURE",
    "DEFAULT_KDA_CHUNK_O_CONSTEXPRS",
    "DEFAULT_KDA_CHUNK_O_SIGNATURE",
    "DEFAULT_KDA_INTER_SOLVE_CONSTEXPRS",
    "DEFAULT_KDA_INTER_SOLVE_SIGNATURE",
    "DEFAULT_KDA_INTRA_TOKEN_CONSTEXPRS",
    "DEFAULT_KDA_INTRA_TOKEN_SIGNATURE",
    "DEFAULT_KDA_RECOMPUTE_W_U_CONSTEXPRS",
    "DEFAULT_KDA_RECOMPUTE_W_U_SIGNATURE",
    "DEFAULT_SIGNATURE",
    "LowerResult",
    "lower_fla_chunk_h",
    "lower_fla_chunk_o",
    "lower_fla_gdn_kkt_solve",
    "lower_fla_gdn_recompute_w_u",
    "lower_fla_kda_chunk_o",
    "lower_fla_kda_inter_solve",
    "lower_fla_kda_intra_token_parallel",
    "lower_fla_kda_recompute_w_u",
    "gdn_fwd_path_d_real_call",
]
