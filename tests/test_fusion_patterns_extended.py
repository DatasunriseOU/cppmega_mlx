"""Smoke tests for the extension fusion patterns added to the
``poc/torch_dynamo`` Dynamo backend.

Covers:
  * ``gemm_softmax`` — attention QK^T softmax (transpose? + matmul +
    softmax) folded into a single TileLang kernel.
  * ``qk_reduce_sm_scale`` — sparse-MLA / DeepSeek-V3 indexed QK reducer
    followed by a scalar ``* sm_scale`` multiply, with the scale baked
    into the reducer's output store.

Two layers of testing
=====================
1. **Pattern-table layer (always runs).** We feed a synthetic op_trace
   list — exactly the shape ``fx_to_tilelang.FXToTileLang`` builds — to
   ``_fusion_patterns.try_match`` and assert the right pattern fires
   and ``_FUSION_HITS`` advances. This is the canonical "did the
   matcher do its job" check; it doesn't need torch / tilelang and
   runs everywhere.

2. **End-to-end ``torch.compile`` layer (xfail until dispatch_lower
   wires the dedicated emitters).** We build a tiny ``nn.Module`` that
   triggers the pattern and run it through ``torch.compile(backend=
   "tilelang")``. The compile is expected to succeed (the sequential
   fallback handles both patterns correctly), and ``_FUSION_HITS``
   should record the hit. This layer is marked ``xfail(strict=True)``
   only on the *perf* axis — TODO: when
   ``cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower`` grows
   the dedicated ``gemm_softmax`` / ``qk_reduce_sm_scale`` emitters,
   flip the marker off and add a perf-floor assertion.

Skipping policy (per memory rule: explicit reasons, no silent skips)
====================================================================
Each test resolves its own skip predicate up-front and emits
``pytest.skip(reason=...)`` with a precise message:

  * "torch unavailable" — minimal env (lint / docs build).
  * "tl_poc_review checkout missing at /private/tmp/tl_poc_review" —
    the POC tree lives outside the cppmega.mlx repo.
  * "torch.compile backend 'tilelang' not registerable on this
    platform" — Metal-only Macs without CUDA still register the
    backend (it falls back to FX eager replay), but the very oldest
    PyTorch wheels lack ``torch._dynamo.register_backend``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap.
# ---------------------------------------------------------------------------
# The patterns module lives in a sibling checkout (``/private/tmp/
# tl_poc_review``) that is intentionally outside the production
# cppmega.mlx tree until RFC §7 Phase 2 stabilises. We add it to
# sys.path on first import only; if the checkout is missing every test
# in this file skips with a precise reason.

_TL_POC_REVIEW_ROOT = Path("/private/tmp/tl_poc_review")


def _poc_torch_dynamo_available() -> bool:
    if not _TL_POC_REVIEW_ROOT.is_dir():
        return False
    if str(_TL_POC_REVIEW_ROOT) not in sys.path:
        sys.path.insert(0, str(_TL_POC_REVIEW_ROOT))
    return importlib.util.find_spec("poc.torch_dynamo._fusion_patterns") is not None


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Layer 1 — pattern-table smoke (no torch / tilelang required).
# ---------------------------------------------------------------------------


def test_gemm_softmax_pattern_fires_on_qkt_softmax_trace() -> None:
    """``transpose + matmul + softmax`` (the QK^T softmax shape) hits
    ``gemm_softmax`` and increments the hit counter."""
    if not _poc_torch_dynamo_available():
        pytest.skip(
            f"tl_poc_review checkout missing at {_TL_POC_REVIEW_ROOT}; "
            "_fusion_patterns.py is the unit under test")

    from poc.torch_dynamo import _fusion_patterns as fp

    # Reset hit counter so we can assert on the delta.
    for k in fp._FUSION_HITS:
        fp._FUSION_HITS[k] = 0

    # Synthetic op_trace mirroring what FXToTileLang appends for
    #   x = torch.matmul(q, k.transpose(-1, -2))
    #   y = torch.softmax(x, dim=-1)
    # ``_TensorSpec`` payloads aren't inspected by the matcher (RFC
    # contract: matchers are payload-agnostic), so empty tuples are OK.
    op_trace = [
        ("transpose", ("k_t", "k_spec", -1, -2)),
        ("matmul", ("x", "q_spec", "k_t_spec")),
        ("softmax", ("y", "x_spec", -1)),
    ]
    result = fp.try_match(op_trace, 0)
    assert result is not None, "gemm_softmax should match QK^T+softmax"
    name, captured, end = result
    assert name == "gemm_softmax", f"expected gemm_softmax, got {name!r}"
    assert end == 3, f"matcher should consume all 3 entries, got end={end}"
    assert len(captured) == 3
    assert fp._FUSION_HITS["gemm_softmax"] == 1, (
        f"hit counter should be 1, got {fp._FUSION_HITS['gemm_softmax']}")

    # 2-op fallback (no upstream transpose) still routes to the legacy
    # softmax_epilogue (declared earlier in FUSION_PATTERNS) so we
    # don't accidentally subsume it.
    op_trace_2 = [
        ("matmul", ("x", "q", "k")),
        ("softmax", ("y", "x", -1)),
    ]
    name2, _, _ = fp.try_match(op_trace_2, 0)
    assert name2 == "softmax_epilogue", (
        f"bare matmul+softmax should still route to softmax_epilogue, "
        f"got {name2!r} (first-match-wins ordering broken?)")


def test_qk_reduce_sm_scale_pattern_fires_on_indexed_reducer_trace() -> None:
    """A sparse-MLA indexed QK reducer followed by a scalar multiply
    hits ``qk_reduce_sm_scale``."""
    if not _poc_torch_dynamo_available():
        pytest.skip(
            f"tl_poc_review checkout missing at {_TL_POC_REVIEW_ROOT}; "
            "_fusion_patterns.py is the unit under test")

    from poc.torch_dynamo import _fusion_patterns as fp

    for k in fp._FUSION_HITS:
        fp._FUSION_HITS[k] = 0

    # Two equivalent qualname spellings should both fire — the matcher
    # canonicalises against ``_QK_REDUCE_OPS``.
    for reducer_name in (
        "qk_reduce",
        "fp8_sparse_mla_indexed_qk_reduce",
    ):
        op_trace = [
            (reducer_name, ("scores", "q", "k_packed", "indices")),
            ("mul", ("scaled", "scores", "sm_scale")),
        ]
        result = fp.try_match(op_trace, 0)
        assert result is not None, (
            f"qk_reduce_sm_scale should match {reducer_name!r} + mul")
        name, captured, end = result
        assert name == "qk_reduce_sm_scale"
        assert end == 2
        assert len(captured) == 2

    assert fp._FUSION_HITS["qk_reduce_sm_scale"] == 2

    # Negative case: a non-QK-reducer custom op followed by mul should
    # NOT match qk_reduce_sm_scale.
    op_trace_neg = [
        ("flash_attention", ("o",)),
        ("mul", ("scaled", "o", "scale")),
    ]
    result_neg = fp.try_match(op_trace_neg, 0)
    # Either no match or a different pattern — but specifically NOT
    # qk_reduce_sm_scale (which would over-fire on every fused-op +
    # multiply pair).
    if result_neg is not None:
        assert result_neg[0] != "qk_reduce_sm_scale", (
            f"qk_reduce_sm_scale over-fired on flash_attention+mul: "
            f"{result_neg!r}")


def test_pending_fusion_warning_class_exported() -> None:
    """The public ``PendingFusionWarning`` symbol exists and is a
    ``UserWarning`` subclass (CI-promotable via ``-W error``)."""
    if not _poc_torch_dynamo_available():
        pytest.skip(
            f"tl_poc_review checkout missing at {_TL_POC_REVIEW_ROOT}")

    from poc.torch_dynamo import _fusion_patterns as fp

    assert hasattr(fp, "PendingFusionWarning")
    assert issubclass(fp.PendingFusionWarning, UserWarning), (
        "PendingFusionWarning must extend UserWarning so it is "
        "filterable via the standard `-W error::UserWarning` machinery")


# ---------------------------------------------------------------------------
# Layer 2 — end-to-end torch.compile (xfail until dispatch_lower wires
# the dedicated emitters).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "gemm_softmax matcher fires (asserted in layer-1 test) but the "
        "dedicated TileLang emitter is not yet wired into "
        "cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower; the "
        "compile currently routes through the sequential fallback, "
        "which produces correct numerics but two kernels. Flip strict=True "
        "and assert _FUSION_HITS['gemm_softmax'] >= 1 once dispatch_lower "
        "emits the fused kernel."),
)
def test_gemm_softmax_compiles_via_torch_compile() -> None:
    """End-to-end: ``torch.compile(backend='tilelang')`` on a tiny
    ``softmax(q @ k^T)`` model fires the gemm_softmax pattern."""
    if not _has("torch"):
        pytest.skip("torch unavailable (lint / docs env)")
    if not _poc_torch_dynamo_available():
        pytest.skip(
            f"tl_poc_review checkout missing at {_TL_POC_REVIEW_ROOT}")
    if not _has("torch._dynamo"):
        pytest.skip("torch._dynamo unavailable (very old PyTorch)")

    import torch
    from torch import nn

    from poc.torch_dynamo import _fusion_patterns as fp
    from poc.torch_dynamo import register

    register()

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    model = TinyQKSoftmax().eval()
    q = torch.randn(2, 8, 16, dtype=torch.float16)
    k = torch.randn(2, 8, 16, dtype=torch.float16)

    # Reset and compile.
    for kk in fp._FUSION_HITS:
        fp._FUSION_HITS[kk] = 0

    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    with torch.no_grad():
        y_ref = model(q, k)
        y = compiled(q, k)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    assert fp._FUSION_HITS["gemm_softmax"] >= 1, (
        f"gemm_softmax pattern did not fire during compile: "
        f"hits={fp._FUSION_HITS}")


@pytest.mark.xfail(
    strict=False,
    reason=(
        "qk_reduce_sm_scale matcher fires (asserted in layer-1 test) "
        "but its emitter still relies on the sparse-MLA path-C kernel "
        "factory `make_fp8_sparse_mla_indexed_qk_reduce_kernel`, which "
        "is not yet exposed through the canonical dispatch_lower path. "
        "Compile currently falls through to the sequential lowering. "
        "Flip strict=True once dispatch_lower learns the qk_reduce "
        "extern-intrinsic handshake (see "
        "poc/extern_intrinsic_examples/simdgroup_mma.py for the same "
        "pattern applied to MMA)."),
)
def test_qk_reduce_sm_scale_compiles_via_torch_compile() -> None:
    """End-to-end: ``torch.compile`` on ``qk_reduce(...) * sm_scale``
    fires the qk_reduce_sm_scale pattern.

    NOTE: this test is doubly fragile because (a) it requires a
    qk_reduce custom op to be registered in the FX op map and (b) the
    sparse-MLA path-C emitter isn't reachable from dispatch_lower yet.
    The xfail is intentional and the layer-1 test above is the real
    coverage gate.
    """
    if not _has("torch"):
        pytest.skip("torch unavailable (lint / docs env)")
    if not _poc_torch_dynamo_available():
        pytest.skip(
            f"tl_poc_review checkout missing at {_TL_POC_REVIEW_ROOT}")
    if not _has("torch._dynamo"):
        pytest.skip("torch._dynamo unavailable (very old PyTorch)")

    import torch

    # We don't have a clean way to build a torch.compile graph that
    # contains a real qk_reduce custom op in the test-only environment
    # (the sparse-MLA reducer needs FP8 kernels). Surface this as the
    # xfail reason rather than silently passing.
    raise NotImplementedError(
        "qk_reduce custom-op registration into the FX op map is "
        "pending — see RFC §7 Phase 2.5; the layer-1 pattern-table "
        "test above provides the real coverage until then.")
