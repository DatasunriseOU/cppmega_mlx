"""Path E eligibility checks — single source of truth.

Path E wraps the vendored mlx-lm ``gated_delta`` Metal kernel (PR #1217). It
has two HARD limits that, if ignored, silently trade correctness or speed:

1. GATE SIGN (correctness).
   Upstream parameterises the per-step decay as
   ``g_decay = exp(-exp(A_log) * softplus(a + dt_bias))`` which is bounded by
   ``1`` for all real inputs. The GDN adapter recovers our FLA log-decay ``g``
   via ``a = softplus_inverse(-g)``, which only exists for ``g <= 0`` (decay
   ``<= 1``). A model whose gate ``g > 0`` (an *amplifying* gate, decay > 1)
   CANNOT be represented; the historic adapter silently clamped ``g`` to ``0``
   and lost information. We instead FAIL-CLOSE: Path E reports itself
   unavailable / raises so the dispatcher falls back to Path B/A, which can
   represent amplifying gates exactly.

   KDA passes ``g_decay = exp(g)`` straight into the kernel, so KDA itself has
   no upper bound on the gate — but the kernel multiplies state by a *positive*
   decay each step, and a NEGATIVE ``g_decay`` (which never arises from
   ``exp``) would be ill-defined. KDA's amplifying gates (``g > 0``) are
   therefore representable; only the GDN parameterisation is bounded.

2. SHAPE (silent slow path).
   The fast Metal kernel only runs when ``Dk % 32 == 0`` and ``Dv % 4 == 0``.
   For other dims the upstream falls back to a pure-MLX ops reference, which is
   correct but SLOW. ``auto_pick`` must not select Path E for such shapes —
   otherwise it picks a "kernel" that is really the slow reference. We record
   the eligibility in the Path E status so ``auto_pick`` skips it.

These helpers are imported by the GDN/KDA adapters (to fail-close at call
time) and by the dispatch ``_path_e_status`` functions (to skip E in
auto-mode). Keeping them here means the rule is defined once.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

# Fast Metal kernel constraints (mlx-lm gated_delta.py ``_make_gated_delta_kernel``).
KERNEL_DK_MULTIPLE = 32
KERNEL_DV_MULTIPLE = 4


@dataclass(frozen=True)
class PathEEligibility:
    """Result of a Path E eligibility probe for a concrete call.

    ``eligible`` is True only when Path E can run on the fast Metal kernel
    AND represents the requested gate exactly. ``reason`` explains a False.
    """

    eligible: bool
    reason: str
    fast_kernel: bool  # shape qualifies for the fast Metal kernel
    gate_representable: bool  # gate sign is representable by this path


def shape_uses_fast_kernel(dk: int, dv: int) -> bool:
    """True iff (Dk, Dv) hit the fast vendored Metal kernel (not slow ops)."""
    return (dk % KERNEL_DK_MULTIPLE == 0) and (dv % KERNEL_DV_MULTIPLE == 0)


def _gate_has_amplifying(g: mx.array) -> bool:
    """True iff any element of the log-decay ``g`` is > 0 (decay > 1)."""
    # tolerance: treat tiny positive jitter as non-amplifying so that
    # round-trip noise on a g==0 gate does not spuriously fail-close.
    return bool(mx.any(g.astype(mx.float32) > 1e-6).item())


def gdn_eligibility(g: mx.array, dk: int, dv: int) -> PathEEligibility:
    """Eligibility for the GDN Path E adapter (bounded, decay <= 1).

    Fails closed when the gate is amplifying (``g > 0``) — the upstream
    parameterisation cannot represent it — or when the shape forces the slow
    ops fallback.
    """
    fast = shape_uses_fast_kernel(dk, dv)
    amplifying = _gate_has_amplifying(g)
    gate_ok = not amplifying
    if amplifying:
        reason = (
            "GDN Path E cannot represent an amplifying gate: g>0 (decay>1) is "
            "outside upstream g_decay=exp(-exp(A_log)*softplus(...)) which is "
            "bounded by 1; falling back so correctness is not silently clamped"
        )
    elif not fast:
        reason = (
            f"GDN Path E shape ineligible: Dk={dk} (need %{KERNEL_DK_MULTIPLE}==0) "
            f"or Dv={dv} (need %{KERNEL_DV_MULTIPLE}==0) would force the slow "
            "upstream ops fallback, not the fast Metal kernel"
        )
    else:
        reason = "GDN Path E eligible: gate g<=0 and fast Metal kernel shape"
    return PathEEligibility(
        eligible=gate_ok and fast,
        reason=reason,
        fast_kernel=fast,
        gate_representable=gate_ok,
    )


def kda_eligibility(dk: int, dv: int) -> PathEEligibility:
    """Eligibility for the KDA Path E adapter.

    KDA passes ``g_decay = exp(g)`` directly, so any finite gate is
    representable (decay is always positive). The only hard constraint is the
    fast-kernel shape; ineligible shapes force the slow ops fallback.
    """
    fast = shape_uses_fast_kernel(dk, dv)
    if not fast:
        reason = (
            f"KDA Path E shape ineligible: Dk={dk} (need %{KERNEL_DK_MULTIPLE}==0) "
            f"or Dv={dv} (need %{KERNEL_DV_MULTIPLE}==0) would force the slow "
            "upstream ops fallback, not the fast Metal kernel"
        )
    else:
        reason = "KDA Path E eligible: fast Metal kernel shape (gate always representable)"
    return PathEEligibility(
        eligible=fast,
        reason=reason,
        fast_kernel=fast,
        gate_representable=True,
    )


class PathEUnavailable(RuntimeError):
    """Raised by a Path E adapter when it cannot run correctly/fast.

    The dispatcher catches this and falls back to Path B/A. Subclassing
    ``RuntimeError`` keeps it compatible with the existing
    ``_dispatch_failure_or_fallback`` handlers.
    """


__all__ = [
    "KERNEL_DK_MULTIPLE",
    "KERNEL_DV_MULTIPLE",
    "PathEEligibility",
    "PathEUnavailable",
    "gdn_eligibility",
    "kda_eligibility",
    "shape_uses_fast_kernel",
]
