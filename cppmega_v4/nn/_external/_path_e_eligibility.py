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

2. SHAPE (forward: unconstrained; backward: Dv%4).
   The fast Metal *forward* kernel now carries an in-MSL Dk remainder-mask
   (``_mlx_lm_gated_delta_vendored.py``), so it runs CORRECTLY for ANY ``Dk``
   — the old ``Dk % 32 == 0`` limit is gone — and ``Dv`` is also unconstrained
   for the forward (non-uniform dispatch + scalar ``v_[dv_idx]``). The
   training/VJP *backward* kernel also lifts ``Dk`` but STILL needs
   ``Dv % 4 == 0`` (four SIMD groups cooperate over a fixed ``[4*Dk]`` tile);
   for ``Dv % 4 != 0`` training fails closed to the Python-ops VJP reference.
   ``shape_uses_fast_kernel`` (forward) / ``shape_uses_fast_kernel_backward``
   (VJP) encode these so dispatch picks the right path.

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
    """True iff (Dk, Dv) hit the fast vendored Metal *forward* kernel.

    The forward inference kernel now carries an in-MSL Dk remainder-mask
    (``_mlx_lm_gated_delta_vendored.py``), so ANY ``Dk`` runs correctly on the
    fast kernel — the old ``Dk % 32 == 0`` requirement is gone. ``Dv`` is also
    unconstrained for the forward: MLX's non-uniform dispatch handles a ragged
    final threadgroup and the forward reads a scalar ``v_[dv_idx]`` per row.
    Forward is therefore fast for all shapes.
    """
    del dk, dv  # forward kernel is shape-agnostic after the remainder-mask
    return True


def shape_uses_fast_kernel_backward(dk: int, dv: int) -> bool:
    """True iff (Dk, Dv) hit the fast Metal *VJP/training backward* kernel.

    The backward (``_mlx_lm_gated_delta_vjp_metal_vendored.py``) also carries
    the Dk remainder-mask, so ``Dk`` is unconstrained. It STILL requires
    ``Dv % 4 == 0`` because four SIMD groups (one per ``dv`` offset 0..3)
    cooperatively reduce ``dq``/``dk``/``dg``/``dbeta`` via a fixed ``[4 * Dk]``
    shared-memory tile; a ragged final ``Dv`` group is not yet handled. When
    this is False, training fails closed to the Python-ops VJP reference.
    """
    del dk  # Dk handled by the remainder-mask
    return dv % KERNEL_DV_MULTIPLE == 0


def _gate_has_amplifying(g: mx.array) -> bool:
    """True iff any element of the log-decay ``g`` is > 0 (decay > 1).

    Behaviour is unchanged from before the Dk remainder-mask work: the GDN
    adapter relies on this to FAIL-CLOSE on an amplifying gate (the GDN
    parameterisation's ``a=softplus_inverse(-g)`` is real only for ``g <= 0``).
    NOTE: ``mx.any(...).item()`` forces a device sync on the hot path; an
    earlier proposal to gate this behind a ``debug`` flag was dropped because
    skipping the probe would silently let an amplifying GDN gate fall through
    to the clamp instead of fail-closing — i.e. it would change gate-sign
    behaviour, which must stay intact.
    """
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
            "GDN Path E cannot represent an amplifying gate under the GDN "
            "parameterisation: GDN recovers the kernel decay via "
            "a=softplus_inverse(-g), which is real only for g<=0 (decay<=1); a "
            "g>0 (decay>1) has no real pre-image, so falling back rather than "
            "silently clamping. (The Metal kernel itself has NO clamp and "
            "multiplies state by any positive decay — KDA feeds exp(g)>1 into "
            "the SAME kernel and amplifies correctly; only the GDN gate "
            "parameterisation is bounded by 1.)"
        )
    elif not fast:
        reason = (
            f"GDN Path E shape ineligible for Dv={dv} (need %{KERNEL_DV_MULTIPLE}"
            "==0 for the training/VJP backward); forward Dk is unconstrained "
            "after the in-MSL remainder-mask"
        )
    else:
        reason = "GDN Path E eligible: gate g<=0 and supported shape"
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
    reason = (
        "KDA Path E eligible: forward fast Metal kernel runs for any Dk "
        "(in-MSL remainder-mask) and any Dv; gate always representable"
    )
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
    "shape_uses_fast_kernel_backward",
]
