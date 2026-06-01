"""Eager (MLX-differentiable) Mamba3 projection fwd/bwd bridge for Path C.

This module is the missing differentiable sub-step that lets the flag-ON
``CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN`` chunked SSD-core region run on the REAL
projected inputs (not the zero-seed that the pre-step owner allocated) and fold
its scan-input grads back into the mamba3 model parameters.

The chunked grid kernels only cover the SSD *scan core*. Their projected inputs
``{brick}_x/B/C/A/dt/z/h0`` are caller-owned. Without this bridge the pre-step
owner allocates them as zeros (so the scan, its cached boundary states, and the
backward grads are all degenerate) AND the chunked grad outputs are w.r.t. the
SSD-core inputs, not the ``mamba3_*_weight`` model params the grad-tree aliases
expect.

FORWARD (``mamba3_projection_forward``)
  Replicates ``Mamba3ReferenceBlock.__call__`` from ``in_proj`` THROUGH the point
  just before the scan dispatch (mamba3.py:728-804), producing the serial scan
  inputs ``x,B,C,z,A,dt,D,h0``. It then maps those serial-convention inputs onto
  the chunked GRID-kernel ABI convention (the kernel folds ``dt`` into the input
  ``inp = dt * x (x) B`` and uses a per-head STATIC ``A`` with ``log_decay =
  A[h]*dt``; OUR serial recurrence uses ``inp = x (x) B`` and a per-timestep
  ``A``). The ABI mapping is EXACT (no approximation): see ``_serial_to_kernel_abi``.

BACKWARD (``mamba3_projection_param_grads``)
  Mirrors ``path_c_prefix_gradient_tree_from_hidden_cotangent``: builds a
  ``(sum of <kernel_abi_input> * <its cotangent>)`` surrogate loss and runs
  ``mx.value_and_grad`` over the mamba block parameters, converting the chunked
  SSD-input cotangents (``dx/dB/dC/dz/dA/ddt/dh0``) into mamba3 param grads
  registered under the EXACT alias names the grad-tree expects.

RULE #1: every path RAISES on a shape/availability mismatch — there is NO silent
fallback and NO aliasing of a mismatched gradient onto the wrong param name.
"""

from __future__ import annotations

from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.mamba3 import (
    Mamba3ReferenceBlock,
    _apply_rope_on_state_dim,
    _broadcast_groups_to_heads,
    _compute_trapezoidal_scale,
    _heads_to_group_scale,
    _rms_norm_last,
    _split_by_sizes,
    causal_depthwise_conv1d,
)


# The kernel-ABI projected-SSD-input suffixes the projection FORWARD produces
# (must match ``_MAMBA3_CHUNKED_PROJECTED_SSD_INPUT_SUFFIXES`` in path_c_fusion
# plus the extra caller-owned ``z`` / ``h0`` the bridge also stages). ``D`` is a
# model param, resolved through the model owner, not produced here.
MAMBA3_KERNEL_ABI_SSD_INPUT_SUFFIXES = ("x", "B", "C", "A", "dt", "z", "h0")

# The kernel-ABI tensors that carry a PARAMETER dependence (so the backward VJP
# accepts a cotangent for them). ``A`` is the per-head STATIC decay the kernel-ABI
# mapping pins to the constant ``-1`` (no parameter dependence: all per-timestep
# decay was folded into ``dt`` = ``-A_s*dt_s``), so it is NOT a backward input —
# the chunked region's ``dlog_decay`` grad (its ``{brick}_dt_grad`` buffer) is for
# that static ``A`` and is intentionally not consumed by the projection VJP.
MAMBA3_KERNEL_ABI_BACKWARD_COTANGENT_KEYS = ("x", "B", "C", "z", "dt", "h0")

# The mamba3 grad-output logical suffixes the projection VJP produces. These are
# the param-grad alias TARGETS (``{brick}_<suffix>``) the model grad-tree reads.
MAMBA3_PROJECTION_PARAM_GRAD_SUFFIXES = (
    "mamba3_in_proj_weight",
    "mamba3_out_proj_weight",
    "mamba3_conv_weight",
    "mamba3_conv_bias",
    "mamba3_dt_bias",
    "mamba3_B_norm_weight",
    "mamba3_C_norm_weight",
    "mamba3_B_bias",
    "mamba3_C_bias",
)

# The mamba3 ``D`` skip param grad is produced DIRECTLY by the chunked region (the
# B2 ``dD`` output, written to ``{brick}_D_grad``), but the model grad-tree alias
# expects the target ``{brick}_mamba3_D_grad`` (with the ``mamba3_`` infix). The
# fold bridges that single rename so coverage completes for ``mamba3_D`` too.
MAMBA3_CHUNKED_D_GRAD_SUFFIX = "D"
MAMBA3_PARAM_D_GRAD_SUFFIX = "mamba3_D"

# Map an MLX mamba block parameter tree name -> its logical grad suffix.
_BLOCK_PARAM_TO_LOGICAL_SUFFIX = {
    "in_proj.weight": "mamba3_in_proj_weight",
    "out_proj.weight": "mamba3_out_proj_weight",
    "conv_weight": "mamba3_conv_weight",
    "conv_bias": "mamba3_conv_bias",
    "dt_bias": "mamba3_dt_bias",
    "B_norm_weight": "mamba3_B_norm_weight",
    "C_norm_weight": "mamba3_C_norm_weight",
    "B_bias": "mamba3_B_bias",
    "C_bias": "mamba3_C_bias",
}


def _entry_normed_hidden(
    hidden: mx.array,
    entry_norm_weight: mx.array | None,
    *,
    eps: float,
) -> mx.array:
    """Apply the brick entry RMSNorm to the raw residual hidden.

    The mamba brick reads ``{brick}_hidden`` as the RAW residual baseline and
    applies its own entry RMSNorm (bound to ``layers.{i}.norm.weight``) to derive
    the projection input. When ``entry_norm_weight`` is ``None`` the caller has
    already supplied the normed hidden (no double-norm).
    """

    if entry_norm_weight is None:
        return hidden
    normed = _rms_norm_last(hidden, eps=eps)
    return normed * entry_norm_weight.astype(normed.dtype)


def _mamba3_projection_serial_inputs(
    block: Mamba3ReferenceBlock,
    hidden: mx.array,
    *,
    entry_norm_weight: mx.array | None,
    entry_norm_eps: float,
) -> dict[str, mx.array]:
    """Projection-only forward: residual hidden -> serial scan inputs.

    Replicates ``Mamba3ReferenceBlock.__call__`` (mamba3.py:728-804) from the
    entry RMSNorm + ``in_proj`` THROUGH (but stopping before) the scan dispatch.
    Returns the serial-convention ``{x,B,C,z,A,dt,h0}`` (D is a model param). Every
    op here is MLX-differentiable so the VJP can flow to the block params.
    """

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be shaped (B,S,D), got {hidden.shape}")
    cfg = block.config
    dims = block.dims
    batch, seq, _ = hidden.shape

    route_input = _entry_normed_hidden(
        hidden, entry_norm_weight, eps=entry_norm_eps
    )

    z, x, B, C, dd_dt, dd_A, trap, angles = block.split_in_proj(
        block.in_proj(route_input)
    )

    xBC = mx.concatenate([x, B, C], axis=-1)
    xBC = causal_depthwise_conv1d(
        xBC,
        block.conv_weight.astype(xBC.dtype),
        block.conv_bias.astype(xBC.dtype),
    )
    x, B, C = _split_by_sizes(nn.silu(xBC), [cfg.d_inner, dims.d_bc, dims.d_bc])
    x = x.reshape(batch, seq, cfg.nheads, cfg.headdim)
    z = z.reshape(batch, seq, cfg.nheads, cfg.headdim)

    B_mimo = B.reshape(batch, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    C_mimo = C.reshape(batch, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    B_mimo, C_mimo = block.transform_bc(B_mimo, C_mimo)
    B = mx.mean(B_mimo, axis=2)
    C = mx.mean(C_mimo, axis=2)

    dt = nn.softplus(dd_dt + block.dt_bias.astype(dd_dt.dtype))
    trap_scale = _compute_trapezoidal_scale(dt, trap)
    B = B * _heads_to_group_scale(trap_scale, cfg.ngroups)[:, :, :, None]

    angles = mx.broadcast_to(
        angles[:, :, None, :],
        (batch, seq, cfg.nheads, dims.num_rope_angles),
    )
    angles_cumsum = mx.cumsum(angles * dt[:, :, :, None], axis=1)
    B = _apply_rope_on_state_dim(B, angles_cumsum)
    C = _apply_rope_on_state_dim(C, angles_cumsum)
    B = _broadcast_groups_to_heads(B, cfg.nheads, "B")
    C = _broadcast_groups_to_heads(C, cfg.nheads, "C")

    A = mx.minimum(-nn.softplus(dd_A), -cfg.A_floor)

    h0 = block.initial_h0(batch, hidden.dtype)

    return {"x": x, "B": B, "C": C, "z": z, "A": A, "dt": dt, "h0": h0}


def _serial_to_kernel_abi(
    serial: Mapping[str, mx.array],
    *,
    nheads: int,
) -> dict[str, mx.array]:
    """Map serial-convention scan inputs onto the chunked GRID-kernel ABI.

    OUR serial recurrence (``_chunked_mamba3_diagonal_scan`` / ``_reference_scan``):
        log_decay[t] = A_s[t] * dt_s[t]          (A_s per-timestep (B,S,H), <0)
        inp[t]       = x_s[t] (x) B_s[t]         (NO dt on the input)
        y[t]         = sum(h[t] * C_s[t]) + D*x_s[t]

    The chunked GRID kernel (chunk_precompute / scan_combine) ABI:
        A_k : (H,) STATIC per-head   dt_k : (B,S,H)
        log_decay[t] = A_k[h] * dt_k[t]
        inp[t]       = dt_k[t] * x_k[t] (x) B_k[t]   (dt folded INTO input)
        y[t]         = sum(h[t] * C_k[t]) + D*x_k[t]

    EXACT mapping (no approximation):
        A_k[h]  = -1                         (static, <= 0)
        dt_k[t] = -A_s[t] * dt_s[t]  (> 0)   => A_k*dt_k = A_s*dt_s   (decay identical)
        x_k     = x_s                        (=> D*x_k = D*x_s skip identical)
        B_k[t]  = B_s[t] / dt_k[t]           (=> dt_k * x_k (x) B_k = x_s (x) B_s)
        C_k     = C_s ;  z_k = z_s ;  h0_k = h0_s

    ``dt_k`` is strictly positive: ``A_s = min(-softplus, -A_floor) <= -A_floor < 0``
    and ``dt_s = softplus(...) > 0``, so ``dt_k = -A_s*dt_s > 0`` and the division
    by ``dt_k`` is well-defined (RULE #1: never a clamp / silent guard).
    """

    x_s = serial["x"]
    B_s = serial["B"]
    C_s = serial["C"]
    z_s = serial["z"]
    A_s = serial["A"]
    dt_s = serial["dt"]
    h0_s = serial["h0"]

    dt_k = (-A_s) * dt_s  # (B,S,H), strictly > 0
    A_k = -mx.ones((nheads,), dtype=A_s.dtype)
    B_k = B_s / dt_k[:, :, :, None]

    return {
        "x": x_s,
        "B": B_k,
        "C": C_s,
        "z": z_s,
        "A": A_k,
        "dt": dt_k,
        "h0": h0_s,
    }


def mamba3_projection_forward(
    block: Mamba3ReferenceBlock,
    hidden: mx.array,
    *,
    entry_norm_weight: mx.array | None = None,
    entry_norm_eps: float = 1e-5,
) -> dict[str, mx.array]:
    """Full projection-only forward in the chunked GRID-kernel ABI convention.

    Returns ``{x,B,C,A,dt,z,h0}`` already mapped to the chunked kernel ABI so
    feeding them to F0/F1/F2 reproduces OUR serial scan output exactly. ``D`` is
    a model parameter (resolved via the model owner), not produced here.
    """

    serial = _mamba3_projection_serial_inputs(
        block,
        hidden,
        entry_norm_weight=entry_norm_weight,
        entry_norm_eps=entry_norm_eps,
    )
    return _serial_to_kernel_abi(serial, nheads=block.config.nheads)


def mamba3_projection_param_grads(
    block: Mamba3ReferenceBlock,
    hidden: mx.array,
    ssd_input_cotangents: Mapping[str, mx.array],
    *,
    entry_norm_weight: mx.array | None = None,
    entry_norm_eps: float = 1e-5,
) -> dict[str, mx.array]:
    """VJP the chunked SSD-input cotangents through the projection -> param grads.

    ``ssd_input_cotangents`` maps a subset of
    ``{x,B,C,A,dt,z,h0}`` (the chunked region's KERNEL-ABI scan-input grad
    buffers) to their cotangent arrays. This builds the surrogate
    ``loss = sum_k sum(kernel_abi[k] * cotangent[k])`` and differentiates it
    w.r.t. the mamba block parameters. Returns a dict keyed by logical grad
    suffix (``mamba3_in_proj_weight`` etc.) WITHOUT the trailing ``_grad`` — the
    caller prefixes the brick name and appends ``_grad``.

    RULE #1: an unknown cotangent key or a cotangent whose shape does not match
    the produced kernel-ABI tensor RAISES (no silent skip / broadcast).
    """

    cotangents = {str(k): v for k, v in ssd_input_cotangents.items()}
    for key in cotangents:
        if key not in MAMBA3_KERNEL_ABI_BACKWARD_COTANGENT_KEYS:
            raise ValueError(
                f"unknown mamba3 SSD-input cotangent {key!r}; expected one of "
                f"{MAMBA3_KERNEL_ABI_BACKWARD_COTANGENT_KEYS}"
            )
    if not cotangents:
        raise ValueError(
            "mamba3_projection_param_grads requires at least one SSD-input cotangent"
        )

    nheads = block.config.nheads

    def surrogate_loss(blk: Mamba3ReferenceBlock) -> mx.array:
        kernel = mamba3_projection_forward(
            blk,
            hidden,
            entry_norm_weight=entry_norm_weight,
            entry_norm_eps=entry_norm_eps,
        )
        loss = mx.array(0.0, dtype=mx.float32)
        for key, cot in cotangents.items():
            produced = kernel[key]
            if tuple(produced.shape) != tuple(cot.shape):
                raise ValueError(
                    f"mamba3 SSD-input cotangent {key!r} shape {tuple(cot.shape)} "
                    f"must match produced kernel-ABI tensor shape "
                    f"{tuple(produced.shape)}"
                )
            loss = loss + (
                produced.astype(mx.float32) * cot.astype(mx.float32)
            ).sum()
        return loss

    grad_fn = nn.value_and_grad(block, surrogate_loss)
    _loss, grad_tree = grad_fn(block)

    from mlx.utils import tree_flatten

    flat = {str(name): value for name, value in tree_flatten(grad_tree)}
    out: dict[str, mx.array] = {}
    for param_name, logical_suffix in _BLOCK_PARAM_TO_LOGICAL_SUFFIX.items():
        value = flat.get(param_name)
        if value is None:
            # The block does not expose this param (e.g. a config without B_bias);
            # skip — the grad-tree alias for it is then absent too.
            continue
        if not isinstance(value, mx.array):
            raise TypeError(
                f"mamba3 projection grad for {param_name!r} is not an mx.array"
            )
        out[logical_suffix] = value
    return out


def mamba3_brick_blocks_for_chain(
    model: Any,
) -> dict[str, tuple[Mamba3ReferenceBlock, int]]:
    """Map every mamba3 brick logical NAME -> (block, layer_index).

    The logical name(s) are exactly the prefixes the grad-tree aliases use:
    ``block.path_c_brick_name`` and ``block.path_c_profile_brick_name`` (both, so
    either alias-target resolves). RULE #1: a mamba layer without a resolvable
    brick name RAISES (the bridge must not silently miss a mamba brick).
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("model has no layers; cannot resolve mamba3 bricks")
    out: dict[str, tuple[Mamba3ReferenceBlock, int]] = {}
    for index, block in enumerate(layers):
        if getattr(block, "backend", None) != "mamba3":
            continue
        inner = getattr(block, "block", None)
        if not isinstance(inner, Mamba3ReferenceBlock):
            raise TypeError(
                f"layer {index} backend=mamba3 but block is "
                f"{type(inner).__name__}, expected Mamba3ReferenceBlock"
            )
        names = []
        brick_name = getattr(block, "path_c_brick_name", None)
        profile_name = getattr(block, "path_c_profile_brick_name", None)
        for candidate in (brick_name, profile_name):
            if candidate is not None and str(candidate) not in names:
                names.append(str(candidate))
        if not names:
            raise ValueError(
                f"mamba3 layer {index} exposes no path_c brick name"
            )
        for name in names:
            out[name] = (inner, index)
    return out
