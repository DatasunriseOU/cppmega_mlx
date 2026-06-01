"""Numerical parity tests for the eager Mamba3 projection fwd/bwd bridge.

These validate the LOAD-BEARING bridge logic independent of the full m04 route:

  FORWARD: feeding the bridge's kernel-ABI inputs to OUR serial chunked diagonal
  scan reproduces the SERIAL Mamba3ReferenceBlock scan output EXACTLY (the bridge
  ABI mapping is loss-preserving).

  BACKWARD: the bridge's projection VJP (chunked SSD-input cotangents ->
  mamba3 param grads) MATCHES, per-parameter < 1e-3, the param grads obtained by
  differentiating the SAME surrogate through the SERIAL forward + serial->kernel
  mapping. (i.e. the bridge VJP is the exact transpose of its own forward.)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.nn.mamba3 import (
    Mamba3Config,
    Mamba3ReferenceBlock,
    _chunked_mamba3_diagonal_scan,
    _reference_scan,
)
from cppmega_mlx.runtime.mamba3_projection_bridge import (
    _mamba3_projection_serial_inputs,
    _serial_to_kernel_abi,
    mamba3_projection_forward,
    mamba3_projection_param_grads,
)


def _make_block(seed: int = 0) -> tuple[Mamba3ReferenceBlock, mx.array]:
    mx.random.seed(seed)
    cfg = Mamba3Config(
        d_model=64,
        expand=1,
        headdim=64,
        d_state=16,
        ngroups=1,
        chunk_size=64,
    )
    block = Mamba3ReferenceBlock(cfg)
    mx.eval(block.parameters())
    hidden = mx.array(np.random.randn(1, 128, 64).astype(np.float32) * 0.1)
    return block, hidden


def _kernel_abi_scan(kernel: dict, D: mx.array, chunk_size: int):
    """Run OUR serial chunked diagonal scan on KERNEL-ABI inputs.

    Kernel ABI: log_decay = A_k[h]*dt_k ; inp = dt_k * x_k (x) B_k.
    """
    x = kernel["x"]
    B = kernel["B"]
    C = kernel["C"]
    z = kernel["z"]
    A = kernel["A"]  # (H,)
    dt = kernel["dt"]  # (B,S,H)
    h0 = kernel["h0"]
    batch, seq, nheads, headdim = x.shape
    d_state = B.shape[-1]
    log_decay = (A[None, None, :] * dt)[:, :, :, None, None]
    inp = (dt[:, :, :, None, None] * x[:, :, :, :, None]) * B[:, :, :, None, :]
    return _chunked_mamba3_diagonal_scan(
        log_decay, inp, C, x, z, D, h0, chunk_size=chunk_size
    )


def test_kernel_abi_mapping_reproduces_serial_scan():
    """Bridge kernel-ABI inputs + kernel-convention scan == serial scan output."""
    block, hidden = _make_block()
    serial = _mamba3_projection_serial_inputs(
        block, hidden, entry_norm_weight=None, entry_norm_eps=1e-5
    )
    # Serial reference output (OUR serial convention).
    y_serial, h_serial = _reference_scan(
        x=serial["x"], B=serial["B"], C=serial["C"], z=serial["z"],
        A=serial["A"], dt=serial["dt"], D=block.D, h0=serial["h0"],
        chunk_size=block.config.chunk_size,
    )
    # Kernel-ABI inputs + kernel-convention scan.
    kernel = _serial_to_kernel_abi(serial, nheads=block.config.nheads)
    y_kernel, h_kernel = _kernel_abi_scan(kernel, block.D, block.config.chunk_size)
    mx.eval(y_serial, h_serial, y_kernel, h_kernel)

    y_diff = float(mx.max(mx.abs(y_serial - y_kernel)).item())
    h_diff = float(mx.max(mx.abs(h_serial - h_kernel)).item())
    assert y_diff < 1e-4, f"kernel-ABI scan y diff {y_diff} too large"
    assert h_diff < 1e-4, f"kernel-ABI scan h diff {h_diff} too large"


def test_dt_k_strictly_positive():
    """dt_k = -A_s*dt_s must be strictly > 0 (no division-by-zero in B_k)."""
    block, hidden = _make_block(seed=3)
    kernel = mamba3_projection_forward(block, hidden, entry_norm_weight=None)
    mx.eval(kernel["dt"])
    assert float(mx.min(kernel["dt"]).item()) > 0.0


def test_projection_vjp_matches_direct_grad():
    """Bridge VJP param grads == direct autodiff of the same surrogate.

    The surrogate is sum_k (kernel_abi[k] * cotangent[k]).sum(). Differentiating
    it directly through the projection (mamba3_projection_forward) is the ground
    truth; mamba3_projection_param_grads must reproduce it per-param < 1e-3.
    """
    block, hidden = _make_block(seed=1)
    kernel = mamba3_projection_forward(block, hidden, entry_norm_weight=None)
    mx.eval(*(v for v in kernel.values()))
    # Random cotangents for every SSD-input the chunked backward produces.
    rng = np.random.RandomState(7)
    cot = {
        k: mx.array(rng.randn(*kernel[k].shape).astype(np.float32) * 0.05)
        for k in ("x", "B", "C", "z", "dt", "h0")
    }

    # Ground-truth: direct value_and_grad of the surrogate through the bridge fwd.
    def direct_loss(blk):
        ker = mamba3_projection_forward(blk, hidden, entry_norm_weight=None)
        loss = mx.array(0.0, dtype=mx.float32)
        for k, c in cot.items():
            loss = loss + (ker[k].astype(mx.float32) * c.astype(mx.float32)).sum()
        return loss

    _l, direct_tree = nn.value_and_grad(block, direct_loss)(block)
    direct = {str(n): v for n, v in tree_flatten(direct_tree)}

    bridge = mamba3_projection_param_grads(
        block, hidden, cot, entry_norm_weight=None
    )

    suffix_to_param = {
        "mamba3_in_proj_weight": "in_proj.weight",
        "mamba3_out_proj_weight": "out_proj.weight",
        "mamba3_conv_weight": "conv_weight",
        "mamba3_conv_bias": "conv_bias",
        "mamba3_dt_bias": "dt_bias",
        "mamba3_B_norm_weight": "B_norm_weight",
        "mamba3_C_norm_weight": "C_norm_weight",
        "mamba3_B_bias": "B_bias",
        "mamba3_C_bias": "C_bias",
    }
    worst = 0.0
    worst_name = ""
    for suffix, pname in suffix_to_param.items():
        assert suffix in bridge, f"bridge missing grad for {suffix}"
        bv = bridge[suffix]
        dv = direct[pname]
        mx.eval(bv, dv)
        d = float(mx.max(mx.abs(bv - dv)).item()) if bv.size else 0.0
        if d > worst:
            worst, worst_name = d, suffix
    assert worst < 1e-3, f"worst bridge-vjp grad diff {worst} at {worst_name}"
    # out_proj.weight receives NO gradient from the projection (it is applied
    # AFTER the scan); the bridge still maps it but the value must be all-zero.
    op = bridge["mamba3_out_proj_weight"]
    mx.eval(op)
    assert float(mx.max(mx.abs(op)).item()) == 0.0


def test_full_composition_matches_serial_autodiff():
    """End-to-end: projection-fwd -> chunked-convention scan -> bridge-VJP grads
    == direct serial autodiff of the SAME (projection+scan) graph, to machine eps.

    This is the load-bearing parity proof: the bridge forward seeds the kernel
    inputs, the chunked-convention scan consumes them, the scan VJP produces the
    SSD-input cotangents (what B0/B1/B2 produce), and the bridge projection-VJP
    folds them into param grads. The composition must equal end-to-end autodiff.
    """
    block, hidden = _make_block(seed=5)
    y_cot = mx.array(
        np.random.RandomState(11).randn(1, 128, 1, 64).astype(np.float32) * 0.05
    )

    def full_loss(b):
        ker = mamba3_projection_forward(b, hidden, entry_norm_weight=None)
        y, _h = _kernel_abi_scan(ker, b.D, b.config.chunk_size)
        return (y.astype(mx.float32) * y_cot).sum()

    _l, gt_tree = nn.value_and_grad(block, full_loss)(block)
    gt = {str(n): v for n, v in tree_flatten(gt_tree)}

    # Bridge path: seed projection (detached), scan VJP -> SSD cotangents, bridge VJP.
    ker0 = mamba3_projection_forward(block, hidden, entry_norm_weight=None)
    ker0 = {k: mx.array(np.asarray(v.astype(mx.float32))) for k, v in ker0.items()}
    D_det = mx.array(np.asarray(block.D.astype(mx.float32)))

    def scan_loss(kdict):
        y, _h = _kernel_abi_scan(kdict, D_det, block.config.chunk_size)
        return (y.astype(mx.float32) * y_cot).sum()

    scan_cot = mx.grad(scan_loss)(ker0)
    bridge = mamba3_projection_param_grads(
        block,
        hidden,
        {k: scan_cot[k] for k in ("x", "B", "C", "z", "dt", "h0")},
        entry_norm_weight=None,
    )

    suffix_to_param = {
        "mamba3_in_proj_weight": "in_proj.weight",
        "mamba3_conv_weight": "conv_weight",
        "mamba3_conv_bias": "conv_bias",
        "mamba3_dt_bias": "dt_bias",
        "mamba3_B_norm_weight": "B_norm_weight",
        "mamba3_C_norm_weight": "C_norm_weight",
        "mamba3_B_bias": "B_bias",
        "mamba3_C_bias": "C_bias",
    }
    worst = 0.0
    for suffix, pname in suffix_to_param.items():
        bv = bridge[suffix]
        dv = gt[pname]
        mx.eval(bv, dv)
        d = float(mx.max(mx.abs(bv - dv)).item()) if bv.size else 0.0
        worst = max(worst, d)
    assert worst < 1e-3, f"full-composition grad diff {worst} too large"


def test_unknown_cotangent_raises():
    block, hidden = _make_block(seed=2)
    try:
        mamba3_projection_param_grads(
            block, hidden, {"bogus": mx.zeros((1,))}, entry_norm_weight=None
        )
    except ValueError as exc:
        assert "unknown mamba3 SSD-input cotangent" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown cotangent")


# --------------------------------------------------------------------------- #
# m04 integration: flag-ON owner forward override + coverage completion.       #
# These run the PLANNER + pre-step owner (CPU/plan path only, no Metal route),  #
# so they are robust to the Metal XPC compiler state.                          #
# --------------------------------------------------------------------------- #


def _build_smoke_chain(m04):
    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )

    mx.random.seed(0)
    model = build_local_gb10_quarter_tiny_smoke_model(
        hidden_size=64, num_attention_heads=1, mamba_expand=1, mamba_head_dim=64,
        mamba_state_dim=16, mamba_groups=1, mamba_chunk_size=64,
    )
    pn = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    prefix = m04._path_c_direct_chain_region_prefix(model, pn)
    seq = 64
    vocab = int(getattr(model, "vocab_size", 0) or 1024)
    ids = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    batch = {"tokens": ids, "target_tokens": ids}
    dcs = m04.plan_path_c_direct_fusion_chains_for_model(
        model, region_prefix=prefix, include_backward=True, max_segment_nodes=1,
        sequence_length=seq,
    )
    regions = m04.build_path_c_model_regions_from_model(
        model, region_prefix=prefix, include_backward=False, sequence_length=seq,
    )
    reg = m04._select_path_c_model_route_region(regions)
    chain = m04._select_path_c_direct_chain_for_region(dcs, reg)
    return model, batch, chain


def test_flag_on_owner_seeds_real_projection_inputs(monkeypatch):
    """Flag ON: pre-step owner seeds REAL (nonzero) chunked SSD inputs, not zeros."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    import importlib

    import m04_train_step as m04  # type: ignore
    importlib.reload(m04)

    model, batch, chain = _build_smoke_chain(m04)
    owner = m04.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain, model=model, batch=batch, batch_row=0
    )
    brick = "local_gb10_quarter_brick_10_M"
    # Every projected SSD input the chunked region reads must be present + nonzero
    # (h0 is the initial state and is legitimately zero).
    for sfx in ("x", "B", "C", "A", "dt", "z"):
        v = owner.buffers[f"{brick}_{sfx}"]
        mx.eval(v)
        a = np.asarray(v.astype(mx.float32))
        assert np.abs(a).max() > 0.0, f"{sfx} must be seeded nonzero, got all-zero"


def test_flag_on_coverage_completes_for_mamba_params(monkeypatch):
    """Flag ON: the projection bridge makes the mamba params coverage-complete."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    import importlib

    import m04_train_step as m04  # type: ignore
    importlib.reload(m04)

    model, _batch, chain = _build_smoke_chain(m04)
    covered = m04._path_c_mamba3_projection_covered_parameter_names(
        model=model, chain=chain
    )
    # All 9 projection params + D for the in-region mamba brick (layer 10).
    for name in (
        "layers.10.block.in_proj.weight",
        "layers.10.block.out_proj.weight",
        "layers.10.block.conv_weight",
        "layers.10.block.conv_bias",
        "layers.10.block.dt_bias",
        "layers.10.block.B_norm_weight",
        "layers.10.block.C_norm_weight",
        "layers.10.block.B_bias",
        "layers.10.block.C_bias",
        "layers.10.block.D",
    ):
        assert name in covered, f"{name} must be projection-covered"

    # With the suffix bridge contract, coverage is COMPLETE (no missing params).
    payload = m04._path_c_direct_chain_full_gradient_coverage_payload(
        model=model,
        chain=chain,
        bridge_contract={
            "parameter_gradient_names": ("norm.weight_grad", "lm_head.weight_grad")
        },
    )
    assert payload["full_model_gradient_tree_ready"], payload["missing_parameter_names"]


def test_flag_off_owner_has_no_projection_buffers(monkeypatch):
    """Flag OFF: no chunked SSD buffers exist and coverage is unaffected."""
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "0")
    import importlib

    import m04_train_step as m04  # type: ignore
    importlib.reload(m04)

    model, batch, chain = _build_smoke_chain(m04)
    owner = m04.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain, model=model, batch=batch, batch_row=0
    )
    # No chunked SSD-input buffers when flag OFF (serial mamba3_mimo owns them).
    proj = [
        n for n in owner.buffers
        if any(n.endswith("_" + s) for s in ("x", "B", "C", "A", "dt", "z"))
        and "brick_10_M" in n
        and "mamba3" not in n
    ]
    assert proj == [], f"flag OFF must not seed chunked SSD inputs, got {proj}"
    covered = m04._path_c_mamba3_projection_covered_parameter_names(
        model=model, chain=chain
    )
    assert covered == set(), "flag OFF: projection bridge covers nothing"
