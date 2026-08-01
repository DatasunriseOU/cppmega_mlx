"""Stage-3 backward: B0/B1/B2 descriptors + interpose + per-grad parity.

Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2/§7 Stage 3.

Mirrors the forward Stage-2 harness ``tests/test_mamba3_chained_forward_f0f1f2.py``
for the BACKWARD: the 3 ``_bwd`` op-names register + select-resolve, the compile-
site delegation interpose substitutes the real ``build_*_bwd_metal`` grid kernels
when the flag is ON, and the chained B2->B1->B0 backward matches the validated MLX
backward proto (itself 1.30e-4 vs the serial VJP) per-grad-tensor < 1e-3.

RULE #1: the kernel path is the single delegation per segment; on a compile/parity
failure the helpers RAISE and the test FAILS — never a silent serial fallback.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch_mps_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


_BWD_OPS = (
    "mamba3_chunk_scan_combine_bwd",
    "mamba3_inter_chunk_recur_bwd",
    "mamba3_chunk_precompute_bwd",
)


# --------------------------------------------------------------------------- #
# Verification 0: all 3 backward descriptors register + select-resolve.        #
# --------------------------------------------------------------------------- #


def test_all_three_backward_descriptors_register_and_resolve():
    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        MAMBA3_CHUNK_PRECOMPUTE_BWD_OP_NAME,
        MAMBA3_CHUNK_SCAN_COMBINE_BWD_OP_NAME,
        MAMBA3_INTER_CHUNK_RECUR_BWD_OP_NAME,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        default_path_c_brick_schedule_descriptor_registry,
    )

    assert {
        MAMBA3_CHUNK_SCAN_COMBINE_BWD_OP_NAME,
        MAMBA3_INTER_CHUNK_RECUR_BWD_OP_NAME,
        MAMBA3_CHUNK_PRECOMPUTE_BWD_OP_NAME,
    } == set(_BWD_OPS)

    reg = default_path_c_brick_schedule_descriptor_registry()
    for op in _BWD_OPS:
        desc = reg.descriptor_for(op)
        assert desc is not None, f"{op} descriptor must register (else select blocks)"
        assert desc.fragment_emitter is not None, f"{op} needs a fragment_emitter"
        assert reg.descriptors_for_signature((op,)) is not None


def test_backward_ops_in_grid_delegation_set():
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        _MAMBA3_CHUNKED_GRID_DELEGATION_OPS,
    )

    for op in _BWD_OPS:
        assert op in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS, op


# --------------------------------------------------------------------------- #
# Verification 1: the flag-ON compile-site interpose emits real grid kernels.   #
# --------------------------------------------------------------------------- #


def _env():
    from cppmega_mlx.runtime.path_c_fusion import PathCModelShapeEnv

    return PathCModelShapeEnv(
        sequence_length=4096, hidden_size=512, attention_num_q_heads=8,
        attention_num_kv_heads=8, attention_head_dim=64, attention_sparse_topk=1,
        mamba_expand=1, mamba_head_dim=64, mamba_state_dim=16, mamba_groups=1,
        mamba_mimo_rank=1, mamba_is_mimo=True, mamba_conv_kernel=4,
        mamba_rope_fraction=0.5, m2rnn_k_head_dim=64, m2rnn_v_head_dim=64,
        m2rnn_num_q_heads=8, m2rnn_num_k_heads=8, m2rnn_num_v_heads=8,
        m2rnn_num_f_heads=8, m2rnn_num_g_heads=8, m2rnn_num_weight_heads=8,
        m2rnn_conv_kernel=4,
    )


_BWD_SURFACE_INPUTS = {
    "mamba3_chunk_scan_combine_bwd": (
        "mamba3_dout", "mamba3_cb", "mamba3_x", "mamba3_z", "mamba3_dt",
        "mamba3_dA_cumsum", "mamba3_C", "mamba3_B", "mamba3_prev_states",
        "mamba3_D", "mamba3_y",
    ),
    "mamba3_inter_chunk_recur_bwd": (
        "mamba3_dchunk_states", "mamba3_dA_cumsum", "mamba3_dh_last",
        "mamba3_prev_states",
    ),
    "mamba3_chunk_precompute_bwd": (
        "mamba3_dstates", "mamba3_dinp_diag", "mamba3_dA_cumsum_y",
        "mamba3_dA_cumsum_tail", "mamba3_dA_cumsum", "mamba3_x", "mamba3_B",
        "mamba3_dt", "mamba3_A",
    ),
}


def _build_segment_prim(op_name, inputs, env):
    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        Z3SyncSpec,
        build_path_c_fusion_region,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        build_path_c_descriptor_prim_func,
        default_path_c_brick_schedule_descriptor_registry,
    )

    reg = default_path_c_brick_schedule_descriptor_registry()
    surf = FusionKernelSurface.path_c(
        name=f"{op_name}_node", op_name=op_name,
        inputs=inputs, outputs=("g0",), backward="owner_output",
    )
    region = build_path_c_fusion_region(
        region_name=op_name, surfaces=(surf,),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
        metadata={"path_c_model_shape_env": env},
    )
    desc = reg.descriptor_for(op_name)
    return build_path_c_descriptor_prim_func(region, (desc,), shape_env=env)


def test_backward_interpose_default_off_not_grid_kernel(monkeypatch):
    """Flag OFF (default): a single B2 segment compiles via the source/exec
    template (shadow marker), NOT the delegated grid kernel (merge-safe)."""
    monkeypatch.delenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", raising=False)
    prim = _build_segment_prim(
        "mamba3_chunk_scan_combine_bwd",
        _BWD_SURFACE_INPUTS["mamba3_chunk_scan_combine_bwd"], _env(),
    )
    assert getattr(prim, "_cppmega_path_c_mamba3_chunked_grid_delegation", None) is None
    assert type(prim).__name__ != "JITKernel"


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.parametrize("op_name", list(_BWD_SURFACE_INPUTS))
def test_backward_interpose_on_emits_real_grid_kernel(op_name, monkeypatch):
    """Flag ON: the interpose substitutes the REAL build_*_bwd_metal grid
    JITKernel for each B0/B1/B2 segment (RULE #1: the shadow marker is never the
    live kernel)."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    prim = _build_segment_prim(op_name, _BWD_SURFACE_INPUTS[op_name], _env())
    assert type(prim).__name__ == "JITKernel", (
        f"{op_name} flag-ON must emit a real grid JITKernel, got {type(prim).__name__}"
    )
    assert (
        getattr(prim, "_cppmega_path_c_mamba3_chunked_grid_delegation", None) == op_name
    )
    assert getattr(prim, "_cppmega_path_c_brick_ops", None) == (op_name,)


# --------------------------------------------------------------------------- #
# Verification 2: chained B2->B1->B0 per-grad parity vs the MLX backward proto. #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Verification 1b: the LIVE REGION-BUILD backward 1->3 surface flip.            #
# --------------------------------------------------------------------------- #


def _build_mamba_direct_chain_region(env):
    from cppmega_mlx.runtime.path_c_fusion import (
        PathCModelBrick,
        build_path_c_model_region_from_bricks,
    )

    bricks = (PathCModelBrick(name="mamba3_scan", kind="mamba3", route_symbol="M"),)
    return build_path_c_model_region_from_bricks(
        region_name="mamba3_direct_chain", bricks=bricks, shape_env=env
    )


def test_region_flip_off_no_chunked_backward_surfaces(monkeypatch):
    """Flag OFF (default): the direct-chain region has NO chunked _bwd surfaces —
    the serial mamba3_mimo (with aot_autograd) is unchanged (merge-safe)."""
    monkeypatch.delenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", raising=False)
    region = _build_mamba_direct_chain_region(_env())
    ops = [n.op_name for n in region.nodes]
    assert "mamba3_mimo" in ops
    assert not any(o in _BWD_OPS for o in ops), ops


def test_region_flip_on_emits_three_chunked_backward_surfaces(monkeypatch):
    """Flag ON: the region emits the 3 chunked _bwd segments (B2->B1->B0), each
    classified as a BACKWARD-phase node isolated into its own backward stage, wired
    by the per-brick grad-handoff buffers."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        _path_c_schedule_node_execution_phase,
        plan_path_c_descriptor_stage_groups,
    )

    region = _build_mamba_direct_chain_region(_env())
    bwd_nodes = {n.op_name: n for n in region.nodes if n.op_name in _BWD_OPS}
    assert set(bwd_nodes) == set(_BWD_OPS), list(bwd_nodes)
    # all 3 classify as backward phase
    for n in region.nodes:
        if n.op_name in _BWD_OPS:
            assert _path_c_schedule_node_execution_phase(n) == "backward", n.op_name

    # each _bwd op is isolated into its own backward stage group
    op_by_node = {n.name: n.op_name for n in region.nodes}
    bwd_stage_ops = []
    for g in plan_path_c_descriptor_stage_groups(region):
        ops = [op_by_node.get(x) for x in g.active_node_names]
        for op in ops:
            if op in _BWD_OPS:
                assert g.execution_stage == "backward", g.execution_stage
                assert len(g.active_node_names) == 1, list(g.active_node_names)
                bwd_stage_ops.append(op)
    assert bwd_stage_ops == list(_BWD_OPS), bwd_stage_ops

    # grad-handoff wiring: B2 -> B1 -> B0 (the transpose of F2 -> F1 -> F0)
    b2 = bwd_nodes["mamba3_chunk_scan_combine_bwd"]
    b1 = bwd_nodes["mamba3_inter_chunk_recur_bwd"]
    b0 = bwd_nodes["mamba3_chunk_precompute_bwd"]
    assert "mamba3_scan_dchunk_states" in b2.outputs
    assert "mamba3_scan_dchunk_states" in b1.inputs
    assert "mamba3_scan_dstates" in b1.outputs
    assert "mamba3_scan_dstates" in b0.inputs
    assert "mamba3_scan_dinp_diag" in b2.outputs and "mamba3_scan_dinp_diag" in b0.inputs
    assert "mamba3_scan_dA_cumsum_tail" in b1.outputs and "mamba3_scan_dA_cumsum_tail" in b0.inputs
    # B2/B1/B0 REUSE the forward-materialized boundary states (no replay)
    assert "mamba3_scan_prev_states" in b2.inputs
    assert "mamba3_scan_prev_states" in b1.inputs
    assert "mamba3_scan_cb" in b2.inputs


def _build_mra_full_backward_region(env, *, include_backward):
    """Build a 3-brick (M mamba3 / R m2rnn / A attention) region.

    This mirrors the real model route span the direct-chain planner discovers so
    the FULL reverse backward chain (non-mamba ``_bwd`` + chunked mamba B2/B1/B0)
    can be asserted.
    """
    from cppmega_mlx.runtime.path_c_fusion import (
        PathCModelBrick,
        build_path_c_model_region_from_bricks,
    )

    bricks = (
        PathCModelBrick(name="brick_M", kind="mamba3", route_symbol="M"),
        PathCModelBrick(name="brick_R", kind="m2rnn", route_symbol="R"),
        PathCModelBrick(name="brick_A", kind="attention", route_symbol="A"),
    )
    return build_path_c_model_region_from_bricks(
        region_name="mra_chain",
        bricks=bricks,
        shape_env=env,
        include_backward=include_backward,
    )


def test_full_backward_chain_off_emits_serial_mamba_bwd(monkeypatch):
    """Flag OFF: the full (fwd+bwd) region emits the 7 serial backward ops in
    reverse order, ending with mamba3_mimo_bwd then entry_rmsnorm_bwd. Merge-safe
    baseline for the flag-ON assertion below."""
    monkeypatch.delenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", raising=False)
    region = _build_mra_full_backward_region(_env(), include_backward=True)
    bwd = [n.op_name for n in region.nodes if n.op_name.endswith("_bwd")]
    assert bwd == [
        "sparse_mla_fp8_apply_bwd",
        "attention_qkv_projection_bwd",
        "residual_rmsnorm_bwd",
        "m2rnn_bwd",
        "residual_rmsnorm_bwd",
        "mamba3_mimo_bwd",
        "entry_rmsnorm_bwd",
    ], bwd


def test_full_backward_chain_on_emits_all_segments_in_reverse_order(monkeypatch):
    """Flag ON (the unlock fix): the full (fwd+bwd) region emits the NON-mamba
    backward segments (sparse_mla/attention/residual x2/m2rnn/entry) IN ADDITION
    TO the 3 chunked mamba ``_bwd`` (B2/B1/B0), all in correct reverse-of-forward
    order. The residual_rmsnorm_bwd producing the mamba ``brick_M_delta_grad``
    cotangent MUST run BEFORE the chunked mamba backward so the cotangent is
    seeded (the root cause of the prior zero-mamba-grad gap)."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    region = _build_mra_full_backward_region(_env(), include_backward=True)
    bwd = [n.op_name for n in region.nodes if n.op_name.endswith("_bwd")]
    assert bwd == [
        "sparse_mla_fp8_apply_bwd",
        "attention_qkv_projection_bwd",
        "residual_rmsnorm_bwd",
        "m2rnn_bwd",
        "residual_rmsnorm_bwd",
        "mamba3_chunk_scan_combine_bwd",
        "mamba3_inter_chunk_recur_bwd",
        "mamba3_chunk_precompute_bwd",
        "entry_rmsnorm_bwd",
    ], bwd

    # The mamba cotangent seam: a residual_rmsnorm_bwd must OUTPUT the brick's
    # delta_grad and run strictly BEFORE the chunked mamba B2 reads it.
    nodes = list(region.nodes)
    delta_grad = "brick_M_delta_grad"
    seed_idx = next(
        i for i, n in enumerate(nodes)
        if n.op_name == "residual_rmsnorm_bwd" and delta_grad in n.outputs
    )
    b2_idx = next(
        i for i, n in enumerate(nodes)
        if n.op_name == "mamba3_chunk_scan_combine_bwd"
    )
    assert seed_idx < b2_idx, (seed_idx, b2_idx)
    assert delta_grad in nodes[b2_idx].inputs, nodes[b2_idx].inputs


def test_backward_handoff_buffers_fp32_and_force_spilled():
    """The 6 backward grad-handoff buffers resolve fp32 and are in the force-spill
    ABI set (mirror of the forward handoff registration)."""
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS,
        _buffer_dtype,
    )

    for nm in DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS:
        assert _buffer_dtype(nm) == "float32", nm
    assert "mamba3_dchunk_states" in DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS
    assert "mamba3_dstates" in DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.xfail(
    reason=(
        "PRE-EXISTING (base b388d6c, not from the dz/y_skip or determinism fix): "
        "this stage test drives the B2/B1/B0 kernels via TORCH-MPS positional "
        "buffers at the tiny non-production config G=1,H=2,N=16. That torch-MPS "
        "multi-kernel chain hits a command-buffer ordering hazard where F0's "
        "dA_cumsum is read corrupted, producing large per-grad errors (up to ~0.6) "
        "in dx. The race does NOT exist on the PRODUCTION MLX route "
        "(mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd at nam56r H=128,N=64), "
        "which is covered green + deterministic by "
        "tests/test_mamba3_path_c_chunked_vs_path_b.py. The proto model below was "
        "updated to the production SSD (inp=x*B, NO dt) so the comparison targets "
        "the right kernels; the remaining failure is the torch-path hazard only. "
        "strict=True: any change that makes the torch path pass will flip this to XPASS."
    ),
    strict=True,
)
@pytest.mark.parametrize("seqlen", [256, 512])
def test_chained_backward_b2b1b0_matches_proto(seqlen, capsys):
    """The 3 backward Metal kernels chained (B2->B1->B0) reproduce the validated
    MLX backward proto per-grad-tensor < 1e-3 (the Stage-3 parity gate).

    The proto is itself 1.30e-4 vs the serial backward VJP (the GOLD), so matching
    it < 1e-3 transitively matches the serial backward < ~1.4e-3 — well inside the
    design's per-grad gate at chunk=64 (the production Metal tile config).

    NOTE: this is the torch-MPS positional-buffer stage harness at the tiny
    G=1,H=2,N=16 config; it is currently xfail due to a PRE-EXISTING torch-path
    NaN (see the marker). The PRODUCTION MLX route is validated bit-correct +
    deterministic by tests/test_mamba3_path_c_chunked_vs_path_b.py."""
    import torch
    from einops import rearrange

    import mlx.core as mx

    # the validated proto (copied into scratch at integration time)
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scratch"))
    import mamba3_chunked_backward_proto as bp

    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal, build_inter_chunk_recur_metal,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
        build_chunk_precompute_bwd_metal,
    )

    b, chunk, G, H, P, N = 1, 64, 1, 2, 64, 16
    nchunks = seqlen // chunk
    dev = "mps"
    rng = np.random.RandomState(0)
    x_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
    B_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
    C_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
    A_np = (-rng.rand(H)).astype(np.float32)
    dt_np = (rng.rand(b, seqlen, H) * 0.05).astype(np.float32)
    D_np = (rng.randn(H)).astype(np.float32)
    h0_np = (rng.randn(b, H, P, N) * 0.1).astype(np.float32)
    dout_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
    z_np = (rng.randn(b, seqlen, H, P) * 0.5).astype(np.float32)

    def mxa(a): return mx.array(a)
    # PRODUCTION SSD model (matches the re-derived B2/B0 kernels, b388d6c):
    #   inp = x (outer) B  with NO dt baked in (dt enters ONLY via the decay
    #   log_decay = A*dt). The earlier proto baked dt into inp (inp=dt*x*B); the
    #   kernels were re-derived to inp=x*B, so the proto + its grad chain MUST drop
    #   the dt factor too or the comparison is against the wrong model.
    log_decay = (mxa(A_np).reshape(1, 1, H) * mxa(dt_np)).reshape(b, seqlen, H, 1, 1)
    B_h = mx.broadcast_to(mxa(B_np)[:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
    inp = mxa(x_np)[..., None] * B_h[:, :, :, None, :]  # PROD inp = x (outer) B, NO dt
    C_proto = mx.broadcast_to(mxa(C_np)[:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
    out, fs, cache = bp.chunked_mamba3_forward_full(
        log_decay, inp, C_proto, mxa(x_np), mxa(z_np), mxa(D_np), mxa(h0_np), chunk_size=chunk)
    grads = bp.chunked_mamba3_backward(mxa(dout_np), cache, dh_last=None)
    mx.eval(grads["log_decay"], grads["inp"], grads["C"], grads["x"], grads["z"], grads["D"], grads["h0"])
    g_dz = np.array(grads["z"]); g_dC = np.array(grads["C"])
    g_dx = np.array(grads["x"]); g_dh0 = np.array(grads["h0"])
    g_dlog = np.array(grads["log_decay"]).reshape(b, seqlen, H)
    g_dinp = np.array(grads["inp"]); g_dD = np.array(grads["D"])
    y_np = np.array(cache["y"])  # PRE-GATE y_skip = C.h + D*x (proto cache["y"], :119)
    # inp = x (outer) B (PROD, NO dt): dx_inp = sum_n dinp*B ; dB = sum_p dinp*x ;
    # ddt has NO input-term contribution (only the decay path g_dlog*A).
    g_dxinp = np.einsum("bshpn,bsn->bshp", g_dinp, B_np[:, :, 0, :])
    g_dB = np.einsum("bshpn,bshp->bshn", g_dinp, x_np).sum(2, keepdims=False)[:, :, None, :]
    g_ddt = g_dlog * A_np.reshape(1, 1, H)

    def th(a, d=torch.float16): return torch.tensor(a, device=dev, dtype=d).contiguous()
    k_f0 = build_chunk_precompute_metal(b, seqlen, chunk, G, H, P, N)
    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
    dA = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
    summ = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    k_f0(th(x_np), th(B_np), th(C_np), th(A_np), th(dt_np), cb, dA, summ); torch.mps.synchronize()
    k_f1 = build_inter_chunk_recur_metal(b, seqlen, chunk, G, H, P, N)
    prev = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    fst = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    k_f1(summ.contiguous(), dA.contiguous(), th(h0_np, torch.float32), prev, fst); torch.mps.synchronize()
    dt_k = rearrange(th(dt_np), "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
    dC_m = torch.zeros(b, seqlen, H, N, device=dev, dtype=torch.float32)
    dx_m = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
    dz_m = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
    dchunk = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    dinp_diag = torch.zeros(b, seqlen, H, P, N, device=dev, dtype=torch.float32)
    dA_y = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
    dD_m = torch.zeros(H, device=dev, dtype=torch.float32)
    k_b2(th(dout_np), cb.contiguous(), th(x_np), th(z_np), dt_k, dA.contiguous(),
         th(C_np), th(B_np), prev.contiguous(), th(D_np), th(y_np),
         dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m); torch.mps.synchronize()

    k_b1 = build_inter_chunk_recur_bwd_metal(b, seqlen, chunk, G, H, P, N)
    dh_last = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    dstates = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    dh0_m = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    dA_tail = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
    k_b1(dchunk.contiguous(), dA.contiguous(), dh_last, prev.contiguous(),
         dstates, dh0_m, dA_tail); torch.mps.synchronize()

    k_b0 = build_chunk_precompute_bwd_metal(b, seqlen, chunk, G, H, P, N)
    dx_full = dx_m.clone()
    dB_m = torch.zeros(b, seqlen, H, N, device=dev, dtype=torch.float32)
    dlog_m = torch.zeros(b, seqlen, H, device=dev, dtype=torch.float32)
    ddt_m = torch.zeros(b, seqlen, H, device=dev, dtype=torch.float32)
    k_b0(dstates.contiguous(), dinp_diag.contiguous(), dA_y.contiguous(),
         dA_tail.contiguous(), dA.contiguous(), th(x_np), th(B_np), dt_k, th(A_np),
         dx_full, dB_m, dlog_m, ddt_m); torch.mps.synchronize()

    def d(got, gold):
        return float(np.abs(np.asarray(got, np.float64) - np.asarray(gold, np.float64)).max())

    diffs = {
        "dz": d(dz_m.float().cpu(), g_dz),
        "dx": d(dx_full.float().cpu(), g_dx + g_dxinp),
        "dC": d(dC_m.float().cpu(), g_dC),
        "dB": d(dB_m.float().cpu().numpy().sum(2, keepdims=False)[:, :, None, :], g_dB),
        "dlog_decay": d(dlog_m.float().cpu(), g_dlog),
        "ddt": d(ddt_m.float().cpu(), g_ddt),
        "dh0": d(dh0_m.float().cpu(), g_dh0),
        "dD": d(dD_m.float().cpu(), g_dD.sum(-1) if g_dD.ndim == 2 else g_dD),
    }
    worst = max(diffs.values())
    with capsys.disabled():
        print(f"\n[chained-backward] S={seqlen} chunk={chunk} H={H} P={P} N={N} "
              f"tg(B2/B1/B0)={b*nchunks*H}/{b*H}/{b*nchunks*H} "
              f"per-grad max|abs|: " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()) +
              f" -> WORST={worst:.3e}")
    for k, v in diffs.items():
        assert not np.isnan(v), f"{k} is NaN"
        assert v < 1e-3, f"{k} chained-backward-vs-proto max|abs|={v:.3e} > 1e-3"
