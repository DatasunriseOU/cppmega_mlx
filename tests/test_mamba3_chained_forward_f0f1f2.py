"""Stage-2 chained-isolation harness: F0 -> F1 -> F2 vs the SERIAL forward.

Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §7 Stage 2.

Stage 1 proved the F2 grid scan+combine kernel in isolation, fed
``cb/dA_cumsum/prev_states`` recomputed EAGERLY (a torch reference precompute
standing in for the not-yet-built F0/F1). Stage 2 builds those two missing
segments as real Metal kernels and chains all three through the caller-owned
handoff buffers, exactly as the Path-C multi-segment chain will:

  F0 ``mamba3_chunk_precompute``  (x,B,C,A,dt) -> cb, dA_cumsum, summary_states
  F1 ``mamba3_inter_chunk_recur`` (summary_states, dA_cumsum, h0) -> prev_states, final_state
  F2 ``mamba3_chunk_scan_combine`` (cb, x, dt, dA_cumsum, C, prev_states, D) -> Output

PARITY GATE (Stage-2 success): the FULL chained output ``max|abs|`` and
``final_state`` (h_last) vs the SERIAL Path-C forward must be ``< 5e-4`` fp16 at
the validated scan-core ``chunk_size == block_M == 64`` (the live feasibility gate
``_mamba3_chunked_forward_scan_feasibility:6363``). The serial reference is an
explicit per-timestep diagonal recurrence (carry ``h`` advanced one step at a
time, seeded by ``h0``) — genuinely serial, NOT a re-expression of the chunked
einsum.

The cross-``chunk_size {64,128,256}`` + non-pow2-S sweep of the design's parity
gate is exercised at the SSD numerical-contract level by
``test_chained_numerical_contract_sweep`` (fp32 reference algebra), since the
Metal F2 kernel's tile config pins ``block_M == 64`` (chunk=64) — a mismatched
chunk RAISES at the feasibility gate, it never silently re-tiles (RULE #1).

RULE #1: the kernel path is the single delegation per segment
(``build_chunk_precompute_metal`` / ``build_inter_chunk_recur_metal`` /
``build_chunk_scan_combine_metal``); on a compile/parity failure the helpers RAISE
with where+what and the test FAILS — it never degrades to the serial scan.
"""

from __future__ import annotations

import pytest

# Reuse the proven Stage-1 reference precompute + serial forward (the numerical
# ground truth), so this harness validates the NEW Metal F0/F1 against the SAME
# contract the Stage-1 eager reference already matched the serial forward to.
from tests.test_mamba3_chunk_scan_combine_f2 import (
    _eager_precompute,
    _serial_forward,
    _torch_mps_available,
)


# --------------------------------------------------------------------------- #
# Verification 0: all 3 forward descriptors register + select-resolve (no Metal).#
# --------------------------------------------------------------------------- #


def test_all_three_forward_descriptors_register_and_resolve():
    """F0/F1/F2 op-names each resolve to a brick descriptor (no ``no descriptor
    target``) — the Stage-2 success criterion: select resolves all 3 signatures."""
    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        MAMBA3_CHUNK_PRECOMPUTE_OP_NAME,
        MAMBA3_INTER_CHUNK_RECUR_OP_NAME,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,
    )
    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        Z3SyncSpec,
        build_path_c_fusion_region,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        default_path_c_brick_schedule_descriptor_registry,
        default_path_c_fusion_schedule_registry,
    )

    reg = default_path_c_brick_schedule_descriptor_registry()
    for op in (
        MAMBA3_CHUNK_PRECOMPUTE_OP_NAME,
        MAMBA3_INTER_CHUNK_RECUR_OP_NAME,
        MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,
    ):
        desc = reg.descriptor_for(op)
        assert desc is not None, f"{op} descriptor must register (else select blocks)"
        assert desc.fragment_emitter is not None, f"{op} needs a fragment_emitter"
        assert reg.descriptors_for_signature((op,)) is not None

    # each single-node region resolves via the schedule registry select
    sched_reg = default_path_c_fusion_schedule_registry()
    surf_inputs = {
        MAMBA3_CHUNK_PRECOMPUTE_OP_NAME: ("mamba3_x", "mamba3_B", "mamba3_C", "mamba3_A", "mamba3_dt"),
        MAMBA3_INTER_CHUNK_RECUR_OP_NAME: ("mamba3_summary_states", "mamba3_dA_cumsum", "mamba3_h0"),
        MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME: (
            "mamba3_cb", "mamba3_x", "mamba3_dt", "mamba3_dA_cumsum",
            "mamba3_C", "mamba3_prev_states", "mamba3_D",
        ),
    }
    for op, inputs in surf_inputs.items():
        surf = FusionKernelSurface.path_c(
            name=f"{op}_node", op_name=op, inputs=inputs,
            outputs=("out0",), backward="owner_output",
        )
        region = build_path_c_fusion_region(
            region_name=op, surfaces=(surf,),
            z3_sync=Z3SyncSpec.minimize_sync_async(),
        )
        assert tuple(n.op_name for n in region.nodes) == (op,)
        target = sched_reg.select(region)
        assert target is not None, f"{op} signature must NOT be blocked by select"
        assert target.op_signature == (op,)


# --------------------------------------------------------------------------- #
# Verification 1: numerical-contract sweep (fp32 reference algebra, no Metal).  #
# chunk {64,128,256} + a non-power-of-2 S — the design's parity-gate shapes.    #
# --------------------------------------------------------------------------- #


def _ref_chunked_forward_fp32(C, Bmat, x, A, dt, h0, D, chunk_size):
    """Fp32 SSD chunked forward (F0+F1+F2 algebra) — the numerical contract.

    Builds cb/dA_cumsum/prev_states via the proven ``_eager_precompute`` (F0+F1
    reference math), then applies the F2 scan+combine algebra (Y_diag+Y_off+D*x)
    AND returns ``final_state`` (h_last). All fp32. This is the object the chained
    Metal kernels must reproduce; it is compared against ``_serial_forward`` to
    confirm the chunked decomposition is faithful at every chunk_size.
    """
    import torch
    from einops import rearrange, repeat

    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    nchunks = seqlen // chunk_size
    h = nheads // ngroups

    # --- F0 reference (fp32, NO fp16 cast — this is the 1e-5 fp32 contract) ---
    Cf, Bf, xf, Af, dtf = C.float(), Bmat.float(), x.float(), A.float(), dt.float()
    a = Af.view(1, 1, nheads) * dtf
    a_c = rearrange(a, "b (c l) h -> b h c l", c=nchunks)
    dA_cumsum = torch.cumsum(a_c, dim=-1)  # (b,h,c,l)
    Cc = rearrange(Cf, "b (c l) g n -> b c l g n", c=nchunks)
    Bc = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks)
    cb = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)  # (b,c,g,l,s)
    Bexp = repeat(Bf, "b (c s) g n -> b c s (g h) n", c=nchunks, h=h)
    xexp = rearrange(xf, "b (c s) hh p -> b c s hh p", c=nchunks)
    dtc = rearrange(dtf, "b (c s) hh -> b hh c s", c=nchunks)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    summary = torch.einsum("bhcs,bhcs,bcshp,bcshn->bchpn", decay_states, dtc, xexp, Bexp)

    # --- F1 reference (fp32 inter-chunk recurrence seeded by h0) ---
    chunk_tail = dA_cumsum[:, :, :, -1]  # (b,nheads,c)
    state = h0.float().clone()  # (b,H,P,N)
    prev_states = torch.zeros(batch, nchunks, nheads, headdim, dstate)
    for c in range(nchunks):
        prev_states[:, c] = state
        state = torch.exp(chunk_tail[:, :, c])[..., None, None] * state + summary[:, c]
    final_state = state

    # --- F2 reference (serial scan+combine over the chunked handoff) ---
    dt_k = dtc.contiguous()
    out = _serial_forward(cb, x, dt_k, dA_cumsum, Cf, prev_states, D, chunk_size)
    return out, final_state


@pytest.mark.parametrize(
    "batch,seqlen,chunk,nheads,headdim,dstate,ngroups",
    [
        (1, 256, 64, 2, 64, 16, 1),
        (1, 512, 128, 2, 64, 16, 1),
        (1, 768, 256, 2, 64, 16, 1),     # chunk=256
        (1, 320, 64, 2, 64, 16, 1),      # non-power-of-2 S (5 chunks)
        (1, 640, 128, 2, 64, 16, 1),     # non-power-of-2 nchunks=5
    ],
)
def test_chained_numerical_contract_sweep(
    batch, seqlen, chunk, nheads, headdim, dstate, ngroups, capsys
):
    """F0+F1+F2 SSD algebra == serial forward < 1e-5 fp32, chunk {64,128,256}+nonpow2.

    Confirms the chunked decomposition (incl. the inter-chunk RoPE-angle-style
    cumsum carried as ``dA_cumsum`` chunk-tail decay) is numerically faithful at
    every chunk granularity and a non-power-of-2 S — independent of the Metal
    tile config (which pins chunk=64).
    """
    import torch

    assert seqlen % chunk == 0
    torch.manual_seed(0)
    dev = "cpu"
    C = torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1
    Bmat = torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1
    x = torch.randn(batch, seqlen, nheads, headdim, device=dev) * 0.1
    A = -torch.rand(nheads, device=dev)
    dt = torch.rand(batch, seqlen, nheads, device=dev) * 0.05
    D = torch.randn(nheads, device=dev)
    h0 = torch.randn(batch, nheads, headdim, dstate, device=dev) * 0.1

    out, final_state = _ref_chunked_forward_fp32(C, Bmat, x, A, dt, h0, D, chunk)

    # Serial ground truth (full per-timestep recurrence seeded by h0), incl
    # h_last. This is independent of the chunked split — the contract anchor.
    out_serial, hlast_serial = _serial_full_forward(C, Bmat, x, A, dt, h0, D)

    assert not bool(torch.isnan(out).any())
    max_abs = float((out - out_serial).abs().max())
    max_hlast = float((final_state - hlast_serial).abs().max())
    # confirm the inter-chunk cumulative-decay MAGNITUDE matches (design gate)
    with capsys.disabled():
        print(
            f"\n[chained-contract] S={seqlen} chunk={chunk} nchunks={seqlen//chunk} "
            f"H={nheads} -> out max|abs|={max_abs:.3e} h_last max|abs|={max_hlast:.3e}"
        )
    assert max_abs < 1e-5, f"chained-vs-serial out max|abs|={max_abs:.3e} > 1e-5"
    assert max_hlast < 1e-5, f"chained-vs-serial h_last max|abs|={max_hlast:.3e} > 1e-5"


def _serial_full_forward(C, Bmat, x, A, dt, h0, D):
    """Reference SERIAL per-timestep diagonal forward over the FULL sequence.

    The OUR recurrence (mamba3.py ``_chunked_mamba3_diagonal_scan``):
      log_decay[t] = A[h]*dt[t]
      h[t]  = exp(log_decay[t]) * h[t-1] + dt[t] * (x[t] outer B[t])
      y[t]  = sum_n h[t] * C[t] + D*x[t]   (no z gate; F2 output contract)
    Seeded by ``h0``. fp32. Returns (out (b,S,H,P), h_last (b,H,P,N)).
    """
    import torch
    from einops import repeat

    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    h = nheads // ngroups
    Cf = repeat(C.float(), "b l g n -> b l (g h) n", h=h)
    Bf = repeat(Bmat.float(), "b l g n -> b l (g h) n", h=h)
    xf = x.float()
    Af = A.float()
    dtf = dt.float()
    state = h0.float().clone()  # (b,H,P,N)
    out = torch.zeros(batch, seqlen, nheads, headdim)
    for t in range(seqlen):
        decay = torch.exp(Af.view(1, nheads) * dtf[:, t])  # (b,H)
        inp = dtf[:, t][:, :, None, None] * (
            xf[:, t][:, :, :, None] * Bf[:, t][:, :, None, :]
        )  # (b,H,P,N)
        state = decay[:, :, None, None] * state + inp
        y = torch.einsum("bhpn,bhn->bhp", state, Cf[:, t])  # (b,H,P)
        out[:, t] = y + D.float().view(1, nheads, 1) * xf[:, t]
    return out, state


# --------------------------------------------------------------------------- #
# Verification 2: Metal F0->F1->F2 chained, vs the SERIAL forward < 5e-4 fp16.  #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.parametrize(
    "batch,seqlen,chunk,nheads,headdim,dstate",
    [
        (1, 256, 64, 2, 64, 16),
        (1, 4096, 64, 8, 64, 16),   # full-scale-style threadgroup counts
    ],
)
def test_chained_metal_f0f1f2_matches_serial_forward(
    batch, seqlen, chunk, nheads, headdim, dstate, capsys
):
    """FULL chained Metal forward (F0->F1->F2) vs SERIAL forward < 5e-4 fp16.

    The Stage-2 end-to-end PARITY GATE. cb/dA_cumsum/summary_states/prev_states
    flow segment->segment EXACTLY as caller-owned handoff buffers (the Path-C
    multi-segment ABI). No eager precompute — F0/F1 are real Metal kernels.
    """
    import torch
    from einops import rearrange

    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal,
        build_inter_chunk_recur_metal,
        chunk_precompute_fwd_grid,
        inter_chunk_recur_fwd_grid,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        MAMBA3_CHUNKED_FWD_BLOCK_M,
        build_chunk_scan_combine_metal,
        chunk_scan_fwd_grid,
    )

    assert chunk == MAMBA3_CHUNKED_FWD_BLOCK_M
    ngroups = 1
    nchunks = seqlen // chunk
    dev = "mps"
    torch.manual_seed(0)
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=dev) * 0.1).half()
    A = -torch.rand(nheads, device=dev).half()
    dt = (torch.rand(batch, seqlen, nheads, device=dev) * 0.05).half()
    D = torch.randn(nheads, device=dev).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=dev) * 0.1).float()

    tg0, g0 = chunk_precompute_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    tg1, g1 = inter_chunk_recur_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    tg2, g2 = chunk_scan_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)

    # ---- F0: precompute ---- (caller-owned handoff buffers, pre-zeroed) ----
    k0 = build_chunk_precompute_metal(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    cb = torch.zeros(batch, nchunks, ngroups, chunk, chunk, device=dev, dtype=torch.float16)
    dA_cumsum = torch.zeros(batch, nheads, nchunks, chunk, device=dev, dtype=torch.float16)
    summary_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=dev, dtype=torch.float32)
    k0(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(), dt.contiguous(),
       cb, dA_cumsum, summary_states)
    torch.mps.synchronize()

    # ---- F1: inter-chunk recurrence ---- (consumes F0 handoff) ----
    k1 = build_inter_chunk_recur_metal(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    prev_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=dev, dtype=torch.float32)
    final_state = torch.zeros(batch, nheads, headdim, dstate, device=dev, dtype=torch.float32)
    k1(summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
       prev_states, final_state)
    torch.mps.synchronize()

    # ---- F2: scan+combine ---- (consumes F0+F1 handoff; prev_states->fp16) ----
    k2 = build_chunk_scan_combine_metal(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    out = torch.zeros(batch, seqlen, nheads, headdim, device=dev, dtype=torch.float16)
    k2(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
       C.contiguous(), prev_states.half().contiguous(), D.contiguous(), out)
    torch.mps.synchronize()

    # ---- SERIAL ground truth (full per-timestep recurrence, seeded by h0) ----
    out_serial, hlast_serial = _serial_full_forward(C.cpu(), Bmat.cpu(), x.cpu(), A.cpu(), dt.cpu(), h0.cpu(), D.cpu())

    assert not bool(torch.isnan(out).any()), "chained Metal forward produced NaN"
    assert not bool(torch.isnan(final_state).any()), "F1 final_state produced NaN"
    max_abs = float((out.float().cpu() - out_serial).abs().max())
    max_hlast = float((final_state.float().cpu() - hlast_serial).abs().max())
    with capsys.disabled():
        print(
            f"\n[chained-metal] S={seqlen} chunk={chunk} H={nheads} P={headdim} "
            f"N={dstate} -> tg(F0/F1/F2)={tg0}/{tg1}/{tg2} grids={g0}/{g1}/{g2} "
            f"out max|abs|(chain vs serial)={max_abs:.3e} "
            f"h_last max|abs|={max_hlast:.3e}"
        )
    assert max_abs < 5e-4, (
        f"chained-Metal-vs-serial out max|abs|={max_abs:.3e} > Stage-2 gate 5e-4 "
        f"at S={seqlen}"
    )
    assert max_hlast < 5e-4, (
        f"chained-Metal-vs-serial h_last max|abs|={max_hlast:.3e} > 5e-4 at S={seqlen}"
    )


# --------------------------------------------------------------------------- #
# Verification 3: the LIVE compile-site DELEGATION INTERPOSE (Stage-2 live flip)#
# - flag OFF: F0/F1/F2 segment compiles via the source/exec template (the SHADOW#
#   marker path) -> a PrimFunc, NOT a delegated grid kernel.                    #
# - flag ON : the interpose BYPASSES exec/source and substitutes the proven     #
#   build_*_metal grid JITKernel (the no-op marker is NEVER the live kernel).   #
# Design: docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md §3.1/§7.                       #
# --------------------------------------------------------------------------- #


def _local_gb10_quarter_env():
    """Quarter-scale (S=4096) Path-C shape env feasible for the chunked scan-core
    (chunk=64, headdim%block_N==0, heads%groups==0)."""
    from cppmega_mlx.runtime.path_c_fusion import PathCModelShapeEnv

    return PathCModelShapeEnv(
        sequence_length=4096,
        hidden_size=3584,
        attention_num_q_heads=28,
        attention_num_kv_heads=4,
        attention_head_dim=128,
        attention_sparse_topk=64,
        mamba_expand=2,
        mamba_head_dim=128,
        mamba_state_dim=128,
        mamba_groups=1,
        mamba_mimo_rank=1,
        mamba_is_mimo=True,
        mamba_conv_kernel=4,
        mamba_rope_fraction=0.5,
        m2rnn_k_head_dim=128,
        m2rnn_v_head_dim=128,
        m2rnn_num_q_heads=8,
        m2rnn_num_k_heads=8,
        m2rnn_num_v_heads=8,
        m2rnn_num_f_heads=8,
        m2rnn_num_g_heads=8,
        m2rnn_num_weight_heads=8,
        m2rnn_conv_kernel=4,
    )


_INTERPOSE_SURFACES = {
    "mamba3_chunk_precompute": (
        ("mamba3_x", "mamba3_B", "mamba3_C", "mamba3_A", "mamba3_dt"),
        ("mamba3_cb", "mamba3_dA_cumsum", "mamba3_summary_states"),
    ),
    "mamba3_inter_chunk_recur": (
        ("mamba3_summary_states", "mamba3_dA_cumsum", "mamba3_h0"),
        ("mamba3_prev_states", "mamba3_final_state"),
    ),
    "mamba3_chunk_scan_combine": (
        ("mamba3_cb", "mamba3_x", "mamba3_dt", "mamba3_dA_cumsum",
         "mamba3_C", "mamba3_prev_states", "mamba3_D"),
        ("mamba3_out",),
    ),
}


def _build_segment_prim(op_name, inputs, outputs, env):
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
        inputs=inputs, outputs=outputs, backward="owner_output",
    )
    region = build_path_c_fusion_region(
        region_name=op_name, surfaces=(surf,),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    desc = reg.descriptor_for(op_name)
    return build_path_c_descriptor_prim_func(region, (desc,), shape_env=env)


def test_chunked_scan_flag_default_off(monkeypatch):
    """The live flip is DEFAULT OFF: a single F0 segment compiles via the
    source/exec template (the shadow marker path), NOT the delegated grid kernel."""
    monkeypatch.delenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", raising=False)
    env = _local_gb10_quarter_env()
    ins, outs = _INTERPOSE_SURFACES["mamba3_chunk_precompute"]
    prim = _build_segment_prim("mamba3_chunk_precompute", ins, outs, env)
    # OFF: NOT a delegated grid kernel.
    assert getattr(prim, "_cppmega_path_c_mamba3_chunked_grid_delegation", None) is None
    assert type(prim).__name__ != "JITKernel"


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.parametrize("op_name", list(_INTERPOSE_SURFACES))
def test_chunked_scan_flag_on_interpose_emits_real_grid_kernel(op_name, monkeypatch):
    """Flag ON: the compile-site interpose substitutes the REAL build_*_metal grid
    JITKernel for the F0/F1/F2 segment — the no-op SHADOW marker is NEVER the
    live emitted kernel (RULE #1)."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    env = _local_gb10_quarter_env()
    ins, outs = _INTERPOSE_SURFACES[op_name]
    prim = _build_segment_prim(op_name, ins, outs, env)
    # ON: the proven grid kernel, tagged with the delegation op-name.
    assert type(prim).__name__ == "JITKernel", (
        f"{op_name} flag-ON must emit a real grid JITKernel, got {type(prim).__name__}"
    )
    assert (
        getattr(prim, "_cppmega_path_c_mamba3_chunked_grid_delegation", None)
        == op_name
    )
    assert getattr(prim, "_cppmega_path_c_brick_ops", None) == (op_name,)


# --------------------------------------------------------------------------- #
# Verification 4: the LIVE REGION-BUILD 1->3 SURFACE FLIP (Stage-2).            #
# - flag OFF: the direct-chain model region emits ONE serial mamba3_mimo        #
#   forward surface (byte-identical to today's behaviour).                      #
# - flag ON : the region emits 3 chunked SSD-core forward surfaces              #
#   (mamba3_chunk_precompute -> mamba3_inter_chunk_recur ->                      #
#   mamba3_chunk_scan_combine) wired by the per-brick handoff buffers, each      #
#   isolated into its own FORWARD stage so the compile-site interpose fires.     #
# Design: docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md §2/§7.                         #
# --------------------------------------------------------------------------- #

_MAMBA3_CHUNKED_REGION_OPS = (
    "mamba3_chunk_precompute",
    "mamba3_inter_chunk_recur",
    "mamba3_chunk_scan_combine",
)


def _build_mamba_direct_chain_region(env):
    from cppmega_mlx.runtime.path_c_fusion import (
        PathCModelBrick,
        build_path_c_model_region_from_bricks,
    )

    bricks = (PathCModelBrick(name="mamba3_scan", kind="mamba3", route_symbol="M"),)
    return build_path_c_model_region_from_bricks(
        region_name="mamba3_direct_chain", bricks=bricks, shape_env=env
    )


def test_region_flip_default_off_single_mamba3_mimo_surface(monkeypatch):
    """Flag OFF (default): the direct-chain region emits ONE mamba3_mimo forward
    surface — byte-identical to today's serial behaviour (RULE #1 merge-safe)."""
    monkeypatch.delenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", raising=False)
    env = _local_gb10_quarter_env()
    region = _build_mamba_direct_chain_region(env)
    forward_ops = [n.op_name for n in region.nodes if n.op_name != "entry_rmsnorm"]
    assert forward_ops == ["mamba3_mimo"], forward_ops
    assert not any(
        n.op_name in _MAMBA3_CHUNKED_REGION_OPS for n in region.nodes
    )


def test_region_flip_on_emits_three_chunked_forward_surfaces(monkeypatch):
    """Flag ON: the direct-chain region replaces the single mamba3_mimo surface
    with the 3 chunked SSD-core forward segments (F0/F1/F2), wired by the
    per-brick handoff buffers, each isolated into its own FORWARD stage group."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        plan_path_c_descriptor_stage_groups,
    )

    env = _local_gb10_quarter_env()
    region = _build_mamba_direct_chain_region(env)
    chunked_ops = [n.op_name for n in region.nodes if n.op_name in _MAMBA3_CHUNKED_REGION_OPS]
    assert chunked_ops == list(_MAMBA3_CHUNKED_REGION_OPS), chunked_ops
    # The serial mamba3_mimo surface is NOT emitted when the flag is ON.
    assert not any(n.op_name == "mamba3_mimo" for n in region.nodes)

    # Each chunked op is isolated into its own FORWARD stage so the compile-site
    # delegation interpose (which requires len(nodes) == 1) fires per segment.
    op_by_node = {n.name: n.op_name for n in region.nodes}
    chunked_stage_ops: list[str] = []
    for group in plan_path_c_descriptor_stage_groups(region):
        ops = [op_by_node.get(nm) for nm in group.active_node_names]
        for op in ops:
            if op in _MAMBA3_CHUNKED_REGION_OPS:
                assert group.execution_stage == "forward", group.execution_stage
                assert len(group.active_node_names) == 1, list(group.active_node_names)
                chunked_stage_ops.append(op)
    assert chunked_stage_ops == list(_MAMBA3_CHUNKED_REGION_OPS), chunked_stage_ops

    # The 3 surfaces are wired by the per-brick handoff buffers: F0 -> F1 -> F2.
    nodes_by_op = {n.op_name: n for n in region.nodes}
    f0 = nodes_by_op["mamba3_chunk_precompute"]
    f1 = nodes_by_op["mamba3_inter_chunk_recur"]
    f2 = nodes_by_op["mamba3_chunk_scan_combine"]
    assert "mamba3_scan_summary_states" in f0.outputs
    assert "mamba3_scan_summary_states" in f1.inputs
    assert "mamba3_scan_prev_states" in f1.outputs
    assert "mamba3_scan_prev_states" in f2.inputs
    assert "mamba3_scan_cb" in f0.outputs and "mamba3_scan_cb" in f2.inputs
    assert f2.outputs == ("mamba3_scan_delta",)


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
def test_region_flip_on_full_scale_region_parity(monkeypatch, capsys):
    """Flag ON, full scale (S=4096, H=8): the LIVE direct-chain region emits 3
    chunked forward segments, each delegates to its proven grid JITKernel, and the
    chained region forward matches the SERIAL Path-C forward < 5e-4 fp16.

    The Stage-2 REGION-build parity gate (mirrors
    ``scripts/repro_fullscale_region_flip.py``). RULE #1: a compile/parity failure
    RAISES — no silent serial fallback."""
    monkeypatch.setenv("CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN", "1")

    import torch
    from einops import rearrange

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        MAMBA3_CHUNKED_FWD_BLOCK_M,
    )
    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        PathCModelShapeEnv,
        Z3SyncSpec,
        build_path_c_fusion_region,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        _MAMBA3_CHUNKED_GRID_DELEGATION_OPS,
        build_path_c_descriptor_prim_func,
        default_path_c_brick_schedule_descriptor_registry,
    )

    b, S, H, P, N, G = 1, 4096, 8, 64, 16, 1
    chunk = MAMBA3_CHUNKED_FWD_BLOCK_M
    nchunks = S // chunk
    dev = "mps"
    env = PathCModelShapeEnv(
        sequence_length=S, hidden_size=H * P, attention_num_q_heads=H,
        attention_num_kv_heads=H, attention_head_dim=P, attention_sparse_topk=1,
        mamba_expand=1, mamba_head_dim=P, mamba_state_dim=N, mamba_groups=G,
        mamba_mimo_rank=1, mamba_is_mimo=True, mamba_conv_kernel=4,
        mamba_rope_fraction=0.5, m2rnn_k_head_dim=P, m2rnn_v_head_dim=P,
        m2rnn_num_q_heads=H, m2rnn_num_k_heads=H, m2rnn_num_v_heads=H,
        m2rnn_num_f_heads=H, m2rnn_num_g_heads=H, m2rnn_num_weight_heads=H,
        m2rnn_conv_kernel=4,
    )
    assert env.mamba_num_heads == H

    region = _build_mamba_direct_chain_region(env)
    # Forward-only chunked nodes (Stage 3 also emits 3 chunked _bwd nodes when the
    # flag is ON; this forward-parity check exercises just F0/F1/F2).
    chunked_nodes = [
        n
        for n in region.nodes
        if n.op_name in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS
        and not n.op_name.endswith("_bwd")
    ]
    assert len(chunked_nodes) == 3

    reg = default_path_c_brick_schedule_descriptor_registry()
    kernels = {}
    for node in chunked_nodes:
        surf = FusionKernelSurface.path_c(
            name=node.name, op_name=node.op_name,
            inputs=node.inputs, outputs=node.outputs, backward="owner_output",
        )
        subregion = build_path_c_fusion_region(
            region_name=node.op_name, surfaces=(surf,),
            z3_sync=Z3SyncSpec.minimize_sync_async(),
            metadata={"path_c_model_shape_env": env},
        )
        prim = build_path_c_descriptor_prim_func(
            subregion, (reg.descriptor_for(node.op_name),), shape_env=env
        )
        assert type(prim).__name__ == "JITKernel", node.op_name
        kernels[node.op_name] = prim

    torch.manual_seed(0)
    C = (torch.randn(b, S, G, N, device=dev) * 0.1).half()
    Bmat = (torch.randn(b, S, G, N, device=dev) * 0.1).half()
    x = (torch.randn(b, S, H, P, device=dev) * 0.1).half()
    A = -torch.rand(H, device=dev).half()
    dt = (torch.rand(b, S, H, device=dev) * 0.05).half()
    D = torch.randn(H, device=dev).half()
    h0 = (torch.randn(b, H, P, N, device=dev) * 0.1).float()

    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
    dA_cumsum = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
    summary_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    kernels["mamba3_chunk_precompute"](
        x.contiguous(), Bmat.contiguous(), C.contiguous(),
        A.contiguous(), dt.contiguous(), cb, dA_cumsum, summary_states)
    torch.mps.synchronize()

    prev_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    final_state = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    kernels["mamba3_inter_chunk_recur"](
        summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
        prev_states, final_state)
    torch.mps.synchronize()

    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    out = torch.zeros(b, S, H, P, device=dev, dtype=torch.float16)
    kernels["mamba3_chunk_scan_combine"](
        cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
        C.contiguous(), prev_states.half().contiguous(), D.contiguous(), out)
    torch.mps.synchronize()

    out_serial, hlast_serial = _serial_full_forward(
        C.cpu(), Bmat.cpu(), x.cpu(), A.cpu(), dt.cpu(), h0.cpu(), D.cpu())
    max_abs = float((out.float().cpu() - out_serial).abs().max())
    max_hlast = float((final_state.float().cpu() - hlast_serial).abs().max())
    with capsys.disabled():
        print(
            f"\n[region-flip parity] S={S} H={H} P={P} N={N} G={G} "
            f"out max|abs|={max_abs:.3e} h_last max|abs|={max_hlast:.3e}"
        )
    assert not bool(torch.isnan(out).any())
    assert max_abs < 5e-4, f"region-flip forward parity {max_abs:.3e} >= 5e-4"
    assert max_hlast < 5e-4, f"region-flip final-state parity {max_hlast:.3e} >= 5e-4"
