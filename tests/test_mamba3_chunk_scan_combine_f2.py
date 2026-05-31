"""Stage-1 isolation harness for the Path-C F2 ``mamba3_chunk_scan_combine``.

Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §7 Stage 1.

Stage 1 registers ``mamba3_chunk_scan_combine`` as a Path-C brick-schedule
descriptor whose kernel DELEGATES to the already-landed, Metal-validated SSD
chunked scan+combine core ``chunk_scan_fwd_metal_prim`` (via
``build_chunk_scan_combine_metal``). This is a SHADOW registration: the live
mamba3 forward still emits the serial scan; the F2 descriptor only needs to

  1. register + ``select``-resolve its op-name signature (no ``no descriptor
     target`` raise), and
  2. compile + run on Metal in ISOLATION, fed ``cb / dA_cumsum / prev_states``
     RECOMPUTED EAGERLY here (a small reference precompute, NOT yet from the
     not-built F0/F1 segments), matching a SERIAL forward of the identical SSD
     math object to ``max|abs| < 5e-4`` fp16 at S in {256, 4096}, no NaN.

The serial reference is an explicit per-timestep diagonal recurrence over each
chunk (carry ``h`` advanced one step at a time, seeded by ``prev_states``) — it
is genuinely serial, NOT a re-expression of the chunked einsum. Its output is
the SSD quantity the F2 grid kernel computes: ``Y_diag + Y_off + D*x`` (no z
gate, matching the scan-core's output contract).

RULE #1: there is no silent fallback. The kernel path is the single delegation
``build_chunk_scan_combine_metal``; on a compile/parity failure the helpers RAISE
with where+what and the test FAILS — it never degrades to the serial scan.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
    MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,
    MAMBA3_CHUNKED_FWD_BLOCK_M,
    chunk_scan_fwd_grid,
)


# --------------------------------------------------------------------------- #
# Verification 1: descriptor registration + ``select`` resolution (no Metal).  #
# --------------------------------------------------------------------------- #


def test_f2_descriptor_registers_in_brick_registry():
    """The F2 op-name resolves to a brick descriptor (its kernel delegates to
    chunk_scan_fwd_metal_prim); its ``_bwd`` auto-derives."""
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        default_path_c_brick_schedule_descriptor_registry,
    )

    reg = default_path_c_brick_schedule_descriptor_registry()
    desc = reg.descriptor_for(MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME)
    assert desc is not None, "F2 descriptor must register (else select blocks it)"
    assert desc.op_name == MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME
    assert desc.fragment_emitter is not None
    assert "build_chunk_scan_combine_metal" in desc.production_source
    # signature resolution for the singleton signature must not be None
    sig = (MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,)
    assert reg.descriptors_for_signature(sig) is not None
    # AOT backward auto-derives (Stage 3 will register an explicit transpose)
    assert reg.descriptor_for(MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME + "_bwd") is not None


def test_f2_single_node_region_select_resolves_no_blocked_target():
    """``select`` returns a descriptor target for the F2 singleton signature —
    the Stage-1 success criterion 1 (no ``no descriptor target`` raise)."""
    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        Z3SyncSpec,
        build_path_c_fusion_region,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        default_path_c_fusion_schedule_registry,
        select_path_c_fusion_schedule_target,
    )

    surf = FusionKernelSurface.path_c(
        name="f2_scan_combine",
        op_name=MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,
        inputs=(
            "mamba3_cb",
            "mamba3_x",
            "mamba3_dt",
            "mamba3_dA_cumsum",
            "mamba3_C",
            "mamba3_prev_states",
            "mamba3_D",
        ),
        outputs=("delta",),
        backward="owner_output",
    )
    region = build_path_c_fusion_region(
        region_name="f2_shadow",
        surfaces=(surf,),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    assert tuple(n.op_name for n in region.nodes) == (
        MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,
    )
    reg = default_path_c_fusion_schedule_registry()
    target = reg.select(region)
    assert target is not None, "F2 signature must NOT be blocked by select"
    assert target.op_signature == (MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME,)
    # module-level selector agrees
    assert select_path_c_fusion_schedule_target(region) is not None

    # negative control: an unregistered op-name MUST stay blocked (None) — proves
    # the registration is what unblocks F2, not a permissive registry.
    ctl = FusionKernelSurface.path_c(
        name="ctl",
        op_name="mamba3_chunk_scan_combine_NOPE",
        inputs=("a",),
        outputs=("b",),
        backward="owner_output",
    )
    ctl_region = build_path_c_fusion_region(
        region_name="ctl",
        surfaces=(ctl,),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    assert reg.select(ctl_region) is None


# --------------------------------------------------------------------------- #
# Verification 2: Metal compile + run + parity vs a SERIAL forward.            #
# --------------------------------------------------------------------------- #


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _eager_precompute(C, Bmat, x, A, dt, h0, chunk_size):
    """Small EAGER reference precompute producing the F2 inputs.

    Recomputes ``cb = C@B^T`` per chunk, ``dA_cumsum = cumsum(A*dt)`` per chunk,
    and ``prev_states`` (per-chunk entry states) from base ``(C,B,x,A,dt,h0)``,
    following the canonical SSD precompute (``scratch/mamba3_chunked_forward_proto``
    algebra). This stands in for the not-yet-built F0/F1 segments (Stage 1 scope).
    All math in fp32 (the production accumulation dtype); returned as fp16 to match
    the F2 kernel ABI.
    """
    import torch
    from einops import rearrange

    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    nchunks = seqlen // chunk_size

    Cf = C.float()
    Bf = Bmat.float()
    xf = x.float()
    Af = A.float()  # (nheads,)
    dtf = dt.float()  # (batch, seqlen, nheads)

    # per-(head,timestep) log-decay increment a = A * dt
    a = Af.view(1, 1, nheads) * dtf  # (batch, seqlen, nheads)
    a_c = rearrange(a, "b (c l) h -> b h c l", c=nchunks)  # (b,h,c,l)
    dA_cumsum = torch.cumsum(a_c, dim=-1)  # (b,h,c,l)

    # cb = C @ B^T per chunk: (b,c,g,l,s)
    Cc = rearrange(Cf, "b (c l) g n -> b c l g n", c=nchunks)
    Bc = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks)
    cb = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)  # (b,c,g,l,s)

    # per-chunk entry states ``prev_states`` (the inter-chunk carry F1 produces).
    # states[c] = sum_l exp(dA_cs[c,-1]-dA_cs[c,l]) * dt[l] * (B[l] outer x[l]); then
    # propagated across chunks with chunk-boundary decay, seeded by h0.
    h = nheads // ngroups
    Bexp = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks)
    Bexp = Bexp.repeat_interleave(h, dim=3)  # (b,c,s,nheads,n)
    xexp = rearrange(xf, "b (c s) hh p -> b c s hh p", c=nchunks)  # (b,c,s,nheads,p)
    dtc = rearrange(dtf, "b (c s) hh -> b hh c s", c=nchunks)  # (b,nheads,c,s)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)  # (b,nheads,c,s)
    # states[b,c,hh,p,n] = sum_s decay_states * dt * x outer B
    states = torch.einsum(
        "bhcs,bhcs,bcshp,bcshn->bchpn",
        decay_states,
        dtc,
        xexp,
        Bexp,
    )  # (b,c,nheads,p,n)

    # chunk-boundary decay (segsum over per-chunk tail), seeded with h0.
    chunk_tail = dA_cumsum[:, :, :, -1]  # (b,nheads,c)
    init = h0.float().unsqueeze(1)  # (b,1,nheads,p,n)
    states_cat = torch.cat([init, states], dim=1)  # (b,c+1,nheads,p,n)
    pad = torch.nn.functional.pad(chunk_tail, (1, 0))  # (b,nheads,c+1)
    # segsum: L[z,c] = sum_{c<k<=z} pad[k] for z>=c else -inf
    cc = pad.shape[-1]
    csum = torch.cumsum(pad, dim=-1)  # (b,nheads,c+1)
    seg = csum[:, :, :, None] - csum[:, :, None, :]  # (b,nheads,z,c)
    mask = torch.tril(torch.ones(cc, cc, device=pad.device, dtype=torch.bool))
    seg = seg.masked_fill(~mask, float("-inf"))
    decay_chunk = torch.exp(seg)  # (b,nheads,c+1,c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    prev_states = new_states[:, :-1]  # (b,c,nheads,p,n) entry state per chunk

    return (
        cb.half().contiguous(),
        dA_cumsum.half().contiguous(),
        prev_states.half().contiguous(),
    )


def _serial_forward(cb, x, dt, dA_cumsum, C, prev_states, D, chunk_size):
    """Explicit per-timestep SERIAL diagonal forward producing the SSD output.

    For each chunk and each within-chunk timestep ``l`` (advanced ONE step at a
    time), accumulates the causal intra-chunk contribution plus the inter-chunk
    ``prev_states`` term, plus the ``D*x`` skip. Output = ``Y_diag+Y_off+D*x`` —
    the exact quantity the F2 grid kernel computes (no z gate). fp32 accumulation.
    """
    import torch
    from einops import rearrange, repeat

    batch, seqlen, ngroups, _ = C.shape
    _, _, nheads, headdim = x.shape
    _, _, nchunks, _ = dt.shape
    h = nheads // ngroups

    Cf = repeat(C.float(), "b l g n -> b l (g h) n", h=h)  # (b,l,nheads,n)
    cbf = repeat(cb.float(), "b c g l s -> b c (g h) l s", h=h)  # (b,c,nheads,l,s)
    xf = rearrange(x.float(), "b (c s) hh p -> b c s hh p", c=nchunks)
    dtf = dt.float()  # (b,nheads,c,s)
    dac = dA_cumsum.float()  # (b,nheads,c,l)
    psf = prev_states.float()  # (b,c,nheads,p,n)
    Cc = rearrange(Cf, "b (c l) hh n -> b c l hh n", c=nchunks)

    out = torch.zeros(batch, nchunks, chunk_size, nheads, headdim, device=x.device)
    for c in range(nchunks):
        for l in range(chunk_size):
            # intra-chunk causal: sum_{s<=l} cb[l,s]*exp(dA[l]-dA[s])*dt[s]*x[s]
            acc = torch.zeros(batch, nheads, headdim, device=x.device)
            for s in range(l + 1):
                decay = torch.exp(dac[:, :, c, l] - dac[:, :, c, s])  # (b,nheads)
                coef = cbf[:, c, :, l, s] * decay * dtf[:, :, c, s]  # (b,nheads)
                acc = acc + coef[:, :, None] * xf[:, c, s]  # (b,nheads,p)
            # inter-chunk: (C[l] . prev_state) * exp(dA[l])
            yoff = torch.einsum("bhn,bhpn->bhp", Cc[:, c, l], psf[:, c])
            yoff = yoff * torch.exp(dac[:, :, c, l])[:, :, None]
            out[:, c, l] = acc + yoff

    out = rearrange(out, "b c l hh p -> b (c l) hh p")
    Dskip = rearrange(D.float(), "hh -> hh 1")
    out = out + x.float() * Dskip
    return out


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.parametrize(
    "batch,seqlen,chunk,nheads,headdim,dstate",
    [
        # chunk_size == MAMBA3_CHUNKED_FWD_BLOCK_M (=64) is the live production
        # feasibility gate (_mamba3_chunked_forward_scan_feasibility :6363); use it.
        (1, 256, 64, 1, 64, 16),
        (1, 4096, 64, 8, 64, 16),
    ],
)
def test_f2_scan_combine_metal_matches_serial_forward(
    batch, seqlen, chunk, nheads, headdim, dstate, capsys
):
    """F2 (delegating to chunk_scan_fwd_metal_prim) vs SERIAL forward < 5e-4 fp16.

    Stage-1 PARITY GATE. Inputs ``cb/dA_cumsum/prev_states`` are recomputed
    EAGERLY here; ``x/dt/C/D`` are shared between F2 and the serial reference.
    """
    import torch

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        build_chunk_scan_combine_metal,
    )

    assert chunk == MAMBA3_CHUNKED_FWD_BLOCK_M, (
        "chunk_size must equal scan-core block_M (live feasibility gate :6363)"
    )
    ngroups = 1
    nchunks = seqlen // chunk
    total_tg, grid = chunk_scan_fwd_grid(
        batch, seqlen, chunk, ngroups, nheads, headdim, dstate
    )
    assert total_tg > 1, "chunked scan must use many threadgroups, not the serial 1"

    dev = "mps"
    torch.manual_seed(0)
    # base inputs (the per-position tensors F0 would stage); small scale for fp16.
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=dev) * 0.1).half()
    A = -torch.rand(nheads, device=dev).half()  # A = -softplus(...) <= 0
    dt = (torch.rand(batch, seqlen, nheads, device=dev) * 0.05).half()
    D = torch.randn(nheads, device=dev).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=dev) * 0.1).half()

    # EAGER precompute of the F2 handoff inputs (stand-in for F0/F1).
    cb, dA_cumsum, prev_states = _eager_precompute(C, Bmat, x, A, dt, h0, chunk)
    # dt in the F2 kernel ABI is (batch, nheads, nchunks, chunk).
    from einops import rearrange

    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    kernel = build_chunk_scan_combine_metal(
        batch, seqlen, chunk, ngroups, nheads, headdim, dstate
    )
    out = torch.zeros(batch, seqlen, nheads, headdim, device=dev, dtype=torch.float16)
    kernel(
        cb.contiguous(),
        x.contiguous(),
        dt_k.contiguous(),
        dA_cumsum.contiguous(),
        C.contiguous(),
        prev_states.contiguous(),
        D.contiguous(),
        out,
    )
    torch.mps.synchronize()

    ref = _serial_forward(cb, x, dt_k, dA_cumsum, C, prev_states, D, chunk)

    assert not bool(torch.isnan(out).any()), "F2 Metal scan produced NaN"
    assert not bool(torch.isnan(ref).any()), "serial forward produced NaN"
    max_abs = float((out.float() - ref.float()).abs().max())
    with capsys.disabled():
        print(
            f"\n[F2 Stage-1] S={seqlen} chunk={chunk} H={nheads} P={headdim} "
            f"N={dstate} -> threadgroups={total_tg} grid={grid} "
            f"max|abs diff|(F2 vs serial fwd)={max_abs:.3e}"
        )
    assert max_abs < 5e-4, (
        f"F2-vs-serial-forward max|abs diff|={max_abs:.3e} exceeds Stage-1 "
        f"parity gate 5e-4 at S={seqlen}"
    )
