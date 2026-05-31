"""Parity + occupancy tests for the PROVEN chunked-parallel forward scan-core.

Validates :mod:`cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core` — the
productionized SSD 4-step chunked forward that replaces the O(S) serial
single-threadgroup Path-C forward scan with a many-threadgroup grid.

Two contracts are checked:
  1. ``chunk_scan_fwd_grid`` reports the multi-threadgroup grid (occupancy
     proof: 2048 threadgroups at full scale vs 1 for the serial forward).
  2. ``chunk_scan_fwd_metal_prim`` COMPILES to MSL and RUNS on Metal, matching
     the torch SSD reference within fp16 tolerance with no NaN.

The Metal run/parity test is skipped when torch+mps is unavailable (CI without
Apple silicon); the grid/occupancy test runs everywhere.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
    MAMBA3_CHUNKED_FWD_BLOCK_M,
    MAMBA3_CHUNKED_FWD_BLOCK_N,
    chunk_scan_fwd_grid,
)


def test_chunk_scan_fwd_grid_is_multi_threadgroup_full_scale():
    """Full-scale (S=4096, H=8) chunked grid is 2048 threadgroups, not 1."""
    total, grid = chunk_scan_fwd_grid(
        batch=1,
        seqlen=4096,
        chunk_size=256,
        ngroups=1,
        nheads=8,
        headdim=64,
        dstate=128,
    )
    assert grid == (8, 16, 16)
    assert total == 2048
    # The whole point of the integration: many threadgroups, not the serial 1.
    assert total > 1


def test_chunk_scan_fwd_grid_scales_with_sequence():
    """Threadgroup count grows with S (gz = batch*nchunks)."""
    t256, _ = chunk_scan_fwd_grid(1, 256, 256, 1, 8, 64, 128)
    t1024, _ = chunk_scan_fwd_grid(1, 1024, 256, 1, 8, 64, 128)
    t4096, _ = chunk_scan_fwd_grid(1, 4096, 256, 1, 8, 64, 128)
    assert t256 < t1024 < t4096


def test_chunk_scan_fwd_grid_requires_divisible_seqlen():
    """RULE #1: no silent padding fallback for non-divisible seqlen."""
    with pytest.raises(ValueError, match="divisible"):
        chunk_scan_fwd_grid(1, 300, 256, 1, 8, 64, 128)


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    import torch
    from einops import rearrange, repeat

    _, _, ngroups, _, _ = cb.shape
    _, _, nheads, _ = x.shape
    _, _, nchunks, chunk_size = dt.shape
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    dss = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dss)
    sd = cb * rearrange(decay, "b h c l s -> b c h l s")
    cm = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), 0
    )
    sd = sd.masked_fill(~cm, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        sd.to(x.dtype),
        dt.to(x.dtype),
        rearrange(x, "b (c s) h p -> b c s h p", c=nchunks),
    )
    sdo = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out = out + torch.einsum(
        "bclhn,bchpn->bclhp",
        rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
        prev_states.to(C.dtype),
    ) * sdo
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out


@pytest.mark.skipif(
    not _torch_mps_available(), reason="requires torch + Metal (mps) backend"
)
@pytest.mark.parametrize(
    "batch,seqlen,chunk,nheads,headdim,dstate,expect_tg",
    [
        (1, 256, 256, 1, 64, 128, 16),
        (1, 1024, 256, 8, 64, 128, 512),
        (1, 4096, 256, 8, 64, 128, 2048),
    ],
)
def test_chunk_scan_fwd_metal_compiles_runs_and_matches_ssd(
    batch, seqlen, chunk, nheads, headdim, dstate, expect_tg
):
    """The chunked scan-core compiles to MSL, runs on Metal, matches SSD ref."""
    import torch

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        compile_chunk_scan_fwd_metal,
    )

    ngroups = 1
    nchunks = seqlen // chunk
    total, _ = chunk_scan_fwd_grid(
        batch, seqlen, chunk, ngroups, nheads, headdim, dstate
    )
    assert total == expect_tg
    assert total > 1, "chunked scan must use many threadgroups, not the serial 1"

    kernel = compile_chunk_scan_fwd_metal(
        batch, seqlen, chunk, ngroups, nheads, headdim, dstate
    )

    dev = "mps"
    torch.manual_seed(0)
    cb = (torch.randn(batch, nchunks, ngroups, chunk, chunk, device=dev) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=dev) * 0.1).half()
    dt = (torch.rand(batch, nheads, nchunks, chunk, device=dev) * 0.05).half()
    dA = torch.cumsum(
        -torch.rand(batch, nheads, nchunks, chunk, device=dev) * 0.05, dim=-1
    ).half()
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=dev) * 0.1).half()
    ps = (torch.randn(batch, nchunks, nheads, headdim, dstate, device=dev) * 0.1).half()
    D = torch.randn(nheads, device=dev).half()

    out = torch.zeros(batch, seqlen, nheads, headdim, device=dev, dtype=torch.float16)
    # Calling convention: explicit pre-zeroed output buffer positionally (8th
    # arg); the torch-mps adapter does not allocate/return via out_idx.
    kernel(
        cb.contiguous(),
        x.contiguous(),
        dt.contiguous(),
        dA.contiguous(),
        C.contiguous(),
        ps.contiguous(),
        D.contiguous(),
        out,
    )
    torch.mps.synchronize()

    ref = _ref_program(cb, x, dt, dA, C, ps, D).to(out.dtype)
    assert not bool(torch.isnan(out).any()), "chunked Metal scan produced NaN"
    max_abs = float((out.float() - ref.float()).abs().max())
    # fp16 reduction-order tolerance (the documented non-bitwise regime).
    assert max_abs < 5e-2, f"chunked-vs-SSD max|abs diff|={max_abs:.3e} exceeds 5e-2"
