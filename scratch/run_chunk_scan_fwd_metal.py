"""Compile + RUN chunk_scan_fwd_metal on the Metal target and check parity
vs the upstream torch SSD ref_program. Replicates the validation behind the
handoff claim (compiles to MSL, 2048 tg, ~1.2e-4 fp16, ~0.27ms @ S=4096).
"""
import sys, time
sys.path.insert(0, "scratch")
import torch
import tilelang
import tilelang.language as T
from einops import rearrange, repeat
from mamba3_chunked_forward_tilelang import chunk_scan_fwd_metal, grid_blocks


def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum("bchls,bhcs,bcshp->bclhp",
                       scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = (torch.einsum("bclhn,bchpn->bclhp",
                rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                prev_states.to(C.dtype)) * state_decay_out)
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out


def run(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, dev):
    nchunks = seqlen // chunk_size
    prim = chunk_scan_fwd_metal(batch, seqlen, chunk_size, ngroups, nheads,
                                headdim, dstate, block_M=64, block_N=16,
                                block_K=64, block_Dstate=128, threads=128)
    from cppmega_mlx.nn._tilelang._msl_transform import _as_metal_target
    mtarget = _as_metal_target("metal -thread_warp_size=32")
    kernel = tilelang.compile(prim, out_idx=[7], target=mtarget)
    torch.manual_seed(0)
    dt_dtype = torch.float16
    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=dev, dtype=dt_dtype) * 0.1
    x = torch.randn(batch, seqlen, nheads, headdim, device=dev, dtype=dt_dtype) * 0.1
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device=dev, dtype=dt_dtype) * 0.05
    # dA_cumsum monotonically increasing within chunk (cumsum of negative a*dt)
    a = -torch.rand(batch, nheads, nchunks, chunk_size, device=dev, dtype=torch.float32) * 0.05
    dA_cumsum = torch.cumsum(a, dim=-1).to(dt_dtype)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=dev, dtype=dt_dtype) * 0.1
    prev_states = torch.randn(batch, nchunks, nheads, headdim, dstate, device=dev, dtype=dt_dtype) * 0.1
    D = torch.randn(nheads, device=dev, dtype=dt_dtype)
    out = torch.zeros(batch, seqlen, nheads, headdim, device=dev, dtype=dt_dtype)
    kernel(cb.contiguous(), x.contiguous(), dt.contiguous(), dA_cumsum.contiguous(),
           C.contiguous(), prev_states.contiguous(), D.contiguous(), out)
    if dev == "mps":
        torch.mps.synchronize()
    ref = ref_program(cb, x, dt, dA_cumsum, C, prev_states, D).to(out.dtype)
    diff = (out.float() - ref.float()).abs()
    mad = float(diff.max()); nan = bool(torch.isnan(out).any())
    total, grid = grid_blocks(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, 64, 16)
    # bench
    iters = 50
    cbc,xc,dtc,dAc,Cc,psc,Dc = (cb.contiguous(),x.contiguous(),dt.contiguous(),
        dA_cumsum.contiguous(),C.contiguous(),prev_states.contiguous(),D.contiguous())
    ob=torch.zeros_like(out)
    for _ in range(5):
        kernel(cbc,xc,dtc,dAc,Cc,psc,Dc,ob)
    if dev == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(iters):
        kernel(cbc,xc,dtc,dAc,Cc,psc,Dc,ob)
    if dev == "mps":
        torch.mps.synchronize()
    ms = (time.time() - t0) / iters * 1e3
    return mad, nan, total, grid, ms


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev}")
    for (B, S, C_, H, P, N) in [(1, 256, 256, 1, 64, 128), (1, 1024, 256, 8, 64, 128), (1, 4096, 256, 8, 64, 128)]:
        mad, nan, total, grid, ms = run(B, S, C_, 1, H, P, N, dev)
        print(f"S={S} chunk={C_} H={H} P={P} N={N} | "
              f"grid={grid} tg={total} | max|d|={mad:.3e} nan={nan} | {ms:.3f} ms/iter | "
              f"{'PASS' if (mad < 5e-2 and not nan) else 'FAIL'}")
