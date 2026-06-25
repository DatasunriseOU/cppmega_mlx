"""LEVER B prototype: O(L) suffix-scan Y_diag (vs O(L^2) tri reduction).

RULE #1: TIMING + PARITY PROBE. The decay separates exactly:
    decay(ll,ss) = exp2((dacs[ll]-dacs[ss])*p) = exp2(dacs[ll]*p) / exp2(dacs[ss]*p)
so  dinp[ss,pp,nn] = sum_{ll>=ss} dY[ll,pp]*C[ll,nn]*decay(ll,ss)
                   = (1/E[ss]) * SUFFIX_{ll>=ss} ( dY[ll,pp]*C[ll,nn]*E[ll] ),  E[k]=exp2(dacs[k]*p)
which is an O(L) running suffix scan over ss (instead of an O(L) reduction per ss => O(L^2)).

NUMERICAL RISK: E[k]=exp2(dacs[k]*p)=exp(dacs[k]) underflows to 0 (and 1/E -> inf) once
|dacs| > ~87 (fp32). MITIGATION (LEVERB_SHIFT): subtract the chunk-max m=max_k dacs[k]
before exponentiating: Es[k]=exp2((dacs[k]-m)*p) in [0,1]; the m cancels in Es[ll]/Es[ss].
We measure BOTH the naive factored form (LEVERB_RAW, may be unstable) and the shifted form
(LEVERB_SHIFT, stable), and the WORST abs diff vs the O(L^2) reference (YD_FULL).

Because the suffix scan SERIALIZES ss, this variant grids (pp,nn) over lanes (P*N work-items)
and each lane runs the L-step suffix scan -- FEWER lanes than the current L*P*N, so the
speedup is a tradeoff (less work, less parallelism). memguard 70.
"""
import os, sys, threading, time
import numpy as np

_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import torch  # noqa
import tilelang  # noqa
import tilelang.language as T  # noqa
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import _LOG2E
from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import _resolve_chunked_compile_target

DEV = "mps"
ITERS = int(os.environ.get("BENCH_ITERS", "40"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))


def build_prim(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, *, threads=128, variant="YD_FULL"):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E
    V = variant

    @T.prim_func
    def main(
        dout: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        gate_buf: T.Tensor((1,), accum_dtype),
        dinp: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            dacs = T.alloc_shared((L,), accum_dtype)
            dY = T.alloc_shared((L, headdim), accum_dtype)
            Esh = T.alloc_shared((L,), accum_dtype)  # exp2((dacs[k]-m)*p), shifted
            mloc = T.alloc_shared((1,), accum_dtype)
            g = gate_buf[0]

            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                dY[ll, pp] = T.Cast(accum_dtype, dout[batch_idx, base + ll, head_idx, pp])
            T.sync_threads()

            # chunk-max of dacs (serial on lane 0; L=64 cheap) for the shifted form
            if V in ("LEVERB_SHIFT", "LEVERB_RAW"):
                if T.get_thread_binding(0) == 0:
                    m = T.alloc_local((1,), accum_dtype)
                    m[0] = dacs[0]
                    if V == "LEVERB_SHIFT":
                        for k in T.serial(1, L):
                            m[0] = T.max(m[0], dacs[k])
                    else:
                        m[0] = T.Cast(accum_dtype, 0)  # RAW: no shift (m=0)
                    mloc[0] = m[0]
                T.sync_threads()
                for k in T.Parallel(L):
                    Esh[k] = T.exp2((dacs[k] - mloc[0]) * p)
                T.sync_threads()

            if V == "YD_FULL":
                # O(L^2) reference: ss over L*P*N lanes, inner ll-sweep with exp2-in-loop
                for spn in T.serial(0, L * headdim * dstate, threads):
                    lane = T.get_thread_binding(0)
                    spni = spn + lane
                    if spni < L * headdim * dstate:
                        ss = spni // (headdim * dstate)
                        rem = spni % (headdim * dstate)
                        pp = rem // dstate
                        nn = rem % dstate
                        acc = T.alloc_local((1,), accum_dtype)
                        acc[0] = T.Cast(accum_dtype, 0)
                        for ll in T.serial(ss, L):
                            lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                            c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                            acc[0] = acc[0] + dY[ll, pp] * c_v * lmat
                        dinp[batch_idx, base + ss, head_idx, pp, nn] = acc[0]
            else:
                # LEVER B: grid (pp,nn) over P*N lanes; each lane suffix-scans ss=L-1..0.
                # S = running suffix sum of G[ll]=dY[ll,pp]*C[ll,nn]*Esh[ll]; dinp[ss]=S/Esh[ss].
                for pn in T.serial(0, headdim * dstate, threads):
                    lane = T.get_thread_binding(0)
                    pni = pn + lane
                    if pni < headdim * dstate:
                        pp = pni // dstate
                        nn = pni % dstate
                        S = T.alloc_local((1,), accum_dtype)
                        S[0] = T.Cast(accum_dtype, 0)
                        for ssr in T.serial(L):
                            ss = L - 1 - ssr
                            c_v = T.Cast(accum_dtype, C[batch_idx, base + ss, group_idx, nn])
                            S[0] = S[0] + dY[ss, pp] * c_v * Esh[ss]
                            dinp[batch_idx, base + ss, head_idx, pp, nn] = S[0] / Esh[ss]

    return main


def compile_variant(shape, variant):
    prim = build_prim(**shape, variant=variant)
    tgt = _resolve_chunked_compile_target(None)
    return tilelang.compile(prim, out_idx=[4], target=tgt)


def make_inputs(shape, seed=0, dacs_scale=1.0):
    b, S, ch, G, H, P, N = (shape["batch"], shape["seqlen"], shape["chunk_size"],
                            shape["ngroups"], shape["nheads"], shape["headdim"], shape["dstate"])
    nchunks = S // ch
    rng = np.random.RandomState(seed)
    def t16(*s): return torch.tensor((rng.randn(*s) * 0.1).astype(np.float32), device=DEV, dtype=torch.float16).contiguous()
    # dA_cumsum: realistic = monotone-decreasing cumsum of negative (A*dt) per chunk.
    dA = np.zeros((b, H, nchunks, ch), np.float32)
    steps = -np.abs(rng.randn(b, H, nchunks, ch).astype(np.float32)) * dacs_scale
    dA = np.cumsum(steps, axis=-1)
    ins = dict(
        dout=t16(b, S, H, P), C=t16(b, S, G, N),
        dA=torch.tensor(dA, device=DEV, dtype=torch.float16).contiguous(),
        gate=torch.ones(1, device=DEV, dtype=torch.float32),
    )
    return ins, (b, S, ch, nchunks, G, H, P, N)


def run_once(k, ins, dims):
    b, S, ch, nchunks, G, H, P, N = dims
    out = torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32)
    k(ins["dout"], ins["C"], ins["dA"], ins["gate"], out)
    torch.mps.synchronize()
    return out


def time_variant(k, ins, dims):
    for _ in range(WARMUP): run_once(k, ins, dims)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); run_once(k, ins, dims); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6), min(ts) * 1e6


SHAPES = {
    "tested-S512": dict(batch=1, seqlen=512, chunk_size=64, ngroups=1, nheads=2, headdim=64, dstate=16),
    "nam56r-H128": dict(batch=1, seqlen=512, chunk_size=64, ngroups=8, nheads=128, headdim=64, dstate=64),
}
VARIANTS = ["YD_FULL", "LEVERB_SHIFT", "LEVERB_RAW"]
# dacs_scale: 1.0 ~ benign (|dacs| small); 3.0 ~ stress (|dacs| can exceed 87 over 64 steps)
for dacs_scale in [1.0, 3.0]:
    print(f"\n############ dacs_scale={dacs_scale} (intra-chunk decay magnitude) ############")
    for sname, shape in SHAPES.items():
        print(f"\n===== {sname} =====")
        ins, dims = make_inputs(shape, dacs_scale=dacs_scale)
        dmin = float(ins["dA"].min().item()); dmax = float(ins["dA"].max().item())
        print(f"  dacs range = [{dmin:.1f}, {dmax:.1f}]  (span {dmax-dmin:.1f}; fp32 exp underflow @ ~-87)")
        res = {}; outs = {}
        for v in VARIANTS:
            k = compile_variant(shape, v)
            med, mn = time_variant(k, ins, dims)
            res[v] = med; outs[v] = run_once(k, ins, dims)
            print(f"  {v:13s} median={med:9.1f}us  min={mn:9.1f}us")
        ref = outs["YD_FULL"]
        denom = float(ref.abs().max().item()) + 1e-12
        for v in ("LEVERB_SHIFT", "LEVERB_RAW"):
            d = outs[v]
            finite = torch.isfinite(d).all().item()
            worst = float((ref - d).abs().max().item()) if finite else float("inf")
            print(f"  -- {v:13s} WORST vs YD_FULL = {worst:.3e} (rel {worst/denom:.3e}) finite={finite} gate<1e-3")
        full = res["YD_FULL"]
        print(f"  -- LEVER B (SHIFT) speedup = {full - res['LEVERB_SHIFT']:8.1f}us "
              f"({100.0*(full-res['LEVERB_SHIFT'])/full:5.1f}%; B={res['LEVERB_SHIFT']:.1f}us vs FULL={full:.1f}us)")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
