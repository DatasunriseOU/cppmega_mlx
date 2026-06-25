"""FOCUSED Y_diag C-read locality probe for B2 (chunk_scan_combine_bwd).

RULE #1: TIMING PROBES, NOT correctness paths. We build a stripped kernel that
runs ONLY the Y_diag dinp transpose block (with the SAME dacs/dY/DYX staging the
production prim does up-front), and vary ONLY the SOURCE of the C operand inside
the hot ll-reduction. Every variant keeps the L*headdim*dstate dinp store AND the
ll-loop live (acc seeded from a runtime-gated global read so the compiler CANNOT
DCE the loop -- the nam56r DCE trap in the block-ablation is avoided here). The
outputs are WRONG on purpose for the CONST/seed variants; they are NEVER fed into
the bit-correct chain, only timed. memguard 70 on. No fabrication: every number is
a real torch.mps-synchronized median over ITERS GPU dispatches.

Variants (Y_diag-only kernel; everything before the block is the prod staging):
  C_GLOBAL : C[base+ll, group, nn] read from DEVICE every ll-iter (prod baseline)
  C_CONST  : C operand replaced by a runtime constant g (==1.0) -> ISOLATES the
             per-ll global C-read cost (loop arithmetic + dY shared-read + store
             all KEPT identical). delta(C_GLOBAL - C_CONST) == the C-read cost.
  C_SHARED : C[L,dstate] staged ONCE into a threadgroup fp16 tile (coalesced
             L*dstate loads, one barrier), then read from SHARED in the ll-loop.
             This is the memory-locality lever; delta(C_GLOBAL - C_SHARED) == the
             redundant-global-traffic win. Per-cell ll accumulation order is
             IDENTICAL to prod (only the C source changes) -> bit-exact in prod.
  C_SHARED_NOSTAGE : same shared READ but WITHOUT the staging copy/barrier (reads
             uninitialised smem -> WRONG, timing only) -> isolates the staging
             (copy+barrier) overhead from the shared-read benefit.
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
ITERS = int(os.environ.get("BENCH_ITERS", "30"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "6"))


def build_ydiag(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, *, threads=128, variant="C_GLOBAL"):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E
    V = variant

    @T.prim_func
    def main(
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        dout: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore  (dY seed)
        gate_buf: T.Tensor((1,), accum_dtype),  # runtime ==1.0; opaque -> no DCE
        dinp: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size
            g = gate_buf[0]

            dacs = T.alloc_shared((L,), accum_dtype)
            dY = T.alloc_shared((L, headdim), accum_dtype)
            Csh = T.alloc_shared((L, dstate), dtype)  # staged C tile (used by C_SHARED*)

            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
            # dY[l,p] seeded from a global read (so the ll-loop cannot be proven
            # constant -> no DCE). In prod dY is the post-split dy_v; here any live
            # per-(l,p) value works for TIMING the read pattern.
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                dY[ll, pp] = T.Cast(accum_dtype, dout[batch_idx, base + ll, head_idx, pp])
            T.sync_threads()

            # stage C[L,dstate] ONCE (only C_SHARED actually consumes it; the copy
            # is what we time). C layout (b, seqlen, ngroups, dstate): row base+ll.
            if "C_SHARED" == V:
                for ln in T.Parallel(L * dstate):
                    ll = ln // dstate
                    nn = ln % dstate
                    Csh[ll, nn] = C[batch_idx, base + ll, group_idx, nn]
                T.sync_threads()

            for spn in T.serial(0, L * headdim * dstate, threads):
                lane = T.get_thread_binding(0)
                spni = spn + lane
                if spni < L * headdim * dstate:
                    ss = spni // (headdim * dstate)
                    rem = spni % (headdim * dstate)
                    pp = rem // dstate
                    nn = rem % dstate
                    sidx = base + ss
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)
                    for ll in T.serial(ss, L):
                        lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                        if V == "C_GLOBAL":
                            c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                        elif V == "C_CONST":
                            c_v = g  # runtime const -> NO global C-read
                        elif V == "C_SHARED":
                            c_v = T.Cast(accum_dtype, Csh[ll, nn])
                        elif V == "C_SHARED_NOSTAGE":
                            c_v = T.Cast(accum_dtype, Csh[ll, nn])  # unstaged smem (wrong, timing only)
                        else:
                            raise ValueError(V)
                        acc[0] = acc[0] + dY[ll, pp] * c_v * lmat
                    dinp[batch_idx, sidx, head_idx, pp, nn] = acc[0]
            T.sync_threads()

    return main


def compile_variant(shape, variant):
    prim = build_ydiag(**shape, variant=variant)
    tgt = _resolve_chunked_compile_target(None)
    # inputs: dA_cumsum, C, dout, gate_buf ; output: dinp (slot 4)
    return tilelang.compile(prim, out_idx=[4], target=tgt)


def make_inputs(shape, seed=0):
    b, S, ch, G, H, P, N = (shape["batch"], shape["seqlen"], shape["chunk_size"],
                            shape["ngroups"], shape["nheads"], shape["headdim"], shape["dstate"])
    nchunks = S // ch
    rng = np.random.RandomState(seed)
    def t16(*s): return torch.tensor((rng.randn(*s) * 0.1).astype(np.float32), device=DEV, dtype=torch.float16).contiguous()
    ins = dict(
        dA=t16(b, H, nchunks, ch), C=t16(b, S, G, N), dout=t16(b, S, H, P),
        gate=torch.ones(1, device=DEV, dtype=torch.float32),
    )
    return ins, (b, S, ch, nchunks, G, H, P, N)


def time_variant(k, ins, dims):
    b, S, ch, nchunks, G, H, P, N = dims
    def run():
        dinp = torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32)
        k(ins["dA"], ins["C"], ins["dout"], ins["gate"], dinp)
        torch.mps.synchronize()
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6), min(ts) * 1e6


SHAPES = {
    "tested-S512": dict(batch=1, seqlen=512, chunk_size=64, ngroups=1, nheads=2, headdim=64, dstate=16),
    "nam56r-H128": dict(batch=1, seqlen=512, chunk_size=64, ngroups=8, nheads=128, headdim=64, dstate=64),
}
VARIANTS = ["C_GLOBAL", "C_CONST", "C_SHARED", "C_SHARED_NOSTAGE"]

for sname, shape in SHAPES.items():
    print(f"\n===== {sname} (N={shape['dstate']}) =====")
    ins, dims = make_inputs(shape)
    base_us = None
    for v in VARIANTS:
        try:
            k = compile_variant(shape, v)
        except Exception as e:
            print(f"  {v:18s} COMPILE-FAIL: {type(e).__name__}: {str(e)[:140]}")
            continue
        med, mn = time_variant(k, ins, dims)
        if v == "C_GLOBAL": base_us = med
        delta = (base_us - med) if base_us is not None else 0.0
        pct = (100.0 * delta / base_us) if base_us else 0.0
        print(f"  {v:18s} median={med:9.1f}us  min={mn:9.1f}us  delta_from_GLOBAL={delta:8.1f}us ({pct:5.1f}%)")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
