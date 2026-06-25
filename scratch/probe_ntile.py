import os, sys, threading, time
_LIM=70*1024*1024
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    while True:
        if _rss()>_LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
import tilelang, tilelang.language as T, re
L,P=64,64

# DYX[l,s] = sum_p dY[l,p]*x[s,p]  -> gemm(dY[L,P], x[L,P], C[L,L], transpose_B)
# N-tile: slice the SECOND operand rows (s) and output cols.
def build(NT):
    @T.prim_func
    def main(dY: T.Tensor((L,P), T.float16), Xm: T.Tensor((L,P), T.float16), Out: T.Tensor((L,L), T.float16)):
        with T.Kernel(1, threads=128) as bx:
            dy=T.alloc_shared((L,P),T.float16); xt=T.alloc_shared((L,P),T.float16)
            DYX=T.alloc_shared((L,L),T.float16)
            cf=T.alloc_fragment((L,NT),T.float32)
            T.copy(dY,dy); T.copy(Xm,xt)
            for n0 in T.serial(0, L, NT):
                T.clear(cf)
                T.gemm(dy, xt[n0:n0+NT,0:P], cf, transpose_B=True)
                T.copy(cf, DYX[0:L, n0:n0+NT])
            for ls in T.Parallel(L*L):
                ll=ls//L; ss=ls%L
                Out[ll,ss]=DYX[ll,ss]
    return main

for NT in (64,32,16):
    try:
        mod=tilelang.lower(build(NT), target="metal")
        src=mod.kernel_source if hasattr(mod,'kernel_source') else str(mod)
        pools=re.findall(r'threadgroup\s+uchar\s+(\w+)\[(\d+)\]', src)
        tot=sum(int(n) for _,n in pools)
        print(f"NT={NT}: pools={pools} total={tot} smma={src.count('simdgroup_multiply_accumulate')}")
    except Exception as e:
        print(f"NT={NT}: RAISED {type(e).__name__}: {str(e)[:150]}")
