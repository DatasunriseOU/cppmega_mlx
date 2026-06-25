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

def build(MT):
    @T.prim_func
    def main(dY: T.Tensor((L,P), T.float16), Xm: T.Tensor((L,P), T.float16), Out: T.Tensor((L,L), T.float16)):
        with T.Kernel(1, threads=128) as bx:
            dy=T.alloc_shared((L,P),T.float16); xt=T.alloc_shared((L,P),T.float16)
            DYX=T.alloc_shared((L,L),T.float16)
            cf=T.alloc_fragment((MT,L),T.float32)
            T.copy(dY,dy); T.copy(Xm,xt)
            for m0 in T.serial(0, L, MT):
                T.clear(cf)
                T.gemm(dy[m0:m0+MT,0:P], xt, cf, transpose_B=True)
                T.copy(cf, DYX[m0:m0+MT,0:L])
            T.copy(DYX, Out)
    return main

for MT in (64,32,16):
    try:
        mod=tilelang.lower(build(MT), target="metal")
        src=mod.kernel_source if hasattr(mod,'kernel_source') else str(mod)
        pools=re.findall(r'threadgroup\s+uchar\s+(\w+)\[(\d+)\]', src)
        tot=sum(int(n) for _,n in pools)
        print(f"MT={MT}: pools={pools} total={tot} smma={src.count('simdgroup_multiply_accumulate')}")
    except Exception as e:
        print(f"MT={MT}: RAISED {type(e).__name__}: {str(e)[:150]}")
