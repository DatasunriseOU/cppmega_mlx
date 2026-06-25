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
L,P,N=64,64,16

# Mimic the B2 prim structure: persistent dY16+DYX, transient opA/opB + frag staging.
def build():
    @T.prim_func
    def main(dY: T.Tensor((L,P), T.float16), Xm: T.Tensor((L,P), T.float16),
             Cm: T.Tensor((L,N), T.float16),
             OutDYX: T.Tensor((L,L), T.float16), OutDC: T.Tensor((P,N), T.float32)):
        with T.Kernel(1, threads=128) as bx:
            dY16=T.alloc_shared((L,P),T.float16)     # persistent
            DYX=T.alloc_shared((L,L),T.float16)      # persistent
            opA=T.alloc_shared((L,P),T.float16)      # transient
            opB=T.alloc_shared((L,N),T.float16)      # transient
            dyxf=T.alloc_fragment((L,L),T.float32)
            dcf=T.alloc_fragment((P,N),T.float32)
            T.copy(dY,dY16); T.copy(Xm,opA)
            T.clear(dyxf); T.gemm(dY16,opA,dyxf,transpose_B=True); T.copy(dyxf,DYX)
            T.copy(Cm,opB)
            T.clear(dcf); T.gemm(opA,opB,dcf,transpose_A=True); T.copy(dcf,OutDC)
            for ls in T.Parallel(L*L):
                ll=ls//L; ss=ls%L
                OutDYX[ll,ss]=DYX[ll,ss]
    return main

mod=tilelang.lower(build(), target="metal")
src=mod.kernel_source if hasattr(mod,'kernel_source') else str(mod)
pools=re.findall(r'threadgroup\s+uchar\s+(\w+)\[(\d+)\]', src)
tot=sum(int(n) for _,n in pools)
print(f"pools={pools} total={tot} smma={src.count('simdgroup_multiply_accumulate')}")
k=tilelang.compile(build(), out_idx=[3,4], target="metal")
print("PSO BUILT OK at", tot, "-> Metal accepts:", k is not None)
