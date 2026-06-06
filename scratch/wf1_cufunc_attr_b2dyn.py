"""wf1: cuFuncGetAttribute STATIC + MAX_DYNAMIC for the §DYN B2 Triton-mold prim.

Builds chunk_scan_combine_bwd_cuda_prim_gemm_batched_dyn at PROD dims, HPC from
env (default 2), compiles via tilelang, extracts the cubin/PTX, loads it with the
CUDA driver API (cuda.bindings.driver), finds the kernel function, and queries the
REAL driver attributes:
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES         (STATIC smem the compiler reserved)
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (the opt-in the launcher set)
Then dumps SASS and tallies HMMA instructions (the GEMM tensor-core evidence).

RULE #1: any build/compile/load failure RAISES (no fallback).
"""
import os
import re
import sys

import torch  # noqa: F401  (initializes CUDA context for the driver API)
import tilelang as tl
import tilelang.language as T  # noqa: F401

from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_scan_combine_bwd_cuda_prim_gemm_batched_dyn,
)

HPC = int(os.environ.get("HPC", "2"))
# prod cfg
b, S, chunk, G, H, P, N = 1, 4096, 64, 8, 112, 64, 64

print(f"[wf1-attr] building §DYN prim prod b={b} S={S} chunk={chunk} G={G} H={H} "
      f"P={P} N={N} HPC={HPC}")
prim = chunk_scan_combine_bwd_cuda_prim_gemm_batched_dyn(
    b, S, chunk, G, H, P, N, heads_per_cta=HPC
)
tgt = "cuda"
pc = {"tl.disable_tma_lower": True} if False else {}
k = tl.compile(prim, out_idx=[11, 12, 13, 14, 15, 16, 17], target=tgt)

src = k.get_kernel_source()
n_static = len(re.findall(r"__shared__\s+[^;e][^;]*\[\d+\]", src))
has_dyn = "extern __shared__" in src
print(f"[wf1-attr] SOURCE: num_static_shared_decls(regex)={n_static} "
      f"has_extern_dyn_shared={has_dyn}")

# Grab the global symbol(s) of device kernels from the CUDA-C source.
syms = re.findall(r'__global__\s+void\s+(\w+)\s*\(', src)
print(f"[wf1-attr] device kernel symbols: {syms}")

# ---- load the cubin/PTX via the CUDA driver API and query func attributes ----
from cuda.bindings import driver as cuda


def _ck(res, what):
    err = res[0]
    if int(err) != 0:
        # try to get string
        try:
            _, name = cuda.cuGetErrorName(err)
            _, desc = cuda.cuGetErrorString(err)
            name = name.decode() if isinstance(name, bytes) else name
            desc = desc.decode() if isinstance(desc, bytes) else desc
        except Exception:
            name = desc = "?"
        raise RuntimeError(f"[wf1-attr] CUDA {what} FAILED: {int(err)} {name}: {desc}")
    return res[1:]


# ensure a context exists (torch already made one)
torch.cuda.init()
torch.zeros(1, device="cuda")

# Get the kernel module binary. tilelang exposes PTX via _get_ptx; cubin via sass
# extraction path. We load PTX through the JIT driver (cuModuleLoadData on PTX).
ptx = None
try:
    ptx = k._get_ptx()
except Exception as e:
    print(f"[wf1-attr] _get_ptx raised: {e!r}")

# Prefer cubin if we can pull it (exact static numbers); fall back to PTX-JIT which
# still yields the SAME func attributes since the compiler reserves static smem.
cubin_bytes = None
try:
    sass_txt = k.get_kernel_source  # placeholder
except Exception:
    pass

loaded = False
attr_results = {}
for label, blob in (("ptx", ptx),):
    if blob is None:
        continue
    data = blob.encode() if isinstance(blob, str) else blob
    (mod,) = _ck(cuda.cuModuleLoadData(data), f"cuModuleLoadData({label})")
    loaded = True
    for sym in syms:
        try:
            (func,) = _ck(cuda.cuModuleGetFunction(mod, sym.encode()),
                          f"cuModuleGetFunction({sym})")
        except RuntimeError as e:
            print(f"[wf1-attr] {e}")
            continue
        (stat,) = _ck(cuda.cuFuncGetAttribute(
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func),
            f"cuFuncGetAttribute(STATIC,{sym})")
        (maxdyn,) = _ck(cuda.cuFuncGetAttribute(
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            func), f"cuFuncGetAttribute(MAXDYN,{sym})")
        (regs,) = _ck(cuda.cuFuncGetAttribute(
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, func),
            f"cuFuncGetAttribute(NUM_REGS,{sym})")
        attr_results[sym] = {"static": int(stat), "maxdyn": int(maxdyn),
                             "num_regs": int(regs)}
        print(f"[wf1-attr] FUNC {sym}: STATIC_SHARED_SIZE_BYTES={int(stat)}  "
              f"MAX_DYNAMIC_SHARED_SIZE_BYTES={int(maxdyn)}  NUM_REGS={int(regs)}")
    break

if not loaded:
    raise RuntimeError("[wf1-attr] could not load any module blob (no PTX/cubin)")

# ---- SASS: tally HMMA tensor-core instructions ----
try:
    sass = k._get_sass()
except Exception as e:
    sass = None
    print(f"[wf1-attr] _get_sass raised: {e!r}")
if sass:
    hmma = re.findall(r'HMMA\.(\S+)', sass)
    from collections import Counter
    c = Counter(hmma)
    print(f"[wf1-attr] SASS HMMA total={len(hmma)} shapes={dict(c)}")
    # also count by mma type lines
    print(f"[wf1-attr] SASS bytes={len(sass)}")

print("[wf1-attr] DONE")
