"""wf1: cuFuncGetAttribute on the §DYN cubin (no PTX-JIT, no version error)."""
import sys
import torch
from cuda.bindings import driver as cuda

torch.zeros(1, device="cuda")  # ensure context

CUBIN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wf1_dyn.cubin"
SYM = b"main_kernel"


def ck(res, what):
    err = res[0]
    if int(err) != 0:
        _, nm = cuda.cuGetErrorName(err)
        _, ds = cuda.cuGetErrorString(err)
        raise RuntimeError(f"{what} FAILED {int(err)} "
                           f"{nm.decode() if isinstance(nm, bytes) else nm}: "
                           f"{ds.decode() if isinstance(ds, bytes) else ds}")
    return res[1:]


with open(CUBIN, "rb") as f:
    data = f.read()
(mod,) = ck(cuda.cuModuleLoadData(data), "cuModuleLoadData(cubin)")
(func,) = ck(cuda.cuModuleGetFunction(mod, SYM), "cuModuleGetFunction(main_kernel)")
A = cuda.CUfunction_attribute
(stat,) = ck(cuda.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func),
             "STATIC")
(maxdyn,) = ck(cuda.cuFuncGetAttribute(
    A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func), "MAXDYN_default")
(regs,) = ck(cuda.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, func), "NUM_REGS")
print(f"[cubin-attr] main_kernel STATIC_SHARED_SIZE_BYTES={int(stat)}  "
      f"MAX_DYNAMIC_SHARED_SIZE_BYTES(default)={int(maxdyn)}  NUM_REGS={int(regs)}")

# Now SET the dynamic opt-in (what the launcher does) to the §DYN 91136 B and confirm
# the driver GRANTS it (HPC=2 launch-feasibility on sm_121).
REQ = 91136
_set = cuda.cuFuncSetAttribute(
    func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, REQ)
r = _set[0] if isinstance(_set, tuple) else _set
if int(r) != 0:
    _, ds = cuda.cuGetErrorString(r)
    print(f"[cubin-attr] cuFuncSetAttribute(MAXDYN={REQ}) REJECTED: {int(r)} "
          f"{ds.decode() if isinstance(ds, bytes) else ds}")
else:
    (maxdyn2,) = ck(cuda.cuFuncGetAttribute(
        A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func), "MAXDYN_after_set")
    print(f"[cubin-attr] cuFuncSetAttribute(MAXDYN={REQ}) GRANTED -> "
          f"MAX_DYNAMIC_SHARED_SIZE_BYTES now {int(maxdyn2)} (HPC=2 LAUNCHES)")

# device optin cap
dev = torch.cuda.current_device()
(cap,) = ck(cuda.cuDeviceGetAttribute(
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    dev), "OPTIN_CAP")
print(f"[cubin-attr] device MAX_SHARED_MEMORY_PER_BLOCK_OPTIN={int(cap)}")
print("[cubin-attr] DONE")
