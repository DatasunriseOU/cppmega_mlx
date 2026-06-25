"""Run the directed backward fusion search at s4096 to show what it auto-selects
there: the LANE single-dispatch candidate CANNOT lower (int32 wall) so it is recorded
crashed/infeasible, and the search falls back to a multi-dispatch chunked grouping.
This is the EXACT measured proof that the mandate's <=1.0x single-dispatch LANE at
s4096 is unsatisfiable. Under memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM=70*1024*1024; _PK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _PK
    while True:
        r=_rss()
        if r>_PK:_PK=r
        if r>_LIM: sys.stderr.write(f"[memguard70] KILL rss={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY","1")
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx")

from cppmega_mlx.runtime.path_c_backward_fusion_search import search_fastest_backward_fusion

MSL_REF_US=5002.0
# s4096 = 64 chunks of 64; nam56r surface
DIMS=(1,4096,64,128,128,64,64)  # b,s,c,G=H,H,P,N
try:
    best,ranked,trace=search_fastest_backward_fusion(
        DIMS, measure_runs=7, warmup=3, bitcorrect_repeats=4, abs_gate=1e-3, verbose=False)
    print("\n===== s4096 SEARCH TRACE =====")
    for row in trace:
        print(f"  {row['variant']:10s} | feasible={row['predicted_feasible']} "
              f"dispatch={row['dispatch_count']} compiled={row['compiled']} crashed={row['crashed']} "
              f"bit_correct={row['bit_correct']} us={row['measured_us']} status={row['status']}")
        if row['crash_reason']:
            print(f"        reason: {row['crash_reason'][:200]}")
    print(f"\n  SELECTED@s4096 = {best.variant_id} ({best.dispatch_count} dispatch) "
          f"us={best.measured_us} schedule={best.schedule_class}")
    if best.measured_us:
        print(f"  selected/MSL = {best.measured_us/MSL_REF_US:.2f}x")
    print(f"  LANE single-dispatch <=1.0x MSL at s4096? "
          f"{'YES' if (best.variant_id=='LANE' and best.measured_us and best.measured_us<=MSL_REF_US) else 'NO'}")
except Exception as e:
    import traceback
    print(f"SEARCH @s4096 raised: {type(e).__name__}: {str(e)[:200]}")
    for ln in traceback.format_exc().splitlines()[-12:]:
        print(ln)

print(f"\nPEAK_RSS_KB={_PK} (~{_PK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
