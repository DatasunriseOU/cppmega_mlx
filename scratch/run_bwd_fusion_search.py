"""Run the FULL directed search (Phase A predict+rank + Phase B measured iterate)
on the chunked Mamba3 backward at the production nam56r surface, under memguard 70.

Reports the search trace per candidate (predicted-feasible? compiled? crashed?
measured us? bit-correct?) and the SELECTED fastest valid variant vs the current
3-dispatch baseline and the MSL 5002us reference. NO fabrication.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# --- memguard 70: self-imposed 70GB RSS killer (mandatory) ------------------
_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK_RSS_KB = 0


def _rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _memguard_thread():
    global _PEAK_RSS_KB
    while True:
        r = _rss_kb()
        if r > _PEAK_RSS_KB:
            _PEAK_RSS_KB = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} (~{r//1048576}GB) > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    search_fastest_backward_fusion,
)

MSL_REF_US = 5002.0  # the production MSL mamba3_mimo_bwd_metal reference @ S=128
DIMS = (1, 128, 64, 128, 128, 64, 64)  # b,s,c,G=H,H,P,N (nam56r)

best, ranked, trace = search_fastest_backward_fusion(
    DIMS, measure_runs=15, warmup=6, bitcorrect_repeats=16, abs_gate=1e-3,
    verbose=True,
)

print("\n\n========== SEARCH TRACE ==========")
for row in trace:
    print(f"  {row['variant']:10s} | predicted_feasible={row['predicted_feasible']} "
          f"dispatch={row['dispatch_count']} compiled={row['compiled']} "
          f"crashed={row['crashed']} bit_correct={row['bit_correct']} "
          f"measured_us={row['measured_us']} status={row['status']}")
    if row['crash_reason']:
        print(f"             reason: {row['crash_reason'][:160]}")

baseline = next(v for v in ranked if v.dispatch_count == 3)
print("\n========== RESULT ==========")
print(f"  baseline 3-dispatch median = {baseline.measured_us:.1f}us "
      f"(bit_correct={baseline.bit_correct})")
print(f"  SELECTED = {best.variant_id} ({best.dispatch_count} dispatch) "
      f"median = {best.measured_us:.1f}us bit_correct={best.bit_correct}")
recovered = baseline.measured_us - best.measured_us
print(f"  recovered = {recovered:.1f}us "
      f"({(3 - best.dispatch_count)} dispatch floors collapsed)")
print(f"  vs MSL ref {MSL_REF_US:.0f}us: selected/MSL = {best.measured_us / MSL_REF_US:.2f}x")
fusion_landed = "3->1" if best.dispatch_count == 1 else ("3->2" if best.dispatch_count == 2 else "none(3)")
print(f"  fusion landed: {fusion_landed}")
print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")
