#!/bin/zsh
# Determinism guard: run the chunked-vs-path_b parity test in N FRESH processes,
# production direct path (cache ENABLED, gate UNSET so direct_pipeline is live),
# memguard via the test's own surface. Each process is a full vjp through the
# direct Metal pipeline. N/N must PASS. Prints DET_RESULT pass/fail counts.
set -u
cd /Volumes/external/sources/cppmega.mlx
N="${1:-50}"
PASS=0
FAIL=0
FAILED_RUNS=""
for i in $(seq 1 $N); do
  TILELANG_DISABLE_CACHE=0 timeout 120 .venv/bin/python -m pytest \
    tests/test_mamba3_path_c_chunked_vs_path_b.py -x -q >/tmp/det_run_$i.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
    FAILED_RUNS="$FAILED_RUNS $i(rc=$rc)"
  fi
  printf "run %d/%d rc=%d pass=%d fail=%d\n" "$i" "$N" "$rc" "$PASS" "$FAIL"
done
echo "DET_RESULT pass=$PASS fail=$FAIL of N=$N"
if [ -n "$FAILED_RUNS" ]; then
  echo "DET_FAILED_RUNS:$FAILED_RUNS"
fi
