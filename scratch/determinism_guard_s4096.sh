#!/bin/zsh
# Determinism guard at s4096 (64 chunks): run the chunked-vs-GOLD bit-correctness
# check in N FRESH processes through the s4096 B1 dispatch boundary (the un-fuse
# fix a99bef7 touched B1). Each process is a full vjp + all-8-grad < 1e-3 assert.
# N/N must PASS. Prints DET4096_RESULT pass/fail counts. memguard 70 inside script.
set -u
cd /Volumes/external/sources/cppmega.mlx
N="${1:-50}"
PASS=0
FAIL=0
FAILED_RUNS=""
for i in $(seq 1 $N); do
  TILELANG_DISABLE_CACHE=0 timeout 180 .venv/bin/python \
    scratch/sweep_seqlen_bitcorrect_phase.py 4096 >/tmp/det4096_run_$i.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
    FAILED_RUNS="$FAILED_RUNS $i(rc=$rc):$(grep -h SEQLEN_RESULT /tmp/det4096_run_$i.log | tail -1)"
  fi
  printf "run %d/%d rc=%d pass=%d fail=%d %s\n" "$i" "$N" "$rc" "$PASS" "$FAIL" \
    "$(grep -h SEQLEN_RESULT /tmp/det4096_run_$i.log | tail -1)"
done
echo "DET4096_RESULT pass=$PASS fail=$FAIL of N=$N"
if [ -n "$FAILED_RUNS" ]; then
  echo "DET4096_FAILED_RUNS:$FAILED_RUNS"
fi
