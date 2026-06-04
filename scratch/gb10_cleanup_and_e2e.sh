#!/usr/bin/env bash
# ==============================================================================
# gb10_cleanup_and_e2e.sh  --  LEVER 3 (cleanup + e2e perf test), RUN ON gb10.
# ==============================================================================
# This script is the committed deliverable for LEVER 3. It is written to run
# ON the GB10 / DGX-Spark box (gx10-9cd4, aarch64 sbsa, sm_121) by the GB10
# phase agent, which is the SOLE GPU owner and runs the levers serially. It is
# NOT executed on the Mac and performs NO destructive op anywhere but gb10.
#
# It does two things, in order:
#   (A) CLEANUP -- re-verify-then-remove the three VERIFIED-SAFE stale items
#       (/home/dave/venv_unsloth, /usr/local/cuda-13.0, /usr/local/cuda-13.1 +
#        their two dangling gds ldconfig confs), capturing freed space, while
#        KEEPING cuda-13.2 (pip cu132 toolkit) + cuda-13.3 (NVRTC builtins) +
#        the cppmega-venv. System toolkits use `sudo -n`; if `sudo -n` is not
#        available it PRINTS the exact sudo command and SKIPS that item -- it
#        NEVER half-deletes (RULE #1: fail loud, no silent partial state).
#   (B) E2E PERF TEST -- run the existing pr6 e2e train_step runner at the
#       MEASURED-GO config (Stage1=8 layers, Stage2=2 layers, bs1 seq=4096 =>
#       4096 tok/step) and print a single machine-parseable RESULT line
#       (step ms, tok/s, peak GB, config) plus the Megatron 3399 tok/s compare.
#
# RULE #1 (NO SILENT FALLBACKS): every removal is gated on a fresh re-verify
# (target must NOT be the live `cuda`/`cuda-13` alternative target, must have
# no open handles via lsof, must have no running process referencing it). If a
# re-verify FAILS, the item is SKIPPED with a loud reason and the script exits
# non-zero at the end -- it never deletes a half-checked target, never falls
# back to a degraded delete, and the e2e perf test RAISES (propagates the
# runner's non-zero exit) rather than reporting a fabricated tok/s.
#
# USAGE (on gb10):   bash scratch/gb10_cleanup_and_e2e.sh
#   env knobs:
#     SKIP_CLEANUP=1   -> run only the e2e perf test (no removals)
#     SKIP_E2E=1       -> run only the cleanup (no GPU work)
#     PR6_STAGE1_LAYERS (default 8), PR6_STAGE2_LAYERS (default 2)
# ==============================================================================

set -u  # nounset; we deliberately do NOT `set -e` so a SKIPPED item cannot
        # abort the whole script -- we want cleanup + e2e to both be attempted
        # and a single explicit non-zero exit at the end if anything was unsafe.

# ---- gb10-local authoritative paths (from LEVER-3 research, verified) --------
CPPMEGA_DIR="/home/dave/source/cppmega_mlx"
PR6_RUNNER="${CPPMEGA_DIR}/scratch/pr6_cuda_e2e_train_step_gb10.py"
VENV_PY="/home/dave/cppmega-venv/bin/python"
TVM_PY="/home/dave/source/tilelang/3rdparty/tvm/python"
TVMFFI_PY="/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python"
TVM_LIBRARY_PATH_VAL="/home/dave/source/tilelang/build/lib"

# Cleanup targets (VERIFIED-SAFE in LEVER-3 research):
VENV_UNSLOTH="/home/dave/venv_unsloth"
CUDA_OLD_1="/usr/local/cuda-13.0"
CUDA_OLD_2="/usr/local/cuda-13.1"
GDS_CONF_1="/etc/ld.so.conf.d/gds-13-0.conf"
GDS_CONF_2="/etc/ld.so.conf.d/gds-13-1.conf"

# Must-keep (NEVER touched):
KEEP_CUDA_132="/usr/local/cuda-13.2"
KEEP_CUDA_133="/usr/local/cuda-13.3"
KEEP_VENV="/home/dave/cppmega-venv"

# Megatron baseline for the compare (MEASURED, bf16+FP8/Muon, 16384 tok/step):
MEGATRON_TOKS=3399

CLEANUP_HAD_SKIP=0   # set to 1 if any cleanup item was skipped/unsafe

log()  { printf '%s\n' "$*"; }
hdr()  { printf '\n========== %s ==========\n' "$*"; }

# du -sh that never aborts the script if the path is already gone.
du_gb() {
  local p="$1"
  if [ -e "$p" ]; then
    du -sb "$p" 2>/dev/null | awk '{printf "%.2f", $1/1073741824.0}'
  else
    printf '0.00'
  fi
}

# ------------------------------------------------------------------------------
# (A) CLEANUP
# ------------------------------------------------------------------------------
cleanup() {
  hdr "LEVER 3 CLEANUP (gb10) -- re-verify-then-remove"

  if [ "$(uname -m)" != "aarch64" ]; then
    log "REFUSE: uname -m = $(uname -m), expected aarch64 (gb10). Not on gb10 -- skipping ALL cleanup (RULE #1: never delete on the wrong host)."
    CLEANUP_HAD_SKIP=1
    return 0
  fi

  # Resolve the LIVE cuda alternative target so we never delete the active one.
  local live_cuda live_cuda13
  live_cuda="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
  live_cuda13="$(readlink -f /usr/local/cuda-13 2>/dev/null || true)"
  log "live /usr/local/cuda    -> ${live_cuda:-<none>}"
  log "live /usr/local/cuda-13 -> ${live_cuda13:-<none>}"

  # Confirm the must-keep toolkits + venv still exist BEFORE we remove anything.
  local k
  for k in "$KEEP_CUDA_132" "$KEEP_CUDA_133" "$KEEP_VENV"; do
    if [ ! -e "$k" ]; then
      log "REFUSE: must-keep '$k' is MISSING -- aborting cleanup (RULE #1: environment is not the expected one)."
      CLEANUP_HAD_SKIP=1
      return 0
    fi
  done
  # The NVRTC harness HARDCODES cuda-13.3 builtins; assert the SONAME is present.
  if [ ! -e "${KEEP_CUDA_133}/targets/sbsa-linux/lib/libnvrtc-builtins.so.13.3" ]; then
    log "REFUSE: ${KEEP_CUDA_133}/.../libnvrtc-builtins.so.13.3 MISSING -- aborting cleanup (NVRTC harness needs it)."
    CLEANUP_HAD_SKIP=1
    return 0
  fi
  log "must-keep OK: cuda-13.2, cuda-13.3 (+ libnvrtc-builtins.so.13.3), cppmega-venv all present."

  # ---- (A1) user-owned venv_unsloth: re-verify no running proc + no open handle
  hdr "A1: ${VENV_UNSLOTH}"
  local freed_unsloth="0.00"
  if [ ! -e "$VENV_UNSLOTH" ]; then
    log "already absent -- nothing to do."
  elif [ "$live_cuda" = "$VENV_UNSLOTH" ] || [ "$live_cuda13" = "$VENV_UNSLOTH" ]; then
    log "SKIP: unexpectedly the live cuda target -- refusing to delete."
    CLEANUP_HAD_SKIP=1
  elif ps aux | grep -F "$VENV_UNSLOTH" | grep -vw grep | grep -q .; then
    log "SKIP: a running process references ${VENV_UNSLOTH} (RULE #1: do not delete an in-use env):"
    ps aux | grep -F "$VENV_UNSLOTH" | grep -vw grep
    CLEANUP_HAD_SKIP=1
  elif command -v lsof >/dev/null 2>&1 && lsof +D "$VENV_UNSLOTH" >/dev/null 2>&1; then
    log "SKIP: lsof reports open handles under ${VENV_UNSLOTH} -- refusing to delete."
    CLEANUP_HAD_SKIP=1
  else
    freed_unsloth="$(du_gb "$VENV_UNSLOTH")"
    log "re-verify OK (no proc, no open handle, not live-cuda). Removing (user-owned)..."
    if rm -rf "$VENV_UNSLOTH"; then
      log "REMOVED ${VENV_UNSLOTH} (freed ${freed_unsloth} GB)."
    else
      log "ERROR: rm -rf ${VENV_UNSLOTH} failed -- left as-is."
      CLEANUP_HAD_SKIP=1
      freed_unsloth="0.00"
    fi
  fi

  # ---- (A2) system toolkits cuda-13.0 / cuda-13.1 + their gds confs (sudo -n)
  hdr "A2: ${CUDA_OLD_1} ${CUDA_OLD_2} (+ gds confs)"
  local have_sudo_n=0
  if sudo -n true 2>/dev/null; then
    have_sudo_n=1
    log "sudo -n: AVAILABLE (unattended system removal enabled)."
  else
    log "sudo -n: NOT available -- will PRINT exact sudo commands and SKIP (RULE #1: never half-delete)."
  fi

  # Pre-measure sizes (for the freed-space report) before any removal.
  local sz_old1 sz_old2
  sz_old1="$(du_gb "$CUDA_OLD_1")"
  sz_old2="$(du_gb "$CUDA_OLD_2")"

  # Per-toolkit re-verify: must NOT be the live cuda target, no proc ref, no open handle.
  local unsafe_sys=0 t
  for t in "$CUDA_OLD_1" "$CUDA_OLD_2"; do
    [ -e "$t" ] || continue
    if [ "$live_cuda" = "$t" ] || [ "$live_cuda13" = "$t" ]; then
      log "SKIP ALL system removal: ${t} IS the live cuda alternative target -- refusing."
      unsafe_sys=1
    fi
    if ps aux | grep -F "$t" | grep -vw grep | grep -q .; then
      log "SKIP ALL system removal: a running process references ${t}:"
      ps aux | grep -F "$t" | grep -vw grep
      unsafe_sys=1
    fi
    if command -v lsof >/dev/null 2>&1 && lsof +D "$t" >/dev/null 2>&1; then
      log "SKIP ALL system removal: lsof reports open handles under ${t} -- refusing."
      unsafe_sys=1
    fi
  done

  local freed_sys="0.00"
  local sudo_cmd="sudo rm -rf ${CUDA_OLD_1} ${CUDA_OLD_2}; sudo rm -f ${GDS_CONF_1} ${GDS_CONF_2}; sudo ldconfig"
  if [ "$unsafe_sys" -ne 0 ]; then
    log "System toolkits NOT removed (a re-verify failed above). RULE #1: failing loud, not deleting."
    CLEANUP_HAD_SKIP=1
  elif [ "$have_sudo_n" -ne 1 ]; then
    log "ACTION REQUIRED (run with real sudo): ${sudo_cmd}"
    CLEANUP_HAD_SKIP=1
  else
    log "re-verify OK for system toolkits. Removing with sudo -n..."
    local ok=1
    sudo -n rm -rf "$CUDA_OLD_1" "$CUDA_OLD_2" || ok=0
    sudo -n rm -f  "$GDS_CONF_1" "$GDS_CONF_2" || ok=0
    sudo -n ldconfig || ok=0
    if [ "$ok" -eq 1 ]; then
      freed_sys="$(awk -v a="$sz_old1" -v b="$sz_old2" 'BEGIN{printf "%.2f", a+b}')"
      log "REMOVED ${CUDA_OLD_1} (${sz_old1} GB) + ${CUDA_OLD_2} (${sz_old2} GB) + gds-13-0/13-1.conf; ldconfig refreshed."
    else
      log "ERROR: a sudo -n step failed. Re-run with real sudo: ${sudo_cmd}"
      CLEANUP_HAD_SKIP=1
    fi
  fi

  # ---- cleanup summary
  hdr "CLEANUP SUMMARY"
  local total
  total="$(awk -v u="$freed_unsloth" -v s="$freed_sys" 'BEGIN{printf "%.2f", u+s}')"
  log "RESULT_CLEANUP freed_unsloth_gb=${freed_unsloth} freed_system_gb=${freed_sys} freed_total_gb=${total} kept=cuda-13.2,cuda-13.3,cppmega-venv had_skip=${CLEANUP_HAD_SKIP}"
  if command -v df >/dev/null 2>&1; then df -h / | tail -1; fi
}

# ------------------------------------------------------------------------------
# (B) E2E PERF TEST (GPU) -- run the existing pr6 runner, parse, print RESULT
# ------------------------------------------------------------------------------
e2e_perf_test() {
  hdr "LEVER 3 E2E PERF TEST (gb10 CUDA) -- pr6 train_step"

  if [ ! -f "$PR6_RUNNER" ]; then
    log "FAIL-LOUD: pr6 runner not found at ${PR6_RUNNER} -- cannot run e2e perf test."
    return 3
  fi
  if [ ! -x "$VENV_PY" ]; then
    log "FAIL-LOUD: cppmega-venv python not found/executable at ${VENV_PY}."
    return 3
  fi

  local s1="${PR6_STAGE1_LAYERS:-8}"
  local s2="${PR6_STAGE2_LAYERS:-2}"
  local ts logf
  ts="$(date +%Y%m%d_%H%M%S)"
  logf="/tmp/pr6_e2e_${ts}.log"

  log "config: PR6_STAGE1_LAYERS=${s1} PR6_STAGE2_LAYERS=${s2} (bs1, seq=4096 => 4096 tok/step)"
  log "log -> ${logf}"

  # The NVRTC harness must be applied BEFORE torch/TE import. The pr6 runner
  # imports tvm directly; we apply the builtins path via a one-line preload of
  # the committed harness so the in-process import order is correct (RULE #1:
  # use the real self-healing harness, not an ad-hoc LD_LIBRARY_PATH hack).
  PYTHONPATH="${CPPMEGA_DIR}:${TVM_PY}:${TVMFFI_PY}" \
  TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH_VAL}" \
  PR6_STAGE1_LAYERS="${s1}" PR6_STAGE2_LAYERS="${s2}" \
  "$VENV_PY" -c "import cppmega_mlx._gb10_nvrtc_env as e; e.ensure_nvrtc_builtins_path(); import runpy, sys; sys.argv=['pr6']; runpy.run_path('${PR6_RUNNER}', run_name='__main__')" \
    2>&1 | tee "$logf"
  local rc="${PIPESTATUS[0]}"

  if [ "$rc" -ne 0 ]; then
    log "FAIL-LOUD: pr6 e2e runner exited ${rc} (see ${logf}). NOT reporting a fabricated tok/s."
    return "$rc"
  fi

  # ---- parse the runner's MEASURED output (do not fabricate). ----
  # Stage 1 line:  "  compile=...s run=<run_s>s"  and the SUMMARY line:
  #   "Stage 1 (...): RUNS=yes loss=... ... measured-free-delta=<peak> GB"
  local run_s peak_gb
  run_s="$(grep -Eo 'run=[0-9.]+s' "$logf" | head -1 | sed -E 's/run=([0-9.]+)s/\1/')"
  peak_gb="$(grep -E 'Stage 1 .*measured-free-delta=' "$logf" | head -1 \
             | grep -Eo 'measured-free-delta=[0-9.]+' | sed -E 's/.*=//')"
  if [ -z "$peak_gb" ]; then
    peak_gb="$(grep -Eo 'MEASURED peak delta=[0-9.]+ GB' "$logf" | head -1 \
               | grep -Eo '[0-9.]+' | head -1)"
  fi

  if [ -z "$run_s" ] || [ -z "$peak_gb" ]; then
    log "FAIL-LOUD: could not parse run_s ('${run_s}') or peak_gb ('${peak_gb}') from ${logf}. NOT fabricating a result."
    return 4
  fi

  # tok/s = 4096 tok/step (bs1, seq=4096) / mean-step-s ; step_ms = run_s*1000.
  local step_ms toks ratio
  step_ms="$(awk -v r="$run_s" 'BEGIN{printf "%.1f", r*1000.0}')"
  toks="$(awk -v r="$run_s" 'BEGIN{printf "%.1f", 4096.0/r}')"
  ratio="$(awk -v t="$toks" -v m="$MEGATRON_TOKS" 'BEGIN{printf "%.3f", t/m}')"

  hdr "E2E RESULT (machine-parseable)"
  # ONE line, key=value, easy for the GB10 phase to grep.
  log "RESULT_E2E config=stage1_${s1}L_stage2_${s2}L_bs1_seq4096 tok_per_step=4096 step_ms=${step_ms} tok_s=${toks} peak_gb=${peak_gb} megatron_tok_s=${MEGATRON_TOKS} ratio_vs_megatron=${ratio} kind=MEASURED note=bs1@${s1}L_GO_config;bs4/28L/16384-tok-step_would_be_EXTRAPOLATION(tok/s_batch-invariant,SSD-scan_launch-bound)"
  return 0
}

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
main() {
  hdr "gb10_cleanup_and_e2e.sh  host=$(hostname 2>/dev/null) arch=$(uname -m) date=$(date)"

  local e2e_rc=0
  if [ "${SKIP_CLEANUP:-0}" = "1" ]; then
    log "SKIP_CLEANUP=1 -> skipping cleanup."
  else
    cleanup
  fi

  if [ "${SKIP_E2E:-0}" = "1" ]; then
    log "SKIP_E2E=1 -> skipping e2e perf test."
  else
    e2e_perf_test
    e2e_rc=$?
  fi

  hdr "DONE"
  log "cleanup_had_skip=${CLEANUP_HAD_SKIP} e2e_rc=${e2e_rc}"
  # Non-zero if a cleanup item was unsafe/skipped OR the e2e perf test failed
  # (RULE #1: surface it; never exit 0 over a half-done/failed run).
  if [ "$CLEANUP_HAD_SKIP" -ne 0 ] || [ "$e2e_rc" -ne 0 ]; then
    return 1
  fi
  return 0
}

main "$@"
