#!/usr/bin/env python3
"""gb10 (NVIDIA GB10, CUDA) 1B speed matrix — FAST FUSED Path-C campaign (20260601).

Extends ``scripts/run_metal_speed_matrix_20260531.py``. Adds the headline axis:
Path-C is run in BOTH modes per (dtype, optimizer):

  - ``path_c``          : flag-OFF (serial mamba3, the prior baseline)
  - ``path_c_chunked``  : flag-ON  (CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1, the new
                          FAST chunked fused mamba3 path)

Plus Path-B as the non-Path-C reference. Each cell runs as a fresh, timeout-
bounded subprocess and records tok/s + a finite-decreasing-loss check
(fail-loud, never silently dropped). The matrix carries a flag-OFF-vs-flag-ON
Path-C speedup column so the fused-mamba3 speedup is explicit.

RULE #1: every cell runs the real config or fails LOUD with the real reason.
No faked cells, no silent degrade.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATESTAMP = "20260601"
# Optional run parameterization (env-driven). Defaults reproduce the original
# batch=1 seq=512 matrix byte-for-byte. MATRIX_TAG suffixes every output path +
# work/cache dir so a batch=4 fair run does NOT clobber the batch=1 artifacts.
TAG = os.environ.get("MATRIX_TAG", "").strip()
_SUF = f"_{TAG}" if TAG else ""
BATCH_SIZE = int(os.environ.get("MATRIX_BATCH_SIZE", "1"))
SEQ_LEN = int(os.environ.get("MATRIX_SEQ_LEN", "512"))
STEPS = int(os.environ.get("MATRIX_STEPS", "10"))
TARGET_PARQUET = (
    ROOT / "data" / "parquet_samples" / "gb10" / "clang_semantic_4k_v10" / "val_00000.parquet"
)
WORK_DIR = Path(f"/tmp/cppmega_1b_speed_matrix_gb10_fastfused_{DATESTAMP}{_SUF}_cells")
CACHE_ROOT = Path(f"/tmp/cppmega_1b_speed_matrix_gb10_fastfused_{DATESTAMP}{_SUF}_tilelang_cache")
MD_OUT = ROOT / "reports" / f"cppmega_1b_speed_matrix_gb10_fastfused_{DATESTAMP}{_SUF}.md"
HTML_OUT = ROOT / "reports" / f"cppmega_1b_speed_matrix_gb10_fastfused_{DATESTAMP}{_SUF}.html"

PYTHON = sys.executable
def _timeout_bin():
    for c in ("/usr/bin/timeout", "/bin/timeout", "/opt/homebrew/bin/timeout"):
        if Path(c).exists():
            return c
    return "timeout"
TIMEOUT_BIN = _timeout_bin()
CELL_TIMEOUT_S = 1800

DTYPES = ("bf16", "fp8", "nvfp4")
OPTIMIZERS = ("muon", "adamw", "lion")
# Bit-widths {8,16}: 16-bit == plain optimizer, 8-bit == int8-quantised optimizer.
BITS = (16, 8)
# paths: Path-B reference, Path-C flag-OFF (serial), Path-C flag-ON (chunked fused).
PATHS = ("path_b", "path_c", "path_c_chunked")

# (optimizer, bits) -> m04 --optimizer arg (mirrors bench harness optimizer_map).
OPT_CLI = {
    ("adamw", 16): "adamw",
    ("adamw", 8): "adam8bit",
    ("lion", 16): "lion",
    ("lion", 8): "lion8bit",
    ("muon", 16): "muon",
    ("muon", 8): "int8",
}


def dtype_arg(dtype: str, path: str) -> str:
    if dtype == "bf16":
        return "bfloat16"
    if dtype == "nvfp4":
        return "nvfp4"
    return "fp8_path_b" if path == "path_b" else "fp8_path_c"


def cell_env(dtype: str, path: str) -> dict[str, str]:
    env = dict(os.environ)
    is_path_c = path in ("path_c", "path_c_chunked")
    kernel_path = "path_c" if is_path_c else "path_b"
    env["CPPMEGA_KERNEL_PATH"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__M2RNN"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] = kernel_path
    if is_path_c:
        # SPLIT/WARM route: Path C fwd + Path B mamba3 bwd, matching prior matrix.
        env["CPPMEGA_MAMBA3_PATH_C_BWD"] = "path_b"
        # HEADLINE: flag-ON chunked fused mamba3 path on path_c_chunked cells.
        env["CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN"] = "1" if path == "path_c_chunked" else "0"
        if dtype == "fp8":
            env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] = "path_c"
            env["CPPMEGA_SPARSE_MLA_FP8_BWD"] = "path_c"
    elif dtype == "fp8":
        env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] = "path_b"
    return env


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


def build_command(dtype: str, optimizer: str, bits: int, path: str, out_json: Path) -> list[str]:
    cmd = [
        TIMEOUT_BIN, str(CELL_TIMEOUT_S),
        PYTHON, "scripts/m04_train_step.py",
        "--model-profile", "local_gb10_quarter",
        "--data-path", str(TARGET_PARQUET.relative_to(ROOT)),
        "--data-format", "parquet",
        "--token-key", "token_ids",
        "--steps", str(STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--seq-len", str(SEQ_LEN),
        "--dtype", dtype_arg(dtype, path),
        "--optimizer", OPT_CLI[(optimizer, bits)],
        "--optimizer-quant-scheme", "dynamic_int8_v1",
        "--lr", "1e-4",
        "--grad-checkpoint",
        "--output", str(out_json),
        "--json",
    ]
    return cmd


def _structured_reason(receipt: dict, status: str) -> str | None:
    training = receipt.get("training")
    if not isinstance(training, dict):
        return None
    route = training.get("fp8_path_c_training_route")
    if not isinstance(route, dict):
        return None
    dc_contract = route.get("direct_fusion_chain_training_runtime_contract")
    if isinstance(dc_contract, dict) and dc_contract.get("status") not in (None, "ok", "ready"):
        fusion = route.get("path_c_fusion") or {}
        dc = fusion.get("direct_chained_fusion") if isinstance(fusion, dict) else {}
        rb = dc.get("runtime_binding") if isinstance(dc, dict) else {}
        mba = dc.get("model_binding_audit") if isinstance(dc, dict) else {}
        parts = [
            f"direct-chain training runtime not implemented (contract={dc_contract.get('status')}",
            f"runtime_binding={rb.get('status') if isinstance(rb, dict) else None}",
            f"model_binding={mba.get('status') if isinstance(mba, dict) else None})",
        ]
        avail = route.get("end_to_end_training_status")
        if avail:
            parts.append(f"split route available={avail}")
        return "; ".join(parts)
    if status == "blocked":
        sel = route.get("selected_action") or route.get("status")
        e2e = route.get("end_to_end_training_status")
        if sel or e2e:
            return f"blocked: selected_action={sel}, end_to_end={e2e}"
    return None


def parse_receipt(receipt: dict, returncode: int, stderr: str, stdout: str) -> dict:
    status = str(receipt.get("status") or "")
    timing = receipt.get("timing") if isinstance(receipt.get("timing"), dict) else {}
    memory = receipt.get("memory") if isinstance(receipt.get("memory"), dict) else {}
    training = receipt.get("training") if isinstance(receipt.get("training"), dict) else {}
    step_times = list(timing.get("step_times_s") or [])
    first_step = float(step_times[0]) if step_times else None
    steady = [float(v) for v in step_times[1:]] if len(step_times) > 1 else []
    step_dur = (sum(steady) / len(steady)) if steady else (
        float(timing["mean_step_time_s"]) if timing.get("mean_step_time_s") is not None else None
    )
    step_sec = (1.0 / step_dur) if step_dur and step_dur > 0 else None
    tok_sec = float(timing["tokens_per_second"]) if timing.get("tokens_per_second") is not None else None
    peak_bytes = int(memory["peak_memory_bytes"]) if memory.get("peak_memory_bytes") is not None else None
    peak_gb = (peak_bytes / (1024 ** 3)) if peak_bytes is not None else None

    losses = [float(x) for x in (training.get("losses") or [])]
    all_finite = training.get("all_finite")
    # finite-decreasing-loss check: every loss finite AND final < initial.
    import math
    finite_ok = bool(losses) and all(math.isfinite(x) for x in losses)
    decreasing = bool(losses) and (losses[-1] < losses[0])
    loss_decreased = training.get("loss_decreased")
    if loss_decreased is None:
        loss_decreased = decreasing
    loss_check = bool(finite_ok and decreasing and (all_finite is not False))

    reason = "ok"
    if status != "ok":
        reason = _structured_reason(receipt, status)
        if reason is None:
            for key in ("status_reason", "failure_reason", "reason", "error", "message", "title"):
                v = receipt.get(key)
                if v:
                    reason = str(v)
                    break
            else:
                tail = (stderr.strip() or stdout.strip() or f"cell exited {returncode}")
                reason = tail[-300:]
    elif all_finite is False or not finite_ok:
        status = "failed"
        reason = "training reported non-finite loss values"
    elif not decreasing:
        status = "failed"
        reason = f"loss did not decrease (first={losses[0]:.4g}, last={losses[-1]:.4g})" if losses else "no losses recorded"
    return {
        "status": status or "failed",
        "tok_sec": tok_sec,
        "step_sec": step_sec,
        "compile_s": first_step,
        "peak_gb": peak_gb,
        "loss_check": loss_check,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "reason": reason,
    }


def run_cell(dtype: str, optimizer: str, bits: int, path: str) -> dict:
    case_id = f"{dtype}_{optimizer}{bits}_{path}"
    out_json = WORK_DIR / f"{case_id}.json"
    if out_json.exists():
        out_json.unlink()
    cmd = build_command(dtype, optimizer, bits, path, out_json)
    env = cell_env(dtype, path)
    cache_dir = CACHE_ROOT / f"{dtype}_{optimizer}{bits}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TILELANG_CACHE_DIR"] = str(cache_dir)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    receipt: dict = {}
    if out_json.exists():
        try:
            receipt = json.loads(out_json.read_text())
        except Exception as exc:  # noqa: BLE001
            receipt = {"status": "failed", "reason": f"unparseable receipt: {exc}"}

    if proc.returncode == 124 and not receipt:
        result = {
            "status": "timeout", "tok_sec": None, "step_sec": None, "compile_s": None,
            "peak_gb": None, "loss_check": False, "initial_loss": None, "final_loss": None,
            "reason": f"cell exceeded {CELL_TIMEOUT_S}s OS timeout (rc=124)",
        }
    elif not receipt:
        tail = (proc.stderr.strip() or proc.stdout.strip() or "")[-600:]
        result = {
            "status": "failed", "tok_sec": None, "step_sec": None, "compile_s": None,
            "peak_gb": None, "loss_check": False, "initial_loss": None, "final_loss": None,
            "reason": f"no JSON receipt (rc={proc.returncode}): {tail}",
        }
    else:
        result = parse_receipt(receipt, proc.returncode, proc.stderr, proc.stdout)

    chunked_flag = "1" if path == "path_c_chunked" else ("0" if path == "path_c" else "")
    result.update({
        "dtype": dtype, "optimizer": optimizer, "bits": bits, "path": path,
        "case_id": case_id, "wall_s": wall, "returncode": proc.returncode,
        "chunked_flag": chunked_flag,
        "command": " ".join(cmd),
    })
    return result


def fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _speedup(results):
    """Map (dtype,optimizer,bits) -> flag-OFF/flag-ON steady step/s speedup."""
    by = {}
    for r in results:
        by[(r["dtype"], r["optimizer"], r["bits"], r["path"])] = r
    out = {}
    for d in DTYPES:
        for o in OPTIMIZERS:
            for b in BITS:
                off = by.get((d, o, b, "path_c"))
                on = by.get((d, o, b, "path_c_chunked"))
                spd = None
                if off and on and off.get("step_sec") and on.get("step_sec"):
                    spd = on["step_sec"] / off["step_sec"]
                # also compute compile (first-step) speedup
                cspd = None
                if off and on and off.get("compile_s") and on.get("compile_s"):
                    cspd = off["compile_s"] / on["compile_s"]
                out[(d, o, b)] = (spd, cspd)
    return out


def write_markdown(results, identity):
    spd = _speedup(results)
    L = []
    L.append("# cppmega 1B Speed Matrix — gb10 (NVIDIA GB10, CUDA) — FAST FUSED Path-C")
    L.append("")
    L.append(f"- Date: {DATESTAMP}")
    L.append("- Host: gb10 (NVIDIA GB10, CUDA sm_121)")
    L.append("- Model profile: local_gb10_quarter")
    L.append(f"- Data: {TARGET_PARQUET.relative_to(ROOT)} (parquet, token_ids)")
    L.append(f"- Settings: seq-len {SEQ_LEN}, batch {BATCH_SIZE}, --steps {STEPS} warm, --grad-checkpoint, --optimizer-quant-scheme dynamic_int8_v1"
             + (f" (tokens/step = {BATCH_SIZE*SEQ_LEN})" if (BATCH_SIZE != 1 or SEQ_LEN != 512) else ""))
    L.append("- Paths: `path_b` (reference); `path_c` = Path-C flag-OFF (serial mamba3, prior baseline); "
             "`path_c_chunked` = Path-C **flag-ON** (CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1, FAST chunked fused mamba3).")
    L.append("- Path-C route: SPLIT/WARM (Path C fwd + Path B mamba3 bwd, mamba3 bwd=path_b).")
    L.append("- loss check = all losses finite AND final<initial (fail-loud per RULE #1).")
    L.append("- nvfp4: accepted route but full training step fails-LOUD (no nvfp4 training kernels yet) — honest blocked cell.")
    L.append("- CUDA note: the CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN flag is target-agnostic (same on Metal+CUDA); per-cell results report whatever CUDA measures.")
    L.append(f"- Per-cell bound: {CELL_TIMEOUT_S}s OS timeout (fail-loud)")
    L.append(f"- cppmega SHA: `{identity['cppmega_sha']}` · TileLang SHA: `{identity['tilelang_sha']}`")
    L.append("")
    L.append("| dtype | optimizer | bits | path | status | tok/s | step/s | compile s | peak GB | loss check | first/last loss | flag |")
    L.append("| ----- | --------- | ---: | ---- | ------ | ----: | -----: | --------: | ------: | ---------- | --------------- | ---- |")
    for r in results:
        ll = ""
        if r.get("initial_loss") is not None and r.get("final_loss") is not None:
            ll = f"{r['initial_loss']:.3g} → {r['final_loss']:.3g}"
        L.append(
            f"| {r['dtype']} | {r['optimizer']} | {r['bits']} | {r['path']} | {r['status']} | "
            f"{fmt(r['tok_sec'])} | {fmt(r['step_sec'])} | {fmt(r['compile_s'])} | "
            f"{fmt(r['peak_gb'])} | {'PASS' if r.get('loss_check') else 'FAIL'} | {ll} | {r.get('chunked_flag','')} |"
        )
    L.append("")
    L.append("## Path-C flag-OFF → flag-ON (chunked fused mamba3) speedup")
    L.append("")
    L.append("- step/s speedup = chunked step/s ÷ serial step/s (steady-state, >1 = chunked faster).")
    L.append("- compile speedup = serial first-step s ÷ chunked first-step s (>1 = chunked compiles/first-steps faster).")
    L.append("")
    L.append("| dtype | optimizer | bits | serial step/s | chunked step/s | step/s speedup | serial compile s | chunked compile s | compile speedup |")
    L.append("| ----- | --------- | ---: | ------------: | -------------: | -------------: | ---------------: | ----------------: | --------------: |")
    by = {(r["dtype"], r["optimizer"], r["bits"], r["path"]): r for r in results}
    for d in DTYPES:
        for o in OPTIMIZERS:
            for b in BITS:
                off = by.get((d, o, b, "path_c"))
                on = by.get((d, o, b, "path_c_chunked"))
                if not (off or on):
                    continue
                s_off = fmt(off["step_sec"]) if off else ""
                s_on = fmt(on["step_sec"]) if on else ""
                c_off = fmt(off["compile_s"]) if off else ""
                c_on = fmt(on["compile_s"]) if on else ""
                sp, cs = spd.get((d, o, b), (None, None))
                L.append(
                    f"| {d} | {o} | {b} | {s_off} | {s_on} | "
                    f"{fmt(sp) + 'x' if sp else ''} | {c_off} | {c_on} | {fmt(cs) + 'x' if cs else ''} |"
                )
    L.append("")
    L.append("## Per-Cell Commands")
    L.append("")
    for r in results:
        L.append(f"- `{r['case_id']}` (flag={r.get('chunked_flag','')}): `{r['command']}`")
    L.append("")
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(L))


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_html(results, identity):
    spd = _speedup(results)
    rows = []
    for r in results:
        cls = "ok" if r["status"] == "ok" and r.get("loss_check") else "bad"
        ll = ""
        if r.get("initial_loss") is not None and r.get("final_loss") is not None:
            ll = f"{r['initial_loss']:.3g} &rarr; {r['final_loss']:.3g}"
        rows.append(
            f'<tr class="{cls}"><td>{_esc(r["dtype"])}</td><td>{_esc(r["optimizer"])}</td>'
            f'<td class="num">{_esc(r["bits"])}</td><td>{_esc(r["path"])}</td>'
            f'<td>{_esc(r["status"])}</td>'
            f'<td class="num">{_esc(fmt(r["tok_sec"]))}</td>'
            f'<td class="num">{_esc(fmt(r["step_sec"]))}</td>'
            f'<td class="num">{_esc(fmt(r["compile_s"]))}</td>'
            f'<td class="num">{_esc(fmt(r["peak_gb"]))}</td>'
            f'<td>{"PASS" if r.get("loss_check") else "FAIL"}</td>'
            f'<td class="num">{ll}</td><td>{_esc(r.get("chunked_flag",""))}</td>'
            f'<td>{_esc(r["reason"])}</td></tr>'
        )
    by = {(r["dtype"], r["optimizer"], r["bits"], r["path"]): r for r in results}
    srows = []
    for d in DTYPES:
        for o in OPTIMIZERS:
            for b in BITS:
                off = by.get((d, o, b, "path_c"))
                on = by.get((d, o, b, "path_c_chunked"))
                if not (off or on):
                    continue
                sp, cs = spd.get((d, o, b), (None, None))
                spcls = "fast" if (sp and sp > 1.02) else ("slow" if (sp and sp < 0.98) else "")
                srows.append(
                    f"<tr><td>{_esc(d)}</td><td>{_esc(o)}</td><td class='num'>{b}</td>"
                    f"<td class='num'>{_esc(fmt(off['step_sec']) if off else '')}</td>"
                    f"<td class='num'>{_esc(fmt(on['step_sec']) if on else '')}</td>"
                    f"<td class='num {spcls}'>{(_esc(fmt(sp)) + 'x') if sp else ''}</td>"
                    f"<td class='num'>{_esc(fmt(off['compile_s']) if off else '')}</td>"
                    f"<td class='num'>{_esc(fmt(on['compile_s']) if on else '')}</td>"
                    f"<td class='num'>{(_esc(fmt(cs)) + 'x') if cs else ''}</td></tr>"
                )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>cppmega 1B Speed Matrix — Metal (M4 Max) FAST FUSED Path-C {DATESTAMP}</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.05rem; margin-top: 1.6rem; }}
 ul.meta {{ font-size: 0.82rem; color: #444; line-height: 1.5; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-bottom: 1rem; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
 th {{ background: #2d2d2d; color: #fff; }}
 td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 td.fast {{ color: #0a7d28; font-weight: 700; }}
 td.slow {{ color: #b40000; font-weight: 700; }}
 tr.ok td:nth-child(5) {{ color: #0a7d28; font-weight: 600; }}
 tr.bad td:nth-child(5) {{ color: #b40000; font-weight: 700; }}
 tr:nth-child(even) {{ background: #f6f6f6; }}
</style></head><body>
<h1>cppmega 1B Speed Matrix — gb10 (NVIDIA GB10, CUDA) — FAST FUSED Path-C</h1>
<ul class="meta">
 <li>Date: {DATESTAMP} &middot; Host: gb10 (NVIDIA GB10, CUDA sm_121) &middot; Profile: local_gb10_quarter</li>
 <li>seq-len {SEQ_LEN}, batch {BATCH_SIZE}, --steps {STEPS} warm, --grad-checkpoint, dynamic_int8_v1 (tokens/step = {BATCH_SIZE*SEQ_LEN})</li>
 <li>path_b=reference &middot; path_c=flag-OFF serial mamba3 &middot; path_c_chunked=flag-ON CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1 (FAST chunked fused mamba3)</li>
 <li>loss check = all losses finite AND final&lt;initial (fail-loud) &middot; per-cell bound {CELL_TIMEOUT_S}s</li>
 <li>cppmega SHA: <code>{_esc(identity['cppmega_sha'])}</code> &middot; TileLang SHA: <code>{_esc(identity['tilelang_sha'])}</code></li>
</ul>
<table>
<thead><tr><th>dtype</th><th>optimizer</th><th>bits</th><th>path</th><th>status</th><th>tok/s</th><th>step/s</th><th>compile s</th><th>peak GB</th><th>loss check</th><th>first/last loss</th><th>flag</th><th>reason</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
<h2>Path-C flag-OFF &rarr; flag-ON (chunked fused mamba3) speedup</h2>
<table>
<thead><tr><th>dtype</th><th>optimizer</th><th>bits</th><th>serial step/s</th><th>chunked step/s</th><th>step/s speedup</th><th>serial compile s</th><th>chunked compile s</th><th>compile speedup</th></tr></thead>
<tbody>
{chr(10).join(srows)}
</tbody></table>
</body></html>
"""
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    identity = {
        "cppmega_sha": git_sha(ROOT),
        "tilelang_sha": git_sha(Path("/Volumes/external/sources/tilelang")),
    }
    only = os.environ.get("MATRIX_ONLY")  # optional CSV of case_ids to run/keep
    cells = [(d, o, b, p) for d in DTYPES for o in OPTIMIZERS for b in BITS for p in PATHS]
    results = []
    # load any existing partial receipts so re-runs are incremental/durable
    existing = {}
    rj = WORK_DIR / "_results.json"
    if rj.exists():
        try:
            for r in json.loads(rj.read_text()):
                existing[r["case_id"]] = r
        except Exception:
            existing = {}
    for i, (d, o, b, p) in enumerate(cells, 1):
        case_id = f"{d}_{o}{b}_{p}"
        if only and case_id not in only.split(","):
            if case_id in existing:
                results.append(existing[case_id])
            continue
        cell_json = WORK_DIR / f"{case_id}.json"
        if case_id in existing and existing[case_id].get("status") == "ok" and not os.environ.get("MATRIX_FORCE"):
            print(f"[{i}/{len(cells)}] SKIP cached ok {case_id}", flush=True)
            results.append(existing[case_id])
            continue
        print(f"[{i}/{len(cells)}] running {case_id} ...", flush=True)
        r = run_cell(d, o, b, p)
        print(f"    -> status={r['status']} tok/s={fmt(r['tok_sec'])} step/s={fmt(r['step_sec'])} "
              f"loss={'PASS' if r.get('loss_check') else 'FAIL'} wall={r['wall_s']:.1f}s reason={r['reason'][:70]}",
              flush=True)
        results.append(r)
        existing[case_id] = r
        write_markdown(results, identity)
        write_html(results, identity)
        rj.write_text(json.dumps(results, indent=2))
    write_markdown(results, identity)
    write_html(results, identity)
    rj.write_text(json.dumps(results, indent=2))
    print(f"\nDONE. md={MD_OUT}\nhtml={HTML_OUT}", flush=True)


if __name__ == "__main__":
    main()
