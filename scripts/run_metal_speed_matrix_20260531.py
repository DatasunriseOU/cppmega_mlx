#!/usr/bin/env python3
"""Local Metal (Apple M4 Max) 1B speed matrix driver — 20260531 campaign.

Reuses the exact per-cell m04 invocation shape of
``scripts/bench_1b_training_matrix.py`` but bounds each cell with an OS-level
1800s timeout and emits BOTH Markdown and HTML matrices. Cells run as fresh
subprocesses; failures surface the real reason (fail-loud, never dropped).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_PARQUET = (
    ROOT / "data" / "parquet_samples" / "gb10" / "clang_semantic_4k_v10" / "val_00000.parquet"
)
WORK_DIR = Path("/tmp/cppmega_1b_speed_matrix_metal_20260531_cells")
CACHE_ROOT = Path("/tmp/cppmega_1b_speed_matrix_metal_20260531_tilelang_cache")
MD_OUT = ROOT / "reports" / "cppmega_1b_speed_matrix_metal_20260531.md"
HTML_OUT = ROOT / "reports" / "cppmega_1b_speed_matrix_metal_20260531.html"

PYTHON = sys.executable  # cppmega .venv python (python3.13)
TIMEOUT_BIN = "/opt/homebrew/bin/timeout"
CELL_TIMEOUT_S = 1800

DTYPES = ("bf16", "fp8")
# request: adamw/adam8bit, lion/lion8bit, muon/muon_int8
OPTIMIZERS = ("adamw", "adam8bit", "lion", "lion8bit", "muon", "muon_int8")
PATHS = ("path_b", "path_c")  # path_c == warm direct-chain route

# optimizer -> m04 --optimizer arg (mirrors bench harness optimizer_map)
OPT_CLI = {
    "adamw": "adamw",
    "adam8bit": "adam8bit",
    "lion": "lion",
    "lion8bit": "lion8bit",
    "muon": "muon",
    "muon_int8": "int8",
}


def dtype_arg(dtype: str, path: str) -> str:
    if dtype == "bf16":
        return "bfloat16"
    return "fp8_path_b" if path == "path_b" else "fp8_path_c"


def cell_env(dtype: str, path: str) -> dict[str, str]:
    env = dict(os.environ)
    kernel_path = "path_b" if path == "path_b" else "path_c"
    env["CPPMEGA_KERNEL_PATH"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__M2RNN"] = kernel_path
    env["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] = kernel_path
    if path == "path_c":
        # SPLIT/WARM route: Path C forward + Path B backward (mamba3 bwd=path_b),
        # exactly matching the prior local matrix's `path_c_warm` cells. This is
        # the performance-safe route; full Path-C backward (mamba3 bwd=path_c) is
        # a separate experiment and is NOT what path_c_warm measured.
        env["CPPMEGA_MAMBA3_PATH_C_BWD"] = "path_b"
        if dtype == "fp8":
            env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] = "path_c"
            env["CPPMEGA_SPARSE_MLA_FP8_BWD"] = "path_c"
    elif dtype == "fp8":
        env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] = "path_b"
    # warm cache: persistent per (dtype,optimizer) cache dir so path_c is warm
    return env


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001 - identity stamp only
        return f"unknown ({exc})"


def build_command(dtype: str, optimizer: str, path: str, out_json: Path) -> list[str]:
    cmd = [
        TIMEOUT_BIN, str(CELL_TIMEOUT_S),
        PYTHON, "scripts/m04_train_step.py",
        "--model-profile", "local_gb10_quarter",
        "--data-path", str(TARGET_PARQUET.relative_to(ROOT)),
        "--data-format", "parquet",
        "--token-key", "token_ids",
        "--steps", "10",
        "--batch-size", "1",
        "--seq-len", "512",
        "--dtype", dtype_arg(dtype, path),
        "--optimizer", OPT_CLI[optimizer],
        "--optimizer-quant-scheme", "dynamic_int8_v1",
        "--lr", "1e-4",
        "--grad-checkpoint",
        "--output", str(out_json),
        "--json",
    ]
    # NOTE: path_c here == SPLIT/WARM route (Path C fwd + Path B bwd), matching
    # the prior local matrix's `path_c_warm`. We deliberately do NOT pass
    # --use-path-c-direct-chain-runtime (direct-chain training runtime is not
    # implemented on Metal and fails-closed to status=blocked).
    return cmd


def _structured_reason(receipt: dict, status: str) -> str | None:
    """Extract the real, structured Path C / training-route blocker reason.

    Avoids dumping TileLang compile logs into the matrix cell; surfaces the
    m04 contract status fields that actually explain a blocked/failed cell.
    """
    training = receipt.get("training")
    if not isinstance(training, dict):
        return None
    route = training.get("fp8_path_c_training_route")
    if not isinstance(route, dict):
        return None
    # Direct-chain runtime contract (the --use-path-c-direct-chain-runtime route)
    dc_contract = route.get("direct_fusion_chain_training_runtime_contract")
    if isinstance(dc_contract, dict) and dc_contract.get("status") not in (None, "ok", "ready"):
        fusion = route.get("path_c_fusion") or {}
        dc = fusion.get("direct_chained_fusion") if isinstance(fusion, dict) else {}
        rb = dc.get("runtime_binding") if isinstance(dc, dict) else {}
        mba = dc.get("model_binding_audit") if isinstance(dc, dict) else {}
        parts = [
            f"direct-chain training runtime not implemented "
            f"(contract={dc_contract.get('status')}",
            f"runtime_binding={rb.get('status') if isinstance(rb, dict) else None}",
            f"model_binding={mba.get('status') if isinstance(mba, dict) else None})",
        ]
        avail = route.get("end_to_end_training_status")
        if avail:
            parts.append(f"split route available={avail}")
        return "; ".join(parts)
    # Generic route-level status if blocked
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
    # "step/s" column == steps per second (matches prior LOCAL matrix semantics)
    step_sec = (1.0 / step_dur) if step_dur and step_dur > 0 else None
    tok_sec = float(timing["tokens_per_second"]) if timing.get("tokens_per_second") is not None else None
    peak_bytes = int(memory["peak_memory_bytes"]) if memory.get("peak_memory_bytes") is not None else None
    peak_gb = (peak_bytes / (1024 ** 3)) if peak_bytes is not None else None

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
                # last resort: short tail of stderr/stdout, NOT a full compile-log dump
                tail = (stderr.strip() or stdout.strip() or f"cell exited {returncode}")
                reason = tail[-300:]
    elif training.get("all_finite") is False:
        status = "failed"
        reason = "training reported non-finite values"
    return {
        "status": status or "failed",
        "tok_sec": tok_sec,
        "step_sec": step_sec,
        "compile_s": first_step,
        "peak_gb": peak_gb,
        "reason": reason,
    }


def run_cell(dtype: str, optimizer: str, path: str) -> dict:
    case_id = f"{dtype}_{optimizer}_{path}"
    out_json = WORK_DIR / f"{case_id}.json"
    if out_json.exists():
        out_json.unlink()
    cmd = build_command(dtype, optimizer, path, out_json)
    env = cell_env(dtype, path)
    # warm per-(dtype,optimizer) tilelang cache so path_c measures warm route
    cache_dir = CACHE_ROOT / f"{dtype}_{optimizer}"
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
            "status": "timeout",
            "tok_sec": None, "step_sec": None, "compile_s": None, "peak_gb": None,
            "reason": f"cell exceeded {CELL_TIMEOUT_S}s OS timeout (rc=124)",
        }
    elif not receipt:
        tail = (proc.stderr.strip() or proc.stdout.strip() or "")[-600:]
        result = {
            "status": "failed",
            "tok_sec": None, "step_sec": None, "compile_s": None, "peak_gb": None,
            "reason": f"no JSON receipt (rc={proc.returncode}): {tail}",
        }
    else:
        result = parse_receipt(receipt, proc.returncode, proc.stderr, proc.stdout)

    cache_hit = receipt.get("path_c_warm_cache_hit_observed")
    result.update({
        "dtype": dtype, "optimizer": optimizer, "path": path,
        "case_id": case_id, "wall_s": wall, "returncode": proc.returncode,
        "cache_hit": cache_hit,
        "command": " ".join(cmd),
    })
    return result


def fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def write_markdown(results, identity):
    lines = []
    lines.append("# cppmega 1B Speed Matrix — Local Metal (Apple M4 Max)")
    lines.append("")
    lines.append(f"- Date: 20260531")
    lines.append(f"- Host: local Mac M4 Max (Metal)")
    lines.append(f"- Model profile: local_gb10_quarter")
    lines.append(f"- Data: {TARGET_PARQUET.relative_to(ROOT)} (parquet, token_ids)")
    lines.append(f"- Settings: seq-len 512, batch 1, --steps 10 warm, --grad-checkpoint, --optimizer-quant-scheme dynamic_int8_v1")
    lines.append(f"- Path C route: SPLIT/WARM (Path C fwd + Path B bwd, mamba3 bwd=path_b); matches prior `path_c_warm`; NO --use-path-c-direct-chain-runtime")
    lines.append(f"- Per-cell bound: {CELL_TIMEOUT_S}s OS timeout (fail-loud)")
    lines.append(f"- cppmega SHA: `{identity['cppmega_sha']}`")
    lines.append(f"- TileLang SHA: `{identity['tilelang_sha']}` (bf16 AtomicAdd fix 3fcfc21c live)")
    lines.append("")
    header = "| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | reason |"
    sep = "| ----- | --------- | ---- | ------ | ----: | -----: | --------: | ------: | --------- | ------ |"
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            f"| {r['dtype']} | {r['optimizer']} | {r['path']} | {r['status']} | "
            f"{fmt(r['tok_sec'])} | {fmt(r['step_sec'])} | {fmt(r['compile_s'])} | "
            f"{fmt(r['peak_gb'])} | {fmt(r['cache_hit'])} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Per-Cell Commands")
    lines.append("")
    for r in results:
        lines.append(f"- `{r['case_id']}`: `{r['command']}`")
    lines.append("")
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(lines))


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_html(results, identity):
    rows = []
    for r in results:
        cls = "ok" if r["status"] == "ok" else "bad"
        rows.append(
            f'<tr class="{cls}"><td>{_esc(r["dtype"])}</td><td>{_esc(r["optimizer"])}</td>'
            f'<td>{_esc(r["path"])}</td><td>{_esc(r["status"])}</td>'
            f'<td class="num">{_esc(fmt(r["tok_sec"]))}</td>'
            f'<td class="num">{_esc(fmt(r["step_sec"]))}</td>'
            f'<td class="num">{_esc(fmt(r["compile_s"]))}</td>'
            f'<td class="num">{_esc(fmt(r["peak_gb"]))}</td>'
            f'<td>{_esc(fmt(r["cache_hit"]))}</td><td>{_esc(r["reason"])}</td></tr>'
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>cppmega 1B Speed Matrix — Local Metal (M4 Max) 20260531</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.3rem; }}
 ul.meta {{ font-size: 0.85rem; color: #444; line-height: 1.5; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
 th {{ background: #2d2d2d; color: #fff; }}
 td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 tr.ok td:nth-child(4) {{ color: #0a7d28; font-weight: 600; }}
 tr.bad td:nth-child(4) {{ color: #b40000; font-weight: 700; }}
 tr:nth-child(even) {{ background: #f6f6f6; }}
</style></head><body>
<h1>cppmega 1B Speed Matrix — Local Metal (Apple M4 Max)</h1>
<ul class="meta">
 <li>Date: 20260531 &middot; Host: local Mac M4 Max (Metal) &middot; Profile: local_gb10_quarter</li>
 <li>seq-len 512, batch 1, --steps 10 warm, --grad-checkpoint, dynamic_int8_v1</li>
 <li>Path C: SPLIT/WARM (Path C fwd + Path B bwd, mamba3 bwd=path_b) &middot; matches prior path_c_warm &middot; per-cell bound {CELL_TIMEOUT_S}s</li>
 <li>cppmega SHA: <code>{_esc(identity['cppmega_sha'])}</code> &middot; TileLang SHA: <code>{_esc(identity['tilelang_sha'])}</code> (bf16 AtomicAdd fix 3fcfc21c live)</li>
</ul>
<table>
<thead><tr><th>dtype</th><th>optimizer</th><th>path</th><th>status</th><th>tok/s</th><th>step/s</th><th>compile s</th><th>peak GB</th><th>cache hit</th><th>reason</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>
"""
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html)


def regen():
    """Rebuild md+html from existing per-cell receipts (no re-run)."""
    identity = {
        "cppmega_sha": git_sha(ROOT),
        "tilelang_sha": git_sha(Path("/Volumes/external/sources/tilelang")),
    }
    results = []
    cells = [(d, o, p) for d in DTYPES for o in OPTIMIZERS for p in PATHS]
    for d, o, p in cells:
        case_id = f"{d}_{o}_{p}"
        rj = WORK_DIR / f"{case_id}.json"
        receipt = json.loads(rj.read_text()) if rj.exists() else {}
        parsed = parse_receipt(receipt, 0, "", "") if receipt else {
            "status": "failed", "tok_sec": None, "step_sec": None,
            "compile_s": None, "peak_gb": None, "reason": "receipt missing on regen",
        }
        parsed.update({
            "dtype": d, "optimizer": o, "path": p, "case_id": case_id,
            "cache_hit": receipt.get("path_c_warm_cache_hit_observed"),
            "command": " ".join(build_command(d, o, p, rj)),
        })
        results.append(parsed)
    write_markdown(results, identity)
    write_html(results, identity)
    print(f"REGEN DONE. md={MD_OUT}\nhtml={HTML_OUT}", flush=True)


def main():
    if "--regen" in sys.argv:
        regen()
        return
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    identity = {
        "cppmega_sha": git_sha(ROOT),
        "tilelang_sha": git_sha(Path("/Volumes/external/sources/tilelang")),
    }
    results = []
    cells = [(d, o, p) for d in DTYPES for o in OPTIMIZERS for p in PATHS]
    for i, (d, o, p) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] running {d}_{o}_{p} ...", flush=True)
        r = run_cell(d, o, p)
        print(f"    -> status={r['status']} tok/s={fmt(r['tok_sec'])} "
              f"peak_gb={fmt(r['peak_gb'])} wall={r['wall_s']:.1f}s reason={r['reason'][:80]}",
              flush=True)
        results.append(r)
        # write partial outputs after every cell so progress is durable
        write_markdown(results, identity)
        write_html(results, identity)
        (WORK_DIR / "_results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, identity)
    write_html(results, identity)
    print(f"\nDONE. md={MD_OUT}\nhtml={HTML_OUT}", flush=True)


if __name__ == "__main__":
    main()
