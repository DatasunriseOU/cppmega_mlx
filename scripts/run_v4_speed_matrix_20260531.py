#!/usr/bin/env python3
"""v4 1B speed matrix driver (op-level GDN/KDA) — 20260531 campaign.

v4 analog of ``scripts/run_metal_speed_matrix_20260531.py``. The v4 harness
(``cppmega_v4/_tilelang/benchmark_matrix.py`` + ``benchmark_receipt.py``)
benchmarks the *linear-attention op* (GDN gated-delta + KDA recurrent) across
the FIVE-path dispatcher A/B/C/D/E -- it does NOT run a full 1B m04 training
step. So this matrix reports op-level forward latency (median seconds) and
derived throughput (elements/s) per (block, path, dtype) cell, fail-loud on
every cell that cannot run the *real* selected path.

Axes (what the harness actually supports + what is representable):
  - block: {gdn, kda}                  (the two v4 linear-attention ops)
  - path:  {a, b, c, d, e}             (the 5-path dispatcher)
  - dtype: {f32, bf16}                 -- f32 is the harness reference; bf16 is
           a REAL cast of q/k/v/beta/g before dispatch. fp8/nvfp4 are NOT
           representable as MLX op-input array dtypes (mlx has only
           to_fp8/from_fp8 packers, no float8 array dtype), so they do not
           apply at the op level and are reported N/A-with-reason, never faked.
           (The v3 fp8 axis was a GEMM storage scheme inside m04 training, a
           different layer than this op-level forward.)

Per cell we record: status (ok/fallback/failed), measured_path, median fwd
seconds, throughput (M elem/s), backend reason on failure. allow_fallback is
DISABLED so a path that cannot run surfaces the real reason (RULE #1).

Each cell runs in a fresh subprocess bounded by an OS timeout; stragglers are
killed; GPU/Metal cleared between cells by process exit.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATESTAMP = "20260531"

# Platform tag chosen by the driver flag.
PLATFORM = os.environ.get("V4_MATRIX_PLATFORM", "metal")  # "metal" | "gb10"
WORK_DIR = Path(f"/tmp/cppmega_v4_speed_matrix_{PLATFORM}_{DATESTAMP}_cells")
MD_OUT = ROOT / "reports" / f"cppmega_v4_speed_matrix_{PLATFORM}_{DATESTAMP}.md"
HTML_OUT = ROOT / "reports" / f"cppmega_v4_speed_matrix_{PLATFORM}_{DATESTAMP}.html"

PYTHON = sys.executable
CELL_TIMEOUT_S = 1800

BLOCKS = ("gdn", "kda")
PATHS = ("path_a", "path_b", "path_c", "path_d", "path_e")
# f32=harness reference, bf16=real cast, fp16=real cast. fp16d = Path D's
# required mixed dtype (fp16 q/k/v + f32 beta/g) so Path D's runtime adapter
# gets a fair measurement. fp8/nvfp4 not op-representable; see docstring.
DTYPES = ("f32", "bf16", "fp16", "fp16d")

# Canonical benchmark shape: Path-E-eligible (Dk%32==0, Dv%4==0) so Path E can
# hit its real Metal kernel; GDN gate forced <=0 so Path E is gate-eligible.
SHAPE = dict(batch=1, seq_len=512, num_heads=8, head_dim_k=64, head_dim_v=64)
WARMUP = 2
ITERS = 5


def _timeout_bin() -> str | None:
    for cand in ("/opt/homebrew/bin/timeout", "/usr/bin/timeout", "/bin/timeout"):
        if Path(cand).exists():
            return cand
    return None


# ---- in-subprocess single-cell measurement (real dtype cast) -------------

CELL_SCRIPT = r'''
import json, os, statistics, sys, time
import mlx.core as mx

block, path, dtype_tag = sys.argv[1], sys.argv[2], sys.argv[3]
B, T, H, DK, DV = __B__, __T__, __H__, __DK__, __DV__
WARMUP, ITERS = __WARMUP__, __ITERS__

from cppmega_v4._tilelang.linear_attention_paths import (
    ENV_VAR as GDN_ENV, gated_delta_recurrent_dispatch, linear_attention_path_statuses,
)
from cppmega_v4._tilelang.kda_paths import (
    ENV_VAR as KDA_ENV, kda_recurrent_dispatch, kda_path_statuses,
)

# fp16d = Path D mixed-dtype combo: fp16 q/k/v, f32 beta/g (gate). All other
# tags cast every input uniformly.
mxdtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp16d": mx.float16}.get(
    dtype_tag, mx.float32)
GATE_DTYPE = mx.float32 if dtype_tag == "fp16d" else mxdtype

def cast(a):
    return a.astype(mxdtype)

def castg(a):  # gate/beta dtype (differs only for fp16d)
    return a.astype(GATE_DTYPE)

def l2n(a):
    return a / mx.sqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-6)

# Inputs are stabilised the way the gated-delta recurrence expects: k is
# L2-normalised, beta in [0,1] via sigmoid, gate <=0 (decay). Raw N(0,1) q/k
# with unbounded beta diverges to NaN over T=512 steps -- that is an input
# pathology, not a path bug, so we feed physically-valid inputs.
if block == "gdn":
    env_var = GDN_ENV
    statuses = linear_attention_path_statuses()
    q = cast(mx.random.normal((B, T, H, DK)) * (DK ** -0.5))
    k = cast(l2n(mx.random.normal((B, T, H, DK))))
    v = cast(mx.random.normal((B, T, H, DV)))
    beta = castg(mx.sigmoid(mx.random.normal((B, T, H))))
    g = castg(-mx.abs(mx.random.normal((B, T, H))) * 0.1)
    def run():
        return gated_delta_recurrent_dispatch(q, k, v, beta, g, path=path, allow_fallback=False)
else:  # kda
    env_var = KDA_ENV
    statuses = kda_path_statuses()
    q = cast(mx.random.normal((B, T, H, DK)) * (DK ** -0.5))
    k = cast(l2n(mx.random.normal((B, T, H, DK))))
    v = cast(mx.random.normal((B, T, H, DV)))
    g = castg(-mx.abs(mx.random.normal((B, T, H, DK))) * 0.05)
    beta = castg(mx.sigmoid(mx.random.normal((B, T, H))))
    def run():
        return kda_recurrent_dispatch(q, k, v, g, beta, path=path, allow_fallback=False)

st = statuses[path]
os.environ[env_var] = path
out = {
    "block": block, "path": path, "dtype": dtype_tag,
    "backend_available": bool(st.available), "backend_reason": st.reason,
    "status": "failed", "median_seconds": None, "throughput_melem_s": None,
    "measured_path": path, "error": None,
}
elements = B * T * H * DV  # output element count
try:
    for _ in range(WARMUP):
        o, _s = run(); mx.eval(o)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        o, _s = run(); mx.eval(o)
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    finite = not bool(mx.any(mx.isnan(o)).item())
    out["median_seconds"] = med
    out["throughput_melem_s"] = (elements / med) / 1e6 if med > 0 else None
    out["status"] = "ok" if finite else "failed"
    if not finite:
        out["error"] = "output contained NaN"
    out["out_shape"] = list(o.shape)
    out["out_dtype"] = str(o.dtype)
except Exception as exc:
    out["status"] = "failed"
    out["error"] = f"{type(exc).__name__}: {exc}"
print("V4CELL_JSON:" + json.dumps(out))
'''


def run_cell(block: str, path: str, dtype: str) -> dict:
    subs = {
        "__B__": SHAPE["batch"], "__T__": SHAPE["seq_len"], "__H__": SHAPE["num_heads"],
        "__DK__": SHAPE["head_dim_k"], "__DV__": SHAPE["head_dim_v"],
        "__WARMUP__": WARMUP, "__ITERS__": ITERS,
    }
    script = CELL_SCRIPT
    for key, val in subs.items():
        script = script.replace(key, str(val))
    cmd = [PYTHON, "-c", script, block, path, dtype]
    tb = _timeout_bin()
    if tb:
        cmd = [tb, str(CELL_TIMEOUT_S)] + cmd
    env = dict(os.environ)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
            timeout=CELL_TIMEOUT_S + 60,
        )
    except subprocess.TimeoutExpired:
        return {"block": block, "path": path, "dtype": dtype, "status": "failed",
                "median_seconds": None, "throughput_melem_s": None,
                "measured_path": path, "backend_available": False,
                "backend_reason": "TIMEOUT", "error": f"timeout >{CELL_TIMEOUT_S}s",
                "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    line = None
    for ln in (p.stdout or "").splitlines():
        if ln.startswith("V4CELL_JSON:"):
            line = ln[len("V4CELL_JSON:"):]
    if line is None:
        tail = ((p.stderr or "") + (p.stdout or "")).strip().splitlines()
        return {"block": block, "path": path, "dtype": dtype, "status": "failed",
                "median_seconds": None, "throughput_melem_s": None,
                "measured_path": path, "backend_available": False,
                "backend_reason": "subprocess produced no result line",
                "error": (tail[-1] if tail else f"rc={p.returncode}"),
                "wall_s": wall}
    rec = json.loads(line)
    rec["wall_s"] = wall
    return rec


def host_info() -> dict:
    info = {"platform": PLATFORM, "uname": platform.platform(),
            "python": sys.version.split()[0]}
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout.strip()
        info["cppmega_sha"] = sha
    except Exception:
        info["cppmega_sha"] = "?"
    return info


def render_md(cells: list[dict], host: dict) -> str:
    title = "Local Metal (Apple M4 Max)" if PLATFORM == "metal" else "gb10 (CUDA sm_121, GB10)"
    lines = [
        f"# cppmega v4 Speed Matrix (op-level GDN/KDA) — {title}",
        "",
        f"- Date: {DATESTAMP}",
        f"- Host: {host['uname']}",
        f"- cppmega SHA: `{host['cppmega_sha']}`",
        f"- Python: {host['python']}",
        f"- Op measured: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), "
        f"NOT full m04 1B training (the v4 benchmark harness is op-level).",
        f"- Shape: B={SHAPE['batch']} T={SHAPE['seq_len']} H={SHAPE['num_heads']} "
        f"Dk={SHAPE['head_dim_k']} Dv={SHAPE['head_dim_v']} (Path-E-eligible: Dk%32==0, Dv%4==0)",
        f"- Sweep: block {{gdn,kda}} x path {{a,b,c,d,e}} x dtype {{f32,bf16,fp16,fp16d}}. "
        f"warmup={WARMUP}, iters={ITERS}, median of {ITERS}. allow_fallback=DISABLED (fail-loud).",
        f"- dtype axis: f32=harness reference; bf16=REAL cast of q/k/v/beta/g. "
        f"fp8/nvfp4 are NOT representable as MLX op-input array dtypes "
        f"(no float8 array dtype; only to_fp8/from_fp8 packers) so they do not apply "
        f"at the op level and are omitted (the v3 fp8 axis was an m04-training GEMM "
        f"storage scheme, a different layer).",
        f"- Per-cell bound: {CELL_TIMEOUT_S}s OS timeout, fresh subprocess (GPU cleared on exit).",
        "",
        "| block | dtype | path | status | measured_path | median fwd ms | throughput Melem/s | reason |",
        "| ----- | ----- | ---- | ------ | ------------- | ------------: | -----------------: | ------ |",
    ]
    for c in cells:
        ms = f"{c['median_seconds']*1e3:.3f}" if c.get("median_seconds") else "—"
        tp = f"{c['throughput_melem_s']:.2f}" if c.get("throughput_melem_s") else "—"
        reason = c.get("error") or c.get("backend_reason") or ""
        reason = reason.replace("\n", " ")[:160]
        lines.append(
            f"| {c['block']} | {c['dtype']} | {c['path'].replace('path_','')} | "
            f"{c['status']} | {c.get('measured_path','?').replace('path_','')} | {ms} | {tp} | {reason} |"
        )
    lines += ["", "## Per-Cell Commands", ""]
    tb = _timeout_bin() or "timeout"
    for c in cells:
        tag = f"{c['block']}_{c['dtype']}_{c['path']}"
        lines.append(
            f"- `{tag}`: `{tb} {CELL_TIMEOUT_S} {PYTHON} -c "
            f"'<cell-script>' {c['block']} {c['path']} {c['dtype']}` "
            f"(driver: scripts/run_v4_speed_matrix_20260531.py)"
        )
    return "\n".join(lines) + "\n"


def render_html(cells: list[dict], host: dict) -> str:
    title = "Local Metal (Apple M4 Max)" if PLATFORM == "metal" else "gb10 (CUDA sm_121, GB10)"
    rows = []
    for c in cells:
        ms = f"{c['median_seconds']*1e3:.3f}" if c.get("median_seconds") else "—"
        tp = f"{c['throughput_melem_s']:.2f}" if c.get("throughput_melem_s") else "—"
        reason = (c.get("error") or c.get("backend_reason") or "").replace("\n", " ")
        cls = "ok" if c["status"] == "ok" else "fail"
        rows.append(
            f"<tr class='{cls}'><td>{c['block']}</td><td>{c['dtype']}</td>"
            f"<td>{c['path'].replace('path_','')}</td><td>{c['status']}</td>"
            f"<td>{c.get('measured_path','?').replace('path_','')}</td>"
            f"<td class='num'>{ms}</td><td class='num'>{tp}</td>"
            f"<td class='reason'>{reason[:240]}</td></tr>"
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>cppmega v4 Speed Matrix — {title} — {DATESTAMP}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a}}
h1{{font-size:20px}} .meta{{color:#555;font-size:13px;line-height:1.5;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top}}
th{{background:#f3f4f6;position:sticky;top:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tr.ok td{{background:#f6fff6}} tr.fail td{{background:#fff6f6}}
td.reason{{max-width:520px;color:#444;font-size:12px}}
</style></head><body>
<h1>cppmega v4 Speed Matrix (op-level GDN/KDA) — {title}</h1>
<div class='meta'>
Date: {DATESTAMP} &middot; Host: {host['uname']} &middot; cppmega SHA: <code>{host['cppmega_sha']}</code><br>
Op: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), op-level not full m04 training.<br>
Shape: B={SHAPE['batch']} T={SHAPE['seq_len']} H={SHAPE['num_heads']} Dk={SHAPE['head_dim_k']} Dv={SHAPE['head_dim_v']}
(Path-E-eligible). Sweep: block&times;path{{a-e}}&times;dtype{{f32,bf16,fp16,fp16d}}. warmup={WARMUP}, iters={ITERS}, median. allow_fallback DISABLED (fail-loud).<br>
dtype: f32=reference, bf16=real cast. fp8/nvfp4 not representable as MLX op-input array dtypes (omitted; that axis lived in m04-training GEMMs, not this op).
</div>
<table><thead><tr><th>block</th><th>dtype</th><th>path</th><th>status</th><th>measured_path</th>
<th>median fwd ms</th><th>throughput Melem/s</th><th>reason</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def main() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    host = host_info()
    cells = []
    for block in BLOCKS:
        for dtype in DTYPES:
            for path in PATHS:
                tag = f"{block}_{dtype}_{path}"
                print(f"[v4-matrix:{PLATFORM}] running {tag} ...", flush=True)
                rec = run_cell(block, path, dtype)
                print(f"    -> status={rec['status']} med="
                      f"{rec.get('median_seconds')} measured={rec.get('measured_path')}",
                      flush=True)
                cells.append(rec)
                (WORK_DIR / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    MD_OUT.write_text(render_md(cells, host))
    HTML_OUT.write_text(render_html(cells, host))
    (WORK_DIR / "all_cells.json").write_text(json.dumps(cells, indent=2))
    print(f"WROTE {MD_OUT}")
    print(f"WROTE {HTML_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
