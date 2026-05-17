#!/usr/bin/env python3
"""Render the 1B training matrix JSON as a static HTML speed report."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/tmp/cppmega_1b_path_matrix.json")
DEFAULT_OUTPUT = Path("/tmp/cppmega_1b_path_matrix.html")
DEFAULT_DTYPES = ("bf16", "fp8")
PATH_ORDER = ("path_b", "path_c_cold", "path_c_warm")


@dataclass(frozen=True)
class Row:
    case_id: str
    dtype: str
    optimizer: str
    path: str
    status: str
    tok_sec: float | None
    step_sec: float | None
    compile_time_s: float | None
    peak_memory_gb: float | None
    cache_hit: bool | None
    pass_fail_reason: str | None
    command: str
    receipt_path: str | None
    selected_schedule: dict[str, Any]
    proof_result: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render scripts/bench_1b_training_matrix.py JSON as HTML.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dtypes",
        default=",".join(DEFAULT_DTYPES),
        help="Comma-separated dtype sections to render.",
    )
    parser.add_argument(
        "--same-speed-tolerance",
        type=float,
        default=0.03,
        help="Path C is default-eligible when warm tok/s is within this fraction of Path B.",
    )
    return parser


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_rows(payload: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    for raw in payload.get("results", []):
        if not isinstance(raw, dict):
            continue
        rows.append(
            Row(
                case_id=str(raw.get("case_id") or ""),
                dtype=str(raw.get("dtype") or ""),
                optimizer=str(raw.get("optimizer") or ""),
                path=str(raw.get("path") or ""),
                status=str(raw.get("status") or ""),
                tok_sec=_number(raw.get("tok_sec")),
                step_sec=_number(raw.get("step_sec")),
                compile_time_s=_number(raw.get("compile_time_s")),
                peak_memory_gb=_number(raw.get("peak_memory_gb")),
                cache_hit=_bool(raw.get("cache_hit")),
                pass_fail_reason=(
                    str(raw.get("pass_fail_reason"))
                    if raw.get("pass_fail_reason") is not None
                    else None
                ),
                command=str(raw.get("command") or ""),
                receipt_path=(
                    str(raw.get("receipt_path"))
                    if raw.get("receipt_path") is not None
                    else None
                ),
                selected_schedule=_dict(raw.get("selected_schedule")),
                proof_result=_dict(raw.get("proof_result")),
            )
        )
    return rows


def parse_dtypes(spec: str) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in spec.split(",") if value.strip())


def h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}x"


def css_class_for_status(status: str) -> str:
    if status == "ok":
        return "status-ok"
    if status == "not_applicable":
        return "status-na"
    return "status-bad"


def path_label(path: str) -> str:
    labels = {
        "path_b": "Path B",
        "path_c_cold": "Path C cold",
        "path_c_warm": "Path C warm",
    }
    return labels.get(path, path)


def rows_by_key(rows: Iterable[Row]) -> dict[tuple[str, str, str], Row]:
    return {(row.dtype, row.optimizer, row.path): row for row in rows}


def optimizer_order(rows: Iterable[Row], dtype: str) -> list[str]:
    order = ("adamw", "lion", "muon", "muon_adamw")
    present = {row.optimizer for row in rows if row.dtype == dtype}
    return [value for value in order if value in present] + sorted(present - set(order))


def speed_ratio(candidate: Row | None, baseline: Row | None) -> float | None:
    if not candidate or not baseline:
        return None
    if candidate.tok_sec is None or baseline.tok_sec in (None, 0):
        return None
    return candidate.tok_sec / baseline.tok_sec


def memory_delta(candidate: Row | None, baseline: Row | None) -> float | None:
    if not candidate or not baseline:
        return None
    if candidate.peak_memory_gb is None or baseline.peak_memory_gb is None:
        return None
    return candidate.peak_memory_gb - baseline.peak_memory_gb


def decision_for(
    *,
    warm: Row | None,
    baseline: Row | None,
    tolerance: float,
) -> tuple[str, str, str]:
    if baseline and baseline.status == "not_applicable":
        if warm and warm.status == "ok":
            return (
                "Path C only",
                "decision-warn",
                "No Path B training surface exists for this dtype.",
            )
        return ("No runnable route", "decision-bad", "Path B is not applicable and Path C is not ok.")
    if not baseline or baseline.status != "ok":
        return ("No baseline", "decision-warn", "Path B did not produce an ok baseline.")
    if not warm or warm.status != "ok":
        return ("Keep Path B", "decision-bad", "Warm Path C did not produce an ok row.")
    ratio = speed_ratio(warm, baseline)
    if ratio is None:
        return ("Keep Path B", "decision-bad", "Missing tok/s for Path B or warm Path C.")
    if ratio >= 1.0:
        return ("Path C default candidate", "decision-good", "Warm Path C is faster than Path B.")
    if ratio >= 1.0 - tolerance:
        return (
            "Path C default candidate",
            "decision-good",
            f"Warm Path C is within {tolerance:.0%} of Path B.",
        )
    return (
        "Keep Path B",
        "decision-bad",
        f"Warm Path C is {(1.0 - ratio):.1%} slower than Path B.",
    )


def render_summary_cards(rows: list[Row], dtypes: tuple[str, ...], tolerance: float) -> str:
    keyed = rows_by_key(rows)
    cards: list[str] = []
    for dtype in dtypes:
        optimizers = optimizer_order(rows, dtype)
        ok_rows = [row for row in rows if row.dtype == dtype and row.status == "ok"]
        candidates = 0
        keep_b = 0
        na = 0
        fastest = max(ok_rows, key=lambda row: row.tok_sec or 0.0, default=None)
        for optimizer in optimizers:
            baseline = keyed.get((dtype, optimizer, "path_b"))
            warm = keyed.get((dtype, optimizer, "path_c_warm"))
            decision, class_name, _ = decision_for(
                warm=warm,
                baseline=baseline,
                tolerance=tolerance,
            )
            if class_name == "decision-good":
                candidates += 1
            elif decision == "Path C only":
                na += 1
            else:
                keep_b += 1
        cards.append(
            """
            <section class="summary-card">
              <div class="card-kicker">{dtype}</div>
              <div class="card-title">{ok_count} runnable cells</div>
              <div class="card-meta">Path C candidates: {candidates} / Keep Path B: {keep_b} / No baseline: {na}</div>
              <div class="card-foot">Fastest: {fastest}</div>
            </section>
            """.format(
                dtype=h(dtype.upper()),
                ok_count=len(ok_rows),
                candidates=candidates,
                keep_b=keep_b,
                na=na,
                fastest=(
                    f"{fastest.optimizer} {path_label(fastest.path)} at {fmt_num(fastest.tok_sec)} tok/s"
                    if fastest
                    else "-"
                ),
            )
        )
    return "\n".join(cards)


def render_comparison_table(rows: list[Row], dtype: str, tolerance: float) -> str:
    keyed = rows_by_key(rows)
    body: list[str] = []
    for optimizer in optimizer_order(rows, dtype):
        baseline = keyed.get((dtype, optimizer, "path_b"))
        cold = keyed.get((dtype, optimizer, "path_c_cold"))
        warm = keyed.get((dtype, optimizer, "path_c_warm"))
        cold_ratio = speed_ratio(cold, baseline)
        warm_ratio = speed_ratio(warm, baseline)
        mem_delta = memory_delta(warm, baseline)
        decision, class_name, reason = decision_for(
            warm=warm,
            baseline=baseline,
            tolerance=tolerance,
        )
        body.append(
            """
            <tr>
              <th>{optimizer}</th>
              <td>{b_tok}</td>
              <td>{cold_tok}</td>
              <td>{warm_tok}</td>
              <td class="{cold_ratio_class}">{cold_ratio}</td>
              <td class="{warm_ratio_class}">{warm_ratio}</td>
              <td>{b_mem}</td>
              <td>{warm_mem}</td>
              <td class="{mem_class}">{mem_delta}</td>
              <td><span class="decision {class_name}">{decision}</span><div class="muted">{reason}</div></td>
            </tr>
            """.format(
                optimizer=h(optimizer),
                b_tok=fmt_num(baseline.tok_sec if baseline else None),
                cold_tok=fmt_num(cold.tok_sec if cold else None),
                warm_tok=fmt_num(warm.tok_sec if warm else None),
                cold_ratio=fmt_ratio(cold_ratio),
                warm_ratio=fmt_ratio(warm_ratio),
                cold_ratio_class=ratio_class(cold_ratio, tolerance),
                warm_ratio_class=ratio_class(warm_ratio, tolerance),
                b_mem=fmt_num(baseline.peak_memory_gb if baseline else None, 2),
                warm_mem=fmt_num(warm.peak_memory_gb if warm else None, 2),
                mem_delta=fmt_signed(mem_delta, " GB"),
                mem_class="bad-number" if mem_delta is not None and mem_delta > 0 else "",
                class_name=class_name,
                decision=h(decision),
                reason=h(reason),
            )
        )
    return """
      <section class="panel">
        <div class="section-head">
          <h2>{dtype} training speed</h2>
          <p>Default rule: warm Path C must be at least {threshold:.0%} of Path B tok/s.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Optimizer</th>
                <th>Path B tok/s</th>
                <th>Path C cold tok/s</th>
                <th>Path C warm tok/s</th>
                <th>Cold / B</th>
                <th>Warm / B</th>
                <th>Path B GB</th>
                <th>Warm C GB</th>
                <th>Memory delta</th>
                <th>Default decision</th>
              </tr>
            </thead>
            <tbody>{body}</tbody>
          </table>
        </div>
      </section>
    """.format(
        dtype=h(dtype.upper()),
        threshold=1.0 - tolerance,
        body="\n".join(body),
    )


def ratio_class(value: float | None, tolerance: float) -> str:
    if value is None:
        return "muted"
    if value >= 1.0 - tolerance:
        return "good-number"
    return "bad-number"


def fmt_signed(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}{suffix}"


def render_methodology(payload: dict[str, Any], tolerance: float) -> str:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    software = payload.get("software", {}) if isinstance(payload.get("software"), dict) else {}
    command = str(payload.get("command") or "")
    threshold = 1.0 - tolerance
    return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>What This Report Compares</h2>
          <p>Full local 1B-class training matrix, not a microkernel-only bench.</p>
        </div>
        <div class="narrative-grid">
          <div>
            <h3>Workload</h3>
            <ul>
              <li>Model profile: <code>local_gb10_quarter</code>, the 13-layer hybrid profile used by the M0.4 GB10 work.</li>
              <li>Dataset: real parquet target shard <code>data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet</code>.</li>
              <li>Shape: batch <code>{batch_size}</code>, sequence length <code>{block_size}</code>, measured training steps <code>{steps}</code>, gradient checkpointing enabled by the matrix command.</li>
              <li>Optimizers: <code>adamw</code>, <code>lion</code>, <code>muon</code>, and <code>muon_adamw</code>.</li>
            </ul>
          </div>
          <div>
            <h3>Measurements</h3>
            <ul>
              <li><strong>tok/s</strong> is the receipt-level mean target-token throughput from <code>scripts/m04_train_step.py</code>.</li>
              <li><strong>compile s</strong> is the first recorded step time, used as the compile/warmup cost indicator.</li>
              <li><strong>peak GB</strong> is MLX/Metal peak memory from the per-cell receipt.</li>
              <li><strong>Warm / B</strong> is <code>Path C warm tok/s / Path B tok/s</code>. Default promotion requires at least <code>{threshold:.0%}</code>.</li>
            </ul>
          </div>
          <div>
            <h3>Software Identity</h3>
            <ul>
              <li>cppmega: <code>{cppmega_sha}</code></li>
              <li>TileLang: <code>{tilelang_sha}</code></li>
              <li>MLX: <code>{mlx_version}</code></li>
              <li>Renderer source: <code>scripts/render_1b_training_matrix_html.py</code></li>
            </ul>
          </div>
        </div>
        <details class="command-details">
          <summary>Matrix command</summary>
          <pre>{command}</pre>
        </details>
      </section>
    """.format(
        batch_size=h(config.get("batch_size", "-")),
        block_size=h(config.get("block_size", "-")),
        steps=h(config.get("steps", "-")),
        threshold=threshold,
        cppmega_sha=h(software.get("cppmega_sha", "-")),
        tilelang_sha=h(software.get("tilelang_sha", "-")),
        mlx_version=h(software.get("mlx_version", "-")),
        command=h(command),
    )


def render_route_legend() -> str:
    return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>Route Definitions</h2>
          <p>The dtype label and path label both matter.</p>
        </div>
        <div class="route-grid">
          <div class="route-card">
            <h3>BF16 Path B</h3>
            <p>Runs <code>--dtype bfloat16</code> with <code>CPPMEGA_KERNEL_PATH=auto</code>. This is the current non-forced baseline route. It may use existing Path B or reference surfaces according to the normal runtime policy.</p>
          </div>
          <div class="route-card">
            <h3>BF16 Path C</h3>
            <p>Runs <code>--dtype bfloat16</code> with <code>CPPMEGA_KERNEL_PATH=path_c</code>. Cold rows clear/use an empty TileLang cache; warm rows reuse the per-dtype optimizer TileLang cache.</p>
          </div>
          <div class="route-card">
            <h3>FP8 Path B</h3>
            <p>Runs <code>--dtype fp8_path_b</code> and forces <code>CPPMEGA_KERNEL_PATH__SPARSE_MLA=path_b</code>. The DSA Sparse-MLA baseline dispatch is recorded as <code>sparse_mla_fp8_reference_path_b</code>; it is an honest non-Path-C FP8 training baseline, not a Path C fallback.</p>
          </div>
          <div class="route-card">
            <h3>FP8 Path C</h3>
            <p>Runs <code>--dtype fp8_path_c</code> and forces Path C for Mamba3, M2RNN, and Sparse-MLA. Sparse-MLA consumes prepared <code>q_fp8/q_scale/kv_fp8/kv_scale</code> buffers through the TileLang/tvm-ffi route.</p>
          </div>
        </div>
      </section>
    """


def render_current_findings(rows: list[Row], dtypes: tuple[str, ...], tolerance: float) -> str:
    keyed = rows_by_key(rows)
    lines: list[str] = []
    for dtype in dtypes:
        for optimizer in optimizer_order(rows, dtype):
            baseline = keyed.get((dtype, optimizer, "path_b"))
            warm = keyed.get((dtype, optimizer, "path_c_warm"))
            ratio = speed_ratio(warm, baseline)
            if ratio is None:
                lines.append(
                    f"<li><code>{h(dtype)} / {h(optimizer)}</code>: no complete Path B versus warm Path C ratio.</li>"
                )
                continue
            decision, class_name, reason = decision_for(
                warm=warm,
                baseline=baseline,
                tolerance=tolerance,
            )
            lines.append(
                """
                <li>
                  <code>{dtype} / {optimizer}</code>: warm Path C is <strong>{ratio}</strong>
                  of Path B; <span class="decision {class_name}">{decision}</span>
                  <span class="muted">{reason}</span>
                </li>
                """.format(
                    dtype=h(dtype),
                    optimizer=h(optimizer),
                    ratio=fmt_ratio(ratio),
                    class_name=class_name,
                    decision=h(decision),
                    reason=h(reason),
                )
            )
    return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>Current Default Decision</h2>
          <p>Generated directly from the matrix ratios below.</p>
        </div>
        <div class="callout">
          <strong>Current result:</strong> no BF16 or FP8 optimizer row qualifies
          for Path C as the default under the {tolerance:.0%} same-speed rule.
          Keep Path B as default until warm Path C reaches at least {threshold:.0%}
          of Path B tok/s on the same 1B training workload.
        </div>
        <ul class="finding-list">{items}</ul>
      </section>
    """.format(
        tolerance=tolerance,
        threshold=1.0 - tolerance,
        items="\n".join(lines),
    )


def render_cell_table(rows: list[Row], dtype: str) -> str:
    dtype_rows = [
        row
        for row in rows
        if row.dtype == dtype
        and row.path in PATH_ORDER
    ]
    path_index = {path: index for index, path in enumerate(PATH_ORDER)}
    dtype_rows.sort(key=lambda row: (row.optimizer, path_index.get(row.path, 99)))
    body = []
    for row in dtype_rows:
        route = row.proof_result.get("fp8_path_c_route_status") or row.proof_result.get("path")
        body.append(
            """
            <tr>
              <th>{case_id}</th>
              <td>{path}</td>
              <td><span class="pill {status_class}">{status}</span></td>
              <td>{tok}</td>
              <td>{compile}</td>
              <td>{peak}</td>
              <td>{cache}</td>
              <td>{route}</td>
              <td>{reason}</td>
            </tr>
            """.format(
                case_id=h(row.case_id),
                path=h(path_label(row.path)),
                status_class=css_class_for_status(row.status),
                status=h(row.status),
                tok=fmt_num(row.tok_sec),
                compile=fmt_num(row.compile_time_s, 2),
                peak=fmt_num(row.peak_memory_gb, 2),
                cache=h(row.cache_hit),
                route=h(route),
                reason=h(row.pass_fail_reason),
            )
        )
    return """
      <section class="panel">
        <div class="section-head">
          <h2>{dtype} cell details</h2>
          <p>Exact cell statuses from the JSON receipt.</p>
        </div>
        <div class="table-wrap">
          <table class="detail-table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Path</th>
                <th>Status</th>
                <th>tok/s</th>
                <th>compile s</th>
                <th>peak GB</th>
                <th>cache hit</th>
                <th>route proof</th>
                <th>reason</th>
              </tr>
            </thead>
            <tbody>{body}</tbody>
          </table>
        </div>
      </section>
    """.format(dtype=h(dtype.upper()), body="\n".join(body))


def render_commands(rows: list[Row], dtypes: tuple[str, ...]) -> str:
    items = []
    for row in rows:
        if row.dtype not in dtypes:
            continue
        items.append(
            "<li><code>{case_id}</code><pre>{command}</pre></li>".format(
                case_id=h(row.case_id),
                command=h(row.command),
            )
        )
    return """
      <section class="panel">
        <details>
          <summary>Cell commands</summary>
          <ol class="commands">{items}</ol>
        </details>
      </section>
    """.format(items="\n".join(items))


def render_html(payload: dict[str, Any], rows: list[Row], dtypes: tuple[str, ...], tolerance: float) -> str:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    software = payload.get("software", {}) if isinstance(payload.get("software"), dict) else {}
    sections: list[str] = [
        render_methodology(payload, tolerance),
        render_route_legend(),
        render_current_findings(rows, dtypes, tolerance),
    ]
    sections.extend(
        render_comparison_table(rows, dtype, tolerance)
        for dtype in dtypes
        if any(row.dtype == dtype for row in rows)
    )
    sections.extend(
        render_cell_table(rows, dtype)
        for dtype in dtypes
        if any(row.dtype == dtype for row in rows)
    )
    sections.append(render_commands(rows, dtypes))
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cppmega 1B Path B vs Path C Training Matrix</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --line-strong: #b8c0cc;
      --good: #0f7a4f;
      --good-bg: #e7f5ee;
      --bad: #b42318;
      --bad-bg: #fde8e5;
      --warn: #93640f;
      --warn-bg: #fff3d6;
      --navy: #1e3a5f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 28px max(24px, calc((100vw - 1240px) / 2)) 22px;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
      line-height: 1.05;
    }}
    h2 {{
      margin: 0;
      font-size: 19px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; color: var(--muted); }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }}
    pre {{
      margin: 8px 0 0;
      overflow-x: auto;
      white-space: pre-wrap;
      color: #344054;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .meta span {{
      border: 1px solid var(--line);
      background: #f9fafb;
      border-radius: 6px;
      padding: 7px 9px;
      color: #344054;
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .summary-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .summary-card {{ padding: 18px; }}
    .card-kicker {{
      color: var(--navy);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .card-title {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 760;
    }}
    .card-meta, .card-foot {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel {{
      margin-top: 18px;
      overflow: hidden;
    }}
    .narrative {{
      padding-bottom: 2px;
    }}
    .narrative-grid, .route-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      padding: 18px;
    }}
    .narrative h3, .route-card h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .narrative ul {{
      margin: 0;
      padding-left: 18px;
      color: #344054;
      font-size: 13px;
    }}
    .narrative li + li {{
      margin-top: 7px;
    }}
    .route-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .route-card p {{
      font-size: 13px;
      color: #344054;
    }}
    .callout {{
      margin: 18px 18px 0;
      border: 1px solid var(--line-strong);
      border-left: 4px solid var(--navy);
      border-radius: 8px;
      background: #f8fbff;
      padding: 13px 14px;
      color: #344054;
      font-size: 14px;
    }}
    .finding-list {{
      padding: 16px 24px 18px 38px;
      margin: 0;
      color: #344054;
      font-size: 13px;
    }}
    .finding-list li + li {{
      margin-top: 8px;
    }}
    .command-details {{
      border-top: 1px solid var(--line);
      margin-top: 2px;
    }}
    .section-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 940px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      text-align: right;
      vertical-align: top;
      font-size: 13px;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{
      background: #f1f4f8;
      border-bottom: 1px solid var(--line-strong);
      color: #475467;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    tbody th {{ font-weight: 700; }}
    tbody tr:hover {{ background: #fafcff; }}
    .detail-table {{ min-width: 1180px; }}
    .decision {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .decision-good {{ background: var(--good-bg); color: var(--good); }}
    .decision-bad {{ background: var(--bad-bg); color: var(--bad); }}
    .decision-warn {{ background: var(--warn-bg); color: var(--warn); }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status-ok {{ background: var(--good-bg); color: var(--good); }}
    .status-na {{ background: var(--warn-bg); color: var(--warn); }}
    .status-bad {{ background: var(--bad-bg); color: var(--bad); }}
    .good-number {{ color: var(--good); font-weight: 700; }}
    .bad-number {{ color: var(--bad); font-weight: 700; }}
    .muted {{ color: var(--muted); font-size: 12px; white-space: normal; }}
    details {{ padding: 16px 18px; }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .commands {{
      margin: 12px 0 0;
      padding-left: 24px;
    }}
    .commands li {{ margin-bottom: 12px; }}
    @media (max-width: 720px) {{
      header {{ padding: 22px 16px; }}
      main {{ padding: 16px; }}
      .section-head {{ display: block; }}
      .section-head p {{ margin-top: 6px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>1B Training Matrix: Path B vs Path C</h1>
    <p>BF16/FP8 optimizer sweep over the local GB10 quarter profile. The default decision is speed-gated by warm Path C versus Path B.</p>
    <div class="meta">
      <span>batch {batch_size}</span>
      <span>seq {block_size}</span>
      <span>steps {steps}</span>
      <span>cppmega {cppmega_sha}</span>
      <span>TileLang {tilelang_sha}</span>
      <span>MLX {mlx_version}</span>
      <span>tolerance {tolerance:.0%}</span>
    </div>
  </header>
  <main>
    <div class="summary-grid">
      {cards}
    </div>
    {sections}
  </main>
</body>
</html>
""".format(
        batch_size=h(config.get("batch_size", "-")),
        block_size=h(config.get("block_size", "-")),
        steps=h(config.get("steps", "-")),
        cppmega_sha=h(software.get("cppmega_sha", "-")),
        tilelang_sha=h(software.get("tilelang_sha", "-")),
        mlx_version=h(software.get("mlx_version", "-")),
        tolerance=tolerance,
        cards=render_summary_cards(rows, dtypes, tolerance),
        sections="\n".join(sections),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("matrix JSON root must be an object")
    dtypes = parse_dtypes(args.dtypes)
    rows = parse_rows(payload)
    html_text = render_html(payload, rows, dtypes, args.same_speed_tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
