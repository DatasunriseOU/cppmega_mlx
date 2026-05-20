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
    cli_optimizer: str | None
    optimizer_key: str | None
    optimizer_name: str | None
    optimizer_class: str | None
    optimizer_source: str | None
    steps_completed: int | None
    first_step_sec: float | None
    median_step_sec: float | None
    tok_sec: float | None
    step_sec: float | None
    compile_time_s: float | None
    peak_memory_gb: float | None
    active_memory_gb: float | None
    cache_memory_gb: float | None
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
        "--compile-receipt",
        type=Path,
        default=None,
        help=(
            "Optional scripts/path_c_fusion_compile_receipt.py JSON. This is "
            "rendered separately because the matrix measures the runtime route."
        ),
    )
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
                cli_optimizer=(
                    str(raw.get("cli_optimizer"))
                    if raw.get("cli_optimizer") is not None
                    else None
                ),
                optimizer_key=(
                    str(raw.get("optimizer_key"))
                    if raw.get("optimizer_key") is not None
                    else None
                ),
                optimizer_name=(
                    str(raw.get("optimizer_name"))
                    if raw.get("optimizer_name") is not None
                    else None
                ),
                optimizer_class=(
                    str(raw.get("optimizer_class"))
                    if raw.get("optimizer_class") is not None
                    else None
                ),
                optimizer_source=(
                    str(raw.get("optimizer_source"))
                    if raw.get("optimizer_source") is not None
                    else None
                ),
                steps_completed=(
                    int(raw.get("steps_completed"))
                    if raw.get("steps_completed") is not None
                    else None
                ),
                first_step_sec=_number(raw.get("first_step_sec")),
                median_step_sec=_number(raw.get("median_step_sec")),
                tok_sec=_number(raw.get("tok_sec")),
                step_sec=_number(raw.get("step_sec")),
                compile_time_s=_number(raw.get("compile_time_s")),
                peak_memory_gb=_number(raw.get("peak_memory_gb")),
                active_memory_gb=_number(raw.get("active_memory_gb")),
                cache_memory_gb=_number(raw.get("cache_memory_gb")),
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
    order = (
        "adamw",
        "adam8bit",
        "lion",
        "lion8bit",
        "muon",
        "muon_adamw",
        "muon_int8",
    )
    present = {row.optimizer for row in rows if row.dtype == dtype}
    return [value for value in order if value in present] + sorted(present - set(order))


def optimizer_label(row: Row | None, *, dtype: str, optimizer: str) -> str:
    """Display the real runtime optimizer, not only the matrix axis name."""

    key = (row.optimizer_key if row else None) or (row.cli_optimizer if row else None)
    name = row.optimizer_name if row else None
    labels = {
        "adamw": "adamw16",
        "adam8bit": "adam8bit",
        "lion": "lion16",
        "lion8bit": "lion8bit",
        "muon": "muon",
        "muon_adamw": "muon_adamw",
        "muon_int8": "muon-int8",
    }
    axis_label = labels.get(optimizer, optimizer)
    expected_keys = {
        "adamw": {"adamw"},
        "adam8bit": {"adam8bit"},
        "lion": {"lion"},
        "lion8bit": {"lion8bit"},
        "muon": {"muon", "muon_adamw"},
        "muon_adamw": {"muon_adamw"},
        "muon_int8": {"int8"},
    }
    if key and key not in expected_keys.get(optimizer, {optimizer}):
        return f"{axis_label} ({key})"
    if name and name.lower() != axis_label.lower():
        return f"{axis_label} ({name})"
    return axis_label


def optimizer_class_hint(row: Row | None) -> str:
    if not row:
        return ""
    parts = []
    if row.optimizer_class:
        parts.append(row.optimizer_class)
    if row.optimizer_source:
        parts.append(row.optimizer_source)
    return " / ".join(parts)


def speed_ratio(candidate: Row | None, baseline: Row | None) -> float | None:
    if not candidate or not baseline:
        return None
    if candidate.tok_sec is None or baseline.tok_sec in (None, 0):
        return None
    return candidate.tok_sec / baseline.tok_sec


def memory_delta(
    candidate: Row | None,
    baseline: Row | None,
    attr: str = "peak_memory_gb",
) -> float | None:
    if not candidate or not baseline:
        return None
    candidate_value = getattr(candidate, attr)
    baseline_value = getattr(baseline, attr)
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def memory_interpretation(
    *,
    peak_delta: float | None,
    active_delta: float | None,
    cache_delta: float | None,
) -> str:
    """Classify allocator deltas so peak high-water is not read as live memory."""

    if peak_delta is None:
        return "-"
    if active_delta is not None and active_delta > 0.25:
        return "live active growth"
    if cache_delta is not None and cache_delta > 0.25:
        return "allocator cache growth"
    if peak_delta >= 1.0:
        return "transient peak/high-water; active is flat"
    if peak_delta > 0:
        return "small peak-only delta"
    return "no peak regression"


def path_c_default_block_reason(payload: dict[str, Any]) -> str | None:
    receipt = payload.get("path_c_fusion_compile_receipt")
    if not isinstance(receipt, dict):
        return None
    reporting = receipt.get("reporting_contract", {})
    if not isinstance(reporting, dict):
        return None
    if reporting.get("path_c_default_allowed") is not False:
        return None
    runtime_fused = reporting.get("runtime_uses_fused_train_block")
    plan = receipt.get("compile_plan", {})
    contract = plan.get("schedule_contract", {}) if isinstance(plan, dict) else {}
    contract_status = (
        contract.get("status") if isinstance(contract, dict) else None
    ) or "unknown"
    return (
        "compile receipt blocks default promotion: "
        f"runtime fused train block is {runtime_fused}; "
        f"schedule contract is {contract_status}."
    )


def decision_for(
    *,
    warm: Row | None,
    baseline: Row | None,
    tolerance: float,
    default_block_reason: str | None = None,
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
        if default_block_reason:
            return (
                "Path C speed candidate",
                "decision-warn",
                f"Warm Path C is faster than Path B, but {default_block_reason}",
            )
        return ("Path C default candidate", "decision-good", "Warm Path C is faster than Path B.")
    if ratio >= 1.0 - tolerance:
        if default_block_reason:
            return (
                "Path C speed candidate",
                "decision-warn",
                f"Warm Path C is within {tolerance:.0%} of Path B, but {default_block_reason}",
            )
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


def render_summary_cards(
    rows: list[Row],
    dtypes: tuple[str, ...],
    tolerance: float,
    default_block_reason: str | None = None,
) -> str:
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
                default_block_reason=default_block_reason,
            )
            if decision in {"Path C default candidate", "Path C speed candidate"}:
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
              <div class="card-meta">{candidate_label}: {candidates} / Keep Path B: {keep_b} / No baseline: {na}</div>
              <div class="card-foot">Fastest: {fastest}</div>
            </section>
            """.format(
                dtype=h(dtype.upper()),
                ok_count=len(ok_rows),
                candidate_label="Path C speed candidates"
                if default_block_reason
                else "Path C candidates",
                candidates=candidates,
                keep_b=keep_b,
                na=na,
                fastest=(
                    f"{optimizer_label(fastest, dtype=dtype, optimizer=fastest.optimizer)} {path_label(fastest.path)} at {fmt_num(fastest.tok_sec)} tok/s"
                    if fastest
                    else "-"
                ),
            )
        )
    return "\n".join(cards)


def render_comparison_table(
    rows: list[Row],
    dtype: str,
    tolerance: float,
    default_block_reason: str | None = None,
) -> str:
    keyed = rows_by_key(rows)
    body: list[str] = []
    for optimizer in optimizer_order(rows, dtype):
        baseline = keyed.get((dtype, optimizer, "path_b"))
        cold = keyed.get((dtype, optimizer, "path_c_cold"))
        warm = keyed.get((dtype, optimizer, "path_c_warm"))
        cold_ratio = speed_ratio(cold, baseline)
        warm_ratio = speed_ratio(warm, baseline)
        peak_delta = memory_delta(warm, baseline, "peak_memory_gb")
        active_delta = memory_delta(warm, baseline, "active_memory_gb")
        cache_delta = memory_delta(warm, baseline, "cache_memory_gb")
        decision, class_name, reason = decision_for(
            warm=warm,
            baseline=baseline,
            tolerance=tolerance,
            default_block_reason=default_block_reason,
        )
        body.append(
            """
            <tr>
              <th>{optimizer}<div class="muted">{optimizer_hint}</div></th>
              <td>{b_tok}</td>
              <td>{cold_tok}</td>
              <td>{warm_tok}</td>
              <td class="{cold_ratio_class}">{cold_ratio}</td>
              <td class="{warm_ratio_class}">{warm_ratio}</td>
              <td>{b_peak}</td>
              <td>{warm_peak}</td>
              <td class="{peak_class}">{peak_delta}</td>
              <td class="{active_class}">{active_delta}</td>
              <td class="{cache_class}">{cache_delta}</td>
              <td>{memory_interpretation}</td>
              <td><span class="decision {class_name}">{decision}</span><div class="muted">{reason}</div></td>
            </tr>
            """.format(
                optimizer=h(optimizer_label(baseline or warm or cold, dtype=dtype, optimizer=optimizer)),
                optimizer_hint=h(optimizer_class_hint(baseline or warm or cold)),
                b_tok=fmt_num(baseline.tok_sec if baseline else None),
                cold_tok=fmt_num(cold.tok_sec if cold else None),
                warm_tok=fmt_num(warm.tok_sec if warm else None),
                cold_ratio=fmt_ratio(cold_ratio),
                warm_ratio=fmt_ratio(warm_ratio),
                cold_ratio_class=ratio_class(cold_ratio, tolerance),
                warm_ratio_class=ratio_class(warm_ratio, tolerance),
                b_peak=fmt_num(baseline.peak_memory_gb if baseline else None, 2),
                warm_peak=fmt_num(warm.peak_memory_gb if warm else None, 2),
                peak_delta=fmt_signed(peak_delta, " GiB"),
                active_delta=fmt_signed(active_delta, " GiB"),
                cache_delta=fmt_signed(cache_delta, " GiB"),
                memory_interpretation=h(
                    memory_interpretation(
                        peak_delta=peak_delta,
                        active_delta=active_delta,
                        cache_delta=cache_delta,
                    )
                ),
                peak_class="bad-number" if peak_delta is not None and peak_delta > 0 else "",
                active_class="bad-number" if active_delta is not None and active_delta > 0.25 else "",
                cache_class="bad-number" if cache_delta is not None and cache_delta > 0 else "",
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
                <th>Path B peak GiB</th>
                <th>Warm C peak GiB</th>
                <th>Peak allocator delta</th>
                <th>Active delta</th>
                <th>Cache delta</th>
                <th>Memory read</th>
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
              <li>Optimizer axes: <code>adamw</code>, <code>adam8bit</code>, <code>lion</code>, <code>lion8bit</code>, <code>muon</code>, <code>muon_adamw</code>, and <code>muon_int8</code>. The tables also show the runtime optimizer class, so <code>lion</code> means fp32-moment Lion16 while <code>lion8bit</code> is the quantized optimizer that previously reached the 900 tok/s-class baseline.</li>
            </ul>
          </div>
          <div>
            <h3>Measurements</h3>
            <ul>
              <li><strong>tok/s</strong> is the receipt-level mean target-token throughput from <code>scripts/m04_train_step.py</code>.</li>
              <li><strong>first step s</strong> is the first recorded step, used as the compile/warmup cost indicator. <strong>steady s</strong> is the mean over the remaining steps.</li>
              <li><strong>peak GiB</strong> is MLX/Metal allocator high-water from the per-cell receipt, not live tensor residency.</li>
              <li><strong>active/cache GiB</strong> are the receipt's post-run active and allocator-cache memory; these separate real live tensor growth from cache/compile pressure.</li>
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


def render_route_legend(payload: dict[str, Any]) -> str:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    mamba3_bwd = str(config.get("mamba3_bwd") or "path_b")
    if mamba3_bwd == "path_c":
        bf16_path_c_detail = (
            "This rendered matrix is the explicit full-Path-C experiment: "
            "Mamba3 forward and backward both run through Path C via "
            "<code>--mamba3-bwd path_c</code>. This is not the default "
            "promotion rule by itself."
        )
    else:
        bf16_path_c_detail = (
            "The default-safe matrix keeps Mamba3 on Path C forward plus Path B "
            "backward via <code>--mamba3-bwd path_b</code>; pass "
            "<code>--mamba3-bwd path_c</code> only for an explicit full-Path-C "
            "experiment."
        )
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
            <p>Runs <code>--dtype bfloat16</code> with <code>CPPMEGA_KERNEL_PATH=path_c</code>. {bf16_path_c_detail}</p>
          </div>
          <div class="route-card">
            <h3>FP8 Path B</h3>
            <p>Runs <code>--dtype fp8_path_b</code> and forces <code>CPPMEGA_KERNEL_PATH__SPARSE_MLA=path_b</code> plus <code>CPPMEGA_SPARSE_MLA_FP8_ROUTE=path_b</code>. The DSA Sparse-MLA baseline dispatch is recorded as <code>sparse_mla_fp8_reference_path_b</code>; it is an honest non-Path-C FP8 training baseline, not a Path C fallback.</p>
          </div>
          <div class="route-card">
            <h3>FP8 Path C</h3>
            <p>Runs <code>--dtype fp8_path_c</code>, <code>CPPMEGA_SPARSE_MLA_FP8_ROUTE=path_c</code>, and the selected Mamba3 backward policy from the matrix command. Sparse-MLA consumes prepared <code>q_fp8/q_scale/kv_fp8/kv_scale</code> buffers through the TileLang/tvm-ffi route.</p>
          </div>
        </div>
      </section>
    """.format(bf16_path_c_detail=bf16_path_c_detail)


def render_compile_receipt(payload: dict[str, Any]) -> str:
    receipt = payload.get("path_c_fusion_compile_receipt")
    if not isinstance(receipt, dict):
        return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>Fused Schedule Compile Receipt</h2>
          <p>No compile receipt was attached to this render.</p>
        </div>
        <div class="callout">
          The tables below still measure the current runtime route. They do not
          prove that the generated fused train-block was compiled or executed.
        </div>
      </section>
    """
    target = receipt.get("schedule_target", {})
    spec = receipt.get("schedule_spec", {})
    plan = receipt.get("compile_plan", {})
    contract = plan.get("schedule_contract", {}) if isinstance(plan, dict) else {}
    artifact = receipt.get("artifact", {})
    source = receipt.get("generated_source", {})
    reporting = receipt.get("reporting_contract", {})
    cache_key = receipt.get("fusion_cache_key", {})
    recompile_audit = receipt.get("cache_key_recompile_audit", {})
    runtime_contract = receipt.get("runtime_execution_contract", {})
    runtime_smoke = receipt.get("runtime_smoke", {})
    runtime_smoke_kernel_source = (
        runtime_smoke.get("kernel_source", {})
        if isinstance(runtime_smoke, dict)
        else {}
    )
    physical_shapes = (
        source.get("physical_buffer_abi_shapes", {})
        if isinstance(source, dict)
        else {}
    )
    spilled_scratch = (
        source.get("spilled_shared_scratch_shapes", {})
        if isinstance(source, dict)
        else {}
    )
    if isinstance(spilled_scratch, dict):
        spill_rows = []
        for name, info in sorted(
            spilled_scratch.items(),
            key=lambda item: (
                int(item[1].get("bytes", 0))
                if isinstance(item[1], dict)
                else 0
            ),
            reverse=True,
        )[:8]:
            if isinstance(info, dict):
                shape = tuple(info.get("shape", ()))
                dtype = info.get("dtype", "?")
                byte_count = info.get("bytes", "?")
                internal = " internal" if info.get("internal_scratch_abi") else ""
                spill_rows.append(
                    f"{name}{shape}:{dtype}:{byte_count}B{internal}"
                )
            else:
                spill_rows.append(str(name))
        spilled_scratch_text = ", ".join(spill_rows)
    else:
        spilled_scratch_text = ""
    internal_scratch_buffers = (
        source.get("internal_scratch_abi_buffers", [])
        if isinstance(source, dict)
        else []
    )
    physical_abi_validation = (
        source.get("physical_abi_validation", {})
        if isinstance(source, dict)
        else {}
    )
    physical_abi_runtime_bridge = (
        source.get("physical_abi_runtime_bridge", {})
        if isinstance(source, dict)
        else {}
    )
    physical_abi_runtime_binding = (
        source.get("physical_abi_runtime_binding", {})
        if isinstance(source, dict)
        else {}
    )
    internal_scratch_text = (
        ", ".join(str(name) for name in internal_scratch_buffers)
        if isinstance(internal_scratch_buffers, list)
        else ""
    )
    if isinstance(physical_shapes, dict):
        physical_shapes_text = ", ".join(
            f"{name}{tuple(shape) if isinstance(shape, list) else shape}"
            for name, shape in sorted(physical_shapes.items())
        )
    else:
        physical_shapes_text = ""
    return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>Fused Schedule Compile Receipt</h2>
          <p>Separate evidence for the generated single-entry TileLang schedule.</p>
        </div>
        <div class="narrative-grid">
          <div>
            <h3>Compile Status</h3>
            <ul>
              <li>Status: <code>{status}</code></li>
              <li>Native compile requested: <code>{native_requested}</code></li>
              <li>Native compile ok: <code>{native_ok}</code></li>
              <li>Artifact: <code>{artifact_type}</code></li>
              <li>Elapsed: <code>{elapsed}</code> s</li>
            </ul>
          </div>
          <div>
            <h3>Selected Schedule</h3>
            <ul>
              <li>Schedule id: <code>{schedule_id}</code></li>
              <li>Implementation: <code>{implementation_kind}</code></li>
              <li>Schedule status: <code>{schedule_status}</code></li>
              <li>Contract status: <code>{contract_status}</code></li>
              <li>Production fragments complete: <code>{fragments_complete}</code></li>
              <li>Real ABI complete: <code>{real_abi_complete}</code></li>
            </ul>
          </div>
          <div>
            <h3>Physical ABI</h3>
            <ul>
              <li>Policy: <code>{physical_abi_policy}</code></li>
              <li>Logical buffers: <code>{logical_parameter_count}</code></li>
              <li>Physical kernel buffers: <code>{physical_parameter_count}</code></li>
              <li>Logical ABI map entries: <code>{logical_buffer_abi_map_count}</code></li>
              <li>Physical ABI validation: <code>{physical_abi_validation_status}</code></li>
              <li>Runtime ABI bridge: <code>{runtime_abi_bridge_status}</code></li>
              <li>Runtime ABI binding: <code>{runtime_abi_binding_status}</code></li>
              <li>Runtime ABI required banks: <code>{runtime_abi_required_banks}</code></li>
              <li>Runtime ABI missing banks: <code>{runtime_abi_missing_banks}</code></li>
              <li>Runtime kernel buffers: <code>{kernel_parameter_count}</code></li>
              <li>Metal buffer limit: <code>{metal_buffer_limit}</code></li>
              <li>Limit exceeded: <code>{metal_limit_exceeded}</code></li>
              <li>Bank shapes: <code>{physical_shapes}</code></li>
              <li>Shared scratch ABI buffers: <code>{spilled_shared_scratch_count}</code></li>
              <li>Shared scratch ABI bytes: <code>{shared_scratch_abi_bytes}</code></li>
              <li>Internal scratch ABI buffers: <code>{internal_scratch_abi_count}</code></li>
              <li>Internal scratch names: <code>{internal_scratch_names}</code></li>
              <li>Top scratch spills: <code>{spilled_scratch}</code></li>
            </ul>
          </div>
          <div>
            <h3>Reporting Boundary</h3>
            <ul>
              <li>Matrix measures current runtime route: <code>{matrix_runtime}</code></li>
              <li>Runtime uses fused train block: <code>{runtime_fused}</code></li>
              <li>Path C default allowed: <code>{default_allowed}</code></li>
              <li>Runtime execution status: <code>{runtime_execution_status}</code></li>
              <li>Plan single-kernel fused: <code>{plan_single_kernel_fused}</code></li>
              <li>Plan schedule status: <code>{plan_schedule_status}</code></li>
              <li>Contract runtime status: <code>{runtime_contract_status}</code></li>
              <li>Contract reason: <code>{runtime_contract_reason}</code></li>
              <li>Declared schedule id: <code>{runtime_contract_schedule_id}</code></li>
              <li>Single-thread generated kernel: <code>{single_thread_kernel}</code></li>
              <li>Lane-0 serialized fragments: <code>{lane0_serial_fragments}</code></li>
              <li>Lane-0 production fragments: <code>{lane0_production_fragments}</code></li>
              <li>Lane-strided row loops: <code>{lane_strided_row_loops}</code></li>
              <li>Fusion cache-key status: <code>{cache_key_status}</code></li>
              <li>Recompile audit status: <code>{recompile_status}</code></li>
              <li>Recompile cache-hit bit: <code>{recompile_cache_hit_status}</code></li>
              <li>Lowered module digest: <code>{lowered_digest}</code></li>
              <li>Generated source: <code>{source_path}</code></li>
              <li>Source SHA256: <code>{source_sha}</code></li>
            </ul>
          </div>
          <div>
            <h3>Runtime Smoke</h3>
            <ul>
              <li>Mode: <code>{runtime_smoke_mode}</code></li>
              <li>Status: <code>{runtime_smoke_status}</code></li>
              <li>Executed: <code>{runtime_smoke_executed}</code></li>
              <li>Smoke schedule id: <code>{runtime_smoke_schedule_id}</code></li>
              <li>Smoke buffers: <code>{runtime_smoke_buffers}</code></li>
              <li>Smoke ABI bytes: <code>{runtime_smoke_bytes}</code></li>
              <li>Smoke MSL bytes: <code>{runtime_smoke_msl_bytes}</code></li>
              <li>Smoke threadgroup bytes: <code>{runtime_smoke_threadgroup_bytes}</code></li>
              <li>Smoke compile: <code>{runtime_smoke_compile_elapsed}</code> s</li>
              <li>Smoke execute: <code>{runtime_smoke_execute_elapsed}</code> s</li>
              <li>Smoke error: <code>{runtime_smoke_error}</code></li>
            </ul>
          </div>
        </div>
        <details class="command-details">
          <summary>Missing or untrusted contract detail</summary>
          <pre>{missing}</pre>
        </details>
      </section>
    """.format(
        status=h(receipt.get("status")),
        native_requested=h(receipt.get("native_compile_requested")),
        native_ok=h(receipt.get("native_compile_ok")),
        artifact_type=h(artifact.get("type") if isinstance(artifact, dict) else None),
        elapsed=fmt_num(_number(receipt.get("elapsed_s")), 3),
        schedule_id=h(target.get("schedule_id") if isinstance(target, dict) else None),
        implementation_kind=h(
            spec.get("implementation_kind") if isinstance(spec, dict) else None
        ),
        schedule_status=h(
            target.get("schedule_status") if isinstance(target, dict) else None
        ),
        contract_status=h(
            contract.get("status") if isinstance(contract, dict) else None
        ),
        fragments_complete=h(
            spec.get("production_fragments_complete")
            if isinstance(spec, dict)
            else None
        ),
        real_abi_complete=h(
            spec.get("real_abi_contract_complete") if isinstance(spec, dict) else None
        ),
        physical_abi_policy=h(
            source.get("physical_abi_policy") if isinstance(source, dict) else None
        ),
        logical_parameter_count=h(
            source.get("logical_parameter_count") if isinstance(source, dict) else None
        ),
        physical_parameter_count=h(
            source.get("physical_parameter_count") if isinstance(source, dict) else None
        ),
        logical_buffer_abi_map_count=h(
            source.get("logical_buffer_abi_map_count")
            if isinstance(source, dict)
            else None
        ),
        physical_abi_validation_status=h(
            physical_abi_validation.get("status")
            if isinstance(physical_abi_validation, dict)
            else None
        ),
        runtime_abi_bridge_status=h(
            physical_abi_runtime_bridge.get("status")
            if isinstance(physical_abi_runtime_bridge, dict)
            else None
        ),
        runtime_abi_binding_status=h(
            physical_abi_runtime_binding.get("status")
            if isinstance(physical_abi_runtime_binding, dict)
            else None
        ),
        runtime_abi_required_banks=h(
            ", ".join(
                str(name)
                for name in physical_abi_runtime_bridge.get("required_bank_buffers", [])
            )
            if isinstance(physical_abi_runtime_bridge, dict)
            else None
        ),
        runtime_abi_missing_banks=h(
            ", ".join(
                str(name)
                for name in physical_abi_runtime_binding.get("missing_bank_buffers", [])
            )
            if isinstance(physical_abi_runtime_binding, dict)
            else None
        ),
        kernel_parameter_count=h(
            runtime_contract.get("kernel_parameter_count")
            if isinstance(runtime_contract, dict)
            else None
        ),
        metal_buffer_limit=h(
            runtime_contract.get("metal_buffer_limit")
            if isinstance(runtime_contract, dict)
            else None
        ),
        metal_limit_exceeded=h(
            runtime_contract.get("metal_buffer_limit_exceeded")
            if isinstance(runtime_contract, dict)
            else None
        ),
        physical_shapes=h(physical_shapes_text),
        spilled_shared_scratch_count=h(
            source.get("spilled_shared_scratch_count")
            if isinstance(source, dict)
            else None
        ),
        shared_scratch_abi_bytes=h(
            source.get("shared_scratch_abi_bytes") if isinstance(source, dict) else None
        ),
        internal_scratch_abi_count=h(
            source.get("internal_scratch_abi_count")
            if isinstance(source, dict)
            else None
        ),
        internal_scratch_names=h(internal_scratch_text),
        spilled_scratch=h(spilled_scratch_text),
        matrix_runtime=h(
            reporting.get("matrix_measures_current_runtime_route")
            if isinstance(reporting, dict)
            else None
        ),
        runtime_fused=h(
            reporting.get("runtime_uses_fused_train_block")
            if isinstance(reporting, dict)
            else None
        ),
        default_allowed=h(
            reporting.get("path_c_default_allowed")
            if isinstance(reporting, dict)
            else None
        ),
        runtime_execution_status=h(
            runtime_contract.get("status")
            if isinstance(runtime_contract, dict)
            else None
        ),
        plan_single_kernel_fused=h(
            runtime_contract.get("plan_single_kernel_fused")
            if isinstance(runtime_contract, dict)
            else None
        ),
        plan_schedule_status=h(
            runtime_contract.get("plan_schedule_status")
            if isinstance(runtime_contract, dict)
            else None
        ),
        runtime_contract_status=h(
            runtime_contract.get("schedule_contract_status")
            if isinstance(runtime_contract, dict)
            else None
        ),
        runtime_contract_reason=h(
            runtime_contract.get("schedule_contract_reason")
            if isinstance(runtime_contract, dict)
            else None
        ),
        runtime_contract_schedule_id=h(
            runtime_contract.get("schedule_contract_declared_schedule_id")
            if isinstance(runtime_contract, dict)
            else None
        ),
        single_thread_kernel=h(
            runtime_contract.get("single_thread_kernel")
            if isinstance(runtime_contract, dict)
            else None
        ),
        lane0_serial_fragments=h(
            runtime_contract.get("lane0_serial_fragments")
            if isinstance(runtime_contract, dict)
            else None
        ),
        lane0_production_fragments=h(
            ", ".join(runtime_contract.get("lane0_production_fragments") or [])
            if isinstance(runtime_contract, dict)
            else None
        ),
        lane_strided_row_loops=h(
            runtime_contract.get("lane_strided_row_loops")
            if isinstance(runtime_contract, dict)
            else None
        ),
        cache_key_status=h(
            cache_key.get("status") if isinstance(cache_key, dict) else None
        ),
        recompile_status=h(
            recompile_audit.get("status")
            if isinstance(recompile_audit, dict)
            else None
        ),
        recompile_cache_hit_status=h(
            recompile_audit.get("cache_hit_status")
            if isinstance(recompile_audit, dict)
            else None
        ),
        lowered_digest=h(
            cache_key.get("lowered_module_digest")
            if isinstance(cache_key, dict)
            else None
        ),
        source_path=h(source.get("path") if isinstance(source, dict) else None),
        source_sha=h(source.get("sha256") if isinstance(source, dict) else None),
        runtime_smoke_mode=h(
            runtime_smoke.get("mode") if isinstance(runtime_smoke, dict) else None
        ),
        runtime_smoke_status=h(
            runtime_smoke.get("status") if isinstance(runtime_smoke, dict) else None
        ),
        runtime_smoke_executed=h(
            runtime_smoke.get("actually_executed")
            if isinstance(runtime_smoke, dict)
            else None
        ),
        runtime_smoke_schedule_id=h(
            runtime_smoke.get("schedule_id")
            if isinstance(runtime_smoke, dict)
            else None
        ),
        runtime_smoke_buffers=h(
            runtime_smoke.get("kernel_parameter_count")
            if isinstance(runtime_smoke, dict)
            else None
        ),
        runtime_smoke_bytes=h(
            runtime_smoke.get("total_buffer_bytes")
            if isinstance(runtime_smoke, dict)
            else None
        ),
        runtime_smoke_msl_bytes=h(
            runtime_smoke_kernel_source.get("kernel_source_bytes")
            if isinstance(runtime_smoke_kernel_source, dict)
            else None
        ),
        runtime_smoke_threadgroup_bytes=h(
            runtime_smoke_kernel_source.get("threadgroup_dynamic_shared_bytes")
            if isinstance(runtime_smoke_kernel_source, dict)
            else None
        ),
        runtime_smoke_compile_elapsed=fmt_num(
            _number(runtime_smoke.get("compile_elapsed_s"))
            if isinstance(runtime_smoke, dict)
            else None,
            3,
        ),
        runtime_smoke_execute_elapsed=fmt_num(
            _number(runtime_smoke.get("execute_elapsed_s"))
            if isinstance(runtime_smoke, dict)
            else None,
            3,
        ),
        runtime_smoke_error=h(
            runtime_smoke.get("error") if isinstance(runtime_smoke, dict) else None
        ),
        missing=h(
            json.dumps(
                {
                    "missing_real_abi_inputs": spec.get("missing_real_abi_inputs")
                    if isinstance(spec, dict)
                    else None,
                    "contract": contract,
                    "cache_key_recompile_audit": recompile_audit,
                    "runtime_execution_contract": runtime_contract,
                    "runtime_smoke": runtime_smoke,
                    "generated_source": {
                        "logical_buffer_abi_map_count": source.get(
                            "logical_buffer_abi_map_count"
                        )
                        if isinstance(source, dict)
                        else None,
                        "physical_abi_validation": source.get(
                            "physical_abi_validation"
                        )
                        if isinstance(source, dict)
                        else None,
                        "physical_abi_runtime_bridge": source.get(
                            "physical_abi_runtime_bridge"
                        )
                        if isinstance(source, dict)
                        else None,
                        "physical_abi_runtime_binding": source.get(
                            "physical_abi_runtime_binding"
                        )
                        if isinstance(source, dict)
                        else None,
                        "spilled_shared_scratch_count": source.get(
                            "spilled_shared_scratch_count"
                        )
                        if isinstance(source, dict)
                        else None,
                        "shared_scratch_abi_bytes": source.get(
                            "shared_scratch_abi_bytes"
                        )
                        if isinstance(source, dict)
                        else None,
                        "internal_scratch_abi_buffers": source.get(
                            "internal_scratch_abi_buffers"
                        )
                        if isinstance(source, dict)
                        else None,
                    },
                    "native_compile_error": receipt.get("native_compile_error"),
                },
                indent=2,
                sort_keys=True,
            )
        ),
    )


def render_current_findings(
    rows: list[Row],
    dtypes: tuple[str, ...],
    tolerance: float,
    default_block_reason: str | None = None,
) -> str:
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
                default_block_reason=default_block_reason,
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
                    optimizer=h(optimizer_label(baseline or warm, dtype=dtype, optimizer=optimizer)),
                    ratio=fmt_ratio(ratio),
                    class_name=class_name,
                    decision=h(decision),
                    reason=h(reason),
                )
            )
    good_rows = []
    for dtype in dtypes:
        for optimizer in optimizer_order(rows, dtype):
            baseline = keyed.get((dtype, optimizer, "path_b"))
            warm = keyed.get((dtype, optimizer, "path_c_warm"))
            decision, class_name, _ = decision_for(
                warm=warm,
                baseline=baseline,
                tolerance=tolerance,
                default_block_reason=default_block_reason,
            )
            if decision in {"Path C default candidate", "Path C speed candidate"}:
                good_rows.append((dtype, optimizer, decision))
    if good_rows:
        if default_block_reason:
            headline = (
                f"{len(good_rows)} row(s) pass the Path C speed rule, but no row "
                f"is a default promotion because {default_block_reason}"
            )
        else:
            headline = (
                f"{len(good_rows)} row(s) qualify for Path C under the "
                f"{tolerance:.0%} same-speed rule; see details below."
            )
    else:
        headline = (
            "No rendered dtype/optimizer row qualifies for Path C as the default "
            f"under the {tolerance:.0%} same-speed rule."
        )
    return """
      <section class="panel narrative">
        <div class="section-head">
          <h2>Current Default Decision</h2>
          <p>Generated directly from the matrix ratios below.</p>
        </div>
        <div class="callout">
          <strong>Current result:</strong> {headline}
          Default promotion requires warm Path C to reach at least {threshold:.0%}
          of Path B tok/s on the same 1B training workload.
        </div>
        <ul class="finding-list">{items}</ul>
      </section>
    """.format(
        headline=h(headline),
        threshold=1.0 - tolerance,
        items="\n".join(lines),
    )


def kernel_counts_text(row: Row | None) -> str:
    if not row:
        return "-"
    counts = row.selected_schedule.get("kernel_counts")
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ", ".join(f"{h(key)} x{h(value)}" for key, value in sorted(counts.items()))


def render_profile_table(rows: list[Row], dtype: str) -> str:
    keyed = rows_by_key(rows)
    body: list[str] = []
    for optimizer in optimizer_order(rows, dtype):
        baseline = keyed.get((dtype, optimizer, "path_b"))
        warm = keyed.get((dtype, optimizer, "path_c_warm"))
        steady_delta = None
        if warm and baseline and warm.step_sec is not None and baseline.step_sec is not None:
            steady_delta = warm.step_sec - baseline.step_sec
        body.append(
            """
            <tr>
              <th>{optimizer}</th>
              <td>{b_first}</td>
              <td>{c_first}</td>
              <td>{b_step}</td>
              <td>{c_step}</td>
              <td>{b_median}</td>
              <td>{c_median}</td>
              <td class="{step_class}">{step_delta}</td>
              <td>{b_kernels}</td>
              <td>{c_kernels}</td>
              <td>{active_delta}</td>
              <td>{cache_delta}</td>
            </tr>
            """.format(
                optimizer=h(optimizer_label(baseline or warm, dtype=dtype, optimizer=optimizer)),
                b_first=fmt_num(baseline.first_step_sec if baseline else None, 3),
                c_first=fmt_num(warm.first_step_sec if warm else None, 3),
                b_step=fmt_num(baseline.step_sec if baseline else None, 3),
                c_step=fmt_num(warm.step_sec if warm else None, 3),
                b_median=fmt_num(baseline.median_step_sec if baseline else None, 3),
                c_median=fmt_num(warm.median_step_sec if warm else None, 3),
                step_delta=fmt_signed(steady_delta, " s"),
                step_class="bad-number" if steady_delta is not None and steady_delta > 0 else "good-number",
                b_kernels=kernel_counts_text(baseline),
                c_kernels=kernel_counts_text(warm),
                active_delta=fmt_signed(memory_delta(warm, baseline, "active_memory_gb"), " GiB"),
                cache_delta=fmt_signed(memory_delta(warm, baseline, "cache_memory_gb"), " GiB"),
            )
        )
    return """
      <section class="panel">
        <div class="section-head">
          <h2>{dtype} dispatch profile</h2>
          <p>Receipt-level profiling: steady step time, runtime kernel dispatch counts, and active/cache memory deltas. GPU-counter traces are separate artifacts; this table is the parsed training receipt evidence.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Optimizer</th>
                <th>Path B first step s</th>
                <th>Warm C first step s</th>
                <th>Path B steady s</th>
                <th>Warm C steady s</th>
                <th>Path B median s</th>
                <th>Warm C median s</th>
                <th>Step delta</th>
                <th>Path B kernels</th>
                <th>Warm C kernels</th>
                <th>Active delta</th>
                <th>Cache delta</th>
              </tr>
            </thead>
            <tbody>{body}</tbody>
          </table>
        </div>
      </section>
    """.format(dtype=h(dtype.upper()), body="\n".join(body))


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
              <td>{optimizer}</td>
              <td>{path}</td>
              <td><span class="pill {status_class}">{status}</span></td>
              <td>{steps_completed}</td>
              <td>{tok}</td>
              <td>{first_step}</td>
              <td>{steady_step}</td>
              <td>{peak}</td>
              <td>{active}</td>
              <td>{cache_mem}</td>
              <td>{cache}</td>
              <td>{route}</td>
              <td>{reason}</td>
            </tr>
            """.format(
                case_id=h(row.case_id),
                optimizer=h(optimizer_label(row, dtype=dtype, optimizer=row.optimizer)),
                path=h(path_label(row.path)),
                status_class=css_class_for_status(row.status),
                status=h(row.status),
                steps_completed=h(row.steps_completed),
                tok=fmt_num(row.tok_sec),
                first_step=fmt_num(row.first_step_sec, 2),
                steady_step=fmt_num(row.step_sec, 2),
                peak=fmt_num(row.peak_memory_gb, 2),
                active=fmt_num(row.active_memory_gb, 2),
                cache_mem=fmt_num(row.cache_memory_gb, 2),
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
                <th>Runtime optimizer</th>
                <th>Path</th>
                <th>Status</th>
                <th>Steps</th>
                <th>tok/s</th>
                <th>first step s</th>
                <th>steady step s</th>
                <th>peak GB</th>
                <th>active GiB</th>
                <th>cache GiB</th>
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
    default_block_reason = path_c_default_block_reason(payload)
    sections: list[str] = [
        render_methodology(payload, tolerance),
        render_route_legend(payload),
        render_compile_receipt(payload),
        render_current_findings(rows, dtypes, tolerance, default_block_reason),
    ]
    sections.extend(
        render_comparison_table(rows, dtype, tolerance, default_block_reason)
        for dtype in dtypes
        if any(row.dtype == dtype for row in rows)
    )
    sections.extend(
        render_profile_table(rows, dtype)
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
        cards=render_summary_cards(rows, dtypes, tolerance, default_block_reason),
        sections="\n".join(sections),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("matrix JSON root must be an object")
    if args.compile_receipt is not None:
        compile_receipt = json.loads(args.compile_receipt.read_text(encoding="utf-8"))
        if not isinstance(compile_receipt, dict):
            raise SystemExit("compile receipt JSON root must be an object")
        payload["path_c_fusion_compile_receipt"] = compile_receipt
    dtypes = parse_dtypes(args.dtypes)
    rows = parse_rows(payload)
    html_text = render_html(payload, rows, dtypes, args.same_speed_tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
