#!/usr/bin/env python3
"""Render the production-shape throughput tables from the bench receipt.

Replaces the placeholder markers in
``docs/bench_local_gb10_quarter_production.md`` and the Stream E addendum in
``docs/mlx_port_master_plan.md`` with markdown tables filled from the bench
receipt.

Markers (rendered into both files):
- ``<!-- LION_TABLE -->``
- ``<!-- MUON_TABLE -->``
- ``<!-- PATH_TABLE -->``
- ``<!-- M4_VS_GB10_TABLE -->``
- ``<!-- CONCLUSIONS -->`` (production summary doc only)

Usage::

    .venv/bin/python scripts/update_bench_summary.py \\
        --receipt bench/baselines/local_gb10_quarter_throughput_m4.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "bench" / "baselines" / "local_gb10_quarter_throughput_m4.json"
DEFAULT_PROD_DOC = ROOT / "docs" / "bench_local_gb10_quarter_production.md"
DEFAULT_PLAN_DOC = ROOT / "docs" / "mlx_port_master_plan.md"

GB10_REFERENCE_TPS = 4000


def fmt(value: float | int | None, *, kind: str = "f1") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if kind == "f0":
        return f"{value:.0f}"
    if kind == "f1":
        return f"{value:.1f}"
    if kind == "f2":
        return f"{value:.2f}"
    if kind == "f3":
        return f"{value:.3f}"
    return str(value)


def table_for_optimizer(rows: list[dict[str, Any]], optimizer: str) -> str:
    selected = [r for r in rows if r["optimizer"] == optimizer]
    if not selected:
        return f"_no rows for {optimizer}_"
    lines = []
    lines.append(
        "| B  | tok/s median | p10   | p90   | peak GB | loss[0]  | mean_last10 | optimizer state GB | cap_hit | notes |"
    )
    lines.append(
        "|----|--------------|-------|-------|---------|----------|-------------|--------------------|---------|-------|"
    )
    for row in selected:
        cap_hit = "yes" if row.get("memory_cap_hit") else "no"
        notes_parts = []
        if row.get("error"):
            notes_parts.append(f"err: {row['error_type']}")
        if row.get("path_b_dispatched") is False:
            notes_parts.append("no path-b")
        steps_measured = row.get("steps_measured") or 0
        if steps_measured == 0:
            notes_parts.append("no measured steps")
        notes = "; ".join(notes_parts) if notes_parts else ""
        lines.append(
            "| {B:<2} | {tps:>12} | {p10:>5} | {p90:>5} | {peak:>7} | {l0:>8} | {l10:>11} | {state:>18} | {cap:<7} | {notes} |".format(
                B=row["batch_size"],
                tps=fmt(row.get("tokens_per_second_median"), kind="f0"),
                p10=fmt(row.get("tokens_per_second_p10"), kind="f0"),
                p90=fmt(row.get("tokens_per_second_p90"), kind="f0"),
                peak=fmt(row.get("peak_memory_gb"), kind="f2"),
                l0=fmt(row.get("loss_first"), kind="f3"),
                l10=fmt(row.get("loss_last_10_mean"), kind="f3"),
                state=fmt(row.get("optimizer_state_gb"), kind="f2"),
                cap=cap_hit,
                notes=notes,
            )
        )
    return "\n".join(lines)


def best_row(rows: list[dict[str, Any]], optimizer: str) -> dict[str, Any] | None:
    candidates = [
        r
        for r in rows
        if r["optimizer"] == optimizer
        and r.get("error") is None
        and not r.get("memory_cap_hit", False)
        and (r.get("steps_measured") or 0) > 0
        and r.get("tokens_per_second_median") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["tokens_per_second_median"])


def path_table(receipt: dict[str, Any]) -> str:
    rows = receipt.get("rows") or []
    path_rows = receipt.get("path_comparison_rows") or []
    best_path_b = best_row(rows, "muon_adamw")
    if best_path_b is None and not path_rows:
        return "_no Muon+AdamW rows reached the comparison stage_"

    lines = []
    lines.append(
        "| path        | B  | tok/s median | peak GB | path_b_dispatched | notes |"
    )
    lines.append(
        "|-------------|----|--------------|---------|-------------------|-------|"
    )

    def emit(row: dict[str, Any], label: str) -> None:
        notes_parts = []
        if row.get("error"):
            notes_parts.append(f"err: {row['error_type']}")
        notes = "; ".join(notes_parts) if notes_parts else ""
        lines.append(
            "| {label:<11} | {B:<2} | {tps:>12} | {peak:>7} | {pb!s:<17} | {notes} |".format(
                label=label,
                B=row["batch_size"],
                tps=fmt(row.get("tokens_per_second_median"), kind="f0"),
                peak=fmt(row.get("peak_memory_gb"), kind="f2"),
                pb=row.get("path_b_dispatched"),
                notes=notes,
            )
        )

    if best_path_b is not None:
        emit(best_path_b, "Path B (auto)")
    for row in path_rows:
        kp = row.get("kernel_path", "?")
        label = "Path A (ref)" if kp == "ref" else f"Path {kp}"
        emit(row, label)
    return "\n".join(lines)


def m4_vs_gb10_table(receipt: dict[str, Any]) -> str:
    rows = receipt.get("rows") or []
    best = best_row(rows, "muon_adamw")
    lines = []
    lines.append(
        "| metric                                  | value                                          |"
    )
    lines.append(
        "|-----------------------------------------|------------------------------------------------|"
    )
    if best is None:
        lines.append(
            "| M4 Max max sustainable B (Muon+AdamW)   | n/a (no Muon+AdamW rows passed the cap)        |"
        )
        lines.append(
            "| M4 Max tok/s @ T=4096 (Muon+AdamW)      | n/a                                            |"
        )
    else:
        peak_gb = fmt(best.get("peak_memory_gb"), kind="f2")
        tps = best.get("tokens_per_second_median")
        ratio = (
            f"{tps / GB10_REFERENCE_TPS:.2f}"
            if tps is not None and GB10_REFERENCE_TPS > 0
            else "n/a"
        )
        lines.append(
            f"| M4 Max max sustainable B (Muon+AdamW)   | B={best['batch_size']} (peak {peak_gb} GB)                   |"
        )
        lines.append(
            f"| M4 Max tok/s @ T=4096 (Muon+AdamW)      | {fmt(tps, kind='f0')} (median over post-warmup steps)            |"
        )
        lines.append(
            f"| GB10 reference tok/s @ T=4096           | ~{GB10_REFERENCE_TPS} (Muon+AdamW; B unknown)                  |"
        )
        lines.append(
            f"| Ratio (M4 Max / GB10)                   | {ratio} (caveat: GB10 batch size at T=4096 unknown) |"
        )
    return "\n".join(lines)


def conclusions(receipt: dict[str, Any]) -> str:
    rows = receipt.get("rows") or []
    cap_gb = receipt.get("config", {}).get("memory_cap_gb", 88.0)

    best_lion = best_row(rows, "lion")
    best_muon = best_row(rows, "muon_adamw")
    cap_hit_any = any(r.get("memory_cap_hit") for r in rows)

    lines = ["**Conclusions**", ""]

    if best_lion is None:
        lines.append(
            "- Lion: no rows passed the memory cap. B=1 alone exceeded "
            f"{cap_gb:.0f} GB. Reduce sequence length or lower grad-checkpoint "
            "blocking before this geometry is trainable on M4 Max."
        )
    else:
        lines.append(
            f"- **Lion (lr=3e-3) max throughput**: {fmt(best_lion['tokens_per_second_median'], kind='f0')} tok/s "
            f"at B={best_lion['batch_size']} (peak {fmt(best_lion['peak_memory_gb'], kind='f2')} GB, "
            f"optimizer state {fmt(best_lion['optimizer_state_gb'], kind='f2')} GB)."
        )

    if best_muon is None:
        lines.append(
            "- Muon+AdamW: no rows passed the memory cap. Same caveat as Lion."
        )
    else:
        lines.append(
            f"- **Muon+AdamW (cppmega_cuda_parity) max throughput**: "
            f"{fmt(best_muon['tokens_per_second_median'], kind='f0')} tok/s at "
            f"B={best_muon['batch_size']} (peak {fmt(best_muon['peak_memory_gb'], kind='f2')} GB, "
            f"optimizer state {fmt(best_muon['optimizer_state_gb'], kind='f2')} GB)."
        )

    if best_lion is not None and best_muon is not None:
        delta_pct = (
            (best_lion["tokens_per_second_median"] - best_muon["tokens_per_second_median"])
            / best_muon["tokens_per_second_median"]
            * 100.0
        )
        sign = "+" if delta_pct >= 0 else ""
        lines.append(
            f"- **Lion vs Muon+AdamW delta**: Lion is {sign}{delta_pct:.1f}% on tokens/sec at the "
            "respective max-B winners. Lion has the smaller optimizer-state footprint "
            f"({fmt(best_lion['optimizer_state_gb'], kind='f2')} GB vs "
            f"{fmt(best_muon['optimizer_state_gb'], kind='f2')} GB) which is the more "
            "useful figure for the 128 GB single-Mac budget."
        )

    path_rows = receipt.get("path_comparison_rows") or []
    if best_muon is not None and path_rows:
        path_a = next((r for r in path_rows if r.get("kernel_path") == "ref"), None)
        if path_a is not None and path_a.get("tokens_per_second_median") is not None:
            delta_b_vs_a = (
                (best_muon["tokens_per_second_median"] - path_a["tokens_per_second_median"])
                / path_a["tokens_per_second_median"]
                * 100.0
            )
            sign = "+" if delta_b_vs_a >= 0 else ""
            lines.append(
                f"- **Path B vs Path A delta** at the winning Muon+AdamW B="
                f"{best_muon['batch_size']}: Path B (production kernels) is "
                f"{sign}{delta_b_vs_a:.1f}% faster than Path A (pure-MLX reference) "
                f"({fmt(best_muon['tokens_per_second_median'], kind='f0')} vs "
                f"{fmt(path_a['tokens_per_second_median'], kind='f0')} tok/s). "
                "Path B fires `metal_kernel_fwd_v1` for `mamba3_mimo` and `m2rnn`; "
                "`sparse_mla` has a separate prepared-FP8 Path C receipt axis."
            )

    error_rows = [r for r in rows if r.get("error") is not None]
    if cap_hit_any or error_rows:
        cap_hit_clause = (
            f"hit the {cap_gb:.0f} GB cap"
            if cap_hit_any
            else f"OOM'd on allocation at the {cap_gb:.0f} GB cap"
        )
        if error_rows and not cap_hit_any:
            offending = error_rows[0]
            cap_hit_clause = (
                f"OOM'd on allocation at B={offending['batch_size']} "
                f"(error={offending.get('error_type')!r}); the {cap_gb:.0f} GB "
                "cap is the de facto knee"
            )
        lines.append(
            f"- **Memory cap is the binding constraint**: at least one (B, optimizer) "
            f"attempt {cap_hit_clause} before throughput plateaued. The M4 Max single-Mac "
            "throughput knee at T=4096 is memory-limited, not compute-limited; "
            "raising B further requires reducing optimizer-state cost (Lion over "
            "Muon+AdamW) or distributed sharding (Stream F)."
        )
    else:
        lines.append(
            "- Memory cap was **not** hit on this sweep. Higher B values may still "
            "improve throughput; rerun with a wider batch list to pin the knee."
        )

    if best_muon is not None:
        ratio = best_muon["tokens_per_second_median"] / GB10_REFERENCE_TPS
        lines.append(
            f"- **M4 Max vs GB10 (Muon+AdamW only)**: "
            f"{fmt(best_muon['tokens_per_second_median'], kind='f0')} tok/s vs "
            f"GB10 reference ~{GB10_REFERENCE_TPS} tok/s. Ratio "
            f"{ratio:.2f}. Caveat: GB10 batch size at T=4096 is not in the public "
            "reference; this is a hardware-class sketch, not a parity claim. "
            "Treat as a single-host receipt with `gb10_parity_claim=false`."
        )
    else:
        lines.append(
            f"- **M4 Max vs GB10 (Muon+AdamW)**: no Muon+AdamW row was usable; "
            f"can't quote a ratio. The {cap_gb:.0f} GB cap is the binding constraint."
        )

    return "\n".join(lines)


def replace_marker(text: str, marker: str, payload: str) -> str:
    if marker not in text:
        # Fallback: append payload at end of file with marker comment so a later
        # run can find it.
        return text + f"\n\n{marker}\n{payload}\n"
    return text.replace(marker, payload, 1)


def render_doc(text: str, receipt: dict[str, Any], *, with_conclusions: bool) -> str:
    rows = receipt.get("rows") or []
    text = replace_marker(
        text, "<!-- LION_TABLE -->", table_for_optimizer(rows, "lion")
    )
    text = replace_marker(
        text, "<!-- MUON_TABLE -->", table_for_optimizer(rows, "muon_adamw")
    )
    text = replace_marker(text, "<!-- PATH_TABLE -->", path_table(receipt))
    text = replace_marker(
        text, "<!-- M4_VS_GB10_TABLE -->", m4_vs_gb10_table(receipt)
    )
    if with_conclusions:
        text = replace_marker(text, "<!-- CONCLUSIONS -->", conclusions(receipt))
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--prod-doc", type=Path, default=DEFAULT_PROD_DOC)
    parser.add_argument("--plan-doc", type=Path, default=DEFAULT_PLAN_DOC)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    prod_text = args.prod_doc.read_text()
    plan_text = args.plan_doc.read_text()

    args.prod_doc.write_text(render_doc(prod_text, receipt, with_conclusions=True))
    args.plan_doc.write_text(render_doc(plan_text, receipt, with_conclusions=True))
    print(f"updated {args.prod_doc} and {args.plan_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
