"""Aggregate Playwright artefact bundles into one human-readable
``e2e_matrix_report.md`` for E2E Coverage Matrix epic (E-6).

Inputs: a directory tree like::

  artifacts/
    preset-matrix-shard-1/
      test-results/.last-run.json
      screenshots/02_preset_matrix/{cell}.png
      logs/{backend,frontend}.{stdout,stderr}.log
    preset-matrix-shard-2/ ...
    specialised-results/ ...
    mini-train-results/ ...

Output: a Markdown summary with shard pass-rate, flaky retries, slowest
scenarios, screenshot counts, and the matrix-size legend.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MATRIX_LEGEND = """\
## Matrix layout

| Spec file | Cells | What it covers |
|---|---|---|
| `02_preset_matrix.spec.ts` | **912 cells** | 57 PRESETS × 4 tokenizers × 4 parquet schemas — smoke pipeline through GUI |
| `03_train_matrix.spec.ts` | **192 cells** | 12 family-rep × 4 tokenizers × 4 parquet — real mlx 2-step gradient |
| `04_tokenizer_playground.spec.ts` | **2 tests** | 3-panel side-by-side compare + chip render |
| `05_data_inspector.spec.ts` | **18 tests** | parquet load (16 = 4 tok × 4 schema) + pagination + channel toggle |
| `06_sharding_proposals.spec.ts` | **3 tests** | proposals fire, accept proposal, fp8 toggle |
| `07_gotchas.spec.ts` | **2 tests** | whole_model compile, empty state |
| `01_canvas_smoke.spec.ts` | **6 tests** | 5 presets + empty-canvas error |

Total scenarios across non-train suites: **25 tests** in specialised set, **6 tests** in canvas smoke.
"""


def _parse_last_run(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _shard_summary(artifacts: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_dir in sorted(artifacts.iterdir()):
        if not shard_dir.is_dir():
            continue
        last_run = shard_dir / "test-results" / ".last-run.json"
        payload = _parse_last_run(last_run)
        if payload is None:
            continue
        png_count = sum(
            1 for p in (shard_dir / "screenshots").rglob("*.png")
            if p.is_file()
        )
        rows.append({
            "name": shard_dir.name,
            "status": payload.get("status", "?"),
            "failed": len(payload.get("failedTests") or []),
            "flaky": len(payload.get("flakyTests") or []),
            "screenshots": png_count,
        })
    return rows


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No artefacts collected._\n"
    header = "| Shard | Status | Failed | Flaky retries | Screenshots |\n"
    sep = "|---|---|---|---|---|\n"
    body = "".join(
        f"| `{r['name']}` | {r['status']} | {r['failed']} | "
        f"{r['flaky']} | {r['screenshots']} |\n"
        for r in rows
    )
    return header + sep + body


def _screenshot_buckets(artifacts: Path) -> Counter:
    counts: Counter[str] = Counter()
    for png in artifacts.rglob("*.png"):
        rel = png.relative_to(artifacts)
        # Bucket is the first dir under the shard's screenshots/ tree.
        parts = rel.parts
        if "screenshots" in parts:
            i = parts.index("screenshots") + 1
            if i < len(parts):
                counts[parts[i]] += 1
    return counts


def build_report(artifacts: Path) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[str] = [
        f"# E2E Coverage Matrix Report",
        "",
        f"_Generated: {now}_",
        "",
        MATRIX_LEGEND,
    ]
    if not artifacts.is_dir() or not any(artifacts.iterdir()):
        out += [
            "## Run summary",
            "",
            "_No artefacts directory found — this report is a stub._",
            "",
        ]
        return "\n".join(out)

    rows = _shard_summary(artifacts)
    total_failed = sum(r["failed"] for r in rows)
    total_flaky = sum(r["flaky"] for r in rows)
    total_pngs = sum(r["screenshots"] for r in rows)

    out += [
        "## Run summary",
        "",
        f"- Playwright run files parsed: **{len(rows)}**",
        f"- Total failed tests (post-retry): **{total_failed}**",
        f"- Total flaky retries (across shards): **{total_flaky}**",
        f"- Screenshots captured: **{total_pngs}**",
        "",
        _format_table(rows),
        "",
        "## Screenshot buckets",
        "",
    ]
    buckets = _screenshot_buckets(artifacts)
    if buckets:
        for name, n in sorted(buckets.items()):
            out.append(f"- `{name}` — {n} files")
    else:
        out.append("_No screenshots._")
    out += [
        "",
        "## Notes",
        "",
        "- Cells expected to fail (xfail) are inverted in the spec files "
        "themselves (e.g. `kimi_linear` and `qwen3_next` in the train "
        "matrix — `kda`/`gdn` bricks have no MLX `vjp` implementation).",
        "- Flaky retries normally clear on the second attempt; persistent "
        "flakies should be investigated through the trace.zip in the "
        "artefact bundle.",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts",
                        help="Directory of downloaded Playwright artefacts.")
    parser.add_argument("--output", default="e2e_matrix_report.md",
                        help="Markdown file to write.")
    args = parser.parse_args(argv)

    report = build_report(Path(args.artifacts))
    Path(args.output).write_text(report)
    print(f"wrote {args.output} ({len(report)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
