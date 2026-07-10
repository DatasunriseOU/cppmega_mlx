from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.nanochat_data.pack_enriched_rows import (
    PACKED_ROW_MACRO_ROUTES_METADATA_KEY,
    PACKED_ROW_MACRO_ROUTES_VERSION,
)
from scripts.report_training_steps import build_report, parse_batch_schedule, _print_summary


def _write_counter_parquet(
    path: Path,
    rows: list[dict[str, int]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "valid_token_count": pa.array([row["valid"] for row in rows], pa.int64()),
            "trained_token_count": pa.array([row["trained"] for row in rows], pa.int64()),
            "num_docs": pa.array([row["docs"] for row in rows], pa.int64()),
        }
    )
    if metadata:
        table = table.replace_schema_metadata(
            {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
        )
    pq.write_table(table, path)


def test_report_keeps_code_commits_main_and_pr_stream_distinct(tmp_path: Path) -> None:
    code = tmp_path / "code"
    commits = tmp_path / "commits"
    pr = tmp_path / "pr"
    _write_counter_parquet(
        code / "1024" / "code.parquet",
        [{"valid": 1000, "trained": 900, "docs": 2}],
    )
    _write_counter_parquet(
        commits / "1024" / "commits.parquet",
        [{"valid": 2200, "trained": 2100, "docs": 3}],
    )
    _write_counter_parquet(
        pr / "1024" / "pr.parquet",
        [{"valid": 700, "trained": 600, "docs": 1}],
    )

    rows = build_report(
        code_root=code,
        commit_root=commits,
        pr_root=pr,
        batch_by_length={1024: 3},
        max_workers=1,
        allow_concurrent_skips=False,
    )
    by_kind = {row.kind: row for row in rows}

    assert by_kind["code_only"].trained_tokens == 900
    assert by_kind["code_only"].steps_by_trained_tokens == 0
    assert by_kind["commits_with_pr_docstring"].trained_tokens == 2100
    assert by_kind["commits_with_pr_docstring"].steps_by_trained_tokens == 0

    main = by_kind["main_code_plus_commits"]
    assert main.trained_tokens == 3000
    assert main.valid_tokens == 3200
    assert main.docs == 5
    assert main.tokens_per_step == 3072
    assert main.steps_by_trained_tokens == 0
    assert main.steps_by_valid_tokens == 1

    assert by_kind["standalone_pr_side_stream"].trained_tokens == 600
    assert by_kind["main_plus_standalone_pr"].trained_tokens == 3600
    assert by_kind["main_plus_standalone_pr"].steps_by_trained_tokens == 1


def test_summary_reports_every_length_once(tmp_path: Path, capsys) -> None:
    code = tmp_path / "code"
    commits = tmp_path / "commits"
    pr = tmp_path / "pr"
    _write_counter_parquet(
        code / "1024" / "code.parquet",
        [{"valid": 4096, "trained": 4096, "docs": 1}],
    )
    _write_counter_parquet(
        commits / "2048" / "commits.parquet",
        [{"valid": 8192, "trained": 8192, "docs": 1}],
    )
    _write_counter_parquet(
        pr / "2048" / "pr.parquet",
        [{"valid": 4096, "trained": 4096, "docs": 1}],
    )

    rows = build_report(
        code_root=code,
        commit_root=commits,
        pr_root=pr,
        batch_by_length={1024: 2, 2048: 2},
        max_workers=1,
        allow_concurrent_skips=False,
    )

    _print_summary(rows)
    out = capsys.readouterr().out

    assert "bucket_len" in out
    assert "tokens_per_step" in out
    assert "main_trained_tokens" in out
    assert "main_plus_standalone_pr_steps" in out
    assert out.count("|       1024 |") == 1
    assert out.count("|       2048 |") == 1
    assert "steps = floor(trained_token_count / tokens_per_step)" in out
    assert "main = code_only + commits_with_pr_docstring" in out

    table_lines = [line for line in out.splitlines() if line.startswith(("+", "|"))]
    assert len({len(line) for line in table_lines}) == 1


def test_summary_handles_no_rows(capsys) -> None:
    _print_summary([])
    out = capsys.readouterr().out

    assert "bucket_len" in out
    assert "steps = floor(trained_token_count / tokens_per_step)" in out


def test_report_can_filter_to_current_macro_route_shards(tmp_path: Path) -> None:
    code = tmp_path / "code"
    commits = tmp_path / "commits"
    pr = tmp_path / "pr"
    required = {
        PACKED_ROW_MACRO_ROUTES_METADATA_KEY: PACKED_ROW_MACRO_ROUTES_VERSION,
    }
    _write_counter_parquet(
        code / "1024" / "old.parquet",
        [{"valid": 1000, "trained": 900, "docs": 2}],
    )
    _write_counter_parquet(
        code / "1024" / "current.parquet",
        [{"valid": 2000, "trained": 1800, "docs": 3}],
        metadata=required,
    )
    _write_counter_parquet(
        commits / "1024" / "wrong.parquet",
        [{"valid": 4000, "trained": 3600, "docs": 4}],
        metadata={PACKED_ROW_MACRO_ROUTES_METADATA_KEY: "old"},
    )
    _write_counter_parquet(
        commits / "1024" / "current.parquet",
        [{"valid": 5000, "trained": 4500, "docs": 5}],
        metadata=required,
    )

    rows = build_report(
        code_root=code,
        commit_root=commits,
        pr_root=pr,
        batch_by_length={1024: 1},
        max_workers=1,
        allow_concurrent_skips=False,
        required_metadata=required,
    )
    by_kind = {row.kind: row for row in rows}

    assert by_kind["code_only"].trained_tokens == 1800
    assert by_kind["code_only"].skipped_files == 1
    assert by_kind["commits_with_pr_docstring"].trained_tokens == 4500
    assert by_kind["commits_with_pr_docstring"].skipped_files == 1
    assert by_kind["main_code_plus_commits"].trained_tokens == 6300


def test_parse_batch_schedule_rejects_bad_items() -> None:
    assert parse_batch_schedule("1024=192,2048=96") == {1024: 192, 2048: 96}

    try:
        parse_batch_schedule("1024")
    except ValueError as exc:
        assert "expected LENGTH=BATCH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid schedule to raise")
