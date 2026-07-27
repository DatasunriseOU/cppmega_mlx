from __future__ import annotations

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_route_by_fit_fails_closed_before_writing_any_fixed_bucket(tmp_path):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import RepoFailure, bucket_for, route_by_fit

    source = tmp_path / "tok.parquet"
    table = pa.Table.from_pylist(
        [
            {"token_ids": [1] * 4, "name": "small"},
            {"token_ids": [2] * 9, "name": "medium"},
            {"token_ids": [3] * 17, "name": "overlong"},
        ]
    )
    pq.write_table(table, source)

    assert bucket_for(4, [8, 16]) == 8
    assert bucket_for(9, [8, 16]) == 16
    assert bucket_for(17, [8, 16]) is None

    route_dir = tmp_path / "routed"
    with pytest.raises(
        RepoFailure,
        match=r"overlong_rows=1 overlong_tokens=17.*fixed_shape_max=16",
    ) as caught:
        route_by_fit(source, [8, 16], route_dir, repo="owner/repo")

    assert caught.value.repo == "owner/repo"
    assert not (route_dir / "route_8.parquet").exists()
    assert not (route_dir / "route_16.parquet").exists()
    assert not (route_dir / "dropped_overlong.json").exists()


def test_token_list_lengths_stay_in_arrow_and_treat_null_as_zero():
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import _token_list_lengths

    values = pa.array([[1, 2, 3], None, [], [4]], type=pa.list_(pa.int64()))

    assert _token_list_lengths(values) == [3, 0, 0, 1]


def test_route_by_fit_releases_arrow_pool_after_streaming(tmp_path, monkeypatch):
    sys.path.insert(0, "scripts")
    import streaming_reindex_commits

    source = tmp_path / "tok.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"token_ids": [1] * 4, "name": "small"},
                {"token_ids": [2] * 9, "name": "medium"},
            ]
        ),
        source,
    )
    calls = []
    monkeypatch.setattr(
        streaming_reindex_commits,
        "release_arrow_unused",
        lambda: calls.append("released"),
    )

    routed = streaming_reindex_commits.route_by_fit(source, [8, 16], tmp_path / "routed")

    assert sorted(routed) == [8, 16]
    assert calls == ["released"]


def test_route_by_fit_preserves_all_rows_when_every_document_fits(tmp_path):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import route_by_fit

    source = tmp_path / "tok.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"token_ids": [1] * 4, "name": "small"},
                {"token_ids": [2] * 9, "name": "medium"},
            ]
        ),
        source,
    )

    route_dir = tmp_path / "routed"
    routed = route_by_fit(source, [8, 16], route_dir)

    assert sorted(routed) == [8, 16]
    assert pq.read_table(routed[8]).column("name").to_pylist() == ["small"]
    assert pq.read_table(routed[16]).column("name").to_pylist() == ["medium"]
    assert not (route_dir / "dropped_overlong.json").exists()


def test_recompress_zstd_max_streams_row_groups_without_full_read(
    tmp_path,
    monkeypatch,
):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import recompress_zstd_max

    source = tmp_path / "packed.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"input_ids": [1, 2, 3], "valid_token_count": 3},
                {"input_ids": [4, 5], "valid_token_count": 2},
                {"input_ids": [6], "valid_token_count": 1},
            ]
        ),
        source,
        row_group_size=1,
    )

    def fail_full_file_read(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("recompress_zstd_max must not read whole parquet")

    monkeypatch.setattr(pq, "read_table", fail_full_file_read)

    recompress_zstd_max(source)

    pf = pq.ParquetFile(source)
    assert pf.num_row_groups == 3
    assert pf.read().column("valid_token_count").to_pylist() == [3, 2, 1]


def test_summarize_done_manifest_aggregates_dropped_overlong_durably(tmp_path):
    """The dropped-overlong audit must survive in the manifest and reach the
    run-level summary, even though the per-range receipt dir is rmtree'd."""
    sys.path.insert(0, "scripts")
    from streaming_reindex import Manifest
    from streaming_reindex_commits import summarize_done_manifest

    lengths = (1024, 2048)

    # Two real done ranges carrying dropped_overlong, plus one legacy entry
    # written before the field existed (must contribute nothing, not crash).
    manifest = Manifest.load(tmp_path / "_done.json")
    manifest.mark_done(
        "repoA::r0",
        {
            "source": "commits",
            "lengths": {
                "1024": {"rows": 3, "valid_tokens": 100, "pad_tokens": 5,
                         "capacity_tokens": 3072},
                "2048": {"rows": 1, "valid_tokens": 2000, "pad_tokens": 48,
                         "capacity_tokens": 2048},
            },
            "dropped_overlong": {"rows": 2, "tokens": 40000},
        },
    )
    manifest.mark_done(
        "repoA::r500",
        {
            "source": "commits",
            "lengths": {
                "1024": {"rows": 2, "valid_tokens": 50, "pad_tokens": 3,
                         "capacity_tokens": 2048},
            },
            "dropped_overlong": {"rows": 1, "tokens": 17000},
        },
    )
    manifest.mark_done(
        "repoLegacy::r0",
        {
            "source": "commits",
            "lengths": {
                "2048": {"rows": 1, "valid_tokens": 1500, "pad_tokens": 548,
                         "capacity_tokens": 2048},
            },
            # no "dropped_overlong" key (legacy)
        },
    )

    # Reload from disk to prove the audit is DURABLE, not just in-memory.
    reloaded = Manifest.load(tmp_path / "_done.json")
    totals, dropped_total = summarize_done_manifest(reloaded.done, lengths)

    assert dropped_total == {"rows": 3, "tokens": 57000}
    assert totals["1024"]["rows"] == 5
    assert totals["1024"]["valid_tokens"] == 150
    assert totals["2048"]["rows"] == 2
    assert totals["2048"]["valid_tokens"] == 3500
