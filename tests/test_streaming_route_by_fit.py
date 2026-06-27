from __future__ import annotations

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def test_route_by_fit_drops_docs_longer_than_largest_fixed_bucket(tmp_path):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import bucket_for, route_by_fit, read_dropped_overlong

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
    routed = route_by_fit(source, [8, 16], route_dir)

    assert sorted(routed) == [8, 16]
    assert pq.read_table(routed[8]).column("name").to_pylist() == ["small"]
    assert pq.read_table(routed[16]).column("name").to_pylist() == ["medium"]

    drop_report = json.loads((route_dir / "dropped_overlong.json").read_text())
    assert drop_report["max_length"] == 16
    assert drop_report["dropped_overlong_rows"] == 1
    assert drop_report["dropped_overlong_tokens"] == 17

    # read_dropped_overlong lifts the receipt counts out so process_range can
    # record them durably before the per-range temp dir is rmtree'd.
    assert read_dropped_overlong(route_dir) == {"rows": 1, "tokens": 17}


def test_read_dropped_overlong_zero_when_all_fit(tmp_path):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import route_by_fit, read_dropped_overlong

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
    # No over-long docs: route_by_fit writes no receipt, read returns zeros
    # (documented contract, not a swallowed error).
    assert not (route_dir / "dropped_overlong.json").exists()
    assert read_dropped_overlong(route_dir) == {"rows": 0, "tokens": 0}


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
