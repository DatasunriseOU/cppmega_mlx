from __future__ import annotations

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def test_route_by_fit_drops_docs_longer_than_largest_fixed_bucket(tmp_path):
    sys.path.insert(0, "scripts")
    from streaming_reindex_commits import bucket_for, route_by_fit

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

    routed = route_by_fit(source, [8, 16], tmp_path / "routed")

    assert sorted(routed) == [8, 16]
    assert pq.read_table(routed[8]).column("name").to_pylist() == ["small"]
    assert pq.read_table(routed[16]).column("name").to_pylist() == ["medium"]

    drop_report = json.loads((tmp_path / "routed" / "dropped_overlong.json").read_text())
    assert drop_report["max_length"] == 16
    assert drop_report["dropped_overlong_rows"] == 1
    assert drop_report["dropped_overlong_tokens"] == 17
