from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.source_identity import source_identity
from cppmega_mlx.training.megatron_objectives import MaterializedMegatronDocument
from scripts import materialize_megatron_objectives as materializer


def _manifest_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_alternating_source_rows_preserve_pool_and_global_replay_state() -> None:
    rows = materializer._AlternatingSourceRows(
        iter(
            [
                ({"value": "primary-0"}, {"epoch": 0, "row_index": 0}),
                ({"value": "primary-1"}, {"epoch": 0, "row_index": 1}),
            ]
        ),
        iter(
            [
                ({"value": "seed-0"}, {"epoch": 2, "row_index": 0}),
                ({"value": "seed-1"}, {"epoch": 2, "row_index": 1}),
            ]
        ),
    )

    yielded = [next(rows) for _ in range(4)]

    assert [row["value"] for row, _cursor in yielded] == [
        "primary-0",
        "seed-0",
        "primary-1",
        "seed-1",
    ]
    assert [cursor["pool_index"] for _row, cursor in yielded] == [0, 1, 0, 1]
    assert yielded[-1][1]["primary_rows_yielded"] == 2
    assert yielded[-1][1]["objective_seed_rows_yielded"] == 2
    assert yielded[-1][1]["next_pool_index"] == 0
    assert rows.last_pool_cursors == [
        {"epoch": 0, "row_index": 1, "source_index": 1},
        {"epoch": 2, "row_index": 1, "source_index": 1},
    ]


def test_two_pool_snapshot_binds_manifest_receipt_bytes_and_pool_cursors(
    tmp_path: Path,
) -> None:
    primary_root = tmp_path / "ci"
    primary = primary_root / "1024" / "ci.parquet"
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1, 2]}), primary)
    seed_root = tmp_path / "seed"
    seed = seed_root / "commits" / "1024" / "seed.parquet"
    seed.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [3, 4]}), seed)
    receipt = {
        "schema": "cppmega_ci_content_store_case5_export_v2",
        "status": "complete",
    }
    receipt_raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    (primary_root / "export_receipt.json").write_bytes(receipt_raw)
    manifest = {
        "schema": materializer._SOURCE_POOL_MANIFEST_SCHEMA,
        "algorithm": materializer._TWO_POOL_SCHEDULE,
        "sequence_lengths": [1024],
        "ci_export": {
            "path": "export_receipt.json",
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "schema": receipt["schema"],
            "status": "complete",
            "source_completion": {
                "schema": receipt["schema"],
                "status": "complete",
            },
        },
        "primary_ci": {
            "files_by_sequence_length": {
                "1024": [_manifest_record(primary, root=primary_root)]
            }
        },
        "objective_seed": {
            "files": [_manifest_record(seed, root=seed_root)]
        },
        "producer": {"script": "fixture"},
    }
    manifest_path = tmp_path / "source_pools.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (
        snapshot,
        signatures,
        primary_shards,
        seed_shards,
        manifest_raw,
        bound_receipt_raw,
    ) = materializer._build_two_pool_source_snapshot(
        manifest_path,
        primary_root=primary_root,
        objective_seed_root=seed_root,
        sequence_length=1024,
        requested_source_rows=4,
        seed=17,
        source_batch_rows=2,
    )

    assert snapshot["schema"] == materializer._TWO_POOL_SOURCE_SNAPSHOT_SCHEMA
    assert snapshot["pool_order"] == ["primary_ci", "objective_seed"]
    assert primary_shards == [str(primary.resolve())]
    assert seed_shards == [str(seed.resolve())]
    assert hashlib.sha256(manifest_raw).hexdigest() == (
        snapshot["source_pool_manifest"]["sha256"]
    )
    assert bound_receipt_raw == receipt_raw

    rows = materializer._AlternatingSourceRows(
        iter(
            [
                ({"value": 1}, {"epoch": 0, "row_index": 0}),
                ({"value": 2}, {"epoch": 0, "row_index": 1}),
            ]
        ),
        iter(
            [
                ({"value": 3}, {"epoch": 0, "row_index": 0}),
                ({"value": 4}, {"epoch": 0, "row_index": 1}),
            ]
        ),
    )
    cursor: dict[str, int] = {}
    for source_index in range(4):
        _row, cursor = next(rows)
        cursor = {**cursor, "source_index": source_index}
    materializer._bind_two_pool_sampling_cursors(
        snapshot,
        rows=rows,
        global_cursor=cursor,
        consumed_samples=4,
    )
    assert snapshot["pools"]["primary_ci"]["sampling"]["requested_samples"] == 2
    assert snapshot["pools"]["objective_seed"]["sampling"]["requested_samples"] == 2

    (primary_root / "export_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed"):
        materializer._require_source_snapshot_unchanged(signatures)


def test_source_iterator_normalizes_packed_row_only_once() -> None:
    identity = source_identity({"repo": "org/repo", "filepath": "src/a.cc"})
    raw_row = {
        "input_ids": [10, 11, 12, 13],
        "valid_token_count": 4,
        "num_docs": 1,
        "doc_ids": [1, 1, 1, 1],
        "token_source_doc_ids": [101, 101, 101, 101],
        "token_source_identity_ids": [identity.source_identity_id] * 4,
        "source_identity_registry": [identity.as_dict()],
        "token_chunk_starts": [],
        "token_chunk_ends": [],
        "token_chunk_kinds": [],
        "token_chunk_dep_levels": [],
        "token_call_edges": [],
        "token_type_edges": [],
        "token_domain_edges": [],
        "token_build_edges": [],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
    }
    iterator = materializer._ObjectiveSourceIterator(
        [], seed=1, source_batch_rows=1
    )
    iterator._rows = iter([(raw_row, {"epoch": 0})])

    source = next(iterator)

    assert source.code_packet is not None
    assert source.code_packet.token_ids.tolist() == [10, 11, 12, 13]
    assert iterator.last_cursor == {"epoch": 0, "source_index": 0}


def test_source_reader_uses_bounded_record_batches_without_whole_shard_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": pa.array(
                    [[value] for value in range(11)],
                    type=pa.list_(pa.uint32()),
                )
            }
        ),
        source,
        row_group_size=4,
    )
    monkeypatch.setattr(
        materializer, "validate_case5_contract_metadata", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        materializer,
        "require_megatron_objective_source_columns",
        lambda *_a, **_k: None,
    )

    original_parquet_file = pq.ParquetFile
    whole_shard_reads = 0

    class GuardedParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self._inner = original_parquet_file(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def read(self, *args, **kwargs):
            nonlocal whole_shard_reads
            whole_shard_reads += 1
            raise AssertionError("whole-shard ParquetFile.read() is forbidden")

    monkeypatch.setattr(materializer.pq, "ParquetFile", GuardedParquetFile)
    observed_batch_rows: list[int] = []
    original_batch_rows = materializer._record_batch_rows

    def record_batch_rows(batch: pa.RecordBatch) -> list[dict[str, object]]:
        observed_batch_rows.append(batch.num_rows)
        return original_batch_rows(batch)

    monkeypatch.setattr(materializer, "_record_batch_rows", record_batch_rows)

    def one_epoch() -> list[int]:
        rows = materializer._iter_parquet_source_rows(
            [str(source)],
            seed=29,
            source_batch_rows=2,
        )
        return [int(next(rows)[0]["token_ids"][0]) for _ in range(11)]  # type: ignore[index]

    first = one_epoch()
    second = one_epoch()
    assert first == second
    assert sorted(first) == list(range(11))
    assert max(observed_batch_rows) <= 2
    assert whole_shard_reads == 0


def test_materialization_loop_releases_each_document_instead_of_retaining_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = 0
    peak = 0
    created = 0

    class TrackedDocument:
        def __init__(self) -> None:
            nonlocal live, peak, created
            live += 1
            created += 1
            peak = max(peak, live)

        def __del__(self) -> None:
            nonlocal live
            live -= 1

    def materialize(*_args, **_kwargs):
        return TrackedDocument()

    class Mixer:
        @staticmethod
        def materialize_window_from_pool(
            sources,
            *,
            output_count: int,
            start_step: int,
            required_assignment=None,
        ):
            del required_assignment
            del start_step
            return [
                SimpleNamespace(source_index=index) for index in range(output_count)
            ]

    class Sink:
        @staticmethod
        def add(_document) -> None:
            assert live == 1

    monkeypatch.setattr(materializer, "materialize_megatron_document", materialize)
    materializer._materialize_stream(
        mixer=Mixer(),  # type: ignore[arg-type]
        source_iter=iter(range(12)),  # type: ignore[arg-type]
        accumulator=Sink(),  # type: ignore[arg-type]
        writer=Sink(),  # type: ignore[arg-type]
        samples=12,
        quota_window_samples=3,
        quota_lookahead_samples=0,
    )

    assert created == 12
    assert peak == 1
    assert live == 0


def test_materialization_stream_uses_bounded_lookahead_and_carries_unused_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sources: list[int] = []

    class Mixer:
        @staticmethod
        def materialize_window_from_pool(
            sources,
            *,
            output_count: int,
            start_step: int,
            required_assignment=None,
        ):
            del required_assignment
            if start_step == 0 and len(sources) < 5:
                raise materializer.ObjectiveQuotaUnsatisfiedError("need rare rows")
            indices = [0, 3, 4] if start_step == 0 else list(range(output_count))
            return [SimpleNamespace(source_index=index) for index in indices]

    class Sink:
        @staticmethod
        def add(_document) -> None:
            return None

    def materialize(_item, source, **_kwargs):
        observed_sources.append(source)
        return object()

    monkeypatch.setattr(materializer, "materialize_megatron_document", materialize)
    receipt = materializer._materialize_stream(
        mixer=Mixer(),  # type: ignore[arg-type]
        source_iter=iter(range(6)),  # type: ignore[arg-type]
        accumulator=Sink(),  # type: ignore[arg-type]
        writer=Sink(),  # type: ignore[arg-type]
        samples=6,
        quota_window_samples=3,
        quota_lookahead_samples=2,
    )

    assert observed_sources == [0, 3, 4, 1, 2, 5]
    assert receipt["schema"] == "cppmega_objective_source_selection_v3"
    assert receipt["algorithm"] == (
        "bounded_eligibility_bipartite_graph_capability_v1"
    )
    assert receipt["output_samples"] == 6
    assert receipt["source_rows_consumed"] == 6
    assert receipt["unused_buffered_sources"] == 0
    assert receipt["quota_window_samples"] == 3
    assert receipt["quota_lookahead_samples"] == 2
    assert receipt["max_source_pool_samples"] == 5
    assert receipt["max_source_pool_observed"] == 5
    assert receipt["required_graph_relations"] == []
    assert [window["selected_source_indices"] for window in receipt["windows"]] == [
        [0, 3, 4],
        [1, 2, 5],
    ]
    assert receipt["schedule"] == {
        "schema": "cppmega_objective_schedule_v1",
        "algorithm": "bounded_eligibility_bipartite_graph_capability_v1",
        "windows_sha256": receipt["windows_sha256"],
    }
    assert receipt["resume"] == {
        "schema": "cppmega_objective_source_resume_v1",
        "cursor_semantics": (
            "replay_buffered_rows_then_continue_after_last_yielded_v1"
        ),
        "last_yielded_cursor": {"source_index": 5},
        "buffered_source_cursors": [],
    }


def test_terminal_lookahead_receipt_preserves_buffered_resume_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CursorIterator:
        def __init__(self) -> None:
            self._next = 0
            self.last_cursor = None

        def __iter__(self):
            return self

        def __next__(self):
            value = self._next
            self._next += 1
            self.last_cursor = {
                "epoch": 0,
                "row_index_in_record_batch": value,
                "source_index": value,
            }
            return value

    class Mixer:
        @staticmethod
        def materialize_window_from_pool(
            sources,
            *,
            output_count: int,
            start_step: int,
            required_assignment=None,
        ):
            del required_assignment, output_count, start_step
            if len(sources) < 5:
                raise materializer.ObjectiveQuotaUnsatisfiedError("need lookahead")
            return [SimpleNamespace(source_index=index) for index in (0, 3, 4)]

    class Sink:
        @staticmethod
        def add(_document) -> None:
            return None

    monkeypatch.setattr(
        materializer,
        "materialize_megatron_document",
        lambda *_args, **_kwargs: object(),
    )
    receipt = materializer._materialize_stream(
        mixer=Mixer(),  # type: ignore[arg-type]
        source_iter=CursorIterator(),  # type: ignore[arg-type]
        accumulator=Sink(),  # type: ignore[arg-type]
        writer=Sink(),  # type: ignore[arg-type]
        samples=3,
        quota_window_samples=3,
        quota_lookahead_samples=2,
    )

    assert receipt["source_rows_consumed"] == 5
    assert receipt["unused_buffered_sources"] == 2
    assert receipt["resume"] == {
        "schema": "cppmega_objective_source_resume_v1",
        "cursor_semantics": (
            "replay_buffered_rows_then_continue_after_last_yielded_v1"
        ),
        "last_yielded_cursor": {
            "epoch": 0,
            "row_index_in_record_batch": 4,
            "source_index": 4,
        },
        "buffered_source_cursors": [
            {
                "epoch": 0,
                "row_index_in_record_batch": 1,
                "source_index": 1,
            },
            {
                "epoch": 0,
                "row_index_in_record_batch": 2,
                "source_index": 2,
            },
        ],
    }


def test_oversized_row_fails_before_padding_and_cleans_partial_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = MaterializedMegatronDocument(
        objective_kind="causal_lm",
        token_ids=[1, 2],
        loss_mask=[1, 0],
        graph_edge_count=0,
        row={"valid_token_count": 2},
    )
    padded = False

    def forbidden_padding(*_args, **_kwargs):
        nonlocal padded
        padded = True
        raise AssertionError("padding must not happen beyond the byte budget")

    monkeypatch.setattr(materializer, "padded_row", forbidden_padding)
    output_dir = tmp_path / "objectives"
    with pytest.raises(MemoryError, match="before Arrow conversion"):
        with materializer._atomic_output_directory(output_dir) as partial:
            (partial / "already-written.partial").write_bytes(b"partial")
            with materializer._StreamingParquetShardWriter(
                partial,
                schema=pa.schema([]),
                capacity=2,
                shard_rows=4,
                write_batch_rows=2,
                max_buffer_bytes=1,
            ) as writer:
                writer.add(document)

    assert not padded
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".objectives.partial-*"))


def test_bounded_sampling_snapshot_records_order_and_replay_cursor(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source, row_group_size=1)
    snapshot, _signatures = materializer._build_source_snapshot(
        [str(source)],
        sequence_length=128,
        requested_samples=5,
        seed=31,
        sampling_mode=materializer._BOUNDED_SAMPLING_MODE,
        source_batch_rows=2,
    )
    sampling = snapshot["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["mode"] == materializer._BOUNDED_SAMPLING_MODE
    assert sampling["record_batch_rows"] == 2
    assert sampling["producer"] == {
        "name": "pyarrow.parquet.ParquetFile.iter_batches",
        "version": 1,
        "row_group_rows": [[1, 1]],
    }
    assert sampling["ordering"] == {
        "permutation": "sha256_sort_key_v1",
        "epochs": "ascending",
        "shards": "seeded_permutation_per_epoch",
        "row_groups": "seeded_permutation_per_shard_epoch",
        "record_batches": "physical_order_within_row_group",
        "rows": "seeded_permutation_within_record_batch",
    }

    cursor = {
        "epoch": 2,
        "shard_position": 0,
        "shard_index": 0,
        "row_group_position": 0,
        "row_group_index": 0,
        "record_batch_index": 0,
        "row_shuffle_position": 0,
        "row_index_in_record_batch": 0,
        "source_index": 4,
    }
    materializer._bind_source_sampling_cursor(snapshot, cursor=cursor)
    assert sampling["final_cursor"] == cursor


def test_source_snapshot_binds_paths_relative_to_explicit_root(tmp_path) -> None:
    source_root = tmp_path / "snapshot"
    source = source_root / "code" / "1024" / "source.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1, 2]}), source, row_group_size=1)

    snapshot, _signatures = materializer._build_source_snapshot(
        [str(source)],
        sequence_length=1024,
        requested_samples=1,
        seed=31,
        sampling_mode=materializer._BOUNDED_SAMPLING_MODE,
        source_batch_rows=2,
        source_root=source_root,
    )

    assert snapshot["files"][0]["path"] == "code/1024/source.parquet"


def test_source_snapshot_rejects_shard_outside_explicit_root(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    source_root = tmp_path / "other"
    source_root.mkdir()
    pq.write_table(pa.table({"value": [1]}), source)

    with pytest.raises(ValueError, match="outside --source-root"):
        materializer._build_source_snapshot(
            [str(source)],
            sequence_length=1024,
            requested_samples=1,
            seed=31,
            sampling_mode=materializer._BOUNDED_SAMPLING_MODE,
            source_batch_rows=2,
            source_root=source_root,
        )


def test_bounded_sampling_cursor_rebinds_actual_lookahead_draw_count(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source, row_group_size=1)
    snapshot, _signatures = materializer._build_source_snapshot(
        [str(source)],
        sequence_length=128,
        requested_samples=5,
        seed=31,
        sampling_mode=materializer._BOUNDED_SAMPLING_MODE,
        source_batch_rows=2,
    )
    cursor = {
        "epoch": 3,
        "shard_position": 0,
        "shard_index": 0,
        "row_group_position": 0,
        "row_group_index": 0,
        "record_batch_index": 0,
        "row_shuffle_position": 0,
        "row_index_in_record_batch": 0,
        "source_index": 6,
    }

    materializer._bind_source_sampling_cursor(
        snapshot,
        cursor=cursor,
        consumed_samples=7,
    )

    sampling = snapshot["sampling"]
    assert sampling["requested_samples"] == 7
    assert sampling["full_passes"] == 3
    assert sampling["tail_rows"] == 1
    assert sampling["min_row_reuse"] == 3
    assert sampling["max_row_reuse"] == 4
    assert sampling["final_cursor"] == cursor
