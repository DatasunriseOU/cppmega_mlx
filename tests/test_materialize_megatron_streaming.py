from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.training.megatron_objectives import MaterializedMegatronDocument
from scripts import materialize_megatron_objectives as materializer


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
    assert receipt == {
        "schema": "cppmega_objective_source_selection_v2",
        "algorithm": "bounded_eligibility_bipartite_pool_v1",
        "output_samples": 6,
        "source_rows_consumed": 6,
        "unused_buffered_sources": 0,
        "quota_window_samples": 3,
        "quota_lookahead_samples": 2,
        "max_source_pool_samples": 5,
        "max_source_pool_observed": 5,
        "required_graph_relations": [],
        "resume": {
            "schema": "cppmega_objective_source_resume_v1",
            "cursor_semantics": (
                "replay_buffered_rows_then_continue_after_last_yielded_v1"
            ),
            "last_yielded_cursor": {"source_index": 5},
            "buffered_source_cursors": [],
        },
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
