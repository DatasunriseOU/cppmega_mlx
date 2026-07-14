from __future__ import annotations

from concurrent.futures import Future
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))


def test_parse_batch_size_caps_huge_repo_ipc_payloads():
    import index_project

    # xemu-scale repos previously used len(files) / parse_workers, so with
    # parse_workers=2 this became ~17.5k files in one returned payload.
    assert index_project.compute_parse_batch_size(35_046, 2) == 25


def test_parse_batch_size_keeps_small_parallel_repos_reasonable():
    import index_project

    assert index_project.compute_parse_batch_size(300, 2) == 25
    assert index_project.compute_parse_batch_size(80, 2) == 25


def test_parse_batch_size_rejects_invalid_worker_count():
    import pytest
    import index_project

    with pytest.raises(ValueError):
        index_project.compute_parse_batch_size(100, 0)


def test_parse_batch_results_bound_in_flight_futures_without_losing_batches(
    monkeypatch,
):
    import index_project

    submitted: list[Future] = []
    completed: list[Future] = []
    batches = [f"batch-{index}" for index in range(7)]

    class FakeExecutor:
        def submit(self, _fn, batch):
            future: Future = Future()
            future.set_result(batch)
            submitted.append(future)
            return future

    def fake_as_completed(futures):
        snapshot = list(futures)
        assert 0 < len(snapshot) <= 2
        selected = snapshot[-1]
        completed.append(selected)
        return iter([selected])

    monkeypatch.setattr(index_project, "as_completed", fake_as_completed)

    results = list(
        index_project._iter_parse_batch_results(
            FakeExecutor(),
            batches,
            max_in_flight=2,
        )
    )

    assert len(submitted) == len(batches)
    assert len(completed) == len(batches)
    assert sorted(results) == batches


def test_parse_batch_results_reject_invalid_submit_window():
    import pytest
    import index_project

    with pytest.raises(ValueError, match="max_in_flight must be positive"):
        list(
            index_project._iter_parse_batch_results(
                object(),
                [],
                max_in_flight=0,
            )
        )
