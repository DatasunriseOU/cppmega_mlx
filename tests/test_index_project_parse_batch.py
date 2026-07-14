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


def test_parse_batch_results_bound_in_flight_futures_without_losing_batches():
    import index_project

    batches = [f"batch-{index}" for index in range(7)]

    class TrackingFuture(Future):
        def __init__(self, executor):
            super().__init__()
            self.executor = executor
            self.consumed = False

        def result(self, timeout=None):
            value = super().result(timeout)
            if not self.consumed:
                self.consumed = True
                self.executor.outstanding -= 1
            return value

    class FakeExecutor:
        def __init__(self):
            self.submitted = 0
            self.outstanding = 0
            self.max_outstanding = 0

        def submit(self, _fn, batch):
            future = TrackingFuture(self)
            self.submitted += 1
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
            future.set_result(batch)
            return future

    executor = FakeExecutor()
    results = list(
        index_project._iter_parse_batch_results(
            executor,
            batches,
            max_in_flight=2,
        )
    )

    assert executor.submitted == len(batches)
    assert executor.max_outstanding == 2
    assert executor.outstanding == 0
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
