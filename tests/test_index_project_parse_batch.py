from __future__ import annotations

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
    assert index_project.compute_parse_batch_size(35_046, 2) == 100


def test_parse_batch_size_keeps_small_parallel_repos_reasonable():
    import index_project

    assert index_project.compute_parse_batch_size(300, 2) == 100
    assert index_project.compute_parse_batch_size(80, 2) == 50


def test_parse_batch_size_rejects_invalid_worker_count():
    import pytest
    import index_project

    with pytest.raises(ValueError):
        index_project.compute_parse_batch_size(100, 0)
