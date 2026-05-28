from __future__ import annotations

from pathlib import Path

from cppmega_v4._tilelang import _path_d_deps


def test_path_d_dependency_discovery_has_no_user_specific_absolute_roots() -> None:
    source = Path(_path_d_deps.__file__).read_text(encoding="utf-8")

    assert 'Path("/Volumes/' not in source
    assert 'Path("/Users/' not in source
    assert 'Path("/private/' not in source
