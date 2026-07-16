from __future__ import annotations

from pathlib import Path

from cppmega_v4._tilelang import _path_d_deps


def test_path_d_dependency_discovery_has_no_user_specific_absolute_roots() -> None:
    source = Path(_path_d_deps.__file__).read_text(encoding="utf-8")

    assert 'Path("/Volumes/' not in source
    assert 'Path("/Users/' not in source
    assert 'Path("/private/' not in source


def test_missing_optional_parent_package_is_treated_as_unavailable(monkeypatch) -> None:
    def missing_parent(_module_name: str):
        raise ModuleNotFoundError("No module named 'poc'")

    monkeypatch.setattr(_path_d_deps.importlib.util, "find_spec", missing_parent)

    assert _path_d_deps._package_root_from_spec("poc.triton_frontend", 2) is None
