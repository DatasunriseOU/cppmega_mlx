"""Per-project C++ standard on the CONSUMING side (tools/clang_indexer).

A repo WITHOUT a compile_commands.json must still be parsed with ITS OWN detected
C++ standard (CMAKE_CXX_STANDARD / -std from the build system), NOT a hardcoded
c++17. These tests pin that behavior with REAL build files and the REAL
detect_build_context -> get_default_compile_args -> _resolve_file_args path that
index_project hands to libclang:

  * a CMAKE_CXX_STANDARD 20 repo (no compile_commands.json) is parsed at c++20,
  * a CMAKE_CXX_STANDARD 14 repo at c++14,
  * a repo with NO build files at all falls back to the documented c++17 default
    (the ONLY case a default is allowed — truly undetectable).

The final test is BEHAVIORAL (RULE #1: exercise the real toolchain, no mocks): it
proves the detected standard genuinely reaches the parser by parsing a C++20
``concept`` translation unit with each repo's resolved args — it parses clean
under the c++20 repo and errors under the c++14 repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

MLX_ROOT = Path(__file__).resolve().parents[1]
for p in (MLX_ROOT / "tools" / "clang_indexer", MLX_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_index_project():
    try:
        import index_project as ip  # type: ignore
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"index_project import failed (libclang?): {exc}")
    return ip


def _make_cmake_repo(tmp: Path, cxx_standard: int) -> Path:
    """A minimal CMake repo with CMAKE_CXX_STANDARD set and NO compile_commands."""
    repo = tmp / f"cmake_cxx{cxx_standard}"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(demo CXX)\n"
        f"set(CMAKE_CXX_STANDARD {cxx_standard})\n"
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        "add_executable(demo main.cpp)\n"
    )
    (repo / "main.cpp").write_text("int main() { return 0; }\n")
    # Sanity: no compile_commands.json anywhere in the repo.
    assert not list(repo.rglob("compile_commands.json"))
    return repo


def _std_flags(args: list[str]) -> list[str]:
    return [a for a in args if a.startswith("-std=")]


def _resolved_file_std(ip, repo: Path) -> list[str]:
    """The -std flags index_project actually hands libclang for repo/main.cpp."""
    default_args = ip._sanitize_compile_args_for_clang(
        ip.get_default_compile_args(str(repo))
    )
    resolved = ip._resolve_file_args(str(repo / "main.cpp"), None, default_args)
    return _std_flags(resolved)


def test_cmake_cxx_standard_20_is_parsed_at_cpp20(tmp_path):
    ip = _load_index_project()
    repo = _make_cmake_repo(tmp_path, 20)

    platform, args, compile_index = ip.detect_build_context(str(repo))
    assert compile_index is None, "no compile_commands.json expected"
    assert platform.get("standard") == "c++20"
    assert _std_flags(args) == ["-std=c++20"]

    # The flag must survive sanitize + per-file adaptation -> reach libclang.
    assert _resolved_file_std(ip, repo) == ["-std=c++20"]


def test_cmake_cxx_standard_14_is_parsed_at_cpp14(tmp_path):
    ip = _load_index_project()
    repo = _make_cmake_repo(tmp_path, 14)

    platform, args, compile_index = ip.detect_build_context(str(repo))
    assert compile_index is None
    assert platform.get("standard") == "c++14"
    assert _std_flags(args) == ["-std=c++14"]
    assert _resolved_file_std(ip, repo) == ["-std=c++14"]


def test_no_build_files_falls_back_to_cpp17_default(tmp_path):
    ip = _load_index_project()
    repo = tmp_path / "bare"
    repo.mkdir()
    (repo / "main.cpp").write_text("int main() { return 0; }\n")

    platform, args, compile_index = ip.detect_build_context(str(repo))
    assert compile_index is None
    assert platform.get("standard") == "c++17"
    assert _std_flags(args) == ["-std=c++17"]
    assert _resolved_file_std(ip, repo) == ["-std=c++17"]


def test_detected_standard_actually_reaches_the_parser(tmp_path):
    """BEHAVIORAL: a C++20 ``concept`` TU parses clean under the c++20 repo's
    resolved args and ERRORS under the c++14 repo's — proving the per-project
    standard genuinely drives libclang (not just the recorded metadata)."""
    ip = _load_index_project()
    try:
        ip._configure_libclang()
        clang_index = ip.Index.create()
    except Exception as exc:  # pragma: no cover - no libclang in this env
        pytest.skip(f"libclang unavailable: {exc}")

    concept_src = tmp_path / "concepts.cpp"
    concept_src.write_text(
        "template <typename T> concept Small = sizeof(T) <= 4;\n"
        "template <Small T> T id(T x) { return x; }\n"
        "int main() { return id(1); }\n"
    )

    def _fatal_or_error_count(repo: Path) -> int:
        default_args = ip._sanitize_compile_args_for_clang(
            ip.get_default_compile_args(str(repo))
        )
        args = ip._resolve_file_args(str(concept_src), None, default_args)
        tu = clang_index.parse(str(concept_src), args=args)
        # severity >= 3 is Error/Fatal in libclang's Diagnostic.severity scale.
        return sum(1 for d in tu.diagnostics if d.severity >= 3)

    repo20 = _make_cmake_repo(tmp_path, 20)
    repo14 = _make_cmake_repo(tmp_path, 14)

    assert _resolved_file_std(ip, repo20) == ["-std=c++20"]
    assert _resolved_file_std(ip, repo14) == ["-std=c++14"]

    # concepts are a C++20 feature: clean under c++20, rejected under c++14.
    assert _fatal_or_error_count(repo20) == 0
    assert _fatal_or_error_count(repo14) > 0
