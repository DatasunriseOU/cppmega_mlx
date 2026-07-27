from __future__ import annotations

import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))


def test_whitespace_build_source_is_preserved_losslessly(tmp_path, capsys):
    from index_project import emit_build_documents
    from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind

    empty_build = tmp_path / "BUILD.bazel"
    empty_build.write_text("  \n\t\n", encoding="utf-8")
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text(
        "add_library(example example.cc)\n"
        "target_link_libraries(example PRIVATE ssl)\n",
        encoding="utf-8",
    )

    docs = emit_build_documents(
        [(str(empty_build), "bazel"), (str(cmake), "cmake")],
        default_build_info=None,
    )

    assert len(docs) == 2
    by_kind = {doc["build_kind"]: doc for doc in docs}
    assert by_kind["bazel"]["text"] == "  \n\t\n"
    assert by_kind["bazel"]["doc_type"] == "build"
    assert len(by_kind["bazel"]["domain_ids"]) == len(by_kind["bazel"]["text"])
    assert by_kind["cmake"]["doc_type"] == "build"
    assert by_kind["cmake"]["domain_kind"] == int(DomainKind.CMAKE)
    assert len(by_kind["cmake"]["domain_ids"]) == len(by_kind["cmake"]["text"])
    assert int(DomainRoleKind.TARGET) in by_kind["cmake"]["domain_role_ids"]
    assert any(
        edge["kind"] == int(DomainEdgeKind.BUILD_TARGET_DEP)
        for edge in by_kind["cmake"]["build_edges"]
    )
    log = capsys.readouterr().err
    assert "source_chars_in=80 source_chars_out=80" in log
    assert "skipped_zero_length=0" in log
