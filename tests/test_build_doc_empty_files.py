from __future__ import annotations

import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))


def test_empty_build_files_are_counted_and_skipped(tmp_path, capsys):
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

    assert len(docs) == 1
    assert docs[0]["doc_type"] == "build"
    assert docs[0]["build_kind"] == "cmake"
    assert docs[0]["domain_kind"] == int(DomainKind.CMAKE)
    assert len(docs[0]["domain_ids"]) == len(docs[0]["text"])
    assert int(DomainRoleKind.TARGET) in docs[0]["domain_role_ids"]
    assert any(
        edge["kind"] == int(DomainEdgeKind.BUILD_TARGET_DEP)
        for edge in docs[0]["build_edges"]
    )
    assert "skipped_empty=1" in capsys.readouterr().err
