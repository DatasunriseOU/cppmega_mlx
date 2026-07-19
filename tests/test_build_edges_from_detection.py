"""Tests for _build_edges_from_detection and build_edges population in build docs."""
from __future__ import annotations

import sys
from pathlib import Path

MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))

CMAKE_FIXTURE = """\
cmake_minimum_required(VERSION 3.16)
project(MyApp LANGUAGES CXX)

add_executable(myapp main.cpp utils.cpp)
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(myapp PRIVATE pthread ssl)
"""

MAKE_FIXTURE = """\
CC = gcc
CFLAGS = -Wall -O2

all: main.o utils.o
\t$(CC) $(CFLAGS) -o app main.o utils.o

main.o: main.c utils.h
\t$(CC) $(CFLAGS) -c main.c

utils.o: utils.c utils.h
\t$(CC) $(CFLAGS) -c utils.c

clean:
\trm -f *.o app
"""

BAZEL_FIXTURE = """\
cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [":base"],
)

cc_binary(
    name = "app",
    srcs = ["main.cc"],
    deps = [":utils"],
)
"""


def test_build_edges_from_detection_cmake():
    """_build_edges_from_detection produces edges for CMake build_info."""
    from index_project import _build_edges_from_detection
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    build_info = {"build_system": "cmake", "compiler": "g++", "standard": "c++17"}
    edges = _build_edges_from_detection(build_info, CMAKE_FIXTURE)

    assert len(edges) > 0, "Expected non-empty build_edges for CMake fixture"
    # Every edge must have the required keys with int values
    for edge in edges:
        assert "from_char" in edge
        assert "to_char" in edge
        assert "kind" in edge
        assert isinstance(edge["from_char"], int)
        assert isinstance(edge["to_char"], int)
        assert isinstance(edge["kind"], int)
    # Should contain at least one BUILD_TARGET_SOURCE or BUILD_TARGET_DEP edge
    kinds = {e["kind"] for e in edges}
    build_kinds = {
        int(DomainEdgeKind.BUILD_TARGET_DEP),
        int(DomainEdgeKind.BUILD_TARGET_SOURCE),
        int(DomainEdgeKind.BUILD_COMMAND_TARGET),
    }
    assert kinds & build_kinds, f"Expected build edge kinds, got {kinds}"


def test_build_edges_from_detection_make():
    """_build_edges_from_detection produces edges for Make build_info."""
    from index_project import _build_edges_from_detection

    build_info = {"build_system": "make"}
    edges = _build_edges_from_detection(build_info, MAKE_FIXTURE)

    assert len(edges) > 0, "Expected non-empty build_edges for Make fixture"
    for edge in edges:
        assert isinstance(edge["from_char"], int)
        assert isinstance(edge["to_char"], int)
        assert isinstance(edge["kind"], int)


def test_build_edges_from_detection_bazel():
    """_build_edges_from_detection produces edges for Bazel build_info."""
    from index_project import _build_edges_from_detection

    build_info = {"build_system": "bazel"}
    edges = _build_edges_from_detection(build_info, BAZEL_FIXTURE)

    assert len(edges) > 0, "Expected non-empty build_edges for Bazel fixture"
    for edge in edges:
        assert isinstance(edge["from_char"], int)
        assert isinstance(edge["to_char"], int)
        assert isinstance(edge["kind"], int)


def test_build_edges_from_detection_empty_inputs():
    """_build_edges_from_detection returns [] for missing/empty inputs."""
    from index_project import _build_edges_from_detection

    assert _build_edges_from_detection(None, CMAKE_FIXTURE) == []
    assert _build_edges_from_detection({"build_system": "cmake"}, "") == []
    assert _build_edges_from_detection({"build_system": "cmake"}, "   \n  ") == []
    assert _build_edges_from_detection({}, CMAKE_FIXTURE) == []
    # Unknown build system
    assert _build_edges_from_detection({"build_system": "scons"}, CMAKE_FIXTURE) == []


def test_build_build_doc_populates_build_edges(tmp_path):
    """build_build_doc with CMakeLists.txt fixture produces non-empty build_edges."""
    from index_project import build_build_doc
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    cmake_path = tmp_path / "CMakeLists.txt"
    cmake_path.write_text(CMAKE_FIXTURE, encoding="utf-8")

    build_info = {"build_system": "cmake", "compiler": "g++", "standard": "c++17"}
    doc = build_build_doc(
        str(cmake_path),
        CMAKE_FIXTURE,
        "cmake",
        source_root=str(tmp_path),
        project_id="test/cmake-project",
        build_info=build_info,
    )

    assert doc["doc_type"] == "build"
    assert doc["build_kind"] == "cmake"
    assert "build_edges" in doc
    assert len(doc["build_edges"]) > 0, "build_build_doc must produce non-empty build_edges"

    # Validate edge structure
    for edge in doc["build_edges"]:
        assert "from_char" in edge
        assert "to_char" in edge
        assert "kind" in edge
        assert isinstance(edge["from_char"], int)
        assert isinstance(edge["to_char"], int)
        assert isinstance(edge["kind"], int)
        # Char offsets must be within text bounds
        assert 0 <= edge["from_char"] < len(CMAKE_FIXTURE)
        assert 0 <= edge["to_char"] < len(CMAKE_FIXTURE)

    # Should have target-dependency edges (target_link_libraries -> pthread, ssl)
    kinds = {e["kind"] for e in doc["build_edges"]}
    assert int(DomainEdgeKind.BUILD_TARGET_DEP) in kinds or \
           int(DomainEdgeKind.BUILD_TARGET_SOURCE) in kinds or \
           int(DomainEdgeKind.BUILD_COMMAND_TARGET) in kinds


def test_emit_build_documents_cmake_build_edges(tmp_path, capsys):
    """emit_build_documents produces docs with non-empty build_edges for CMake."""
    from index_project import emit_build_documents
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    cmake_path = tmp_path / "CMakeLists.txt"
    cmake_path.write_text(CMAKE_FIXTURE, encoding="utf-8")

    docs = emit_build_documents(
        [(str(cmake_path), "cmake")],
        source_root=str(tmp_path),
        project_id="test/cmake-edges",
        default_build_info={"build_system": "cmake", "compiler": "g++"},
    )

    assert len(docs) == 1
    doc = docs[0]
    assert doc["build_kind"] == "cmake"
    assert len(doc["build_edges"]) > 0, "emit_build_documents must produce non-empty build_edges"
    assert any(
        edge["kind"] == int(DomainEdgeKind.BUILD_TARGET_DEP)
        for edge in doc["build_edges"]
    )
