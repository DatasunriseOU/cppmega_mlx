from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _load_index_project_or_skip():
    from tools.clang_indexer import index_project

    try:
        index_project._configure_libclang(None)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"libclang unavailable: {exc}")
    return index_project


def test_build_context_ignores_build_directory_named_like_bazel_file(tmp_path: Path) -> None:
    from cppmega_mlx.data.nanochat_pipeline.build_context import detect_build_context

    (tmp_path / "Build").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")

    context, default_args, compile_index = detect_build_context(str(tmp_path))

    assert compile_index is None
    assert context["build_system"] == "default"
    assert "-std=c++17" in default_args


def test_windows_source_dumps_are_not_code_worktree_repos() -> None:
    from scripts.streaming_reindex import is_code_worktree_repo

    assert not is_code_worktree_repo("repo.bare")
    assert not is_code_worktree_repo("windows_10_shared_source_kit")
    assert not is_code_worktree_repo("windows_ce_5_121231")
    assert is_code_worktree_repo("llvm-project")


def test_empty_index_log_classifies_dedup_exhausted() -> None:
    from scripts.streaming_reindex import _classify_empty_index_project_log

    reason = _classify_empty_index_project_log(
        "\n".join(
            [
                "  Found 45 C/C++ source files",
                "  Found 3 build/compilation files",
                "  Function-level dedup: kept_roots=0 dropped_exact=950 dropped_near=11",
                "  Semantic chunk claims: root=0 dep=0 crosslib=0 type=0",
                "  Generated 0 code training documents",
                "  Build docs: emitted=0 dropped_dup=3 skipped_empty=0",
                "  Generated 0 total training documents",
            ]
        )
    )

    assert reason == "dedup_exhausted"


def test_process_one_repo_fails_for_empty_training_docs(tmp_path: Path) -> None:
    from scripts import streaming_reindex

    repo_dir = tmp_path / "src"
    repo_dir.mkdir()

    with pytest.raises(streaming_reindex.RepoFailure, match="no training docs"):
        streaming_reindex.process_one_repo(
            "repo",
            repo_dir,
            (1024,),
            tmp_path / "work",
            dedup_db=None,
            dedup_near=True,
            parse_workers=1,
            project_id="tests/problem-fixture",
        )


def test_run_commits_half_fails_for_no_commit_records(tmp_path: Path) -> None:
    from scripts import streaming_conveyor

    manifest = streaming_conveyor.ConcurrentManifest.load(tmp_path / "_done.json")
    repo_dir = tmp_path / "repo" / "_src"
    repo_dir.mkdir(parents=True)

    def no_records(*_args, **_kwargs):
        raise streaming_conveyor.RepoNoCommitRecords(
            "repo",
            reason="no_matching_commits",
            detail="no --no-merges --diff-filter=M commits",
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(streaming_conveyor.RepoFailure, match="no commit records"):
            streaming_conveyor.run_commits_half(
                repo="repo",
                repo_dir=repo_dir,
                repo_work=tmp_path / "repo",
                work_root=tmp_path / "work",
                work_parent=tmp_path / "parent",
                lengths_commits=(1024,),
                range_size=500,
                pool=pool,
                manifest=manifest,
                manifest_lock=streaming_conveyor.threading.Lock(),
                resume=True,
                cumulative={"valid": 0},
                dedup_db=None,
                dedup_near=True,
                pr_store=None,
                repo_list=None,
                commit_records_override=no_records,
            )

    assert "repo::commits" not in streaming_conveyor.ConcurrentManifest.load(
        tmp_path / "_done.json"
    ).done


def test_header_only_class_template_emits_standalone_header_doc(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "box.hpp").write_text(
        "#pragma once\n"
        "template <class T>\n"
        "struct Box {\n"
        "  T value;\n"
        "  constexpr T get() const { return value; }\n"
        "};\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    header_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "type"
    ]
    assert header_docs
    template_doc = next(doc for doc in header_docs if "template <class T>" in doc["text"])
    assert template_doc["filepath"].endswith("include/box.hpp")
    assert template_doc["domain_kind"] == 1  # DomainKind.CPP
    assert any(boundary["kind"] == 4 for boundary in template_doc["chunk_boundaries"])
    assert any(value == 4 for value in template_doc["structure_ids"])
    assert any(value == 2 for value in template_doc["ast_node_type"])


def test_header_type_provenance_survives_source_first_parse_order(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    src = tmp_path / "src"
    include.mkdir()
    src.mkdir()
    header = include / "box.hpp"
    source = src / "main.cpp"
    header.write_text(
        "#pragma once\n"
        "template <class T>\n"
        "struct CppMegaBox {\n"
        "  T value;\n"
        "  constexpr T get() const { return value; }\n"
        "};\n",
        encoding="utf-8",
    )
    source.write_text(
        "#include \"../include/box.hpp\"\n"
        "int cppmega_box_value() {\n"
        "  CppMegaBox<int> box{7};\n"
        "  return box.get();\n"
        "}\n",
        encoding="utf-8",
    )

    clang_index = index_project.Index.create()
    project_index = index_project.ProjectIndex()
    args = ["-std=c++23", f"-I{include}"]

    for path in [source, header]:
        functions, typedefs = index_project.parse_translation_unit(
            str(path),
            clang_index,
            index_project._adapt_args_for_file(args, str(path)),
            str(tmp_path),
            project_id="tests/problem-fixture",
        )
        for func in functions:
            project_index.add_function(func)
        for typedef in typedefs:
            project_index.add_typedef(typedef)

    docs: list[dict] = []
    index_project.build_training_documents(
        project_index,
        max_tokens=16384,
        enriched=True,
        project_dir=str(tmp_path),
        default_args=args,
        header_files=[str(header)],
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    header_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "type"
        and "template <class T>" in doc.get("text", "")
        and "CppMegaBox" in doc.get("text", "")
    ]
    assert header_docs
    assert header_docs[0]["filepath"].endswith("include/box.hpp")


def test_register_header_macros_reuses_directive_parse_cache_for_shared_includes(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    common = tmp_path / "common.h"
    root_a = tmp_path / "root_a.cpp"
    root_b = tmp_path / "root_b.cpp"
    common.write_text(
        "#define CPPMEGA_SHARED_MACRO(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    root_a.write_text('#include "common.h"\nint a() { return CPPMEGA_SHARED_MACRO(1); }\n', encoding="utf-8")
    root_b.write_text('#include "common.h"\nint b() { return CPPMEGA_SHARED_MACRO(2); }\n', encoding="utf-8")

    index = index_project.ProjectIndex()
    stats = index_project.register_header_macros(
        index,
        [str(root_a), str(root_b)],
        project_dir=str(tmp_path),
        project_id="tests/problem-fixture",
        include_dirs=[str(tmp_path)],
    )

    assert stats["directive_file_reads"] == 3
    assert stats["directive_cache_hits"] >= 1
    visible_roots = {
        macro.visible_in_file
        for macro in index.macros_by_name["CPPMEGA_SHARED_MACRO"]
    }
    assert visible_roots == {"root_a.cpp", "root_b.cpp"}


def test_register_header_macros_caps_per_root_include_fanout(tmp_path: Path) -> None:
    from tools.clang_indexer import index_project

    root = tmp_path / "root.cpp"
    common = tmp_path / "common.h"
    extra = tmp_path / "extra.h"
    root.write_text(
        '#include "common.h"\n'
        '#include "extra.h"\n'
        "int use_macro() { return CPPMEGA_SHARED_MACRO(1); }\n",
        encoding="utf-8",
    )
    common.write_text(
        "#define CPPMEGA_SHARED_MACRO(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    extra.write_text(
        "#define CPPMEGA_EXTRA_MACRO(x) ((x) + 2)\n",
        encoding="utf-8",
    )

    index = index_project.ProjectIndex()
    stats = index_project.register_header_macros(
        index,
        [str(root)],
        project_dir=str(tmp_path),
        project_id="tests/problem-fixture",
        include_dirs=[str(tmp_path)],
        max_include_files_per_root=2,
    )

    assert "CPPMEGA_SHARED_MACRO" in index.macros_by_name
    assert "CPPMEGA_EXTRA_MACRO" not in index.macros_by_name
    assert stats["skipped_include_file_cap"] == 1


def test_resolve_local_include_uses_project_file_index() -> None:
    from tools.clang_indexer import index_project

    project_dir = "/repo"
    header = "/repo/include/macro.hpp"

    resolved = index_project._resolve_local_include(
        "macro.hpp",
        current_abs="/repo/src/root.cpp",
        project_dir=project_dir,
        include_dirs=[f"/repo/missing_{idx}" for idx in range(128)] + ["/repo/include"],
        known_local_files={header},
    )

    assert resolved == header


def test_platform_detection_skips_macro_regexes_when_literal_is_absent() -> None:
    from tools.v5_gke_orchestrator import platform_detect

    class CountingRegex:
        def __init__(self) -> None:
            self.searches = 0

        def search(self, _source: str):
            self.searches += 1
            return None

    regexes = {
        "__CPPMEGA_NEVER_PRESENT__": CountingRegex(),
        "__CPPMEGA_ALSO_ABSENT__": CountingRegex(),
    }
    macro_set = {key: [("os", "test")] for key in regexes}

    assert platform_detect.detect_platforms(
        "int main() { return 0; }\n",
        macro_regexes=regexes,
        macro_set=macro_set,
    ) is None
    assert sum(regex.searches for regex in regexes.values()) == 0


def test_header_function_template_emits_trainable_cpp_doc(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "algo.hpp").write_text(
        "#pragma once\n"
        "template <class T>\n"
        "constexpr T cppmega_add(T lhs, T rhs) {\n"
        "  return lhs + rhs;\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    template_docs = [
        doc for doc in docs
        if "cppmega_add" in doc.get("text", "")
        and "template <class T>" in doc.get("text", "")
    ]
    assert template_docs
    doc = template_docs[0]
    assert doc["filepath"].endswith("include/algo.hpp")
    assert doc["doc_type"] == "code_header"
    assert doc["header_fragment_kind"] == "function_template"
    assert doc["domain_kind"] == 1  # DomainKind.CPP
    assert any(boundary["name"] == "cppmega_add" for boundary in doc["chunk_boundaries"])


def test_header_cxx20_concept_and_inline_variable_template_emit_docs(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(cppmega_header_fixture LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n",
        encoding="utf-8",
    )
    include = tmp_path / "include"
    include.mkdir()
    (include / "traits.hpp").write_text(
        "#pragma once\n"
        "#include <type_traits>\n"
        "template <class T>\n"
        "concept CppMegaAddable = requires(T lhs, T rhs) {\n"
        "  lhs + rhs;\n"
        "};\n"
        "template <class T>\n"
        "inline constexpr bool cppmega_small_v = sizeof(T) <= 8;\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    concept_docs = [doc for doc in docs if "concept CppMegaAddable" in doc.get("text", "")]
    variable_template_docs = [
        doc for doc in docs
        if "cppmega_small_v" in doc.get("text", "")
        and "inline constexpr bool" in doc.get("text", "")
    ]
    assert concept_docs
    assert variable_template_docs
    for doc in [concept_docs[0], variable_template_docs[0]]:
        assert doc["doc_type"] == "code_header"
        assert doc["header_fragment_kind"] == "header_decl"
        assert doc["filepath"].endswith("include/traits.hpp")
        assert doc["domain_kind"] == 1  # DomainKind.CPP


def test_macro_only_header_emits_macro_doc_and_cpp_delimiters(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind, delimiter_token_ids
    from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
        materialize_tokenized_enriched_batch,
    )
    from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
        TOKEN_IDS_COLUMN,
        TOKEN_DOMAIN_EDGES_COLUMN,
        TOKEN_ROLE_IDS_COLUMN,
        TOKEN_STRUCTURE_IDS_COLUMN,
    )
    from scripts.nanochat_data.token_budget import load_tokenizer

    assert getattr(DomainEdgeKind, "MACRO_PARAM_USE", None) is not None

    include = tmp_path / "include"
    include.mkdir()
    (include / "macros.hpp").write_text(
        "#ifndef CPPMEGA_MACROS_HPP\n"
        "#define CPPMEGA_MACROS_HPP\n"
        "#define CPPMEGA_MACROS_H_\n"
        "#define CPPMEGA_MAX(a, b) \\\n"
        "  ((a) > (b) ? (a) : (b))\n"
        "#define CPPMEGA_FLAG 0x1\n"
        "#endif\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    macro_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "macro"
    ]
    assert macro_docs
    assert not any(doc["text"].strip() == "#define CPPMEGA_MACROS_HPP" for doc in macro_docs)
    assert not any(doc["text"].strip() == "#define CPPMEGA_MACROS_H_" for doc in macro_docs)
    multiline = next(doc for doc in macro_docs if "CPPMEGA_MAX" in doc["text"])
    assert "\\\n" in multiline["text"]
    assert set(multiline["structure_ids"]) == {index_project.MACRO_KIND}
    assert set(multiline["ast_node_type"]) == {81}
    assert int(DomainRoleKind.IDENTIFIER) in set(multiline["domain_role_ids"])
    assert int(DomainRoleKind.VARIABLE) in set(multiline["domain_role_ids"])
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_PARAM_USE)
        for edge in multiline["domain_edges"]
    )

    row = materialize_tokenized_enriched_batch([multiline], load_tokenizer())[0]
    cpp_start, cpp_end = delimiter_token_ids(DomainKind.CPP)
    assert row[TOKEN_IDS_COLUMN][1] == cpp_start
    assert row[TOKEN_IDS_COLUMN][-1] == cpp_end
    assert row[TOKEN_STRUCTURE_IDS_COLUMN][2] == index_project.MACRO_KIND
    assert int(DomainRoleKind.VARIABLE) in set(row[TOKEN_ROLE_IDS_COLUMN])
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_PARAM_USE)
        for edge in row[TOKEN_DOMAIN_EDGES_COLUMN]
    )


def test_function_doc_pulls_used_macro_and_routes_invocation_to_definition(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    assert getattr(DomainEdgeKind, "MACRO_INVOCATION", None) is not None

    include = tmp_path / "include"
    include.mkdir()
    (include / "math.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_SQUARE(x) ((x) * (x))\n"
        "inline int cppmega_area(int side) {\n"
        "  return CPPMEGA_SQUARE(side);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed_docs = [
        doc for doc in docs
        if "cppmega_area" in doc.get("text", "")
        and "#define CPPMEGA_SQUARE" in doc.get("text", "")
    ]
    assert routed_docs
    routed = routed_docs[0]
    assert any(boundary["kind"] == index_project.MACRO_KIND for boundary in routed["chunk_boundaries"])
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        for edge in routed["domain_edges"]
    )


def test_conditional_macro_route_preserves_condition_stack(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    assert getattr(DomainEdgeKind, "MACRO_CONDITION", None) is not None

    include = tmp_path / "include"
    include.mkdir()
    (include / "feature.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_FEATURE 1\n"
        "#if CPPMEGA_FEATURE\n"
        "#define CPPMEGA_VALUE(x) ((x) + 1)\n"
        "#else\n"
        "#define CPPMEGA_VALUE(x) ((x) - 1)\n"
        "#endif\n"
        "inline int cppmega_value(int x) {\n"
        "  return CPPMEGA_VALUE(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_value" in doc.get("text", "")
        and "#define CPPMEGA_VALUE" in doc.get("text", "")
    )
    assert "#if CPPMEGA_FEATURE" in routed["text"]
    assert "#define CPPMEGA_FEATURE 1" in routed["text"]
    feature_name = routed["text"].find("#define CPPMEGA_FEATURE") + len("#define ")
    value_name = routed["text"].find("#define CPPMEGA_VALUE") + len("#define ")
    assert feature_name >= len("#define ")
    assert value_name >= len("#define ")
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_CONDITION)
        and edge["from_char"] == value_name
        and edge["to_char"] == feature_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_CONDITION)
        for edge in routed["domain_edges"]
    )


def test_macro_redefinition_window_routes_to_latest_visible_definition(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    assert getattr(DomainEdgeKind, "MACRO_REDEFINITION", None) is not None

    include = tmp_path / "include"
    include.mkdir()
    (include / "select.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_SELECT(x) ((x) + 1)\n"
        "#undef CPPMEGA_SELECT\n"
        "#define CPPMEGA_SELECT(x) ((x) + 2)\n"
        "inline int cppmega_select(int x) {\n"
        "  return CPPMEGA_SELECT(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_select" in doc.get("text", "")
        and "#define CPPMEGA_SELECT" in doc.get("text", "")
    )
    first = routed["text"].find("#define CPPMEGA_SELECT(x) ((x) + 1)")
    second = routed["text"].find("#define CPPMEGA_SELECT(x) ((x) + 2)")
    use = routed["text"].find("return CPPMEGA_SELECT(x)")
    assert first != -1 and second != -1 and use != -1
    assert first < second < use
    first_name = first + len("#define ")
    second_name = second + len("#define ")
    use_name = routed["text"].find("CPPMEGA_SELECT", use)
    assert "#undef CPPMEGA_SELECT" in routed["text"]
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_REDEFINITION)
        and edge["from_char"] == second_name
        and edge["to_char"] == first_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == use_name
        and edge["to_char"] == second_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_REDEFINITION)
        for edge in routed["domain_edges"]
    )


def test_pulled_dependency_macro_invocation_keeps_source_redefinition_window(
    tmp_path: Path,
) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    include = tmp_path / "include"
    include.mkdir()
    (include / "window.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_WINDOW(x) ((x) + 1)\n"
        "inline int cppmega_window_before(int x) {\n"
        "  return CPPMEGA_WINDOW(x);\n"
        "}\n"
        "#undef CPPMEGA_WINDOW\n"
        "#define CPPMEGA_WINDOW(x) ((x) + 2)\n"
        "inline int cppmega_window_after(int x) {\n"
        "  return cppmega_window_before(x) + CPPMEGA_WINDOW(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_window_before" in doc.get("text", "")
        and "cppmega_window_after" in doc.get("text", "")
        and "#define CPPMEGA_WINDOW(x) ((x) + 1)" in doc.get("text", "")
        and "#define CPPMEGA_WINDOW(x) ((x) + 2)" in doc.get("text", "")
    )
    first = routed["text"].find("#define CPPMEGA_WINDOW(x) ((x) + 1)")
    second = routed["text"].find("#define CPPMEGA_WINDOW(x) ((x) + 2)")
    before_func = routed["text"].find("cppmega_window_before")
    after_func = routed["text"].find("cppmega_window_after")
    before_return = routed["text"].find("return CPPMEGA_WINDOW(x);", before_func)
    after_return = routed["text"].find("return cppmega_window_before(x)", after_func)
    before_use = routed["text"].find("CPPMEGA_WINDOW", before_return)
    after_use = routed["text"].find("CPPMEGA_WINDOW", after_return)
    assert -1 not in {first, second, before_use, after_use}
    first_name = first + len("#define ")
    second_name = second + len("#define ")

    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == before_use
        and edge["to_char"] == first_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == after_use
        and edge["to_char"] == second_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_REDEFINITION)
        and edge["from_char"] == after_use
        and edge["to_char"] == first_name
        for edge in routed["domain_edges"]
    )


def test_macro_include_order_routes_across_local_headers(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    assert getattr(DomainEdgeKind, "MACRO_INCLUDE_ORDER", None) is not None

    include = tmp_path / "include"
    include.mkdir()
    (include / "a.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_PICK(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    (include / "b.hpp").write_text(
        "#pragma once\n"
        "#undef CPPMEGA_PICK\n"
        "#define CPPMEGA_PICK(x) ((x) + 2)\n",
        encoding="utf-8",
    )
    (include / "use.hpp").write_text(
        "#pragma once\n"
        "#include \"a.hpp\"\n"
        "#include \"b.hpp\"\n"
        "inline int cppmega_pick(int x) {\n"
        "  return CPPMEGA_PICK(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_pick" in doc.get("text", "")
        and "#define CPPMEGA_PICK" in doc.get("text", "")
    )
    first = routed["text"].find("#define CPPMEGA_PICK(x) ((x) + 1)")
    second = routed["text"].find("#define CPPMEGA_PICK(x) ((x) + 2)")
    use = routed["text"].find("return CPPMEGA_PICK(x)")
    assert first != -1 and second != -1 and use != -1
    assert first < second < use
    first_name = first + len("#define ")
    second_name = second + len("#define ")
    use_name = routed["text"].find("CPPMEGA_PICK", use)
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INCLUDE_ORDER)
        and edge["from_char"] == second_name
        and edge["to_char"] == first_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == use_name
        and edge["to_char"] == second_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INCLUDE_ORDER)
        for edge in routed["domain_edges"]
    )


def test_macro_body_dependency_is_pulled_and_expansion_routes_to_included_macro(
    tmp_path: Path,
) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    include = tmp_path / "include"
    include.mkdir()
    (include / "base.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_BASE(x) ((x) + 3)\n",
        encoding="utf-8",
    )
    (include / "wrap.hpp").write_text(
        "#pragma once\n"
        '#include "base.hpp"\n'
        "#define CPPMEGA_WRAP(x) CPPMEGA_BASE(x)\n"
        "inline int cppmega_wrap(int x) {\n"
        "  return CPPMEGA_WRAP(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_wrap" in doc.get("text", "")
        and "#define CPPMEGA_WRAP" in doc.get("text", "")
    )
    base = routed["text"].find("#define CPPMEGA_BASE")
    wrap = routed["text"].find("#define CPPMEGA_WRAP")
    use_line = routed["text"].find("return CPPMEGA_WRAP(x)")
    use = routed["text"].find("CPPMEGA_WRAP", use_line)
    assert -1 not in {base, wrap, use}
    base_name = base + len("#define ")
    wrap_name = wrap + len("#define ")
    assert base < wrap < use

    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == use
        and edge["to_char"] == wrap_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER)
        and edge["from_char"] == use
        and edge["to_char"] == base_name
        for edge in routed["domain_edges"]
    )


def test_macro_condition_dependency_cycle_does_not_recurse_forever(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    include = tmp_path / "include"
    include.mkdir()
    (include / "cycle.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_CYCLE 1\n"
        "#if CPPMEGA_CYCLE\n"
        "#define CPPMEGA_CYCLE 2\n"
        "#endif\n"
        "inline int cppmega_cycle() {\n"
        "  return CPPMEGA_CYCLE;\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_cycle" in doc.get("text", "")
        and "#define CPPMEGA_CYCLE" in doc.get("text", "")
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_CONDITION)
        for edge in routed["domain_edges"]
    )


def test_function_like_macro_condition_does_not_emit_python_syntax_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "callish_condition.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_ENABLED 1\n"
        "#if CPPMEGA_ENABLED(0)\n"
        "#define CPPMEGA_CALLISH_CONDITION(x) ((x) + 1)\n"
        "#endif\n"
        "inline int cppmega_callish_condition(int x) {\n"
        "  return CPPMEGA_CALLISH_CONDITION(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    captured = capsys.readouterr()
    assert "SyntaxWarning" not in captured.err
    assert any(
        "cppmega_callish_condition" in doc.get("text", "")
        and "#define CPPMEGA_CALLISH_CONDITION" in doc.get("text", "")
        for doc in docs
    )


def test_macro_elif_branch_can_become_active_after_false_if(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "choice.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_MODE 2\n"
        "#if CPPMEGA_MODE == 1\n"
        "#define CPPMEGA_CHOICE(x) ((x) + 1)\n"
        "#elif CPPMEGA_MODE == 2\n"
        "#define CPPMEGA_CHOICE(x) ((x) + 2)\n"
        "#else\n"
        "#define CPPMEGA_CHOICE(x) ((x) + 3)\n"
        "#endif\n"
        "inline int cppmega_choice(int x) {\n"
        "  return CPPMEGA_CHOICE(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_choice" in doc.get("text", "")
        and "return CPPMEGA_CHOICE" in doc.get("text", "")
    )
    assert "#elif CPPMEGA_MODE == 2" in routed["text"]
    assert "#define CPPMEGA_CHOICE(x) ((x) + 2)" in routed["text"]
    assert "#define CPPMEGA_CHOICE(x) ((x) + 1)" not in routed["text"]
    assert "#define CPPMEGA_CHOICE(x) ((x) + 3)" not in routed["text"]


def test_macro_else_branch_routes_to_branch_directive(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind
    from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
        materialize_tokenized_enriched_batch,
    )
    from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
        TOKEN_DOMAIN_EDGES_COLUMN,
    )
    from scripts.nanochat_data.token_budget import load_tokenizer

    include = tmp_path / "include"
    include.mkdir()
    (include / "else_choice.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_ELSE_MODE 0\n"
        "#if CPPMEGA_ELSE_MODE\n"
        "#define CPPMEGA_ELSE_PICK(x) ((x) + 1)\n"
        "#else\n"
        "#define CPPMEGA_ELSE_PICK(x) ((x) + 2)\n"
        "#endif\n"
        "inline int cppmega_else_pick(int x) {\n"
        "  return CPPMEGA_ELSE_PICK(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_else_pick" in doc.get("text", "")
        and "return CPPMEGA_ELSE_PICK" in doc.get("text", "")
    )
    assert "#else" in routed["text"]
    assert "#endif" in routed["text"]
    assert "#define CPPMEGA_ELSE_PICK(x) ((x) + 1)" not in routed["text"]
    assert "#define CPPMEGA_ELSE_PICK(x) ((x) + 2)" in routed["text"]
    define_start = routed["text"].find("#define CPPMEGA_ELSE_PICK(x) ((x) + 2)")
    use_line = routed["text"].find("return CPPMEGA_ELSE_PICK(x)")
    assert define_start != -1 and use_line != -1
    define_name = define_start + len("#define ")
    use_name = routed["text"].find("CPPMEGA_ELSE_PICK", use_line)
    else_keyword = routed["text"].find("#else") + 1
    endif_keyword = routed["text"].find("#endif") + 1

    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_CONDITION)
        and edge["from_char"] == define_name
        and edge["to_char"] == else_keyword
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_CONDITION)
        and edge["from_char"] == use_name
        and edge["to_char"] == else_keyword
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_CONDITION)
        and edge["from_char"] == define_name
        and edge["to_char"] == endif_keyword
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_CONDITION)
        and edge["from_char"] == use_name
        and edge["to_char"] == endif_keyword
        for edge in routed["domain_edges"]
    )

    tokenized = materialize_tokenized_enriched_batch(
        [routed],
        load_tokenizer(),
        num_threads=1,
    )[0]
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_CONDITION)
        for edge in tokenized[TOKEN_DOMAIN_EDGES_COLUMN]
    )


def test_macro_if_not_defined_condition_is_respected(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "fallback.hpp").write_text(
        "#pragma once\n"
        "#if !defined(CPPMEGA_HAVE_FALLBACK)\n"
        "#define CPPMEGA_FALLBACK(x) ((x) + 4)\n"
        "#endif\n"
        "inline int cppmega_fallback(int x) {\n"
        "  return CPPMEGA_FALLBACK(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_fallback" in doc.get("text", "")
        and "return CPPMEGA_FALLBACK" in doc.get("text", "")
    )
    assert "#if !defined(CPPMEGA_HAVE_FALLBACK)" in routed["text"]
    assert "#define CPPMEGA_FALLBACK(x) ((x) + 4)" in routed["text"]


def test_include_guard_defines_affect_repeated_local_include_scan(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "guarded.hpp").write_text(
        "#ifndef CPPMEGA_GUARDED_HPP\n"
        "#define CPPMEGA_GUARDED_HPP\n"
        "#define CPPMEGA_GUARDED_VALUE(x) ((x) + 1)\n"
        "#endif\n",
        encoding="utf-8",
    )
    (include / "use.hpp").write_text(
        "#pragma once\n"
        "#include \"guarded.hpp\"\n"
        "#include \"guarded.hpp\"\n"
        "inline int cppmega_guarded(int x) {\n"
        "  return CPPMEGA_GUARDED_VALUE(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_guarded" in doc.get("text", "")
        and "return CPPMEGA_GUARDED_VALUE" in doc.get("text", "")
    )
    assert routed["text"].count("#define CPPMEGA_GUARDED_VALUE(x) ((x) + 1)") == 1


def test_source_include_order_uses_only_headers_visible_to_source(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (include / "a.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_SOURCE_PICK(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    (include / "z.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_SOURCE_PICK(x) ((x) + 2)\n",
        encoding="utf-8",
    )
    (src / "main.cpp").write_text(
        "#include \"../include/a.hpp\"\n"
        "int cppmega_source_pick(int x) {\n"
        "  int y = CPPMEGA_SOURCE_PICK(x);\n"
        "  y += CPPMEGA_SOURCE_PICK(y);\n"
        "  y += 3;\n"
        "  return y + 10;\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_source_pick" in doc.get("text", "")
        and "CPPMEGA_SOURCE_PICK" in doc.get("text", "")
    )
    assert "#define CPPMEGA_SOURCE_PICK(x) ((x) + 1)" in routed["text"]
    assert "#define CPPMEGA_SOURCE_PICK(x) ((x) + 2)" not in routed["text"]


def test_source_include_order_redefinition_window_routes_each_use_by_source_line(
    tmp_path: Path,
) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind

    include = tmp_path / "include"
    include.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (include / "a.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_LINE_PICK(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    (include / "b.hpp").write_text(
        "#pragma once\n"
        "#undef CPPMEGA_LINE_PICK\n"
        "#define CPPMEGA_LINE_PICK(x) ((x) + 2)\n",
        encoding="utf-8",
    )
    (src / "main.cpp").write_text(
        "#include \"../include/a.hpp\"\n"
        "int cppmega_line_pick(int x) {\n"
        "  int first = CPPMEGA_LINE_PICK(x);\n"
        "#include \"../include/b.hpp\"\n"
        "  int second = CPPMEGA_LINE_PICK(first);\n"
        "  return first + second;\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_line_pick" in doc.get("text", "")
        and "#define CPPMEGA_LINE_PICK(x) ((x) + 1)" in doc.get("text", "")
        and "#define CPPMEGA_LINE_PICK(x) ((x) + 2)" in doc.get("text", "")
    )
    first_def = routed["text"].find("#define CPPMEGA_LINE_PICK(x) ((x) + 1)")
    second_def = routed["text"].find("#define CPPMEGA_LINE_PICK(x) ((x) + 2)")
    first_use_line = routed["text"].find("int first = CPPMEGA_LINE_PICK(x)")
    second_use_line = routed["text"].find("int second = CPPMEGA_LINE_PICK(first)")
    assert -1 not in {first_def, second_def, first_use_line, second_use_line}

    first_def_name = first_def + len("#define ")
    second_def_name = second_def + len("#define ")
    first_use = routed["text"].find("CPPMEGA_LINE_PICK", first_use_line)
    second_use = routed["text"].find("CPPMEGA_LINE_PICK", second_use_line)

    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == first_use
        and edge["to_char"] == first_def_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        and edge["from_char"] == second_use
        and edge["to_char"] == second_def_name
        for edge in routed["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_REDEFINITION)
        and edge["from_char"] == second_use
        and edge["to_char"] == first_def_name
        for edge in routed["domain_edges"]
    )


def test_macro_include_resolution_uses_compile_include_dirs(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    generated = tmp_path / "generated"
    generated.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (generated / "config.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_CONFIG_PICK(x) ((x) + 7)\n",
        encoding="utf-8",
    )
    (src / "main.cpp").write_text(
        "#include \"config.hpp\"\n"
        "int cppmega_config_pick(int x) {\n"
        "  int y = CPPMEGA_CONFIG_PICK(x);\n"
        "  y += CPPMEGA_CONFIG_PICK(y);\n"
        "  y += 5;\n"
        "  return y + 10;\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "compile_commands.json").write_text(
        "[{"
        f"\"directory\": \"{tmp_path}\", "
        "\"command\": \"clang++ -I generated -std=c++23 -c src/main.cpp\", "
        f"\"file\": \"{src / 'main.cpp'}\""
        "}]\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = next(
        doc for doc in docs
        if "cppmega_config_pick" in doc.get("text", "")
        and "CPPMEGA_CONFIG_PICK" in doc.get("text", "")
    )
    assert "#define CPPMEGA_CONFIG_PICK(x) ((x) + 7)" in routed["text"]


def test_standalone_header_macros_survive_chunk_claims_when_used_as_deps(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    (include / "macro.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_CLAIMED_MACRO(x) ((x) + 1)\n"
        "inline int cppmega_claimed_macro(int x) {\n"
        "  return CPPMEGA_CLAIMED_MACRO(x);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        tokenizer_path=str(Path("cppmega_mlx/tokenizer/tokenizer.json").resolve()),
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    routed = [
        doc for doc in docs
        if "cppmega_claimed_macro" in doc.get("text", "")
        and "return CPPMEGA_CLAIMED_MACRO" in doc.get("text", "")
    ]
    macro_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "macro"
        and "#define CPPMEGA_CLAIMED_MACRO" in doc.get("text", "")
    ]
    assert routed
    assert macro_docs


@pytest.mark.parametrize("extension", [".h", ".hpp", ".hxx", ".hh", ".ipp", ".tpp", ".txx"])
def test_header_template_extensions_emit_template_and_macro_docs(
    tmp_path: Path,
    extension: str,
) -> None:
    index_project = _load_index_project_or_skip()

    include = tmp_path / "include"
    include.mkdir()
    header = include / f"algo{extension}"
    header.write_text(
        "#pragma once\n"
        "#define CPPMEGA_EXT_WRAP(x) ((x) + 1)\n"
        "template <class T>\n"
        "constexpr T cppmega_ext_add(T lhs, T rhs) {\n"
        "  return CPPMEGA_EXT_WRAP(lhs + rhs);\n"
        "}\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    template_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "function_template"
        and doc.get("filepath", "").endswith(f"include/algo{extension}")
        and "cppmega_ext_add" in doc.get("text", "")
    ]
    macro_docs = [
        doc for doc in docs
        if doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "macro"
        and doc.get("filepath", "").endswith(f"include/algo{extension}")
        and "#define CPPMEGA_EXT_WRAP" in doc.get("text", "")
    ]
    assert template_docs
    assert macro_docs


def test_header_template_docs_survive_tokenized_route_and_pack_e2e(tmp_path: Path) -> None:
    index_project = _load_index_project_or_skip()
    from cppmega_mlx.data.domain_schema import DomainEdgeKind
    from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
        INPUT_IDS_COLUMN,
        SOURCE_DOC_TYPES_COLUMN,
        SOURCE_HEADER_FRAGMENT_KINDS_COLUMN,
        TOKEN_CHUNK_KINDS_COLUMN,
        TOKEN_DOMAIN_EDGES_COLUMN,
        TOKEN_STRUCTURE_IDS_COLUMN,
    )
    from scripts.nanochat_data.clang_enriched_to_parquet import (
        convert_local_jsonl_to_parquet,
    )
    from scripts.nanochat_data.pack_enriched_rows import pack_parquet_dataset
    from scripts.nanochat_data.token_budget import load_tokenizer
    from scripts.streaming_reindex_commits import route_by_fit
    import pyarrow.parquet as pq

    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(cppmega_header_e2e LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n",
        encoding="utf-8",
    )
    include = tmp_path / "include"
    include.mkdir()
    (include / "box.hpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_E2E_ID(x) (x)\n"
        "template <class T>\n"
        "struct CppMegaE2EBox {\n"
        "  T value;\n"
        "};\n",
        encoding="utf-8",
    )
    (include / "algo.tpp").write_text(
        "#pragma once\n"
        "#define CPPMEGA_E2E_BASE(x) ((x) + 1)\n"
        "#define CPPMEGA_E2E_ADD(x, y) ((x) + (y))\n"
        "template <class T>\n"
        "constexpr T cppmega_e2e_add(T lhs, T rhs) {\n"
        "  return CPPMEGA_E2E_ADD(CPPMEGA_E2E_BASE(lhs), rhs);\n"
        "}\n",
        encoding="utf-8",
    )
    (include / "concepts.hpp").write_text(
        "#pragma once\n"
        "template <class T>\n"
        "concept CppMegaE2EAddable = requires(T lhs, T rhs) {\n"
        "  lhs + rhs;\n"
        "};\n"
        "template <class T>\n"
        "inline constexpr bool cppmega_e2e_small_v = sizeof(T) <= 8;\n",
        encoding="utf-8",
    )

    docs: list[dict] = []
    index_project.process_project(
        str(tmp_path),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        tokenizer_path=str(Path("cppmega_mlx/tokenizer/tokenizer.json").resolve()),
        emit_doc=docs.append,
        project_id="tests/problem-fixture",
    )

    assert any(
        doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "type"
        and "CppMegaE2EBox" in doc.get("text", "")
        for doc in docs
    )
    assert any(
        doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "function_template"
        and "cppmega_e2e_add" in doc.get("text", "")
        for doc in docs
    )
    assert any(
        doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "macro"
        and "CPPMEGA_E2E_ADD" in doc.get("text", "")
        for doc in docs
    )
    assert any(
        doc.get("doc_type") == "code_header"
        and doc.get("header_fragment_kind") == "header_decl"
        and "concept CppMegaE2EAddable" in doc.get("text", "")
        for doc in docs
    )

    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, sort_keys=True))
            handle.write("\n")

    tokenized = tmp_path / "tokenized.parquet"
    convert_local_jsonl_to_parquet(
        jsonl,
        tokenized,
        tokenizer=load_tokenizer(),
        max_tokens=65536,
        overflow_policy="drop",
        materialize_tokenized_enriched=True,
            default_repo="tests/header-e2e",
    )

    route_dir = tmp_path / "routes"
    routed = route_by_fit(tokenized, [1024, 2048, 4096, 8192, 16384], route_dir)
    assert routed

    packed_paths: list[Path] = []
    for target_length, route_path in routed.items():
        packed = tmp_path / f"packed_{target_length}.parquet"
        summary = pack_parquet_dataset(
            route_path,
            packed,
            target_length=target_length,
            strategy="best_fit",
        )
        assert summary["overflow_docs"] == 0
        packed_paths.append(packed)

    rows: list[dict] = []
    for packed in packed_paths:
        rows.extend(pq.read_table(packed).to_pylist())

    source_doc_types = {
        item
        for row in rows
        for item in row[SOURCE_DOC_TYPES_COLUMN]
        if item is not None
    }
    source_fragment_kinds = {
        item
        for row in rows
        for item in row[SOURCE_HEADER_FRAGMENT_KINDS_COLUMN]
        if item is not None
    }
    assert "code_header" in source_doc_types
    assert {"type", "function_template", "macro", "header_decl"} <= source_fragment_kinds

    assert any(index_project.MACRO_KIND in row[TOKEN_CHUNK_KINDS_COLUMN] for row in rows)
    assert any(index_project.MACRO_KIND in row[TOKEN_STRUCTURE_IDS_COLUMN] for row in rows)
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_PARAM_USE)
        for row in rows
        for edge in row[TOKEN_DOMAIN_EDGES_COLUMN]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        for row in rows
        for edge in row[TOKEN_DOMAIN_EDGES_COLUMN]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER)
        for row in rows
        for edge in row[TOKEN_DOMAIN_EDGES_COLUMN]
    )
    for row in rows:
        row_length = len(row[INPUT_IDS_COLUMN])
        for column in (TOKEN_STRUCTURE_IDS_COLUMN,):
            assert len(row[column]) == row_length
