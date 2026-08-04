from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.clang_indexer import index_project as ip
from tools.clang_indexer import process_commits as pc


ROOT = Path(__file__).resolve().parents[1]
CASE3_FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"


def _clang_index():
    pytest.importorskip("clang.cindex")
    ip._configure_libclang()
    return ip.Index.create()


def _std_flags(args: list[str]) -> list[str]:
    return [arg for arg in args if arg.startswith("-std=")]


def test_autoconf_compile_context_keeps_operands_and_truthful_standard() -> None:
    from cppmega_mlx.data.nanochat_pipeline.build_context import (
        detect_build_context_from_loader,
    )

    configure = (
        "AC_PROG_CC\n"
        "CFLAGS='-std=iso9899:201x -I include -D FEATURE=1 -U OLD "
        "-include config.h -imacrosdefs.h --target x86_64-linux-gnu'\n"
    )

    platform, args, compile_index = detect_build_context_from_loader(
        lambda name: configure if name == "configure.ac" else None
    )

    assert compile_index is None
    assert platform["compiler"] == "gcc"
    assert platform["standard"] == "c11"
    assert args[:2] == ["-x", "c"]
    assert "-std=iso9899:201x" in args
    for option, operand in (
        ("-I", "include"),
        ("-D", "FEATURE=1"),
        ("-U", "OLD"),
        ("-include", "config.h"),
        ("--target", "x86_64-linux-gnu"),
    ):
        option_index = args.index(option)
        assert args[option_index + 1] == operand
    assert "-imacrosdefs.h" in args
    assert args[-2:] == ["-fsyntax-only", "-Wno-everything"]


def test_detected_build_without_standard_does_not_claim_cpp17() -> None:
    from cppmega_mlx.data.nanochat_pipeline.build_context import (
        detect_build_context_from_loader,
    )

    platform, args, compile_index = detect_build_context_from_loader(
        lambda name: "AC_PROG_CC\n" if name == "configure.ac" else None
    )

    assert compile_index is None
    assert platform["build_system"] == "autoconf"
    assert platform["compiler"] == "gcc"
    assert "standard" not in platform
    assert args[:2] == ["-x", "c"]
    assert not _std_flags(args)


@pytest.mark.parametrize(
    "compile_args",
    (
        [
            "-x",
            "c",
            "-std=iso9899:199x",
            "-D",
            "FEATURE=1",
            "-U",
            "OLD",
            "-include",
            "config.h",
        ],
        ["-xc++", "--std=c++20", "-m64"],
        ["-x", "cl", "-cl-std=CL3.0"],
    ),
)
def test_strict_compile_arg_validation_accepts_coherent_contexts(
    compile_args: list[str],
) -> None:
    assert ip._is_sane_compile_args(compile_args)


@pytest.mark.parametrize(
    "compile_args",
    (
        [],
        ["-march=@<:@^"],
        ["-DVERSION=@VERSION@"],
        ["-D"],
        ["-x", "not-a-clang-language"],
        ["-m32", "-m64"],
        ["-x", "c++", "-std=c++17", "-std=c++20"],
        ["-x", "c++", "-std=c11"],
        ["-cl-std=CL3.0"],
    ),
)
def test_strict_compile_arg_validation_rejects_unusable_aggregates(
    compile_args: list[str],
) -> None:
    assert not ip._is_sane_compile_args(compile_args)


def test_unusable_default_context_falls_back_atomically(tmp_path: Path) -> None:
    args, build_info = ip._resolve_default_compile_context(
        str(tmp_path),
        {
            "build_system": "autoconf",
            "source": "build_files",
            "compiler": "g++",
            "standard": "c++20",
        },
        ["-x", "c++", "-std=c++17", "-std=c++20"],
    )

    assert args[:2] == ["-fsyntax-only", "-Wno-everything"]
    assert build_info == {
        "build_system": "autoconf",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }


def test_per_file_context_matches_adapted_args_and_preserves_legacy_cpp_c(
    tmp_path: Path,
) -> None:
    plain_c = tmp_path / "plain.c"
    plain_c.write_text("int plain(void) { return 1; }\n", encoding="utf-8")
    legacy_cpp_c = tmp_path / "compiler-regression.c"
    legacy_cpp_c.write_text(
        "char *legacy(void) { return(::new char[2] ('a', 'b')); }\n",
        encoding="utf-8",
    )
    default_build_info = {
        "build_system": "cmake",
        "source": "build_files",
        "compiler": "clang++",
        "standard": "c++20",
    }

    plain_args, plain_build_info = ip._compile_context_for_rel_file(
        plain_c.name,
        project_dir=str(tmp_path),
        compile_db=None,
        default_args=["-std=c++20"],
        default_build_info=default_build_info,
    )
    legacy_args, legacy_build_info = ip._compile_context_for_rel_file(
        legacy_cpp_c.name,
        project_dir=str(tmp_path),
        compile_db=None,
        default_args=["-std=c++20"],
        default_build_info=default_build_info,
    )

    assert plain_args[:3] == ["-x", "c", "-std=c11"]
    assert plain_build_info["standard"] == "c11"
    assert legacy_args[:2] == ["-x", "c++"]
    assert "-std=c++20" in legacy_args
    assert legacy_build_info["standard"] == "c++20"

    compile_db_args, compile_db_build_info = ip._compile_context_for_rel_file(
        plain_c.name,
        project_dir=str(tmp_path),
        compile_db={
            str(plain_c): {
                "compile_args": ["-x", "c", "-std=iso9899:199x"],
                "build_info": {
                    "build_system": "compile_commands",
                    "source": "compile_commands",
                    "compiler": "clang",
                    "standard": "c++20",
                },
            }
        },
        default_args=["-std=c++20"],
        default_build_info=default_build_info,
    )
    assert compile_db_args[:2] == ["-x", "c"]
    assert "-std=iso9899:199x" in compile_db_args
    assert compile_db_build_info["standard"] == "c99"


def test_open_watcom_plusplus_c_path_forces_cpp_without_c_standard(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bld" / "plusplus" / "bugs" / "zcc02.c"
    source.parent.mkdir(parents=True)
    source.write_text(
        "char *foo( void )\n"
        "{\n"
        "    return( ::new char[10] ( 'a', 'b', 'c', '\\0' ) );\n"
        "}\n",
        encoding="utf-8",
    )

    adapted = ip._adapt_args_for_file(
        ["-std=c11", "-std=c++20", "-fsyntax-only", "-Wno-everything"],
        str(source),
    )

    assert adapted[:3] == ["-x", "c++", "-std=c++20"]
    assert "-std=c11" not in adapted
    translation_unit = ip._load_translation_unit(
        str(source),
        _clang_index(),
        adapted,
    )
    assert translation_unit.spelling == str(source)

    fallback_adapted = ip._adapt_args_for_file(
        ["-std=c11", "-fsyntax-only", "-Wno-everything"],
        str(source),
    )
    assert fallback_adapted[:3] == ["-x", "c++", "-std=c++17"]
    assert "-std=c11" not in fallback_adapted

    ordinary_c = tmp_path / "src" / "ordinary.c"
    ordinary_c.parent.mkdir(parents=True)
    ordinary_c.write_text("int ordinary(void) { return 0; }\n", encoding="utf-8")
    assert ip._adapt_args_for_file(
        ["-std=c++20", "-Wno-everything"],
        str(ordinary_c),
    )[:3] == ["-x", "c", "-std=c11"]


def test_emitted_function_sidecar_matches_per_file_adapted_args(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain.c"
    function_text = (
        "int plain(int first, int second, int third) { "
        "int pair = first + second; int bounded = third < 0 ? 0 : third; "
        "return pair + bounded; }"
    )
    source.write_text(f"{function_text}\n", encoding="utf-8")
    index = ip.ProjectIndex()
    index.add_function(
        ip.FunctionDef(
            name="plain",
            qualified_name="plain",
            file=source.name,
            line=1,
            text=function_text,
            callees=[],
        )
    )

    documents = ip.build_training_documents(
        index,
        enriched=True,
        project_dir=str(tmp_path),
        project_id="tests/source-sidecar-port",
        default_args=["-std=c++20"],
        default_build_info={
            "build_system": "cmake",
            "source": "build_files",
            "compiler": "clang++",
            "standard": "c++20",
        },
    )

    assert len(documents) == 1
    assert documents[0]["build_info"]["standard"] == "c11"
    assert documents[0]["language_info"]["primary_language"] == "c"
    assert documents[0]["language_info"]["primary_standard"] == "c11"


def test_recovery_include_discovery_includes_vex_public_headers(
    tmp_path: Path,
) -> None:
    vex_public = tmp_path / "VEX" / "pub"
    vex_public.mkdir(parents=True)
    (vex_public / "libvex_basictypes.h").write_text(
        "typedef unsigned long UWord;\n",
        encoding="utf-8",
    )
    source = tmp_path / "drd" / "client.h"
    source.parent.mkdir()
    source.write_text(
        '#include "libvex_basictypes.h"\n',
        encoding="utf-8",
    )

    assert str(vex_public) in ip._discover_recovery_include_dirs(str(tmp_path))
    recovered_args, added_dirs, unresolved = ip._include_recovery_args(
        ["-x", "c-header"],
        filepath=str(source),
        project_dir=str(tmp_path),
        include_names=["libvex_basictypes.h"],
    )
    assert added_dirs == ["VEX/pub"]
    assert recovered_args[-1] == f"-I{vex_public}"
    assert unresolved == []


def test_unknown_libclang_cursor_kind_is_opaque_not_fatal() -> None:
    class UnknownCursor:
        _kind_id = 437

        @property
        def kind(self):
            raise ValueError("Unknown template argument kind 437")

    cursor = UnknownCursor()
    assert ip._cursor_kind(cursor) is None
    assert ip._cursor_kind_name(cursor) == "UNKNOWN_CURSOR_437"
    assert ip._bucket_clang_cursor_kind(ip._cursor_kind(cursor)) == 255


def test_document_symbol_registry_keeps_boundary_only_symbol_claims() -> None:
    index = ip.ProjectIndex()
    function = ip.FunctionDef(
        name="declaration_only",
        qualified_name="api::declaration_only",
        file="include/api.hpp",
        line=7,
        text="int declaration_only();",
        callees=[],
        symbol_key="v3|project=tests/clang-indexer-regressions|file=include/api.hpp|line=7|kind=FUNCTION_DECL|signature=int declaration_only()",
        canonical_signature="int declaration_only()",
        symbol_kind="FUNCTION_DECL",
        is_definition=False,
    )
    index.add_function(function)

    records = ip._document_symbol_identities(
        [ip._function_part(function)],
        index,
        [],
        [],
        [],
        source="boundary-only regression",
    )

    assert records == [
        {
            "symbol_id": function.symbol_id,
            "symbol_key": function.symbol_key,
        }
    ]


def test_semantic_metadata_uses_name_tokens_and_template_type_edges(
    tmp_path: Path,
) -> None:
    source = """\
namespace api {
template<class T> struct Box { T value; };
int plus(int value) { return value + 1; }
}
int caller(api::Box<int> box) { return api::plus(box.value); }
"""
    source_path = tmp_path / "ranges.cpp"
    source_path.write_text(source, encoding="utf-8")

    functions, types = ip.parse_translation_unit(
        str(source_path),
        _clang_index(),
        ["-std=c++20"],
        str(tmp_path),
        project_id="tests/clang-indexer-regressions",
    )
    caller = next(function for function in functions if function.name == "caller")
    plus = next(function for function in functions if function.name == "plus")
    box_type = next(type_def for type_def in types if type_def.name == "Box")

    caller_name = caller.text.index("caller")
    caller_open_paren = caller_name + len("caller")
    assert caller.semantic_symbol_ids[caller_name] == caller.symbol_id
    assert caller.semantic_symbol_ids[caller_open_paren] == 0
    assert caller.semantic_def_use[caller_open_paren] == ip.DEF_USE_NONE

    call_name = caller.text.index("plus")
    call_open_paren = call_name + len("plus")
    argument_name = caller.text.index("box.value")
    assert set(caller.semantic_call_targets[call_name:call_open_paren]) == {
        plus.symbol_id
    }
    assert caller.semantic_call_targets[call_open_paren] == 0
    assert caller.semantic_call_targets[argument_name] == 0

    namespace_use = caller.text.index("api::Box")
    box_use = namespace_use + len("api::")
    template_open = box_use + len("Box")
    assert caller.semantic_type_refs[namespace_use] == 0
    assert caller.semantic_type_refs[box_use] == box_type.symbol_id
    assert caller.semantic_type_refs[template_open] == 0
    assert caller.referenced_type_keys == [box_type.symbol_key]


def test_offline_semantic_fallback_marks_only_definition_name() -> None:
    text = "int fallback(int value) { return value; }"
    function = ip.FunctionDef(
        name="fallback",
        qualified_name="fallback",
        file="src/fallback.cpp",
        line=1,
        text=text,
        callees=[],
    )
    index = ip.ProjectIndex()
    index.add_function(function)

    doc = ip.build_enriched_doc(
        [ip._function_part(function)],
        index,
        filepath=function.file,
        project_id="tests/clang-indexer-regressions",
    )
    name_start = text.index("fallback")
    name_positions = set(range(name_start, name_start + len("fallback")))

    assert {
        position
        for position, symbol_id in enumerate(doc["symbol_ids"])
        if symbol_id == function.symbol_id
    } == name_positions
    assert {
        position
        for position, def_use in enumerate(doc["def_use"])
        if def_use == ip.DEF_USE_DEF
    } == name_positions


@pytest.mark.parametrize("standard", [20, 23, 26])
def test_header_parser_preserves_per_project_standard(
    tmp_path: Path,
    standard: int,
) -> None:
    repo = tmp_path / f"cmake-cxx{standard}"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(demo CXX)\n"
        f"set(CMAKE_CXX_STANDARD {standard})\n",
        encoding="utf-8",
    )
    header = repo / "api.hpp"
    header.write_text("struct Api {};\n", encoding="utf-8")

    default_args = ip._sanitize_compile_args_for_clang(
        ip.get_default_compile_args(str(repo))
    )
    static_args = ip._resolve_file_args(str(header), None, default_args)
    commit_args = pc._analysis_compile_args(
        str(header),
        default_args,
        str(repo),
    )

    assert static_args[:2] == ["-x", "c++-header"]
    assert commit_args[:2] == ["-x", "c++-header"]
    assert _std_flags(static_args) == [f"-std=c++{standard}"]
    assert _std_flags(commit_args) == [f"-std=c++{standard}"]


def test_header_parser_preserves_compile_commands_cpp26(tmp_path: Path) -> None:
    repo = tmp_path / "compile-db"
    repo.mkdir()
    source = repo / "main.cpp"
    header = repo / "api.hpp"
    source.write_text('#include "api.hpp"\n', encoding="utf-8")
    header.write_text("struct Api {};\n", encoding="utf-8")
    (repo / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(repo),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        "-std=c++26",
                        "-I",
                        str(repo),
                        "-c",
                        str(source),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    _platform, default_args, compile_index = ip.detect_build_context(str(repo))
    assert compile_index is not None
    static_args = ip._resolve_file_args(
        str(header),
        ip.load_compile_commands(str(repo)),
        ip._sanitize_compile_args_for_clang(default_args),
    )
    resolver = pc.BuildContextResolver(repo_root=str(repo))
    repo_root, commit_compile_args, _build_info = resolver.resolve(
        {"filepath": "api.hpp"}
    )
    commit_args = pc._analysis_compile_args(
        str(header),
        commit_compile_args,
        repo_root,
    )

    assert _std_flags(static_args) == ["-std=c++26"]
    assert _std_flags(commit_args) == ["-std=c++26"]


def test_macro_scanner_ignores_comments_and_string_literals(tmp_path: Path) -> None:
    source = r'''/*
#define COMMENTED_OUT 1
*/
const char *ordinary = "#define QUOTED 1";
const char *raw = R"tag(
#define RAW_STRING 1
)tag";
// continued comment \
#define LINE_COMMENT_CONTINUATION 1
// continued comment with two slashes \\
#define DOUBLE_SLASH_CONTINUATION 1
#define REAL_MACRO(x) ((x) + 1)
'''
    blocks = ip.extract_macro_blocks(source)
    assert [name for _start, _end, name, _text in blocks] == ["REAL_MACRO"]

    header = tmp_path / "api.hpp"
    header.write_text(source, encoding="utf-8")
    index = ip.ProjectIndex()
    ip.register_header_macros(
        index,
        [str(header)],
        project_dir=str(tmp_path),
        project_id="tests/clang-indexer-regressions",
    )
    assert set(index.macros_by_name) == {"REAL_MACRO"}


def test_macro_scanner_keeps_raw_and_masked_lines_aligned_across_form_feed(
    tmp_path: Path,
) -> None:
    header = tmp_path / "bsd_page.h"
    header.write_text(
        "#define BEFORE 1\n"
        "/* old BSD page\fbreak */\n"
        "#define AFTER(x) ((x) + BEFORE)\n"
        "int use_after(void) { return AFTER(1); }\n",
        encoding="utf-8",
    )

    index = ip.ProjectIndex()
    ip.register_header_macros(
        index,
        [str(header)],
        project_dir=str(tmp_path),
        project_id="tests/clang-indexer-regressions",
    )

    assert set(index.macros_by_name) == {"BEFORE", "AFTER"}
    assert [
        name
        for _start, _end, name, _text in ip.extract_macro_blocks(
            header.read_text(encoding="utf-8")
        )
    ] == ["BEFORE", "AFTER"]


def test_macro_references_ignore_comments_and_literals() -> None:
    index = ip.ProjectIndex()
    macro = ip.MacroDef(
        "ROUTE",
        "api.hpp",
        1,
        "#define ROUTE(x) (x)\n",
        ["x"],
        project_id="tests/clang-indexer-regressions",
        visible_in_file="api.cpp",
        sequence=1,
    )
    index.add_macro(macro)
    non_code = r'''// ROUTE(1)
/* ROUTE(2) */
const char *ordinary = "ROUTE(3)";
const char *raw = R"tag(ROUTE(4))tag";
'''

    assert ip._used_macro_defs(index, [non_code], target_file="api.cpp") == []
    assert ip._macro_invocation_route_parts(
        non_code,
        offset=0,
        index=index,
        target_file="api.cpp",
        start_line=1,
    ) == []
    assert ip._macro_body_dependency_names(
        '#define OUTER "ROUTE" /* ROUTE */\n',
        macro_name="OUTER",
    ) == []

    mixed = non_code + "int value = ROUTE(5);\n"
    assert ip._used_macro_defs(index, [mixed], target_file="api.cpp") == [macro]
    routes = ip._macro_invocation_route_parts(
        mixed,
        offset=0,
        index=index,
        target_file="api.cpp",
        start_line=1,
    )
    assert routes == [
        {
            "name": "ROUTE",
            "start": mixed.rindex("ROUTE"),
            "end": mixed.rindex("ROUTE") + len("ROUTE"),
            "target_sequence": macro.sequence,
        }
    ]


def test_inline_namespace_normalization_is_consistent_in_local_records() -> None:
    common = {
        "qname": "api::consume",
        "kind": "FUNCTION_DECL",
        "project": "tests/clang-indexer-regressions",
        "file": "api.cpp",
    }
    public_identity = ip.canonical_symbol_identity(
        **common,
        canonical_signature="void (std::vector<int>)",
    )
    inline_identity = ip.canonical_symbol_identity(
        **common,
        canonical_signature="void (std::__1::vector<int>)",
    )
    assert inline_identity == public_identity

    function = ip.FunctionDef(
        name="size",
        qualified_name="std::__1::vector::size",
        file="vector.hpp",
        line=1,
        text="int size() const { return 0; }",
        callees=["std::__cxx11::helper"],
        referenced_types=["std::__1::vector"],
    )
    class_def = pc.ClassDef(
        name="vector",
        qualified_name="std::__1::vector",
        text="struct vector { int size; };",
        start_line=1,
        end_line=1,
    )

    assert function.qualified_name == "std::vector::size"
    assert function.callees == ["std::helper"]
    assert function.referenced_types == ["std::vector"]
    assert class_def.qualified_name == "std::vector"


def test_macro_registry_retains_only_used_closure_across_many_roots(
    tmp_path: Path,
) -> None:
    common = tmp_path / "common.hpp"
    macro_count = 128
    root_count = 48
    common.write_text(
        "".join(
            f"#define UNUSED_{index}(x) ((x) + {index})\n"
            for index in range(macro_count - 1)
        )
        + "#define HOT_MACRO(x) ((x) + 1)\n",
        encoding="utf-8",
    )
    roots = []
    for index in range(root_count):
        root = tmp_path / f"root_{index}.cpp"
        root.write_text(
            '#include "common.hpp"\n'
            f"int root_{index}() {{ return HOT_MACRO({index}); }}\n",
            encoding="utf-8",
        )
        roots.append(str(root))

    index = ip.ProjectIndex()
    stats = ip.register_header_macros(
        index,
        [*roots, *roots],
        project_dir=str(tmp_path),
        project_id="tests/clang-indexer-regressions",
        include_dirs=[str(tmp_path)],
        directive_cache_entries=4,
        resolve_cache_entries=8,
        max_macro_candidates_per_root=macro_count + 1,
        max_retained_macros=root_count + 1,
    )

    assert stats["discovered_macro_occurrences"] == root_count * macro_count
    assert stats["registered_macros"] == root_count
    assert stats["pruned_macro_occurrences"] == root_count * (macro_count - 1)
    assert stats["peak_root_macro_candidates"] == 1
    assert stats["peak_root_relevant_macro_names"] < 16
    assert stats["peak_root_retained_macros"] == 1
    assert stats["directive_cache_peak_entries"] <= 4
    assert stats["resolve_cache_peak_entries"] <= 8
    assert stats["skipped_duplicate_roots"] == root_count
    assert set(index.macros_by_name) == {"HOT_MACRO"}
    assert len(index.macro_definitions) == root_count


def test_macro_registry_candidate_limit_fails_loud(tmp_path: Path) -> None:
    header = tmp_path / "many.hpp"
    header.write_text(
        "".join(f"#define VALUE_{index} {index}\n" for index in range(8)),
        encoding="utf-8",
    )

    with pytest.raises(MemoryError, match="per-root candidate bound"):
        ip.register_header_macros(
            ip.ProjectIndex(),
            [str(header)],
            project_dir=str(tmp_path),
            project_id="tests/clang-indexer-regressions",
            max_macro_candidates_per_root=4,
        )


def test_case3_indexer_output_has_exact_v3_source_and_symbol_provenance(
    tmp_path: Path,
) -> None:
    docs: list[dict[str, object]] = []
    project_id = "tests/case3-prompt-repo"
    ip.process_project(
        str(CASE3_FIXTURE),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id=project_id,
    )

    assert docs
    assert any(doc["doc_type"] == "build" for doc in docs)
    assert any(len(doc["source_identity_registry"]) > 1 for doc in docs)
    for doc in docs:
        assert doc["symbol_identity_schema_version"] == 3
        assert "symbol_identities" in doc
        if doc["doc_type"] in {"build", "shell", "diagnostic"}:
            assert doc["symbol_identities"] == []
        assert doc["repo"] == project_id
        assert doc["repo_stable_id"] == ip._stable_repo_id(project_id)
        assert doc["filepath_stable_id"] == ip._stable_filepath_id(
            project_id,
            str(doc["filepath"]),
        )

        registry = {
            int(entry["source_identity_id"]): json.loads(str(entry["source"]))
            for entry in doc["source_identity_registry"]
        }
        identity_ids = [int(value) for value in doc["domain_source_identity_ids"]]
        assert len(identity_ids) == len(str(doc["text"]))
        assert identity_ids and min(identity_ids) > 0
        assert set(identity_ids) <= set(registry)
        for identity in registry.values():
            assert identity["repo"] == project_id
            assert identity["repo_stable_id"] == ip._stable_repo_id(project_id)
            assert identity["filepath_stable_id"] == ip._stable_filepath_id(
                project_id,
                identity["filepath"],
            )


def test_case3_mixed_indexer_batch_reaches_converter_parquet(tmp_path: Path) -> None:
    from scripts.nanochat_data import clang_enriched_to_parquet as converter
    import pyarrow.parquet as pq

    docs: list[dict[str, object]] = []
    ip.process_project(
        str(CASE3_FIXTURE),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/case3-prompt-repo",
    )
    input_path = tmp_path / "case3.jsonl"
    output_path = tmp_path / "case3.parquet"
    input_path.write_text(
        "".join(json.dumps(doc) + "\n" for doc in docs),
        encoding="utf-8",
    )
    summary = converter.convert_local_jsonl_to_parquet(
        input_path,
        output_path,
        tokenizer=converter.load_tokenizer(
            str(ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json")
        ),
        max_tokens=16384,
        overflow_policy="drop",
        materialize_tokenized_enriched=True,
    )
    table = pq.read_table(output_path)

    assert summary["docs_in"] == len(docs)
    assert summary["docs_out"] == len(docs)
    assert summary["source_docs_emitted"] == len(docs)
    assert summary["dropped_input_docs"] == 0
    assert summary["materialized_rows"] == len(docs)
    assert table.num_rows == len(docs)
    assert table.schema.metadata[
        converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    ] == b"3"


def test_case3_production_token_spans_have_exact_source_identity() -> None:
    from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched
    from scripts.nanochat_data import clang_enriched_to_parquet as converter

    docs: list[dict[str, object]] = []
    ip.process_project(
        str(CASE3_FIXTURE),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/case3-prompt-repo",
    )
    source_doc = next(
        doc for doc in docs if len(doc["source_identity_registry"]) > 1
    )
    tokenizer = converter.load_tokenizer(
        str(ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json")
    )
    converted = converter.process_record_with_policy(
        source_doc,
        tokenizer,
        max_tokens=16384,
        overflow_policy="drop",
    )[0]
    token_ids, token_spans = (
        tokenized_enriched._encode_batch_with_optional_char_spans(
            tokenizer,
            [str(converted["text"])],
            prepend=tokenizer.bos_token_id,
        )
    )
    projected = tokenized_enriched._chars_to_tokens_structure_ids(
        converted["domain_source_identity_ids"],
        "",
        token_spans[0],
    )
    registry_ids = {
        int(entry["source_identity_id"])
        for entry in converted["source_identity_registry"]
    }

    assert len(projected) == len(token_ids[0])
    assert [
        (index, span)
        for index, (identity_id, span) in enumerate(
            zip(projected, token_spans[0], strict=True)
        )
        if identity_id == 0
    ] == [(0, (0, 0))]
    assert {
        identity_id
        for identity_id, (start, end) in zip(
            projected,
            token_spans[0],
            strict=True,
        )
        if end > start
    } <= registry_ids
