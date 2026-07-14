from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = REPO_ROOT / "tools" / "clang_indexer"
for path in (REPO_ROOT, CLANG_INDEXER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.clang_indexer import index_project as ip  # noqa: E402
from tools.clang_indexer import process_commits as pc  # noqa: E402
from scripts.crossrepo import build_global_symbol_index as gsi  # noqa: E402


def _clang_index():
    pytest.importorskip("clang.cindex")
    ip._configure_libclang()
    return ip.Index.create()


def _parse_project(tmp_path: Path, files: dict[str, str]) -> tuple[ip.ProjectIndex, dict[str, list]]:
    clang_index = _clang_index()
    project = ip.ProjectIndex()
    parsed: dict[str, list] = {}
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        functions, types = ip.parse_translation_unit(
            str(path),
            clang_index,
            ["-std=c++20"],
            str(tmp_path),
            project_id="tests/clang-fixture",
        )
        parsed[relative] = functions
        for function in functions:
            project.add_function(function)
        for type_def in types:
            project.add_typedef(type_def)
    project.compute_dep_levels()
    return project, parsed


def _function(functions: list, *, name: str, signature_fragment: str):
    return next(
        function
        for function in functions
        if function.name == name and signature_fragment in function.canonical_signature
    )


def test_canonical_identity_namespaces_usr_and_fallback_uses_normalized_qname() -> None:
    shared = {
        "qname": "api::route",
        "kind": "FUNCTION_DECL",
        "usr": "c:@N@api@F@route#I#",
        "canonical_signature": "display=route(int)|type=int (int)",
    }
    repo_a = ip.canonical_symbol_identity(
        **shared, project="owner/repo-a", file="a.cpp"
    )
    repo_b = ip.canonical_symbol_identity(
        **shared, project="owner/repo-b", file="b.cpp"
    )
    repo_a_other_file = ip.canonical_symbol_identity(
        **shared, project="owner/repo-a", file="other.cpp"
    )

    assert repo_a != repo_b
    assert repo_a == repo_a_other_file

    fallback = {
        "kind": "FUNCTION_DECL",
        "canonical_signature": "display=route(int)|type=int (int)",
        "project": "owner/repo-a",
        "file": "route.cpp",
    }
    assert ip.canonical_symbol_identity(qname="left::route", **fallback) != (
        ip.canonical_symbol_identity(qname="right::route", **fallback)
    )
    assert ip.canonical_symbol_identity(qname=" left :: route ", **fallback) == (
        ip.canonical_symbol_identity(qname="left::route", **fallback)
    )
    assert ip.canonical_symbol_identity(
        qname="api::route",
        **{**fallback, "canonical_signature": "display=route(double)|type=double (double)"},
    ) != ip.canonical_symbol_identity(qname="api::route", **fallback)


def test_fallback_identity_and_symbol_ids_separate_qnames_and_overloads() -> None:
    shared = {
        "kind": "FUNCTION_DECL",
        "project": "owner/repo-a",
        "file": "route.cpp",
    }
    left_int = ip.canonical_symbol_identity(
        qname="left::route",
        canonical_signature="display=route(int)|type=int (int)",
        **shared,
    )
    right_int = ip.canonical_symbol_identity(
        qname="right::route",
        canonical_signature="display=route(int)|type=int (int)",
        **shared,
    )
    left_double = ip.canonical_symbol_identity(
        qname="left::route",
        canonical_signature="display=route(double)|type=double (double)",
        **shared,
    )

    assert len({left_int, right_int, left_double}) == 3
    symbol_ids = {ip._compute_symbol_id(key) for key in (left_int, right_int, left_double)}
    assert len(symbol_ids) == 3
    assert max(symbol_ids) > 0xFFFFFFFF
    assert all(0 < symbol_id <= 0xFFFFFFFFFFFFFFFF for symbol_id in symbol_ids)


def test_strong_usr_and_signature_identity_ids_remain_stable() -> None:
    usr_key = ip.canonical_symbol_identity(
        qname="api::route",
        kind="FUNCTION_DECL",
        usr="c:@N@api@F@route#I#",
        canonical_signature="display=route(int)|type=int (int)",
        project="owner/repo",
        file="one.cpp",
        line=7,
    )
    signature_key = ip.canonical_symbol_identity(
        qname="api::route",
        kind="FUNCTION_DECL",
        canonical_signature="display=route(int)|type=int (int)",
        file="../sdk/route.hpp",
        line=7,
    )

    assert usr_key == (
        "usr:schema=v3\x1fproject=owner/repo\x1fusr=c:@N@api@F@route#I#"
    )
    assert ip._compute_symbol_id(usr_key) == 9970187544792576286
    assert signature_key == (
        "fallback:schema=v3\x1fqname=api::route\x1fkind=FUNCTION_DECL\x1f"
        "sig=display=route(int)|type=int (int)"
    )
    assert ip._compute_symbol_id(signature_key) == 8106072181455702189


def _fallback_cursor(
    path: Path,
    *,
    storage: str = "NONE",
    parent_kind: str = "NAMESPACE",
    with_signature: bool = True,
    exception_name: str | None = "NONE",
):
    type_info = SimpleNamespace(
        spelling="int (int)" if with_signature else "",
        get_canonical=lambda: type_info,
    )
    result_info = SimpleNamespace(
        spelling="int" if with_signature else "",
        get_canonical=lambda: result_info,
    )
    parent = SimpleNamespace(
        kind=SimpleNamespace(name=parent_kind),
        spelling="api" if parent_kind == "NAMESPACE" else "caller",
        semantic_parent=None,
    )
    return SimpleNamespace(
        kind=SimpleNamespace(name="FUNCTION_DECL"),
        spelling="route",
        displayname="route(int)" if with_signature else "",
        semantic_parent=parent,
        lexical_parent=parent,
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(path)),
            line=7,
            column=4,
        ),
        linkage=SimpleNamespace(name="EXTERNAL"),
        storage_class=SimpleNamespace(name=storage),
        type=type_info,
        result_type=result_info,
        get_arguments=lambda: [],
        get_usr=lambda: "",
        exception_specification_kind=(
            None
            if exception_name is None
            else SimpleNamespace(name=exception_name)
        ),
    )


def test_external_no_usr_fallback_is_global_but_static_and_local_are_file_scoped(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    external_a = _fallback_cursor(tmp_path / "sdk-a" / "route.hpp")
    external_b = _fallback_cursor(tmp_path / "sdk-b" / "route.hpp")
    external_key_a, _, _ = ip.symbol_identity_for_cursor(
        external_a, project_dir=str(project_dir), project="owner/repo"
    )
    external_key_b, _, _ = ip.symbol_identity_for_cursor(
        external_b, project_dir=str(project_dir), project="owner/repo"
    )
    assert external_key_a == external_key_b
    assert "file=" not in external_key_a

    static_keys = []
    local_keys = []
    for name in ("one.cpp", "two.cpp"):
        path = project_dir / name
        static_keys.append(
            ip.symbol_identity_for_cursor(
                _fallback_cursor(path, storage="STATIC"),
                project_dir=str(project_dir),
                project="owner/repo",
            )[0]
        )
        local_keys.append(
            ip.symbol_identity_for_cursor(
                _fallback_cursor(path, parent_kind="FUNCTION_DECL"),
                project_dir=str(project_dir),
                project="owner/repo",
            )[0]
        )
    assert len(set(static_keys)) == 2
    assert len(set(local_keys)) == 2
    assert all("file=" in key for key in static_keys + local_keys)


def test_unknown_external_without_usr_or_signature_fails_closed(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    external_a = _fallback_cursor(
        tmp_path / "sdk-a" / "route.hpp", with_signature=False
    )
    external_b = _fallback_cursor(
        tmp_path / "sdk-b" / "route.hpp", with_signature=False
    )

    for cursor in (external_a, external_b):
        with pytest.raises(ip.SymbolIdentityError, match="stable provider identity"):
            ip.symbol_identity_for_cursor(
                cursor, project_dir=str(project_dir), project="owner/repo"
            )


def test_provider_location_identity_is_checkout_independent(tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    left = _fallback_cursor(
        tmp_path / "sdk-a" / "libcxx" / "include" / "vector",
        with_signature=False,
    )
    right = _fallback_cursor(
        tmp_path / "sdk-b" / "libcxx" / "include" / "vector",
        with_signature=False,
    )

    left_key, _, _ = ip.symbol_identity_for_cursor(
        left, project_dir=str(project_dir), project="owner/repo"
    )
    right_key, _, _ = ip.symbol_identity_for_cursor(
        right, project_dir=str(project_dir), project="owner/repo"
    )

    assert left_key == right_key
    assert "project=llvm/llvm-project" in left_key
    assert "file=@provider/libc++/vector" in left_key

    reference = ip.symbol_reference_for_cursor(
        left,
        project_dir=str(project_dir),
        project_id="owner/repo",
    )
    assert reference["project"] == "llvm/llvm-project"
    assert reference["file"] == "@provider/libc++/vector"
    assert reference["provider"] == "libc++"
    assert reference["include_provenance"] == "vector"


def test_exception_only_signature_uses_location_fallback(tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    cursor = _fallback_cursor(
        project_dir / "include" / "route.hpp",
        with_signature=False,
        exception_name="UNPARSED",
    )

    key, _usr, signature = ip.symbol_identity_for_cursor(
        cursor,
        project_dir=str(project_dir),
        project="owner/repo",
    )

    assert signature == ""
    assert key.startswith("repo_file_location:")
    assert "file=include/route.hpp" in key


def test_repo_file_location_identity_is_checkout_independent_and_normalized(
    tmp_path: Path,
) -> None:
    checkout_a = tmp_path / "checkout-a" / "repo"
    checkout_b = tmp_path / "checkout-b" / "repo"
    cursor_a = _fallback_cursor(
        checkout_a / "include" / "route.hpp", with_signature=False
    )
    cursor_b = _fallback_cursor(
        checkout_b / "include" / "route.hpp", with_signature=False
    )
    cursor_other_line = _fallback_cursor(
        checkout_b / "include" / "route.hpp", with_signature=False
    )
    cursor_other_line.location.line = 8
    cursor_other_column = _fallback_cursor(
        checkout_b / "include" / "route.hpp", with_signature=False
    )
    cursor_other_column.location.column = 9

    key_a = ip.symbol_identity_for_cursor(
        cursor_a, project_dir=str(checkout_a), project="owner/repo"
    )[0]
    key_b = ip.symbol_identity_for_cursor(
        cursor_b, project_dir=str(checkout_b), project="owner/repo"
    )[0]
    key_other_line = ip.symbol_identity_for_cursor(
        cursor_other_line,
        project_dir=str(checkout_b),
        project="owner/repo",
    )[0]
    key_other_column = ip.symbol_identity_for_cursor(
        cursor_other_column,
        project_dir=str(checkout_b),
        project="owner/repo",
    )[0]
    windows_key = ip.symbol_identity_for_cursor(
        _fallback_cursor(
            Path(r"C:\checkout\repo\include\route.hpp"),
            with_signature=False,
        ),
        project_dir=r"C:\checkout\repo",
        project="owner/repo",
    )[0]

    assert key_a == key_b == windows_key
    assert key_a != key_other_line
    assert key_a != key_other_column
    assert key_a.startswith("repo_file_location:")
    assert "project=owner/repo" in key_a
    assert "file=include/route.hpp" in key_a
    assert "line=7" in key_a
    assert "column=4" in key_a
    assert str(tmp_path) not in key_a
    assert "\\" not in key_a

    windows_casefolded = ip.symbol_identity_for_cursor(
        _fallback_cursor(
            Path(r"c:\\CHECKOUT\\REPO\\INCLUDE\\ROUTE.hpp"),
            with_signature=False,
        ),
        project_dir=r"C:\\checkout\\repo",
        project="owner/repo",
    )[0]
    assert windows_casefolded == windows_key


def test_usr_extraction_error_fails_loud(tmp_path: Path) -> None:
    cursor = _fallback_cursor(tmp_path / "repo" / "route.hpp", with_signature=False)

    def raise_usr_error():
        raise RuntimeError("synthetic get_usr failure")

    cursor.get_usr = raise_usr_error

    with pytest.raises(ip.SymbolIdentityError, match="clang USR extraction failed"):
        ip.symbol_identity_for_cursor(
            cursor,
            project_dir=str(tmp_path / "repo"),
            project="owner/repo",
        )


def test_canonical_signature_extraction_error_fails_loud(tmp_path: Path) -> None:
    cursor = _fallback_cursor(tmp_path / "repo" / "route.hpp")

    def raise_canonical_error():
        raise RuntimeError("synthetic canonical type failure")

    cursor.type = SimpleNamespace(
        spelling="int (int)",
        get_canonical=raise_canonical_error,
    )

    with pytest.raises(
        ip.SymbolIdentityError,
        match="clang canonical signature extraction failed",
    ):
        ip.symbol_identity_for_cursor(
            cursor,
            project_dir=str(tmp_path / "repo"),
            project="owner/repo",
        )


def test_corpus_registry_fails_closed_on_cross_project_symbol_id_collision() -> None:
    first_key = ip.canonical_symbol_identity(
        qname="left::route",
        kind="FUNCTION_DECL",
        canonical_signature="display=route(int)|type=int (int)",
        project="owner/repo-a",
        file="route.cpp",
    )
    second_key = ip.canonical_symbol_identity(
        qname="right::route",
        kind="FUNCTION_DECL",
        canonical_signature="display=route(int)|type=int (int)",
        project="owner/repo-b",
        file="route.cpp",
    )
    claimed_id = ip._compute_symbol_id(first_key)
    registry = ip.SymbolIdentityRegistry()
    registry.register(first_key, symbol_id=claimed_id, source="owner/repo-a")

    with pytest.raises(RuntimeError, match="symbol ID collision.*owner/repo-a.*owner/repo-b"):
        registry.register(second_key, symbol_id=claimed_id, source="owner/repo-b")


def test_qname_only_part_never_synthesizes_semantic_identity() -> None:
    text = "int route(int value) { return value; }"
    metadata = ip.extract_semantic_metadata_from_parts(
        text,
        [(text, 2, 0, "route", "api::route")],
        ip.ProjectIndex(),
    )

    assert set(metadata["symbol_ids"]) == {0}
    assert set(metadata["call_targets"]) == {0}
    assert set(metadata["type_refs"]) == {0}


def test_static_indexer_surfaces_translation_unit_parse_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "broken.cpp"
    source.write_text("int broken(int value) { return value; }", encoding="utf-8")

    class BrokenIndex:
        def parse(self, *_args, **_kwargs):
            raise RuntimeError("synthetic clang failure")

    monkeypatch.setattr(ip, "TranslationUnit", SimpleNamespace(PARSE_INCOMPLETE=0))
    with pytest.raises(RuntimeError, match=r"libclang parse failed.*broken\.cpp"):
        ip.parse_translation_unit(
            str(source),
            BrokenIndex(),
            ["-std=c++20"],
            str(tmp_path),
            project_id="owner/repo-a",
        )


def test_real_clang_usr_separates_overloads_templates_and_specializations(tmp_path: Path) -> None:
    source = """
namespace api {
int route(int value) { return value + 1; }
double route(double value) { return value + 0.5; }
template<class T> T route(T value) { return value; }
template<> long route<long>(long value) { return value + 2; }
int caller(int value) { return route(value); }
}
"""
    project, parsed = _parse_project(tmp_path, {"route.cpp": source})
    routes = [function for function in parsed["route.cpp"] if function.name == "route"]

    assert len(routes) == 4
    assert len({function.usr for function in routes}) == 4
    assert len({function.symbol_key for function in routes}) == 4
    assert len({function.symbol_id for function in routes}) == 4
    assert len(project.function_qnames["api::route"]) == 4
    assert project.resolve_function_key("api::route") is None

    caller = next(function for function in parsed["route.cpp"] if function.name == "caller")
    int_route = _function(routes, name="route", signature_fragment="args=(int)")
    assert caller.callee_keys == [int_route.symbol_key]
    assert int_route.symbol_key in project._function_callee_keys(caller)


def test_real_clang_usr_separates_cv_ref_noexcept_and_conversion_operators(
    tmp_path: Path,
) -> None:
    source = r"""
namespace api {
struct Box {
  int value;
  int read() & { return value; }
  int read() const & noexcept { return value; }
  int read() && { return value + 1; }
  explicit operator int() const & noexcept { return value; }
  explicit operator double() && { return static_cast<double>(value); }
  template<class T> T cast(T value) const noexcept { return static_cast<T>(value); }
};
}
"""
    _project, parsed = _parse_project(tmp_path, {"operators.cpp": source})
    methods = [
        function
        for function in parsed["operators.cpp"]
        if function.name in {"read", "operator int", "operator double", "cast"}
    ]
    reads = [function for function in methods if function.name == "read"]
    conversions = [
        function for function in methods if function.name.startswith("operator ")
    ]
    assert len(reads) == 3
    assert len({function.usr for function in reads}) == 3
    assert len({function.symbol_key for function in reads}) == 3
    assert len(conversions) == 2
    assert len({function.usr for function in conversions}) == 2
    assert any("NOEXCEPT" in function.canonical_signature.upper() for function in methods)


def test_real_clang_same_usr_is_project_scoped_across_repositories(tmp_path: Path) -> None:
    source = "namespace api { int route(int value) { return value + 1; } }\n"
    clang_index = _clang_index()
    parsed = []
    for project_id in ("owner/repo-a", "owner/repo-b"):
        project_dir = tmp_path / project_id.rsplit("/", 1)[-1]
        project_dir.mkdir()
        path = project_dir / "route.cpp"
        path.write_text(source, encoding="utf-8")
        functions, _types = ip.parse_translation_unit(
            str(path),
            clang_index,
            ["-std=c++20"],
            str(project_dir),
            project_id=project_id,
        )
        parsed.append(next(function for function in functions if function.name == "route"))

    assert parsed[0].usr == parsed[1].usr
    assert parsed[0].symbol_key != parsed[1].symbol_key
    assert parsed[0].symbol_id != parsed[1].symbol_id


def test_real_clang_identity_keeps_inline_namespace_static_and_local_symbols_distinct(
    tmp_path: Path,
) -> None:
    source_template = """
namespace api {{
inline namespace {inline_ns} {{
struct Box {{ int value; }};
int versioned(Box box) {{ return box.value; }}
}}
static int hidden(int value) {{ return value + {increment}; }}
int caller() {{
    int local = {increment};
    return hidden(local);
}}
}}
"""
    project, parsed = _parse_project(
        tmp_path,
        {
            "one.cpp": source_template.format(inline_ns="v1", increment=1),
            "two.cpp": source_template.format(inline_ns="v2", increment=2),
        },
    )

    hidden = [
        function
        for functions in parsed.values()
        for function in functions
        if function.name == "hidden"
    ]
    callers = [
        function
        for functions in parsed.values()
        for function in functions
        if function.name == "caller"
    ]
    versioned = [
        function
        for functions in parsed.values()
        for function in functions
        if function.name == "versioned"
    ]

    assert len({function.symbol_key for function in hidden}) == 2
    assert len(project.function_qnames["api::hidden"]) == 2
    assert project.resolve_function_key("api::hidden") is None
    assert {function.qualified_name for function in versioned} == {
        "api::v1::versioned",
        "api::v2::versioned",
    }

    for caller, hidden_function in zip(callers, hidden, strict=True):
        assert caller.callee_keys == [hidden_function.symbol_key]
        local_offset = caller.text.index("local")
        local_id = caller.semantic_symbol_ids[local_offset]
        assert local_id != 0
    assert callers[0].semantic_symbol_ids[callers[0].text.index("local")] != callers[
        1
    ].semantic_symbol_ids[callers[1].text.index("local")]


def test_exact_clang_ranges_route_overloaded_calls_and_type_uses(tmp_path: Path) -> None:
    source = """
namespace api {
struct Item { int value; };
int convert(int value) { return value + 1; }
double convert(double value) { return value + 0.5; }
int caller(Item item) {
    double floating = convert(1.0);
    return convert(item.value) + static_cast<int>(floating);
}
}
"""
    _project, parsed = _parse_project(tmp_path, {"ranges.cpp": source})
    functions = parsed["ranges.cpp"]
    caller = next(function for function in functions if function.name == "caller")
    int_convert = _function(functions, name="convert", signature_fragment="args=(int)")
    double_convert = _function(functions, name="convert", signature_fragment="args=(double)")

    first_call = caller.text.index("convert")
    second_call = caller.text.index("convert", first_call + 1)
    assert caller.semantic_call_targets[first_call] == double_convert.symbol_id
    assert caller.semantic_call_targets[second_call] == int_convert.symbol_id
    assert caller.semantic_call_targets[first_call] != caller.semantic_call_targets[second_call]

    item_type = next(type_def for type_def in _project.typedefs.values() if type_def.name == "Item")
    item_use = caller.text.index("Item")
    assert caller.semantic_type_refs[item_use] == item_type.symbol_id
    assert len(caller.semantic_symbol_ids) == len(caller.text)
    assert len(caller.semantic_call_targets) == len(caller.text)
    assert len(caller.semantic_type_refs) == len(caller.text)
    assert len(caller.semantic_def_use) == len(caller.text)


def test_macro_identity_is_definition_scoped_and_part_metadata_is_versioned(tmp_path: Path) -> None:
    first = ip.MacroDef(
        "ROUTE", "one.hpp", 4, "#define ROUTE(x) (x)", ["x"],
        project_id="owner/repo",
    )
    second = ip.MacroDef(
        "ROUTE", "two.hpp", 4, "#define ROUTE(x) ((x) + 1)", ["x"],
        project_id="owner/repo",
    )
    repo_a = ip.MacroDef(
        "ROUTE",
        "include/route.hpp",
        4,
        "#define ROUTE(x) (x)",
        ["x"],
        project_id="owner/repo-a",
    )
    repo_b = ip.MacroDef(
        "ROUTE",
        "include/route.hpp",
        4,
        "#define ROUTE(x) (x)",
        ["x"],
        project_id="owner/repo-b",
    )

    assert first.symbol_key != second.symbol_key
    assert first.symbol_id != second.symbol_id
    assert repo_a.symbol_key != repo_b.symbol_key
    assert repo_a.symbol_id != repo_b.symbol_id

    part = ip._macro_part(first)
    metadata = ip._part_symbol_metadata(part)
    assert metadata is not None
    assert metadata["symbol_key"] == first.symbol_key
    assert metadata["symbol_id"] == first.symbol_id
    assert metadata["symbol_identity_schema_version"] == ip.SYMBOL_IDENTITY_SCHEMA_VERSION
    assert ip._part_macro_provenance(part)["file"] == "one.hpp"


def test_commit_old_new_matching_uses_usr_and_preserves_exact_call_targets(tmp_path: Path) -> None:
    old_source = """
namespace api {
int route(int value) { return value + 1; }
double route(double value) { return value + 0.5; }
int caller(int value) { return route(value); }
}
"""
    new_source = old_source.replace("value + 1", "value + 2")
    clang_index = _clang_index()
    old_analysis = pc.analyze_file_clang(
        old_source,
        "route.cpp",
        clang_index,
        str(tmp_path / "old"),
        compile_args=["-std=c++20"],
        repo_root=str(tmp_path),
        build_info={},
        project_id="owner/repo",
    )
    new_analysis = pc.analyze_file_clang(
        new_source,
        "route.cpp",
        clang_index,
        str(tmp_path / "new"),
        compile_args=["-std=c++20"],
        repo_root=str(tmp_path),
        build_info={},
        project_id="owner/repo",
    )

    old_routes = [function for function in old_analysis.functions if function.name == "route"]
    new_routes = [function for function in new_analysis.functions if function.name == "route"]
    assert len(old_routes) == len(new_routes) == 2
    assert {function.symbol_key for function in old_routes} == {
        function.symbol_key for function in new_routes
    }
    assert len({function.symbol_key for function in old_routes}) == 2

    record = {
        "repo": "owner/repo",
        "filepath": "route.cpp",
        "old_content": old_source,
        "new_content": new_source,
        "diff": "@@ -3,1 +3,1 @@\n-int route(int value) { return value + 1; }\n+int route(int value) { return value + 2; }",
        "subject": "change int overload",
    }
    doc = pc.format_chain_document(
        record,
        old_analysis,
        new_analysis,
        pc.parse_hunk_ranges(record["diff"]),
    )
    assert doc is not None
    assert doc["symbol_identity_schema_version"] == ip.SYMBOL_IDENTITY_SCHEMA_VERSION
    int_route = _function(new_routes, name="route", signature_fragment="args=(int)")
    assert doc["changed_symbol_ids"] == [int_route.symbol_id]
    assert any(value == int_route.symbol_id for value in doc["symbol_ids"])


def _create_composite_v1_store(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE symbols (
                symbol_uid TEXT NOT NULL PRIMARY KEY,
                qname TEXT NOT NULL,
                base_lib TEXT NOT NULL,
                base_repo TEXT NOT NULL,
                kind INTEGER NOT NULL,
                sym_type TEXT NOT NULL,
                file TEXT NOT NULL,
                line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                is_public INTEGER NOT NULL,
                token_est INTEGER NOT NULL,
                body_len INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE symbol_index_libs (
                lib TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                n_files INTEGER NOT NULL DEFAULT 0,
                n_funcs INTEGER NOT NULL DEFAULT 0,
                n_types INTEGER NOT NULL DEFAULT 0,
                n_errors INTEGER NOT NULL DEFAULT 0,
                elapsed_s REAL NOT NULL DEFAULT 0,
                subtrees TEXT NOT NULL DEFAULT '',
                finished_utc TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.executemany(
            "INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "uid-a",
                    "api::route",
                    "api",
                    "repo-a",
                    2,
                    "func",
                    "a.cpp",
                    1,
                    1,
                    1,
                    8,
                    32,
                    "int route(int value) { return value; }",
                ),
                (
                    "uid-b",
                    "api::route",
                    "api",
                    "repo-b",
                    2,
                    "func",
                    "b.cpp",
                    1,
                    1,
                    1,
                    8,
                    32,
                    "double route(double value) { return value; }",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _global_record(
    *,
    qname: str = "api::route",
    project: str = "owner/repo-a",
    file: str = "route.cpp",
    usr: str = "",
    signature: str = "display=route(int)|type=int (int)|result=int|args=(int)",
    symbol_kind: str = "FUNCTION_DECL",
    base_lib: str = "api",
    provider: str | None = None,
    include_provenance: str | None = None,
    text: str = "int route(int value) { return value; }",
) -> gsi.GlobalSymbolRecord:
    symbol_key = ip.canonical_symbol_identity(
        qname=qname,
        kind=symbol_kind,
        usr=usr,
        canonical_signature=signature,
        project=project,
        file=file,
    )
    return gsi.GlobalSymbolRecord(
        qname=qname,
        base_lib=base_lib,
        base_repo=project,
        kind=2,
        sym_type="func",
        file=file,
        line=1,
        end_line=1,
        is_public=1,
        token_est=8,
        body_len=32,
        text=text,
        symbol_key=symbol_key,
        usr=usr,
        canonical_signature=signature,
        symbol_kind=symbol_kind,
        provider=provider or project,
        include_provenance=include_provenance or file,
    )


def test_global_store_rejects_legacy_schema_without_fabricating_identity(
    tmp_path: Path,
) -> None:
    db = tmp_path / "symbols.sqlite"
    _create_composite_v1_store(db)

    with pytest.raises(ip.SymbolIdentityError, match="cannot be migrated"):
        gsi.GlobalSymbolStore(str(db))

    conn = sqlite3.connect(db)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        metadata_table = conn.execute(
            "SELECT COUNT(*) FROM symbol_index_libs"
        ).fetchone()[0]
    finally:
        conn.close()
    assert version == 0
    assert "symbol_key" not in columns
    assert metadata_table == 0


def test_global_store_fails_closed_on_cross_project_symbol_id_collision(
    tmp_path: Path,
) -> None:
    db = tmp_path / "symbols.sqlite"
    signature = "display=route(int)|type=int (int)|result=int|args=(int)"
    first_key = ip.canonical_symbol_identity(
        qname="left::route",
        kind="FUNCTION_DECL",
        canonical_signature=signature,
        project="owner/repo-a",
        file="route.cpp",
    )
    second_key = ip.canonical_symbol_identity(
        qname="right::route",
        kind="FUNCTION_DECL",
        canonical_signature=signature,
        project="owner/repo-b",
        file="route.cpp",
    )
    claimed_id = ip._compute_symbol_id(first_key)
    first = gsi.GlobalSymbolRecord(
        qname="left::route",
        base_lib="api",
        base_repo="owner/repo-a",
        kind=2,
        sym_type="func",
        file="route.cpp",
        line=1,
        end_line=1,
        is_public=1,
        token_est=8,
        body_len=32,
        text="int route(int value) { return value; }",
        symbol_key=first_key,
        symbol_id=claimed_id,
        canonical_signature=signature,
        symbol_kind="FUNCTION_DECL",
        provider="owner/repo-a",
        include_provenance="route.cpp",
    )
    second = gsi.GlobalSymbolRecord(
        qname="right::route",
        base_lib="api",
        base_repo="owner/repo-b",
        kind=2,
        sym_type="func",
        file="route.cpp",
        line=1,
        end_line=1,
        is_public=1,
        token_est=8,
        body_len=32,
        text="int route(int value) { return value; }",
        symbol_key=second_key,
        symbol_id=claimed_id,
        canonical_signature=signature,
        symbol_kind="FUNCTION_DECL",
        provider="owner/repo-b",
        include_provenance="route.cpp",
    )

    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols([first])
    with pytest.raises(RuntimeError, match="symbol ID collision"):
        store.insert_symbols([second])
    store.close()


def test_global_store_rejects_stale_usr_row_without_rewriting_metadata(
    tmp_path: Path,
) -> None:
    db = tmp_path / "symbols.sqlite"
    usr = "c:@N@api@F@route#I#"
    signature = "display=route(int)|type=int (int)|result=int|args=(int)"
    expected_key = ip.canonical_symbol_identity(
        qname="api::route",
        kind="FUNCTION_DECL",
        usr=usr,
        canonical_signature=signature,
        project="owner/repo-a",
        file="route.cpp",
    )
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            gsi.GlobalSymbolRecord(
                qname="api::route",
                base_lib="api",
                base_repo="owner/repo-a",
                kind=2,
                sym_type="func",
                file="route.cpp",
                line=1,
                end_line=1,
                is_public=1,
                token_est=8,
                body_len=32,
                text="int route(int value) { return value; }",
                symbol_key=expected_key,
                usr=usr,
                canonical_signature=signature,
                symbol_kind="FUNCTION_DECL",
                provider="owner/repo-a",
                include_provenance="route.cpp",
            )
        ]
    )
    store.commit()
    store.close()
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE symbols SET symbol_key='old-key', identity_schema_version=1"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ip.SymbolIdentityError, match="cannot be promoted"):
        gsi.GlobalSymbolStore(str(db))
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT symbol_key, usr, canonical_signature, identity_schema_version "
            "FROM symbols"
        ).fetchone()
    finally:
        conn.close()
    assert row == (
        "old-key",
        usr,
        signature,
        1,
    )


def test_global_store_refuses_to_downgrade_newer_schema(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    store = gsi.GlobalSymbolStore(str(db))
    store.close()
    conn = sqlite3.connect(db)
    try:
        conn.execute(f"PRAGMA user_version={gsi.GLOBAL_SYMBOL_DB_SCHEMA_VERSION + 1}")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="newer global symbol schema"):
        gsi.GlobalSymbolStore(str(db))


def test_global_reader_rejects_stale_identity_rows_in_current_db(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols([_global_record()])
    store.commit()
    store.close()
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE symbols SET identity_schema_version=1")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="incompatible symbol identity rows"):
        ip.GlobalSymbolReader(str(db))


def test_structured_lookup_never_falls_back_to_lone_legacy_qname(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols([_global_record()])
    store.commit()
    store.close()

    reader = ip.GlobalSymbolReader(str(db))
    try:
        assert reader.lookup(
            "api::route",
            usr="c:@N@api@F@route#I#",
            canonical_signature="display=route(int)|type=int (int)",
            symbol_kind="FUNCTION_DECL",
        ) is None
    finally:
        reader.close()


def test_global_reader_returns_candidates_and_never_selects_ambiguous_qname(
    tmp_path: Path,
) -> None:
    db = tmp_path / "symbols.sqlite"
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            _global_record(
                project="owner/repo-a",
                file="a.cpp",
                usr="c:@N@api@F@route#I#",
                signature="type=int (int)|result=int|args=(int)",
            ),
            _global_record(
                project="owner/repo-b",
                file="b.cpp",
                usr="c:@N@api@F@route#d#",
                signature="type=double (double)|result=double|args=(double)",
                text="double route(double value) { return value; }",
            ),
            _global_record(
                project="owner/repo-c",
                file="c.cpp",
                usr="c:@N@api@F@route#I#",
                signature="type=int (int)|result=int|args=(int)",
                text="int route(int value) { return value + 1; }",
            ),
        ]
    )
    store.commit()
    store.close()

    reader = ip.GlobalSymbolReader(str(db))
    try:
        candidates = reader.lookup_candidates("api::route")
        assert len(candidates) == 3
        with pytest.raises(ip.SymbolIdentityError, match="ambiguous global symbol"):
            reader.lookup("api::route")
        with pytest.raises(ip.SymbolIdentityError, match="ambiguous global symbol"):
            reader.lookup(
                "api::route",
                usr="c:@N@api@F@route#I#",
                canonical_signature="intentionally wrong fallback signature",
                symbol_kind="WRONG_KIND",
            )
        resolved = reader.lookup(
            "api::route",
            symbol_key="intentionally wrong canonical key",
            usr="c:@N@api@F@route#I#",
            canonical_signature="intentionally wrong fallback signature",
            symbol_kind="WRONG_KIND",
            project="owner/repo-a",
        )
        assert resolved is not None
        assert resolved["base_repo"] == "owner/repo-a"
    finally:
        reader.close()


def test_crosslink_enrichment_records_ambiguity_without_guessing_provider() -> None:
    class AmbiguousReader:
        def lookup(self, _qname: str, **_kwargs):
            raise ip.AmbiguousGlobalSymbolError("two authoritative providers")

    reference_key = ip.canonical_symbol_identity(
        qname="EVP_PKEY_CTX_free",
        kind="FUNCTION_DECL",
        usr="c:@F@EVP_PKEY_CTX_free",
        canonical_signature="void (EVP_PKEY_CTX *)",
    )
    caller = ip.FunctionDef(
        name="release",
        qualified_name="app::release",
        file="release.cpp",
        line=1,
        text="void release() { EVP_PKEY_CTX_free(nullptr); }",
        callees=["EVP_PKEY_CTX_free"],
        callee_refs=[
            {
                "symbol_key": reference_key,
                "qname": "EVP_PKEY_CTX_free",
                "usr": "c:@F@EVP_PKEY_CTX_free",
                "canonical_signature": "void (EVP_PKEY_CTX *)",
                "symbol_kind": "FUNCTION_DECL",
                "project": "",
                "file": "/usr/include/openssl/evp.h",
            }
        ],
    )
    project = ip.ProjectIndex()
    project.add_function(caller)
    visited: dict[str, dict[str, object]] = {}
    budget = ip.CrossLinkBudget()

    deps = ip.collect_transitive_deps(
        caller.symbol_key,
        project,
        global_symbols=AmbiguousReader(),
        crosslink_visited=visited,
        crosslink_budget=budget,
    )

    assert deps == []
    assert visited == {}
    assert budget.ambiguous_lookups == 1
    assert budget.ambiguity_examples == ["EVP_PKEY_CTX_free"]


def test_std_lookup_requires_authoritative_provider_provenance(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    usr = "c:@N@std@FT@move>#t0.0#&&t0.0#"
    signature = "display=move(T &&)|type=T &&(T &&)"
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            _global_record(
                qname="std::move",
                project="llvm/llvm-project",
                file="llvm-project/libcxx/include/vector",
                usr=usr,
                signature=signature,
                symbol_kind="FUNCTION_TEMPLATE",
                base_lib="std",
                provider="libc++",
                include_provenance="vector",
            ),
            _global_record(
                qname="std::move",
                project="gcc-mirror/gcc",
                file="gcc-mirror/libstdc++-v3/include/bits/stl_vector.h",
                usr=usr,
                signature=signature,
                symbol_kind="FUNCTION_TEMPLATE",
                base_lib="std",
                provider="libstdc++",
                include_provenance="bits/stl_vector.h",
            ),
        ]
    )
    store.commit()
    store.close()

    reader = ip.GlobalSymbolReader(str(db))
    try:
        with pytest.raises(ip.SymbolIdentityError, match="ambiguous global symbol"):
            reader.lookup(
                "std::move",
                usr=usr,
                canonical_signature="wrong fallback signature",
                symbol_kind="WRONG_KIND",
            )
        resolved = reader.lookup(
            "std::move",
            usr=usr,
            canonical_signature="wrong fallback signature",
            symbol_kind="WRONG_KIND",
            provider="libc++",
            include_provenance="vector",
        )
        assert resolved is not None
        assert resolved["base_repo"] == "llvm/llvm-project"
    finally:
        reader.close()


def test_crossrepo_lookup_routes_external_usr_without_caller_project_alias(
    tmp_path: Path,
) -> None:
    db = tmp_path / "symbols.sqlite"
    usr = "c:@N@boost@N@algorithm@F@trim#&1$@S@Range#"
    signature = "display=trim(Range &)|type=void (Range &)"
    global_key = ip.canonical_symbol_identity(
        qname="boost::algorithm::trim",
        kind="FUNCTION_TEMPLATE",
        usr=usr,
        canonical_signature=signature,
        project="boostorg/boost",
        file="boost/algorithm/string/trim.hpp",
    )
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            gsi.GlobalSymbolRecord(
                qname="boost::algorithm::trim",
                base_lib="boost",
                base_repo="boostorg/boost",
                kind=2,
                sym_type="func",
                file="boost/algorithm/string/trim.hpp",
                line=10,
                end_line=12,
                is_public=1,
                token_est=12,
                body_len=48,
                text="template<class R> void trim(R& value) { value.clear(); }",
                symbol_key=global_key,
                usr=usr,
                canonical_signature=signature,
                symbol_kind="FUNCTION_TEMPLATE",
                provider="boost",
                include_provenance="boost/algorithm/string/trim.hpp",
            )
        ]
    )
    store.commit()
    store.close()

    external_key = ip.canonical_symbol_identity(
        qname="boost::algorithm::trim",
        kind="FUNCTION_TEMPLATE",
        usr=usr,
        canonical_signature=signature,
    )
    caller_text = "void caller(Range& value) { boost::algorithm::trim(value); }"
    external_id = ip._compute_symbol_id(external_key)
    symbol_ids = [0] * len(caller_text)
    call_targets = [0] * len(caller_text)
    trim_start = caller_text.index("trim")
    trim_end = trim_start + len("trim")
    symbol_ids[trim_start:trim_end] = [external_id] * len("trim")
    call_targets[trim_start:trim_end] = [external_id] * len("trim")
    caller = ip.FunctionDef(
        name="caller",
        qualified_name="app::caller",
        file="caller.cpp",
        line=1,
        text=caller_text,
        callees=[],
        baselib_callees=["boost::algorithm::trim"],
        baselib_callee_refs=[
            {
                "symbol_key": external_key,
                "symbol_id": external_id,
                "qname": "boost::algorithm::trim",
                "usr": usr,
                "canonical_signature": signature,
                "symbol_kind": "FUNCTION_TEMPLATE",
                "project": "",
                "file": "/opt/boost/include/boost/algorithm/string/trim.hpp",
                "line": 10,
                "provider": "boost",
                "include_provenance": "boost/algorithm/string/trim.hpp",
            }
        ],
        semantic_symbol_ids=symbol_ids,
        semantic_call_targets=call_targets,
        semantic_type_refs=[0] * len(caller_text),
        semantic_def_use=[0] * len(caller_text),
    )
    project = ip.ProjectIndex()
    project.add_function(caller)
    reader = ip.GlobalSymbolReader(str(db))
    visited: dict[str, dict[str, object]] = {}
    try:
        ip.collect_transitive_deps(
            caller.symbol_key,
            project,
            global_symbols=reader,
            crosslink_visited=visited,
            crosslink_budget=ip.CrossLinkBudget(),
        )
    finally:
        reader.close()

    assert len(visited) == 1
    assert next(iter(visited.values()))["symbol_key"] == global_key

    reader = ip.GlobalSymbolReader(str(db))
    try:
        docs = ip.build_training_documents(
            project,
            max_tokens=4096,
            enriched=True,
            global_symbols=reader,
        )
    finally:
        reader.close()
    caller_doc = next(doc for doc in docs if "void caller" in doc["text"])
    caller_chunk = next(
        index
        for index, boundary in enumerate(caller_doc["chunk_boundaries"])
        if boundary["name"] == "caller"
    )
    target_chunk = next(
        index
        for index, boundary in enumerate(caller_doc["chunk_boundaries"])
        if boundary.get("dep_source") == "crosslib:boostorg/boost"
    )
    assert {"from": caller_chunk, "to": target_chunk} in caller_doc["call_edges"]
    assembled_trim = caller_doc["text"].index("trim(value)")
    global_id = ip._compute_symbol_id(global_key)
    assert caller_doc["call_targets"][assembled_trim] == global_id
    assert caller_doc["symbol_ids"][assembled_trim] == global_id
