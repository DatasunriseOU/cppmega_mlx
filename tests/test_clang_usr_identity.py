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
    repo_a = ip.canonical_symbol_identity(**shared, project="repo-a", file="a.cpp")
    repo_b = ip.canonical_symbol_identity(**shared, project="repo-b", file="b.cpp")
    repo_a_other_file = ip.canonical_symbol_identity(
        **shared, project="repo-a", file="other.cpp"
    )

    assert repo_a != repo_b
    assert repo_a == repo_a_other_file

    fallback = {
        "kind": "FUNCTION_DECL",
        "canonical_signature": "display=route(int)|type=int (int)",
        "project": "repo-a",
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
        "project": "repo-a",
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
            project_id="repo-a",
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
    first = ip.MacroDef("ROUTE", "one.hpp", 4, "#define ROUTE(x) (x)", ["x"])
    second = ip.MacroDef("ROUTE", "two.hpp", 4, "#define ROUTE(x) ((x) + 1)", ["x"])
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


def test_global_store_migrates_composite_schema_and_backfills_identity(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    _create_composite_v1_store(db)

    store = gsi.GlobalSymbolStore(str(db))
    store.close()

    conn = sqlite3.connect(db)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        rows = conn.execute(
            "SELECT symbol_key, canonical_signature, identity_schema_version "
            "FROM symbols ORDER BY base_repo"
        ).fetchall()
    finally:
        conn.close()
    assert version == gsi.GLOBAL_SYMBOL_DB_SCHEMA_VERSION
    assert len({row[0] for row in rows}) == 2
    assert all(row[0] for row in rows)
    assert all(row[1] for row in rows)
    assert {row[2] for row in rows} == {ip.SYMBOL_IDENTITY_SCHEMA_VERSION}


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
    )

    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols([first])
    with pytest.raises(RuntimeError, match="symbol ID collision"):
        store.insert_symbols([second])
    store.close()


def test_global_store_migration_preserves_recoverable_usr_identity(tmp_path: Path) -> None:
    db = tmp_path / "symbols.sqlite"
    usr = "c:@N@api@F@route#I#"
    signature = "display=route(int)|type=int (int)|result=int|args=(int)"
    expected_key = ip.canonical_symbol_identity(
        qname="api::route",
        kind="FUNCTION_DECL",
        usr=usr,
        canonical_signature=signature,
        project="repo-a",
        file="route.cpp",
    )
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            gsi.GlobalSymbolRecord(
                qname="api::route",
                base_lib="api",
                base_repo="repo-a",
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

    store = gsi.GlobalSymbolStore(str(db))
    store.close()
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT symbol_key, usr, canonical_signature, identity_schema_version "
            "FROM symbols"
        ).fetchone()
    finally:
        conn.close()
    assert row == (
        expected_key,
        usr,
        signature,
        ip.SYMBOL_IDENTITY_SCHEMA_VERSION,
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
    store.insert_symbols(
        [
            (
                "api::route",
                "api",
                "repo-a",
                2,
                "func",
                "route.cpp",
                1,
                1,
                1,
                8,
                32,
                "int route(int value) { return value; }",
            )
        ]
    )
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
    store.insert_symbols(
        [
            (
                "api::route",
                "api",
                "repo-a",
                2,
                "func",
                "route.cpp",
                1,
                1,
                1,
                8,
                32,
                "int route(int value) { return value; }",
            )
        ]
    )
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
            gsi.GlobalSymbolRecord(
                qname="api::route",
                base_lib="api",
                base_repo="repo-a",
                kind=2,
                sym_type="func",
                file="a.cpp",
                line=1,
                end_line=1,
                is_public=1,
                token_est=8,
                body_len=32,
                text="int route(int value) { return value; }",
                symbol_key="usr:c:@N@api@F@route#I#",
                usr="c:@N@api@F@route#I#",
                canonical_signature="type=int (int)|result=int|args=(int)",
                symbol_kind="FUNCTION_DECL",
            ),
            gsi.GlobalSymbolRecord(
                qname="api::route",
                base_lib="api",
                base_repo="repo-b",
                kind=2,
                sym_type="func",
                file="b.cpp",
                line=1,
                end_line=1,
                is_public=1,
                token_est=8,
                body_len=32,
                text="double route(double value) { return value; }",
                symbol_key="usr:c:@N@api@F@route#d#",
                usr="c:@N@api@F@route#d#",
                canonical_signature="type=double (double)|result=double|args=(double)",
                symbol_kind="FUNCTION_DECL",
            ),
            gsi.GlobalSymbolRecord(
                qname="api::route",
                base_lib="api",
                base_repo="repo-c",
                kind=2,
                sym_type="func",
                file="c.cpp",
                line=1,
                end_line=1,
                is_public=1,
                token_est=8,
                body_len=32,
                text="int route(int value) { return value + 1; }",
                symbol_key="usr:c:@N@api@F@route#I#",
                usr="c:@N@api@F@route#I#",
                canonical_signature="type=int (int)|result=int|args=(int)",
                symbol_kind="FUNCTION_DECL",
            ),
        ]
    )
    store.commit()
    store.close()

    reader = ip.GlobalSymbolReader(str(db))
    try:
        candidates = reader.lookup_candidates("api::route")
        assert len(candidates) == 3
        assert reader.lookup("api::route") is None
        assert reader.lookup(
            "api::route",
            usr="c:@N@api@F@route#I#",
            canonical_signature="type=int (int)|result=int|args=(int)",
            symbol_kind="FUNCTION_DECL",
        ) is None
        resolved = reader.lookup(
            "api::route",
            usr="c:@N@api@F@route#I#",
            canonical_signature="type=int (int)|result=int|args=(int)",
            symbol_kind="FUNCTION_DECL",
            project="repo-a",
        )
        assert resolved is not None
        assert resolved["base_repo"] == "repo-a"
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
        project="boost",
        file="boost/algorithm/string/trim.hpp",
    )
    store = gsi.GlobalSymbolStore(str(db))
    store.insert_symbols(
        [
            gsi.GlobalSymbolRecord(
                qname="boost::algorithm::trim",
                base_lib="boost",
                base_repo="boost",
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
        if boundary.get("dep_source") == "crosslib:boost"
    )
    assert {"from": caller_chunk, "to": target_chunk} in caller_doc["call_edges"]
    assembled_trim = caller_doc["text"].index("trim(value)")
    global_id = ip._compute_symbol_id(global_key)
    assert caller_doc["call_targets"][assembled_trim] == global_id
    assert caller_doc["symbol_ids"][assembled_trim] == global_id
