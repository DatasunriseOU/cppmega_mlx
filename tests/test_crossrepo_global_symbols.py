"""Cross-repo base-lib symbol linking — unit coverage (no libclang, no tarball).

These tests exercise the PURE pieces of the cross-link feature end to end:

  * GlobalSymbolStore write -> GlobalSymbolReader read round-trips a base-lib
    function definition by qualified name.
  * is_public_symbol gates internal/reserved names and (for A2 public_only libs)
    non-header files.
  * build_training_documents(..., global_symbols=reader) PULLS an unresolved
    base-lib callee in as a DEEPEST dep chunk tagged dep_source='crosslib:<repo>'
    and leaves behavior UNCHANGED when global_symbols is None.
  * CrossLinkBudget enforces the per-doc count + token bounds.

RULE #1: no mocks of the real store — we build a real (tiny) SQLite store.
"""
from __future__ import annotations

import sys
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

MLX_ROOT = Path(__file__).resolve().parents[1]
for p in (MLX_ROOT / "tools" / "clang_indexer", MLX_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _sleeping_parse_worker(args: tuple[float]):
    (delay_s,) = args
    time.sleep(delay_s)
    return [], []


def _load_index_project():
    try:
        import index_project as ip  # type: ignore
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"index_project import failed (libclang?): {exc}")
    return ip


def _load_builder():
    from scripts.crossrepo import build_global_symbol_index as b  # type: ignore
    return b


def test_boost_provider_spec_keeps_boost_type_definitions() -> None:
    b = _load_builder()
    assert b.BASE_LIBS["boost"]["allow_system_types"] is True


def test_boost_provider_worker_emits_boost_type_definition(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    b = _load_builder()
    root = tmp_path / "provider"
    header = root / "boost" / "include" / "boost" / "route.hpp"
    header.parent.mkdir(parents=True)
    header.write_text(
        "namespace boost { struct route_type { int value; }; }\n",
        encoding="utf-8",
    )

    _functions, types = b._parse_file_worker(
        (
            str(header),
            str(root),
            b.project_identity_for_subtree("boost"),
            "-std=c++17",
            "c++",
            [str(root), str(root / "boost" / "include")],
            {
                "allow_system_types": b.BASE_LIBS["boost"]["allow_system_types"],
            },
        )
    )

    assert any(row["qualified_name"] == "boost::route_type" for row in types)


def _make_store(tmp: Path, *, reference: dict[str, object] | None = None):
    b = _load_builder()
    db = str(tmp / "gsi.sqlite")
    store = b.GlobalSymbolStore(db)
    body = ("template<class R> void trim(R& r) "
            "{ /* boost trim impl, long enough body to keep */ }")
    reference = reference or {}
    usr = str(reference.get("usr") or "")
    signature = str(
        reference.get("canonical_signature")
        or "display=trim(R &)|type=void (R &)"
    )
    symbol_kind = str(reference.get("symbol_kind") or "FUNCTION_TEMPLATE")
    record = b.GlobalSymbolRecord(
        qname="boost::algorithm::trim",
        base_lib="boost",
        base_repo="boostorg/boost",
        kind=2,
        sym_type="func",
        file="boost/algorithm/string/trim.hpp",
        line=5,
        end_line=40,
        is_public=1,
        token_est=len(body) // 4,
        body_len=len(body),
        text=body,
        symbol_key=b.ip.canonical_symbol_identity(
            qname="boost::algorithm::trim",
            kind=symbol_kind,
            usr=usr,
            canonical_signature=signature,
            project="boostorg/boost",
            file="boost/algorithm/string/trim.hpp",
        ),
        usr=usr,
        canonical_signature=signature,
        symbol_kind=symbol_kind,
        provider="boost",
        include_provenance="boost/algorithm/string/trim.hpp",
    )
    store.insert_symbols([record])
    store.commit()
    store.close()
    return db, body


def test_store_reader_roundtrip(tmp_path):
    ip = _load_index_project()
    db, body = _make_store(tmp_path)
    reader = ip.GlobalSymbolReader(db)
    rec = reader.lookup("boost::algorithm::trim")
    assert rec is not None
    assert rec["base_repo"] == "boostorg/boost"
    assert rec["base_lib"] == "boost"
    assert "trim impl" in rec["text"]
    assert reader.lookup("does::not::exist") is None
    reader.close()


def test_store_reader_normalizes_std_inline_namespace(tmp_path):
    ip = _load_index_project()
    b = _load_builder()
    db = str(tmp_path / "gsi.sqlite")
    store = b.GlobalSymbolStore(db)
    body = "size_t basic_string_size() { return 0; }"
    signature = "display=size()|type=size_t ()"
    store.insert_symbols([b.GlobalSymbolRecord(
        qname="std::basic_string::size",
        base_lib="std",
        base_repo="microsoft/STL",
        kind=2,
        sym_type="func",
        file="STL/stl/inc/string",
        line=1,
        end_line=1,
        is_public=1,
        token_est=len(body) // 4,
        body_len=len(body),
        text=body,
        symbol_key=b.ip.canonical_symbol_identity(
            qname="std::basic_string::size",
            kind="CXX_METHOD",
            canonical_signature=signature,
            project="microsoft/STL",
            file="STL/stl/inc/string",
        ),
        canonical_signature=signature,
        symbol_kind="CXX_METHOD",
        provider="msvc-stl",
        include_provenance="string",
    )])
    store.commit()
    store.close()

    reader = ip.GlobalSymbolReader(db)
    rec = reader.lookup("std::__cxx11::basic_string::size")
    assert rec is not None
    assert rec["base_repo"] == "microsoft/STL"
    assert "basic_string_size" in rec["text"]
    assert ip.normalize_inline_namespace_qname("std::__1::vector::push_back") == (
        "std::vector::push_back"
    )
    reader.close()


def test_store_preserves_distinct_definitions_with_same_qname(tmp_path):
    b = _load_builder()
    db = str(tmp_path / "gsi.sqlite")
    store = b.GlobalSymbolStore(db)
    msvc_signature = "display=append(const char *)|type=void (const char *)"
    libcxx_signature = "display=append(size_type, char)|type=void (size_type, char)"
    store.insert_symbols([
        b.GlobalSymbolRecord(
            qname="std::basic_string::append", base_lib="std",
            base_repo="microsoft/STL", kind=2, sym_type="func",
            file="STL/stl/inc/xstring", line=10, end_line=18,
            is_public=1, token_est=16, body_len=64,
            text="void append(const char*) { /* overload one */ }",
            symbol_key=b.ip.canonical_symbol_identity(
                qname="std::basic_string::append", kind="CXX_METHOD",
                canonical_signature=msvc_signature, project="microsoft/STL",
                file="STL/stl/inc/xstring",
            ),
            canonical_signature=msvc_signature, symbol_kind="CXX_METHOD",
            provider="msvc-stl", include_provenance="xstring",
        ),
        b.GlobalSymbolRecord(
            qname="std::basic_string::append", base_lib="std",
            base_repo="llvm/llvm-project", kind=2, sym_type="func",
            file="llvm-project/libcxx/include/string", line=200, end_line=215,
            is_public=1, token_est=32, body_len=128,
            text="void append(size_type, char) { /* overload two */ }",
            symbol_key=b.ip.canonical_symbol_identity(
                qname="std::basic_string::append", kind="CXX_METHOD",
                canonical_signature=libcxx_signature, project="llvm/llvm-project",
                file="llvm-project/libcxx/include/string",
            ),
            canonical_signature=libcxx_signature, symbol_kind="CXX_METHOD",
            provider="libc++", include_provenance="string",
        ),
    ])
    store.commit()
    store.close()

    import sqlite3

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT base_repo, file, line FROM symbols "
            "WHERE qname='std::basic_string::append' AND sym_type='func' "
            "ORDER BY base_repo, file, line"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("llvm/llvm-project", "llvm-project/libcxx/include/string", 200),
        ("microsoft/STL", "STL/stl/inc/xstring", 10),
    ]


def test_store_read_only_rejects_old_qname_only_schema(tmp_path):
    b = _load_builder()
    db = tmp_path / "old.sqlite"

    import sqlite3

    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE symbols (
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
                text TEXT NOT NULL,
                PRIMARY KEY (qname, sym_type)
            );
            """
        )
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="old qname-only schema"):
        b.GlobalSymbolStore(str(db), read_only=True)


def test_extraction_cache_reuses_complete_lib_without_restreaming(tmp_path):
    b = _load_builder()
    zstd = shutil.which("zstd")
    if not zstd:
        pytest.skip("zstd is required to exercise the real extraction path")

    raw_tar = tmp_path / "corpus.tar"
    tarball = tmp_path / "corpus.tar.zst"
    src = tmp_path / "src.hpp"
    src.write_text("namespace boost { inline int cached_symbol() { return 7; } }\n")
    with tarfile.open(raw_tar, "w") as tf:
        tf.add(src, arcname="cpp_all/boost/include/boost/cached_symbol.hpp")
    subprocess.run([zstd, "-q", "-f", str(raw_tar), "-o", str(tarball)], check=True)

    specs = {"boost": b.BASE_LIBS["boost"]}
    cache_root = tmp_path / "extract_cache"
    dirs, counts = b.prepare_extraction_cache(
        tarball, specs, cache_root, max_files=10, member_cap=None
    )
    cached_file = dirs["boost"] / "boost" / "include" / "boost" / "cached_symbol.hpp"
    assert counts == {"boost": 1}
    assert cached_file.read_text().startswith("namespace boost")

    old_mode = tarball.stat().st_mode
    os.chmod(tarball, 0)
    try:
        dirs2, counts2 = b.prepare_extraction_cache(
            tarball, specs, cache_root, max_files=10, member_cap=None
        )
    finally:
        os.chmod(tarball, old_mode)

    assert dirs2 == dirs
    assert counts2 == counts
    assert cached_file.read_text().startswith("namespace boost")


def test_extraction_cache_hardlinks_complete_code_cache(tmp_path):
    b = _load_builder()
    tarball = tmp_path / "corpus.tar.zst"
    tarball.write_bytes(b"stable corpus fingerprint")
    source_root = tmp_path / "source_cache"
    boost_root = source_root / "boost"
    wanted = boost_root / "include" / "boost" / "cached_symbol.hpp"
    wanted.parent.mkdir(parents=True)
    wanted.write_text("namespace boost { inline int cached_symbol() { return 7; } }\n")
    (boost_root / "README.txt").write_text("not an index input\n")
    (boost_root / ".cppmega_source_cache_complete.json").write_text(json.dumps({
        "repo": "Boost",
        "source": str(tarball),
        "completed_at": "2026-07-14T00:00:00",
    }))
    alias = source_root / "BOOST"
    if not alias.exists():
        alias.symlink_to(boost_root, target_is_directory=True)
    spec = dict(b.BASE_LIBS["boost"])
    spec["subtrees"] = ["boost", "BOOST"]

    cache_root = tmp_path / "extract_cache"
    counts = b.populate_extraction_cache_from_source_cache(
        tarball, {"boost": spec}, source_root, cache_root,
        max_files=10,
    )
    cached = cache_root / "boost/boost/include/boost/cached_symbol.hpp"
    assert counts == {"boost": 1}
    assert cached.read_text() == wanted.read_text()
    assert cached.stat().st_ino == wanted.stat().st_ino
    manifest = json.loads(
        (cache_root / "boost/.gsi_extract_complete.json").read_text())
    assert manifest["publication"] == "hardlink"

    dirs, reused = b.prepare_extraction_cache(
        tarball, {"boost": spec}, cache_root, max_files=10)
    assert reused == counts
    assert dirs["boost"] == cache_root / "boost"


def test_is_public_symbol_filters():
    b = _load_builder()
    # A1 (public_only=False): accept normal qnames, reject internal segments.
    assert b.is_public_symbol(
        "boost::algorithm::trim", "a.hpp", False, ("boost::",)
    )
    assert not b.is_public_symbol(
        "boost::detail::do_trim", "a.hpp", False, ("boost::",)
    )
    assert not b.is_public_symbol(
        "absl::internal::Foo", "a.h", False, ("absl::",)
    )
    # A2 (public_only=True): require a header file AND reject reserved names.
    assert b.is_public_symbol(
        "std::vector::push_back", "bits/vector.h", True, ("std::",)
    )
    assert b.is_public_symbol(
        "std::__1::vector::push_back", "include/vector", True, ("std::",)
    )
    assert not b.is_public_symbol(
        "std::vector::push_back", "src/vector.cpp", True, ("std::",)
    )
    assert not b.is_public_symbol(
        "std::__copy_impl", "bits/algo.h", True, ("std::",)
    )
    # C2 regression: std A2 must not index compiler/libiberty or other random
    # C++ header symbols just because they live under the selected subtree.
    assert not b.is_public_symbol(
        "libiberty::demangle_component", "libiberty/demangle.h", True, ("std::",)
    )
    assert not b.is_public_symbol(
        "__gnu_cxx::__normal_iterator", "bits/stl_iterator.h", True, ("std::",)
    )
    assert (
        b.normalize_inline_namespace_qname("std::__cxx11::basic_string::size")
        == "std::basic_string::size"
    )
    assert b.normalize_inline_namespace_qname("boost::__1::route") == (
        "boost::__1::route"
    )


def test_global_symbol_rebuild_rolls_back_late_parse_failure(
    tmp_path,
    monkeypatch,
):
    b = _load_builder()
    store = b.GlobalSymbolStore(str(tmp_path / "symbols.sqlite"))
    with store.rebuild_lib("boost", "A1", ["boost"]):
        store.mark_lib_done(
            "boost",
            "A1",
            n_files=1,
            n_funcs=0,
            n_types=0,
            n_errors=0,
            elapsed_s=0.1,
            subtrees=["boost"],
        )
    assert store.lib_done("boost")

    source_root = tmp_path / "source"
    source_file = source_root / "boost" / "route.hpp"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("namespace boost { int route(int); }\n")
    monkeypatch.setattr(b.ip, "find_cpp_files", lambda _root: [str(source_file)])
    monkeypatch.setattr(b, "SYMBOL_BATCH_SIZE", 1)

    def late_failure(tasks, **_kwargs):
        filepath, project_id, _worker_args = next(iter(tasks))
        qname = "boost::route"
        signature = "display=route(int)|type=int (int)"
        symbol_key = b.ip.canonical_symbol_identity(
            qname=qname,
            kind="FUNCTION_DECL",
            canonical_signature=signature,
            project=project_id,
            file="boost/route.hpp",
        )
        yield filepath, project_id, (
            [
                {
                    "qualified_name": qname,
                    "file": "boost/route.hpp",
                    "line": 1,
                    "end_line": 1,
                    "text": "int route(int value) { return value + 1; }",
                    "symbol_key": symbol_key,
                    "usr": "",
                    "canonical_signature": signature,
                    "symbol_kind": "FUNCTION_DECL",
                }
            ],
            [],
        )
        raise RuntimeError("late parse failure")

    monkeypatch.setattr(b, "_iter_parse_results", late_failure)
    with pytest.raises(RuntimeError, match="late parse failure"):
        b.index_lib_from_dir(
            "boost",
            b.BASE_LIBS["boost"],
            source_root,
            store,
            workers=1,
            std_arg="-std=c++17",
        )

    assert not store.lib_done("boost")
    assert store.conn.execute(
        "SELECT COUNT(*) FROM symbols WHERE base_lib='boost'"
    ).fetchone()[0] == 0
    store.close()


def test_parse_worker_timeout_is_real_wall_clock():
    b = _load_builder()
    started = time.monotonic()
    with pytest.raises(b.ParseWorkerTimeout, match="slow.cpp"):
        list(
            b._iter_parse_results(
                [("slow.cpp", "owner/repo", (30.0,))],
                workers=1,
                timeout_s=0.2,
                worker_fn=_sleeping_parse_worker,
            )
        )
    assert time.monotonic() - started < 8.0


def test_parse_timeout_is_explicitly_configurable():
    b = _load_builder()
    args = b.parse_args(["--parse-timeout-seconds", "300"])
    assert args.parse_timeout_seconds == 300.0


def test_abseil_global_api_index_excludes_test_and_benchmark_tus() -> None:
    b = _load_builder()
    paths = [
        "/src/absl/flags/flag.cc",
        "/src/absl/flags/flag_test.cc",
        "/src/absl/flags/flag_benchmark.cc",
        "/src/absl/container/inline.hpp",
    ]
    kept, excluded = b._filter_non_provider_sources(paths, b.BASE_LIBS["abseil"])
    assert kept == [paths[0], paths[3]]
    assert excluded == 2


def test_global_api_index_excludes_non_provider_trees_and_libc_sources() -> None:
    b = _load_builder()
    folly_paths = [
        "/src/folly/container/Map.cpp",
        "/src/folly/container/test/MapTest.cpp",
        "/src/folly/benchmark/MapBenchmark.cpp",
    ]
    kept, excluded = b._filter_non_provider_sources(
        folly_paths, b.BASE_LIBS["folly"]
    )
    assert kept == [folly_paths[0]]
    assert excluded == 2

    libc_paths = [
        "/src/glibc/include/stdio.h",
        "/src/musl/include/stdlib.h",
        "/src/glibc/stdlib/strtol.c",
        "/src/musl/tests/stdio.c",
    ]
    kept, excluded = b._filter_non_provider_sources(
        libc_paths, b.BASE_LIBS["libc"]
    )
    assert kept == libc_paths[:2]
    assert excluded == 2


def test_crosslink_budget_bounds():
    ip = _load_index_project()
    b = ip.CrossLinkBudget(max_deps=2, token_budget=1000)
    assert b.has_room()
    assert b.can_afford(500)
    b.spend(500)
    b.spend(400)
    assert not b.has_room()  # 2 deps used
    b2 = ip.CrossLinkBudget(max_deps=100, token_budget=100)
    assert b2.can_afford(100)
    assert not b2.can_afford(101)
    b2.spend(100)
    assert not b2.has_room()


def _tiny_index(ip):
    idx = ip.ProjectIndex()
    signature = "display=trim(R &)|type=void (R &)"
    external_key = ip.canonical_symbol_identity(
        qname="boost::algorithm::trim",
        kind="FUNCTION_TEMPLATE",
        canonical_signature=signature,
    )
    root = ip.FunctionDef(
        name="do_work", qualified_name="myrepo::do_work", file="a.cpp", line=1,
        text=("void do_work() {\n  boost::algorithm::trim(s);\n"
              "  myrepo::local_helper();\n}"),
        callees=["myrepo::local_helper"],
        baselib_callees=["boost::algorithm::trim"],
        baselib_callee_refs=[{
            "symbol_key": external_key,
            "symbol_id": ip._compute_symbol_id(external_key),
            "qname": "boost::algorithm::trim",
            "usr": "",
            "canonical_signature": signature,
            "symbol_kind": "FUNCTION_TEMPLATE",
            "project": "",
            "file": "/opt/boost/include/boost/algorithm/string/trim.hpp",
            "line": 5,
            "provider": "boost",
            "include_provenance": "boost/algorithm/string/trim.hpp",
        }],
        is_definition=True,
    )
    helper = ip.FunctionDef(
        name="local_helper", qualified_name="myrepo::local_helper", file="a.cpp",
        line=10,
        text="void local_helper() { /* a real local helper body, long enough */ }",
        callees=[], is_definition=True,
    )
    idx.add_function(root)
    idx.add_function(helper)
    return idx


def test_crosslink_off_unchanged(tmp_path):
    ip = _load_index_project()
    idx = _tiny_index(ip)
    docs = ip.build_training_documents(idx, max_tokens=16384, enriched=True)
    root_doc = [d for d in docs if "do_work" in d["text"]][0]
    assert not any(b.get("dep_source") for b in root_doc["chunk_boundaries"])
    assert "trim impl" not in root_doc["text"]


def test_build_training_documents_streams_without_accumulating(tmp_path):
    ip = _load_index_project()
    idx = _tiny_index(ip)
    emitted = []

    docs = ip.build_training_documents(
        idx,
        max_tokens=16384,
        enriched=True,
        emit_doc=emitted.append,
    )

    assert docs == []
    assert emitted
    assert any("do_work" in doc["text"] for doc in emitted)


def test_crosslink_on_pulls_tagged_chunk(tmp_path):
    ip = _load_index_project()
    db, _ = _make_store(tmp_path)
    reader = ip.GlobalSymbolReader(db)
    idx = _tiny_index(ip)
    docs = ip.build_training_documents(
        idx, max_tokens=16384, enriched=True, global_symbols=reader)
    root_doc = [d for d in docs if "do_work" in d["text"]][0]
    boundaries = root_doc["chunk_boundaries"]
    crosslib_idxs = [i for i, b in enumerate(boundaries) if b.get("dep_source")]
    assert len(crosslib_idxs) == 1
    cl_idx = crosslib_idxs[0]
    assert boundaries[cl_idx]["dep_source"] == "crosslib:boostorg/boost"
    assert boundaries[cl_idx]["name"] == "trim"
    assert "trim impl" in root_doc["text"]
    # The root function chunk's call_edge to the unresolved base-lib callee must
    # resolve to the pulled cross-lib chunk (root -> crosslib).
    root_idx = next(i for i, b in enumerate(boundaries) if b["name"] == "do_work")
    edges = root_doc["call_edges"]
    assert {"from": root_idx, "to": cl_idx} in edges, (
        f"expected root->crosslib call_edge; edges={edges} root={root_idx} cl={cl_idx}"
    )
    reader.close()


# --------------------------------------------------------------------------- #
# REAL libclang parse-path regression. The unit tests above hand-build
# FunctionDef.callees with a 'boost::' qname, which bypasses extract_callees and
# its SYSTEM_PREFIXES filter. In the LIVE pipeline, base-lib callees go through
# extract_callees first -- and 'std::'/'boost::' are in SYSTEM_PREFIXES, so they
# were silently dropped from `callees` and never reached the cross-lib linker
# (the feature was dead for boost/std). This test parses real source with
# libclang to prove the cross-linkable base-lib callee survives into
# FunctionDef.baselib_callees and is pulled when the index is enabled.
# --------------------------------------------------------------------------- #
def _parse_one(ip, src_text, tmp_path):
    ip._configure_libclang()
    from clang.cindex import Index  # type: ignore
    src = tmp_path / "u.cpp"
    src.write_text(src_text)
    idx = Index.create()
    funcs, _types = ip.parse_translation_unit(
        str(src), idx, ["-std=c++17"], str(tmp_path),
        project_id="tests/crossrepo-fixture",
    )
    return funcs


def test_extract_callees_splits_baselib_from_normal():
    ip = _load_index_project()
    pytest.importorskip("clang.cindex")
    out = ip.extract_callees  # signature check: returns a 2-tuple
    assert ip.is_crosslinkable_baselib("boost::beast::make_printable")
    assert ip.is_crosslinkable_baselib("std::sort")
    assert not ip.is_crosslinkable_baselib("myrepo::helper")
    assert not ip.is_crosslinkable_baselib("memcpy")
    assert callable(out)


def test_real_parse_baselib_callee_survives_and_pulls(tmp_path):
    ip = _load_index_project()
    try:
        import clang.cindex  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"libclang unavailable: {exc}")

    # Root calls a boost:: symbol (filtered out of `callees`) + a local helper.
    src = (
        "namespace boost { namespace algorithm { template<class R> "
        "void trim(R& r); } }\n"
        "static int local_helper(int x) { return x + 1; }\n"
        "int do_work(int s) {\n"
        "    boost::algorithm::trim(s);\n"
        "    return local_helper(s);\n"
        "}\n"
    )
    funcs = _parse_one(ip, src, tmp_path)
    root = next(f for f in funcs if f.name == "do_work")

    # REGRESSION CORE: boost:: callee is NOT in `callees` (SYSTEM_PREFIXES) but
    # IS captured in baselib_callees so the cross-lib linker can see it.
    assert not any(c.startswith("boost::") for c in root.callees), root.callees
    assert "boost::algorithm::trim" in root.baselib_callees, root.baselib_callees
    # IPC round-trip preserves the new field.
    assert ip.FunctionDef.from_dict(root.to_dict()).baselib_callees == \
        root.baselib_callees

    # Now build a store with the boost def and confirm an actual PULL.
    db, body = _make_store(tmp_path, reference=root.baselib_callee_refs[0])
    reader = ip.GlobalSymbolReader(db)
    idx = ip.ProjectIndex()
    for f in funcs:
        idx.add_function(f)
    docs = ip.build_training_documents(
        idx, max_tokens=16384, enriched=True, global_symbols=reader)
    root_doc = [d for d in docs if "do_work" in d["text"]][0]
    cl = [b for b in root_doc["chunk_boundaries"]
          if (b.get("dep_source") or "").startswith("crosslib:")]
    assert len(cl) == 1, cl
    assert cl[0]["dep_source"] == "crosslib:boostorg/boost"
    assert cl[0]["name"] == "trim"
    assert "trim impl" in root_doc["text"]
    reader.close()


def test_real_parse_off_has_no_crosslink(tmp_path):
    """With global_symbols=None the base-lib callee is captured but never pulled
    (OFF behavior unchanged: no crosslib chunk, no provenance)."""
    ip = _load_index_project()
    try:
        import clang.cindex  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"libclang unavailable: {exc}")
    src = (
        "namespace boost { namespace algorithm { template<class R> "
        "void trim(R& r); } }\n"
        "int do_work(int seed_value_for_token_floor) {\n"
        "    int accumulator = seed_value_for_token_floor + 1;\n"
        "    boost::algorithm::trim(accumulator);\n"
        "    return accumulator + seed_value_for_token_floor;\n"
        "}\n"
    )
    funcs = _parse_one(ip, src, tmp_path)
    idx = ip.ProjectIndex()
    for f in funcs:
        idx.add_function(f)
    docs = ip.build_training_documents(idx, max_tokens=16384, enriched=True)
    root_doc = [d for d in docs if "do_work" in d["text"]][0]
    assert not any(b.get("dep_source") for b in root_doc["chunk_boundaries"])
    assert "trim impl" not in root_doc["text"]
