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
import tempfile
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


def _load_builder():
    from scripts.crossrepo import build_global_symbol_index as b  # type: ignore
    return b


def _make_store(tmp: Path):
    b = _load_builder()
    db = str(tmp / "gsi.sqlite")
    store = b.GlobalSymbolStore(db)
    body = ("template<class R> void trim(R& r) "
            "{ /* boost trim impl, long enough body to keep */ }")
    store.insert_symbols([(
        "boost::algorithm::trim", "boost", "boost", 2, "func",
        "boost/algorithm/string/trim.hpp", 5, 40, 1,
        len(body) // 4, len(body), body,
    )])
    store.commit()
    store.close()
    return db, body


def test_store_reader_roundtrip(tmp_path):
    ip = _load_index_project()
    db, body = _make_store(tmp_path)
    reader = ip.GlobalSymbolReader(db)
    rec = reader.lookup("boost::algorithm::trim")
    assert rec is not None
    assert rec["base_repo"] == "boost"
    assert rec["base_lib"] == "boost"
    assert "trim impl" in rec["text"]
    assert reader.lookup("does::not::exist") is None
    reader.close()


def test_is_public_symbol_filters():
    b = _load_builder()
    # A1 (public_only=False): accept normal qnames, reject internal segments.
    assert b.is_public_symbol("boost::algorithm::trim", "a.hpp", False)
    assert not b.is_public_symbol("boost::detail::do_trim", "a.hpp", False)
    assert not b.is_public_symbol("absl::internal::Foo", "a.h", False)
    # A2 (public_only=True): require a header file AND reject reserved names.
    assert b.is_public_symbol("std::vector::push_back", "bits/vector.h", True)
    assert not b.is_public_symbol("std::vector::push_back", "src/vector.cpp", True)
    assert not b.is_public_symbol("std::__copy_impl", "bits/algo.h", True)


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
    root = ip.FunctionDef(
        name="do_work", qualified_name="myrepo::do_work", file="a.cpp", line=1,
        text=("void do_work() {\n  boost::algorithm::trim(s);\n"
              "  myrepo::local_helper();\n}"),
        callees=["boost::algorithm::trim", "myrepo::local_helper"],
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
    assert boundaries[cl_idx]["dep_source"] == "crosslib:boost"
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
        str(src), idx, ["-std=c++17"], str(tmp_path))
    return funcs


def test_extract_callees_splits_baselib_from_normal():
    ip = _load_index_project()
    import clang.cindex  # noqa: F401  (skip cleanly if libclang missing)
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
    db, body = _make_store(tmp_path)
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
    assert cl[0]["dep_source"] == "crosslib:boost"
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
