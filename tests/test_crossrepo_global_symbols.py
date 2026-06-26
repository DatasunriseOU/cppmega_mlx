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
