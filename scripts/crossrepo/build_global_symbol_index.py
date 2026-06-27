#!/usr/bin/env python3
"""Build the GLOBAL cross-repo base-library symbol index (BuildIndex phase).

Goal
----
We selectively cross-repo link a small set of *base libraries* that are used in
many places across the corpus. When a function in repo X calls e.g. a
``boost::``/``absl::``/``std::`` symbol whose definition is NOT in repo X's own
index, the offline doc builder (``index_project.collect_transitive_deps``) can
consult THIS global symbol index and PULL the called base-lib function definition
in as a deepest-level dependency chunk (tagged ``dep_source='crosslib:<repo>'``).

This script BUILDS that index. It does NOT modify the corpus or the running
conveyor. It:

  1. Stream-extracts ONLY the selected base-lib subtrees from the corpus tarball
     (``zstd -dc --long=31 | tar`` with a member filter) into a bounded temp
     working dir, one base-lib at a time (resumable).
  2. Indexes each base-lib's PUBLIC / exported function + type DEFINITIONS via
     libclang (reusing ``index_project``'s proven parse helpers).
  3. Stores ``qualified_name -> {base_repo, file, text, kind, signature,
     is_public, token_estimate, ...}`` in a single SQLite store at
     ``outputs/crossrepo/global_symbols.sqlite``.

Selection (the careful cut — see outputs/crossrepo/base_lib_crosslink_candidates.md)
-----------------------------------------------------------------------------------
A1 (high-value, tractable; FULL public-API index):
    boost, abseil-cpp, folly, openssl, boringssl, protobuf, eigen, fmt, glib
A2 (huge / template-heavy; PUBLIC symbols only, report cost):
    std  via  STL, stl (microsoft/STL) + gcc-mirror (libstdc++) + llvm-project (libc++)
    libc via  glibc + musl

RULE #1 (fail loud, bounded)
----------------------------
ONE clear path per base-lib. A parse error on a single file is counted and the
file is skipped (parse failures in 3rd-party headers are EXPECTED and not a bug
in OUR pipeline), but a missing tarball, a libclang load failure, a bad subtree
name, or a SQLite open failure RAISES immediately. No silent degraded index.

Resumability
------------
``symbol_index_libs(lib TEXT PRIMARY KEY, done INT, n_funcs INT, n_types INT,
elapsed_s REAL, ...)`` records each completed base-lib. ``--resume`` (default)
skips libs already marked done. Per-lib extraction goes to a temp dir that is
deleted after the lib is indexed (bounded disk: ~1 base-lib of source at a time).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Locate the indexer module so we reuse its libclang parse helpers verbatim.
# --------------------------------------------------------------------------- #
MLX_ROOT = Path(__file__).resolve().parents[2]
INDEXER_DIR = MLX_ROOT / "tools" / "clang_indexer"
if str(INDEXER_DIR) not in sys.path:
    sys.path.insert(0, str(INDEXER_DIR))

# These imports REQUIRE libclang to be importable; a failure RAISES (RULE #1).
import index_project as ip  # noqa: E402

DEFAULT_TARBALL = Path("/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst")
DEFAULT_OUTPUT = MLX_ROOT / "outputs" / "crossrepo" / "global_symbols.sqlite"

# --------------------------------------------------------------------------- #
# Selected base libraries -> their tarball subtree name(s) under cpp_all/<name>/.
# Subtree names are AUTHORITATIVE from the corpus census provider lists
# (outputs/crossrepo/base_lib_usage_ranked.json) — verified present in tarball.
#
# Each entry: lib-key -> dict(
#   subtrees=[<cpp_all subtree dir names>],
#   tier='A1'|'A2',
#   public_only=bool,           # A2 libs: only header-declared/exported symbols
#   namespace_prefixes=[...],   # symbol-namespace cleanliness (prefixed cross-link)
#   include_path_markers=[...], # (optional) RESTRICT extraction to members whose
#                               #   path contains one of these — for the huge
#                               #   gcc/llvm monorepos this pins the cut to the
#                               #   C++ stdlib header trees and OFF gcc's compiler
#                               #   / libiberty internals (the C2 root cause).
#   index_extensionless_headers=bool,  # (optional) also index extensionless std
#                               #   headers (<vector>, <type_traits>, ...).
# )
# A1 = tractable full public-API index. A2 = huge, public symbols only.
# --------------------------------------------------------------------------- #
# ``lang`` controls how headers (.h/.inc) are parsed: 'c++' (default for the C++
# libs — header-only template libs like boost/eigen/fmt MUST parse as C++ or the
# templated public API yields zero symbols) or 'c' (glib/openssl/libc are C).
BASE_LIBS: dict[str, dict] = {
    # ---- A1 (high-value, tractable) ----
    "boost":      {"subtrees": ["boost"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["boost::"]},
    "abseil":     {"subtrees": ["abseil-cpp"],   "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["absl::"]},
    "folly":      {"subtrees": ["folly"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["folly::"]},
    "openssl":    {"subtrees": ["openssl"],      "tier": "A1", "public_only": False, "lang": "c",   "namespace_prefixes": []},
    "boringssl":  {"subtrees": ["boringssl"],    "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": []},
    "protobuf":   {"subtrees": ["protobuf"],     "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["google::protobuf::"]},
    "eigen":      {"subtrees": ["eigen"],        "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["Eigen::"]},
    "fmt":        {"subtrees": ["fmt"],          "tier": "A1", "public_only": False, "lang": "c++", "namespace_prefixes": ["fmt::"]},
    "glib":       {"subtrees": ["glib"],         "tier": "A1", "public_only": False, "lang": "c",   "namespace_prefixes": []},
    # ---- A2 (huge / template-heavy) — PUBLIC symbols only ----
    # std: gcc-mirror and llvm-project are FULL compiler monorepos (the entire GCC
    # / LLVM source trees). Without include_path_markers the per-lib file cap is
    # spent on gcc's compiler + libiberty internals and ZERO real C++ standard
    # library headers get indexed (the C2 defect). Pin the cut to the libstdc++ /
    # libc++ / microsoft-STL public header trees only, and index their
    # EXTENSIONLESS public headers (<vector>, <type_traits>, ...).
    "std":        {"subtrees": ["STL", "stl", "gcc-mirror", "llvm-project"],
                   "tier": "A2", "public_only": True, "lang": "c++",
                   "namespace_prefixes": ["std::"],
                   "include_path_markers": [
                       "/libstdc++-v3/include/",    # gcc-mirror: libstdc++ public headers
                       "/libstdc++-v3/libsupc++/",  # gcc-mirror: <typeinfo>/<exception>/<new>
                       "/libcxx/include/",          # llvm-project: libc++ public headers
                       "/stl/inc/",                 # microsoft/STL public headers
                   ],
                   "index_extensionless_headers": True},
    # libc: C libraries have NO namespace; keep by-name public symbols (no prefix
    # filter). glibc/musl public surface lives in normal .h headers.
    "libc":       {"subtrees": ["glibc", "musl"],
                   "tier": "A2", "public_only": True, "lang": "c", "namespace_prefixes": []},
}

A1_LIBS = [k for k, v in BASE_LIBS.items() if v["tier"] == "A1"]
A2_LIBS = [k for k, v in BASE_LIBS.items() if v["tier"] == "A2"]

# Per-lib hard bounds (BuildIndex cost containment; RULE #1: bounded, no explosion)
MAX_FILES_PER_LIB_DEFAULT = 60_000       # cap files indexed per base-lib
MAX_BYTES_PER_FILE = 500_000             # mirror index_project's per-file cap
PARSE_TIMEOUT_S = 60                      # per-file libclang timeout
PROGRESS_EVERY = 2000

# A2 std/libc: index ONLY symbols declared in headers (public surface). For std
# the public headers are the libstdc++ <bits/...>/<...> includes + libc++'s; for
# libc the installed headers. We treat any file under an "include"-ish path or a
# header extension as public-API-bearing. Bodies of huge template impls in .tcc
# files are still indexed when public_only and they declare exported symbols.


# --------------------------------------------------------------------------- #
# Public-symbol classification.
# --------------------------------------------------------------------------- #
_HEADER_EXTS = {".h", ".hpp", ".hxx", ".hh", ".h++", ".inl", ".inc", ".ipp", ".tcc"}

# Tokens that mark a symbol as INTERNAL/private — never part of the public API.
# Conservative: a leading underscore-uppercase or a "detail"/"internal"/"impl"
# namespace segment in the qualified name.
_INTERNAL_QNAME_SEG = re.compile(
    r"(^|::)(detail|internal|impl|_internal|__detail|__internal)(::|$)", re.IGNORECASE
)
_RESERVED_LEADING = re.compile(r"(^|::)(_[A-Z]|__)")  # __foo / _Foo reserved names
_INLINE_NAMESPACE_SEGMENTS = {
    "__1",
    "__2",
    "__3",
    "__cxx11",
}


def normalize_inline_namespace_qname(qname: str) -> str:
    """Canonicalize common C++ inline ABI namespaces in qualified names.

    libstdc++/libc++ expose public symbols through inline implementation
    namespaces such as ``std::__1`` or ``std::__cxx11``.  The cross-repo lookup
    should index those under the stable public spelling so callers resolving
    ``std::basic_string`` or ``std::vector`` do not miss them because of a
    library-specific ABI namespace.
    """
    if not qname:
        return qname
    parts = [part for part in qname.split("::") if part not in _INLINE_NAMESPACE_SEGMENTS]
    return "::".join(parts)


def _is_public_header_path(rel_file: str) -> bool:
    ext = os.path.splitext(rel_file)[1].lower()
    if ext in _HEADER_EXTS:
        return True
    parts = [part for part in rel_file.replace("\\", "/").split("/") if part]
    return "include" in parts


def is_public_symbol(
    qname: str,
    rel_file: str,
    public_only: bool,
    namespace_prefixes: Sequence[str] = (),
) -> bool:
    """Decide whether a parsed symbol belongs in the cross-link index.

    For A1 libs (public_only=False) we accept any non-internal qualified name
    that belongs to the lib's configured namespace prefixes, when provided.
    For A2 libs (public_only=True) we additionally REQUIRE the defining file to
    be a header (public API surface) and reject reserved/internal names — this
    bounds the std/libc index to the public symbol surface and keeps cost down.
    """
    qname = normalize_inline_namespace_qname(qname)
    if not qname:
        return False
    if namespace_prefixes and not any(qname.startswith(prefix) for prefix in namespace_prefixes):
        return False
    if _INTERNAL_QNAME_SEG.search(qname):
        return False
    if public_only:
        if not _is_public_header_path(rel_file):
            return False
        # Reserved-name implementation symbols (__copy, _M_..., _S_...) are not
        # the public API the model should learn; drop them for the A2 surface.
        if _RESERVED_LEADING.search(qname):
            return False
    return True


# --------------------------------------------------------------------------- #
# Trivial / tiny inline std body filter (skip — too small to be worth a chunk).
# --------------------------------------------------------------------------- #
def normalized_body_len(text: str) -> int:
    s = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return len(s)


# --------------------------------------------------------------------------- #
# SQLite global symbol store.
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    qname        TEXT NOT NULL,
    base_lib     TEXT NOT NULL,   -- logical lib key (boost/std/libc/...)
    base_repo    TEXT NOT NULL,   -- the cpp_all subtree the def came from
    kind         INTEGER NOT NULL,-- 2=function, 4=record/enum, 7=typedef/using
    sym_type     TEXT NOT NULL,   -- 'func' | 'type'
    file         TEXT NOT NULL,
    line         INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    is_public    INTEGER NOT NULL,
    token_est    INTEGER NOT NULL,
    body_len     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (qname, sym_type)
);
CREATE INDEX IF NOT EXISTS idx_symbols_qname ON symbols(qname);
CREATE INDEX IF NOT EXISTS idx_symbols_lib   ON symbols(base_lib);

CREATE TABLE IF NOT EXISTS symbol_index_libs (
    lib       TEXT PRIMARY KEY,
    tier      TEXT NOT NULL,
    done      INTEGER NOT NULL DEFAULT 0,
    n_files   INTEGER NOT NULL DEFAULT 0,
    n_funcs   INTEGER NOT NULL DEFAULT 0,
    n_types   INTEGER NOT NULL DEFAULT 0,
    n_errors  INTEGER NOT NULL DEFAULT 0,
    elapsed_s REAL    NOT NULL DEFAULT 0,
    subtrees  TEXT    NOT NULL DEFAULT '',
    finished_utc TEXT NOT NULL DEFAULT ''
);
"""


class GlobalSymbolStore:
    """Resumable SQLite store for cross-repo base-lib symbol definitions.

    Read path used by index_project at doc-build time is intentionally tiny:
    ``lookup(qname)`` returns the function def (or None). The write path is used
    only by THIS builder.
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        if read_only:
            uri = f"file:{path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=60.0)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---- write path (builder) ----
    def lib_done(self, lib: str) -> bool:
        row = self.conn.execute(
            "SELECT done FROM symbol_index_libs WHERE lib=?", (lib,)
        ).fetchone()
        return bool(row and row[0])

    def insert_symbols(self, rows: list[tuple]) -> None:
        # rows: (qname, base_lib, base_repo, kind, sym_type, file, line,
        #        end_line, is_public, token_est, body_len, text)
        self.conn.executemany(
            "INSERT OR IGNORE INTO symbols "
            "(qname, base_lib, base_repo, kind, sym_type, file, line, end_line, "
            " is_public, token_est, body_len, text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def mark_lib_done(self, lib: str, tier: str, *, n_files: int, n_funcs: int,
                      n_types: int, n_errors: int, elapsed_s: float,
                      subtrees: list[str]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO symbol_index_libs "
            "(lib, tier, done, n_files, n_funcs, n_types, n_errors, elapsed_s, "
            " subtrees, finished_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lib, tier, 1, n_files, n_funcs, n_types, n_errors, elapsed_s,
             ",".join(subtrees), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ---- read path (consulted by index_project) ----
    def lookup(self, qname: str) -> dict | None:
        row = self.conn.execute(
            "SELECT qname, base_lib, base_repo, kind, sym_type, file, line, "
            "end_line, is_public, token_est, body_len, text "
            "FROM symbols WHERE qname=? AND sym_type='func' LIMIT 1",
            (qname,),
        ).fetchone()
        if row is None:
            return None
        return {
            "qname": row[0], "base_lib": row[1], "base_repo": row[2],
            "kind": row[3], "sym_type": row[4], "file": row[5], "line": row[6],
            "end_line": row[7], "is_public": row[8], "token_est": row[9],
            "body_len": row[10], "text": row[11],
        }

    def counts(self) -> dict:
        cur = self.conn.execute("SELECT base_lib, sym_type, COUNT(*) FROM symbols "
                                "GROUP BY base_lib, sym_type")
        per: dict[str, dict[str, int]] = {}
        for lib, st, n in cur.fetchall():
            per.setdefault(lib, {"func": 0, "type": 0})[st] = n
        return per

    def lib_stats(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT lib, tier, done, n_files, n_funcs, n_types, n_errors, "
            "elapsed_s, subtrees FROM symbol_index_libs ORDER BY tier, lib")
        out = []
        for r in cur.fetchall():
            out.append({"lib": r[0], "tier": r[1], "done": r[2], "n_files": r[3],
                        "n_funcs": r[4], "n_types": r[5], "n_errors": r[6],
                        "elapsed_s": r[7], "subtrees": r[8]})
        return out


# --------------------------------------------------------------------------- #
# Streaming extraction of selected subtrees from the tarball.
# --------------------------------------------------------------------------- #
# Extensionless files that are documentation / build / metadata, NOT headers.
_NON_HEADER_BASENAMES = frozenset({
    "readme", "license", "licence", "copying", "copyright", "authors", "news",
    "todo", "changelog", "changes", "install", "notice", "version", "credits",
    "thanks", "bugs", "faq", "makefile", "gnumakefile", "dockerfile", "owners",
    "contributing", "manifest",
})
# Path components that hold the EXTENSIONLESS C++ stdlib public headers:
#   libstdc++/libc++/microsoft-STL use ``include``/``inc``; libstdc++ also keeps
#   <typeinfo>/<exception>/<new> under ``libsupc++``.
_HEADER_DIR_COMPONENTS = frozenset({"include", "inc", "libsupc++"})


def _is_extensionless_header(name: str) -> bool:
    """True if ``name`` is an EXTENSIONLESS C++ standard-library header.

    The libstdc++/libc++ public surface (<vector>, <type_traits>, <memory>,
    <typeinfo>, ...) has NO file extension, so the normal extension filter skips
    it entirely. We only treat an extensionless file as a header when it lives
    under a stdlib header dir component AND its basename is not a well-known
    doc/build file — this avoids grabbing extensionless license/readme/makefile
    noise.
    """
    base = os.path.basename(name)
    if not base or base.startswith("."):
        return False
    if os.path.splitext(base)[1] != "":
        return False  # has a real extension
    if base.lower() in _NON_HEADER_BASENAMES:
        return False
    parts = set(Path(name.replace("\\", "/")).parts)
    return bool(parts & _HEADER_DIR_COMPONENTS)


def _member_is_wanted(name: str, index_exts, *, path_markers, extensionless: bool) -> bool:
    """Decide whether a tarball member (by path) should be extracted for a lib.

    ``path_markers`` (when non-empty) RESTRICTS extraction to members whose path
    contains one of the markers. This is the C2 fix for the huge gcc/llvm
    monorepos: ``std`` pulls ONLY the C++ standard-library header trees
    (``/libstdc++-v3/include/``, ``/libcxx/include/``, ``/stl/inc/``) and NEVER
    gcc's compiler / libiberty internals, so the per-lib file cap is spent on real
    ``std::`` headers instead of compiler-internal noise.

    ``extensionless`` additionally admits extensionless std headers (<vector>…).
    """
    if path_markers and not any(m in name for m in path_markers):
        return False
    ext = os.path.splitext(name)[1].lower()
    if ext in index_exts:
        return True
    if extensionless and ext == "" and _is_extensionless_header(name):
        return True
    return False


def _find_extensionless_headers(dest: Path) -> list[str]:
    """Discover EXTENSIONLESS public std headers under ``dest`` (<vector>, …).

    ``ip.find_cpp_files`` filters by INDEX_EXTENSIONS and therefore skips the
    extensionless libstdc++/libc++ public headers entirely. We augment discovery
    with them (bounded by the same per-file size cap) so they are parsed.
    """
    out: list[str] = []
    for root, _dirs, files in os.walk(dest):
        for fn in files:
            full = os.path.join(root, fn)
            if not _is_extensionless_header(full):
                continue
            try:
                if os.path.getsize(full) > MAX_BYTES_PER_FILE:
                    continue
            except OSError:
                continue
            out.append(full)
    return out


def extract_subtrees(tarball: Path, subtrees: list[str], dest: Path,
                     *, max_files: int, member_cap: int | None = None,
                     path_markers: list[str] | None = None,
                     extensionless: bool = False) -> int:
    """Stream the tarball and extract members under cpp_all/<subtree>/ that are
    C/C++ source/header files, up to ``max_files`` per call.

    Returns the number of files extracted. Streaming (mode 'r|') so we never
    materialize the 235 GiB tarball; we filter by member name on the fly.
    RULE #1: a missing/zero subtree match RAISES at the caller (we just report 0).
    """
    if not tarball.exists():
        raise FileNotFoundError(f"tarball not found: {tarball}")
    wanted_prefixes = tuple(f"cpp_all/{s}/" for s in subtrees)
    dest.mkdir(parents=True, exist_ok=True)
    index_exts = ip.INDEX_EXTENSIONS

    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(tarball)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    extracted = 0
    members_seen = 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        for member in tf:
            members_seen += 1
            if member_cap is not None and members_seen > member_cap:
                break
            if not member.isfile():
                continue
            name = member.name
            if not name.startswith(wanted_prefixes):
                continue
            if not _member_is_wanted(name, index_exts, path_markers=path_markers,
                                     extensionless=extensionless):
                continue
            if member.size > MAX_BYTES_PER_FILE:
                continue
            # Sanitize path: strip leading cpp_all/ to keep dest shallow.
            rel = name[len("cpp_all/"):]
            out_path = dest / rel
            if ".." in Path(rel).parts:
                continue  # never escape dest
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            with open(out_path, "wb") as w:
                w.write(fobj.read())
            extracted += 1
            if extracted >= max_files:
                break
            if extracted % PROGRESS_EVERY == 0:
                print(f"    extracted {extracted} files "
                      f"({members_seen} members streamed)...", file=sys.stderr,
                      flush=True)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return extracted


def extract_many_subtrees(tarball: Path, lib_specs: dict[str, dict], dest_root: Path,
                          *, max_files: int, member_cap: int | None = None) -> dict[str, int]:
    """Extract the subtrees of MULTIPLE libs in ONE streaming tarball pass.

    Routes each matching member to ``dest_root/<lib>/<rel>``. This is the path
    used for a full multi-lib build so the 235 GiB tarball is streamed exactly
    ONCE instead of once per lib. Returns {lib: n_files_extracted}.
    """
    if not tarball.exists():
        raise FileNotFoundError(f"tarball not found: {tarball}")
    index_exts = ip.INDEX_EXTENSIONS
    # subtree-name -> lib (a subtree maps to exactly one lib here)
    prefix_to_lib: dict[str, str] = {}
    for lib, spec in lib_specs.items():
        for sub in spec["subtrees"]:
            prefix_to_lib[f"cpp_all/{sub}/"] = lib
    prefixes = tuple(prefix_to_lib.keys())
    counts: dict[str, int] = {lib: 0 for lib in lib_specs}
    dest_root.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(tarball)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    members_seen = 0
    total = 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        for member in tf:
            members_seen += 1
            if member_cap is not None and members_seen > member_cap:
                break
            if not member.isfile():
                continue
            name = member.name
            matched_prefix = None
            for p in prefixes:
                if name.startswith(p):
                    matched_prefix = p
                    break
            if matched_prefix is None:
                continue
            lib = prefix_to_lib[matched_prefix]
            if counts[lib] >= max_files:
                continue
            lspec = lib_specs[lib]
            if not _member_is_wanted(
                    name, index_exts,
                    path_markers=lspec.get("include_path_markers"),
                    extensionless=bool(lspec.get("index_extensionless_headers"))):
                continue
            if member.size > MAX_BYTES_PER_FILE:
                continue
            rel = name[len("cpp_all/"):]
            if ".." in Path(rel).parts:
                continue
            out_path = dest_root / lib / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            with open(out_path, "wb") as w:
                w.write(fobj.read())
            counts[lib] += 1
            total += 1
            if total % PROGRESS_EVERY == 0:
                done = {k: v for k, v in counts.items() if v}
                print(f"    extracted {total} files ({members_seen} members "
                      f"streamed): {done}", file=sys.stderr, flush=True)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return counts


# --------------------------------------------------------------------------- #
# Per-file libclang parse worker (isolated subprocess so a segfault is local).
# --------------------------------------------------------------------------- #
def _parse_file_worker(args_tuple):
    filepath, project_dir, std_arg, lang, include_dirs = args_tuple
    sys.setrecursionlimit(50000)
    ip._configure_libclang()
    clang_index = ip.Index.create()
    ext = os.path.splitext(filepath)[1].lower()
    # Language is decided PER LIB (lang), not per extension: header-only C++
    # template libs (boost/eigen/fmt) MUST parse as C++ or their public API
    # yields zero symbols. A genuine .c file in a C++ lib is still parsed as C.
    if ext == ".c" or (lang == "c" and ext in {".h", ".inc", ".inl"}):
        compile_args = ["-x", "c", "-std=c11"]
    else:
        compile_args = ["-x", "c++", std_arg]
    # Self-include dirs so a header resolves its own siblings (cuts parse errors
    # + time massively for header-heavy libs). PARSE_INCOMPLETE keeps going past
    # unresolved system includes.
    for inc in include_dirs:
        compile_args += ["-I", inc]
    funcs, types = ip.parse_translation_unit(
        filepath, clang_index, compile_args, project_dir
    )
    return (
        [f.to_dict() for f in funcs],
        [t.to_dict() for t in types],
    )


@dataclass
class LibResult:
    lib: str
    tier: str
    subtrees: list[str]
    n_files: int = 0
    n_funcs: int = 0
    n_types: int = 0
    n_errors: int = 0
    elapsed_s: float = 0.0
    func_qnames: set = field(default_factory=set)


def _discover_include_dirs(dest: Path, subtrees: list[str]) -> list[str]:
    """Common self-include roots so a base-lib header resolves its siblings.

    Adds the extraction root, each subtree root, and any ``include``/``src``
    subdir found one or two levels down. Bounded (no deep walk).
    """
    inc: list[str] = [str(dest)]
    for sub in subtrees:
        base = dest / sub
        if base.is_dir():
            inc.append(str(base))
            for cand in ("include", "src", "src/include"):
                p = base / cand
                if p.is_dir():
                    inc.append(str(p))
            # one level down: <sub>/<pkg>/include (e.g. abseil-cpp/absl/...)
            try:
                for child in base.iterdir():
                    if child.is_dir() and (child / "include").is_dir():
                        inc.append(str(child / "include"))
            except OSError:
                pass
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for d in inc:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def index_lib_from_dir(lib: str, spec: dict, dest: Path, store: GlobalSymbolStore,
                       *, workers: int, std_arg: str) -> LibResult:
    """Parse an ALREADY-EXTRACTED base-lib dir and store its public symbols."""
    t0 = time.time()
    subtrees = spec["subtrees"]
    tier = spec["tier"]
    public_only = spec["public_only"]
    namespace_prefixes = tuple(spec.get("namespace_prefixes") or ())
    lang = spec.get("lang", "c++")
    res = LibResult(lib=lib, tier=tier, subtrees=subtrees)

    cpp_files = ip.find_cpp_files(str(dest))
    if spec.get("index_extensionless_headers"):
        # ip.find_cpp_files skips extensionless files; the libstdc++/libc++ public
        # headers (<vector>, <type_traits>, ...) are extensionless, so add them or
        # the std public surface is never parsed (a C2 root cause).
        seen = set(cpp_files)
        for fp in _find_extensionless_headers(dest):
            if fp not in seen:
                cpp_files.append(fp)
                seen.add(fp)
    res.n_files = len(cpp_files)
    if res.n_files == 0:
        raise RuntimeError(
            f"[{lib}] 0 parseable files in {dest} — extraction produced nothing "
            f"(subtree name(s) {subtrees} wrong/absent?). RULE #1: fail loud."
        )
    project_dir = str(dest)
    include_dirs = _discover_include_dirs(dest, subtrees)
    print(f"  [{lib}] lang={lang} include_dirs={len(include_dirs)} "
          f"files_to_parse={res.n_files} ({workers} workers)",
          file=sys.stderr, flush=True)

    batch: list[tuple] = []
    BATCH_COMMIT = 5000
    parsed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(_parse_file_worker,
                      (fp, project_dir, std_arg, lang, include_dirs)): fp
            for fp in cpp_files
        }
        for fut in as_completed(futures):
            fp = futures[fut]
            rel = os.path.relpath(fp, project_dir)
            base_repo = rel.split(os.sep)[0] if os.sep in rel else subtrees[0]
            try:
                func_dicts, type_dicts = fut.result(timeout=PARSE_TIMEOUT_S * 4)
            except TimeoutError:
                res.n_errors += 1
                continue
            except Exception:
                res.n_errors += 1
                continue
            for d in func_dicts:
                qn = normalize_inline_namespace_qname(d["qualified_name"])
                frel = d["file"]
                if not is_public_symbol(qn, frel, public_only, namespace_prefixes):
                    continue
                body = d["text"]
                blen = normalized_body_len(body)
                # Skip trivial/tiny inline std bodies (not worth a chunk).
                if public_only and blen < 24:
                    continue
                if blen < 12:
                    continue
                tok = ip.estimate_tokens(body)
                batch.append((qn, lib, base_repo, 2, "func", frel,
                              d["line"], d.get("end_line", 0), 1, tok, blen, body))
                res.n_funcs += 1
                res.func_qnames.add(qn)
            for d in type_dicts:
                qn = normalize_inline_namespace_qname(d["qualified_name"])
                frel = d["file"]
                if not is_public_symbol(qn, frel, public_only, namespace_prefixes):
                    continue
                body = d["text"]
                blen = normalized_body_len(body)
                if blen < 8:
                    continue
                tok = ip.estimate_tokens(body)
                batch.append((qn, lib, base_repo, d["kind"], "type", frel,
                              d["line"], d.get("end_line", 0), 1, tok, blen, body))
                res.n_types += 1
            parsed += 1
            if len(batch) >= BATCH_COMMIT:
                store.insert_symbols(batch)
                store.commit()
                batch = []
            if parsed % PROGRESS_EVERY == 0:
                print(f"    [{lib}] parsed {parsed}/{res.n_files} files, "
                      f"{res.n_funcs} funcs / {res.n_types} types, "
                      f"{res.n_errors} errors", file=sys.stderr, flush=True)
    if batch:
        store.insert_symbols(batch)
        store.commit()

    res.elapsed_s = time.time() - t0
    return res


def index_one_lib(lib: str, spec: dict, tarball: Path, store: GlobalSymbolStore,
                  *, workers: int, max_files: int, std_arg: str,
                  member_cap: int | None) -> LibResult:
    """Extract (own tarball pass) + index ONE base-lib. Used for single-lib runs."""
    with tempfile.TemporaryDirectory(prefix=f"gsi_{lib}_") as td:
        dest = Path(td)
        print(f"  [{lib}] extracting subtrees {spec['subtrees']} "
              f"(cap {max_files} files)...", file=sys.stderr, flush=True)
        n_extracted = extract_subtrees(
            tarball, spec["subtrees"], dest, max_files=max_files,
            member_cap=member_cap,
            path_markers=spec.get("include_path_markers"),
            extensionless=bool(spec.get("index_extensionless_headers")),
        )
        if n_extracted == 0:
            raise RuntimeError(
                f"[{lib}] extracted 0 files from subtrees {spec['subtrees']} — "
                f"subtree name(s) wrong or absent from tarball (RULE #1)."
            )
        print(f"  [{lib}] extracted {n_extracted} files; indexing...",
              file=sys.stderr, flush=True)
        return index_lib_from_dir(lib, spec, dest, store,
                                  workers=workers, std_arg=std_arg)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tarball", default=str(DEFAULT_TARBALL),
                   help="Path to the corpus tarball (zstd).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Path to the global symbol SQLite store.")
    p.add_argument("--libs", default="A1",
                   help="Comma list of lib keys to index, or 'A1'/'A2'/'all'. "
                        "(default A1 — the tractable high-value set).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                   help="Parallel parse workers.")
    p.add_argument("--max-files-per-lib", type=int, default=MAX_FILES_PER_LIB_DEFAULT,
                   help=f"Cap files indexed per base-lib (default {MAX_FILES_PER_LIB_DEFAULT}).")
    p.add_argument("--std-arg", default="-std=c++17",
                   help="C++ std flag for parsing (default -std=c++17).")
    p.add_argument("--member-cap", type=int, default=None,
                   help="Stop streaming the tarball after N members (debug/bounded).")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-index libs even if already marked done.")
    p.add_argument("--per-lib-pass", action="store_true",
                   help="Stream the tarball ONCE PER LIB instead of one shared "
                        "pass (slower; for targeted single-lib debugging).")
    p.add_argument("--keep-extract", action="store_true",
                   help="Keep the extracted source dirs after indexing (debug).")
    p.add_argument("--work-dir", default=None,
                   help="Staging root for extracted source (default: a fresh "
                        "mkdtemp deleted after the run).")
    p.add_argument("--libclang-path", default=None)
    p.add_argument("--report-only", action="store_true",
                   help="Print store counts/cost and exit (no indexing).")
    return p.parse_args(argv)


def resolve_libs(spec: str) -> list[str]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(BASE_LIBS.keys())
    if spec.upper() == "A1":
        return list(A1_LIBS)
    if spec.upper() == "A2":
        return list(A2_LIBS)
    libs = [x.strip() for x in spec.split(",") if x.strip()]
    for lk in libs:
        if lk not in BASE_LIBS:
            raise SystemExit(f"unknown lib key {lk!r}; known: {list(BASE_LIBS)}")
    return libs


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ip._configure_libclang(args.libclang_path)

    tarball = Path(args.tarball)
    store = GlobalSymbolStore(args.output, read_only=args.report_only)

    if args.report_only:
        _print_report(store)
        store.close()
        return 0

    libs = resolve_libs(args.libs)
    print(f"Building GLOBAL symbol index at {args.output}", file=sys.stderr)
    print(f"  tarball : {tarball}", file=sys.stderr)
    print(f"  libs    : {libs}", file=sys.stderr)
    print(f"  workers : {args.workers}  max_files/lib: {args.max_files_per_lib}",
          file=sys.stderr)

    resume = not args.no_resume
    todo = [lib for lib in libs if not (resume and store.lib_done(lib))]
    for lib in libs:
        if lib not in todo:
            print(f"  SKIP (done) {lib}", file=sys.stderr)
    if not todo:
        print("All requested libs already done.", file=sys.stderr)
        _print_report(store)
        store.close()
        return 0

    results: list[LibResult] = []

    def _finish(lib: str, spec: dict, res: LibResult) -> None:
        store.mark_lib_done(
            lib, spec["tier"], n_files=res.n_files, n_funcs=res.n_funcs,
            n_types=res.n_types, n_errors=res.n_errors, elapsed_s=res.elapsed_s,
            subtrees=res.subtrees,
        )
        results.append(res)
        print(f"  DONE {lib}: files={res.n_files} funcs={res.n_funcs} "
              f"types={res.n_types} errors={res.n_errors} "
              f"elapsed={res.elapsed_s:.1f}s", file=sys.stderr, flush=True)

    # ONE tarball pass extracts ALL todo libs to a shared staging root (the
    # 235 GiB tarball is streamed ONCE, not once per lib), unless --per-lib-pass.
    if len(todo) > 1 and not args.per_lib_pass:
        extract_root = Path(args.work_dir) if args.work_dir else Path(
            tempfile.mkdtemp(prefix="gsi_extract_"))
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            todo_specs = {lib: BASE_LIBS[lib] for lib in todo}
            print(f"\n=== single-pass extraction of {todo} -> {extract_root} ===",
                  file=sys.stderr, flush=True)
            t_ext = time.time()
            counts = extract_many_subtrees(
                tarball, todo_specs, extract_root,
                max_files=args.max_files_per_lib, member_cap=args.member_cap,
            )
            print(f"  extraction done in {time.time()-t_ext:.1f}s: {counts}",
                  file=sys.stderr, flush=True)
            for lib in todo:
                spec = BASE_LIBS[lib]
                if counts.get(lib, 0) == 0:
                    raise RuntimeError(
                        f"[{lib}] extracted 0 files from {spec['subtrees']} — "
                        f"subtree name(s) wrong/absent (RULE #1: fail loud)."
                    )
                print(f"\n=== indexing base-lib: {lib} (tier {spec['tier']}) ===",
                      file=sys.stderr, flush=True)
                res = index_lib_from_dir(
                    lib, spec, extract_root / lib, store,
                    workers=args.workers, std_arg=args.std_arg,
                )
                _finish(lib, spec, res)
                if not args.keep_extract:
                    import shutil
                    shutil.rmtree(extract_root / lib, ignore_errors=True)
        finally:
            if not args.keep_extract and not args.work_dir:
                import shutil
                shutil.rmtree(extract_root, ignore_errors=True)
    else:
        for lib in todo:
            spec = BASE_LIBS[lib]
            print(f"\n=== indexing base-lib: {lib} (tier {spec['tier']}) ===",
                  file=sys.stderr, flush=True)
            res = index_one_lib(
                lib, spec, tarball, store,
                workers=args.workers, max_files=args.max_files_per_lib,
                std_arg=args.std_arg, member_cap=args.member_cap,
            )
            _finish(lib, spec, res)

    print(f"\n{'='*64}", file=sys.stderr)
    _print_report(store)
    store.close()
    return 0


def _print_report(store: GlobalSymbolStore) -> None:
    counts = store.counts()
    lib_stats = store.lib_stats()
    report = {
        "store_path": store.path,
        "per_lib_symbol_counts": counts,
        "lib_index_stats": lib_stats,
        "totals": {
            "funcs": sum(c.get("func", 0) for c in counts.values()),
            "types": sum(c.get("type", 0) for c in counts.values()),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
