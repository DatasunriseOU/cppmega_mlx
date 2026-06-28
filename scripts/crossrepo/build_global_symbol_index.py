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
import hashlib
import json
import os
import re
import shutil
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
    # The std umbrella headers DO NOT parse with the bare A1 args — they need a
    # real C++ stdlib compile env (clang builtins + a C sysroot + >= c++20) AND
    # the parser must be told to KEEP std:: type names (allow_system_types), or
    # libclang yields only a few free functions and ZERO class templates (the old
    # "std=202 funcs/0 types" defect). See _cxx_stdlib_sysroot_args / the std
    # experiment in build notes: with this env libc++ <vector> yields std::vector
    # + std::vector::push_back (and every container's public members).
    #
    # PRIMARY target = libc++ (llvm-project/libcxx/include): self-contained
    # (ships its own __config), needs ONLY clang builtins + a C lib -> it parses
    # and supplies the std:: surface.
    # libstdc++ (gcc-mirror/libstdc++-v3/include): its <bits/c++config.h> is
    # GENERATED by configure and is NOT in the source tree, so most libstdc++
    # umbrella headers fail to parse standalone here -> they are BEST-EFFORT
    # (per-file parse errors, counted in the report, never silently dropped). We
    # deliberately do NOT synthesize a c++config.h: libc++ already provides the
    # full std:: surface and dedup (INSERT OR IGNORE) merges any overlap.
    # MS-STL (microsoft/STL/stl/inc): needs MSVC intrinsics -> BEST-EFFORT via
    # -fms-compatibility (msvc_path_marker); headers needing genuine MSVC
    # builtins fail per-file (counted), not silently skipped.
    "std":        {"subtrees": ["STL", "stl", "gcc-mirror", "llvm-project"],
                   "tier": "A2", "public_only": True, "lang": "c++",
                   "namespace_prefixes": ["std::"],
                   "include_path_markers": [
                       "/libstdc++-v3/include/",    # gcc-mirror: libstdc++ public headers
                       "/libstdc++-v3/libsupc++/",  # gcc-mirror: <typeinfo>/<exception>/<new>
                       "/libcxx/include/",          # llvm-project: libc++ public headers
                       "/stl/inc/",                 # microsoft/STL public headers
                   ],
                   "index_extensionless_headers": True,
                   # ---- per-lib C++ standard-library compile environment ----
                   # cxx_std is chosen at BUILD TIME by an explicit probe (see
                   # _probe_highest_cxx_std): we try the candidates below highest
                   # first and pick the highest one libclang ACTUALLY accepts on a
                   # real libc++ <ranges>/<format> parse. A SUPERSET std captures
                   # the c++23/26 standard surface (ranges/concepts/format/expected/
                   # flat_map). The static "cxx_std" is the documented FLOOR used
                   # only if no probe candidates are configured; the probe (which
                   # includes c++23 as its lowest rung) RAISES rather than silently
                   # dropping below c++23 (RULE #1).
                   "cxx_std": "-std=c++23",          # floor; probe bumps to c++26/2c when accepted
                   "cxx_std_probe_candidates": ("-std=c++26", "-std=c++2c", "-std=c++23"),
                   "needs_cxx_stdlib_env": True,      # inject clang builtins + C sysroot
                   "allow_system_types": True,        # KEEP std:: TYPE definitions
                   "stdlib_include_markers": [        # per-FILE -I root (no cross-tree mix)
                       "/libcxx/include/",
                       "/libstdc++-v3/include/",
                       "/libstdc++-v3/libsupc++/",
                       "/stl/inc/",
                   ],
                   "msvc_path_marker": "/stl/inc/"},  # MS-STL best-effort MSVC compat
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
    symbol_uid  TEXT NOT NULL PRIMARY KEY,
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
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_qname ON symbols(qname);
CREATE INDEX IF NOT EXISTS idx_symbols_qname_type ON symbols(qname, sym_type);
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
            self._require_current_schema()
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=60.0)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.executescript(SCHEMA)
            self._require_current_schema()
            self.conn.commit()

    def _require_current_schema(self) -> None:
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(symbols)").fetchall()
        }
        if "symbol_uid" not in cols:
            raise RuntimeError(
                f"{self.path}: global symbol index uses old qname-only schema; "
                "delete/rebuild it with scripts/crossrepo/build_global_symbol_index.py"
            )

    @staticmethod
    def _symbol_uid(row: tuple) -> str:
        qname, base_lib, base_repo, _kind, sym_type, file, line, end_line, *_rest = row
        text = row[-1]
        payload = "\x1f".join(
            str(part)
            for part in (
                base_lib,
                base_repo,
                sym_type,
                qname,
                file,
                line,
                end_line,
                hashlib.sha1(str(text).encode("utf-8")).hexdigest(),
            )
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # ---- write path (builder) ----
    def lib_done(self, lib: str) -> bool:
        row = self.conn.execute(
            "SELECT done FROM symbol_index_libs WHERE lib=?", (lib,)
        ).fetchone()
        return bool(row and row[0])

    def insert_symbols(self, rows: list[tuple]) -> None:
        # rows: (qname, base_lib, base_repo, kind, sym_type, file, line,
        #        end_line, is_public, token_est, body_len, text)
        keyed_rows = [(self._symbol_uid(row), *row) for row in rows]
        self.conn.executemany(
            "INSERT OR IGNORE INTO symbols "
            "(symbol_uid, qname, base_lib, base_repo, kind, sym_type, file, line, end_line, "
            " is_public, token_est, body_len, text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            keyed_rows,
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


def _tarball_fingerprint(tarball: Path) -> dict[str, object]:
    st = tarball.stat()
    return {
        "path": str(tarball.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _extract_cache_signature(
    tarball: Path,
    spec: dict,
    *,
    max_files: int,
    member_cap: int | None,
) -> dict[str, object]:
    return {
        "version": 1,
        "tarball": _tarball_fingerprint(tarball),
        "subtrees": list(spec["subtrees"]),
        "include_path_markers": list(spec.get("include_path_markers") or []),
        "index_extensionless_headers": bool(spec.get("index_extensionless_headers")),
        "max_files": max_files,
        "member_cap": member_cap,
        "max_bytes_per_file": MAX_BYTES_PER_FILE,
    }


def _extract_cache_manifest_path(cache_root: Path, lib: str) -> Path:
    return cache_root / lib / ".gsi_extract_complete.json"


def _read_extract_cache_manifest(cache_root: Path, lib: str) -> dict | None:
    path = _extract_cache_manifest_path(cache_root, lib)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt extraction cache manifest: {path}: {exc}") from exc


def _write_extract_cache_manifest(
    cache_root: Path,
    lib: str,
    signature: dict[str, object],
    *,
    count: int,
) -> None:
    path = _extract_cache_manifest_path(cache_root, lib)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(signature)
    payload.update({
        "lib": lib,
        "count": count,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _extract_cache_hit(
    tarball: Path,
    spec: dict,
    cache_root: Path,
    lib: str,
    *,
    max_files: int,
    member_cap: int | None,
) -> int | None:
    manifest = _read_extract_cache_manifest(cache_root, lib)
    if manifest is None:
        return None
    signature = _extract_cache_signature(
        tarball, spec, max_files=max_files, member_cap=member_cap
    )
    for key, value in signature.items():
        if manifest.get(key) != value:
            return None
    count = int(manifest.get("count", 0))
    if count <= 0:
        return None
    if not (cache_root / lib).is_dir():
        return None
    return count


def prepare_extraction_cache(
    tarball: Path,
    lib_specs: dict[str, dict],
    cache_root: Path,
    *,
    max_files: int,
    member_cap: int | None = None,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Return extracted source dirs, populating a reusable extraction cache.

    The expensive operation is the 235 GiB ``zstd | tar`` stream. This cache
    makes that step one-time per (tarball fingerprint, lib spec, max_files,
    member_cap): later rebuilds re-index the cached source dirs without reading
    the tarball again. The cache stores SOURCE FILES only; symbol extraction still
    uses the current parser/indexer code, so C++20/23/26 parser fixes take effect
    on every rebuild without re-extraction.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    dirs: dict[str, Path] = {}
    missing: dict[str, dict] = {}

    for lib, spec in lib_specs.items():
        hit = _extract_cache_hit(
            tarball, spec, cache_root, lib,
            max_files=max_files, member_cap=member_cap,
        )
        dirs[lib] = cache_root / lib
        if hit is None:
            missing[lib] = spec
        else:
            counts[lib] = hit
            print(f"  [{lib}] extraction cache hit: {hit} files",
                  file=sys.stderr, flush=True)

    if missing:
        for lib in missing:
            shutil.rmtree(cache_root / lib, ignore_errors=True)
        print(f"  extraction cache miss for {list(missing)} -> {cache_root}",
              file=sys.stderr, flush=True)
        extracted = extract_many_subtrees(
            tarball, missing, cache_root,
            max_files=max_files, member_cap=member_cap,
        )
        for lib, count in extracted.items():
            counts[lib] = count
            if count > 0:
                _write_extract_cache_manifest(
                    cache_root, lib,
                    _extract_cache_signature(
                        tarball, missing[lib], max_files=max_files,
                        member_cap=member_cap,
                    ),
                    count=count,
                )

    return dirs, counts


# --------------------------------------------------------------------------- #
# Per-lib C++ standard-library compile environment (the std cross-link fix).
#
# The std umbrella headers (<vector>, <string>, <map>, ...) of libstdc++/libc++
# DO NOT parse standalone with the bare ['-x','c++','-std=cXX'] + self-include
# args that work for the A1 libs: they pull in compiler BUILTIN headers
# (<stddef.h>, <stdint.h>, <stdarg.h>, __stddef_*.h) that live in clang's
# RESOURCE-DIR, and C library headers (<cstddef>->stddef.h, <cstdint>->stdint.h)
# that live in a C sysroot. Without those, libclang only ever sees a handful of
# free functions and ZERO class templates (the "std=202 funcs/0 types" defect).
#
# We therefore build, ONCE per build (in the parent), the extra include args:
#   -isystem <clang-resource-dir>/include   (compiler builtins)
#   -isystem <sysroot>/usr/include + -isysroot <sysroot>  (C library headers)
# and pass them to every std parse worker. PRIMARY std target is libc++ (it ships
# its own __config and needs only builtins + a C lib). See BASE_LIBS["std"].
# --------------------------------------------------------------------------- #
def _find_clang_resource_include() -> str | None:
    """Locate a clang RESOURCE-DIR ``include`` that holds the builtin headers
    (stddef.h/stdint.h/stdarg.h). Tries ``clang -print-resource-dir`` for any
    clang on PATH / known toolchains, then well-known Homebrew/Xcode locations.

    The bundled libclang (pip ``libclang``) ships NO resource headers, so we must
    borrow them from an installed toolchain. A minor clang-version skew between
    the libclang parser and the resource headers is tolerated by the modular
    __stddef_*/__stdarg_* split (clang >= 18).
    """
    import glob as _glob
    cands: list[str] = []
    for clang_bin in ("clang", "/opt/homebrew/opt/llvm/bin/clang",
                      "/usr/bin/clang", "cc"):
        try:
            rd = subprocess.check_output(
                [clang_bin, "-print-resource-dir"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if rd:
                cands.append(rd)
        except Exception:
            continue
    # Newest-first so a current toolchain (closest to modern libc++) wins.
    cands += sorted(_glob.glob("/opt/homebrew/Cellar/llvm/*/lib/clang/*"),
                    reverse=True)
    cands += sorted(_glob.glob(
        "/Applications/Xcode.app/Contents/Developer/Toolchains/"
        "XcodeDefault.xctoolchain/usr/lib/clang/*"), reverse=True)
    cands += sorted(_glob.glob("/usr/lib/clang/*"), reverse=True)
    for rd in cands:
        inc = os.path.join(rd, "include")
        if os.path.isfile(os.path.join(inc, "stddef.h")):
            return inc
    return None


def _find_c_sysroot() -> str | None:
    """Locate a C library sysroot for <cstddef>/<cstdint>/... resolution.

    macOS: the SDK from ``xcrun --show-sdk-path`` (headers under <sdk>/usr/include).
    Linux: ``/`` (headers under /usr/include). Returns the sysroot ROOT (the
    caller appends usr/include).
    """
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True,
            stderr=subprocess.DEVNULL).strip()
        if sdk and os.path.isdir(os.path.join(sdk, "usr", "include")):
            return sdk
    except Exception:
        pass
    for sdk in ("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk", "/"):
        if os.path.isdir(os.path.join(sdk, "usr", "include")):
            return sdk
    return None


def _cxx_stdlib_sysroot_args() -> list[str]:
    """Extra libclang args giving the C++ stdlib umbrella headers their compiler
    builtins + a C library. RAISES if no clang resource-dir is found (RULE #1:
    std cannot be indexed without builtins — fail loud, never silently degrade)."""
    res_inc = _find_clang_resource_include()
    if not res_inc:
        raise RuntimeError(
            "std cross-link build requires clang builtin headers (stddef.h/"
            "stdint.h) from a clang RESOURCE-DIR; none found via "
            "`clang -print-resource-dir` or known Homebrew/Xcode paths. "
            "Install LLVM or Xcode command-line tools. RULE #1: fail loud."
        )
    args = ["-isystem", res_inc]
    sysroot = _find_c_sysroot()
    if sysroot:
        args += ["-isystem", os.path.join(sysroot, "usr", "include")]
        if sysroot != "/":
            args += ["-isysroot", sysroot]
    return args


def _stdlib_include_root_for(filepath: str, markers: Sequence[str] | None) -> str | None:
    """For a std header, the SINGLE include root of the stdlib tree it lives in.

    The std lib bundles THREE separate C++ standard libraries (libc++, libstdc++,
    MS-STL). Their umbrella header names collide (<vector> exists in all three),
    so a libc++ header must be compiled with ONLY libc++'s include root on -I or
    it could resolve a libstdc++/MS-STL sibling (cross-tree contamination). Given
    e.g. ``.../llvm-project/libcxx/include/__vector/vector.h`` and marker
    ``/libcxx/include/`` this returns ``.../llvm-project/libcxx/include``.
    """
    if not markers:
        return None
    p = filepath.replace("\\", "/")
    for m in markers:
        idx = p.find(m)
        if idx != -1:
            return p[:idx] + m.rstrip("/")
    return None


def _find_libcxx_include_root(cpp_files: Sequence[str]) -> str | None:
    """The libc++ public-header include root (``.../libcxx/include``) from the
    already-discovered std source files. Used as the ``-I`` root for the cxx-std
    probe so ``#include <ranges>`` / ``#include <format>`` resolve to the REAL
    extracted libc++ umbrella headers (not a system libc++)."""
    for fp in cpp_files:
        root = _stdlib_include_root_for(fp, ("/libcxx/include/",))
        if root:
            return root
    return None


def _probe_highest_cxx_std(
    candidates: Sequence[str],
    include_root: str | None,
    sysroot_args: Sequence[str],
    *,
    lib: str,
) -> str:
    """Pick the HIGHEST ``-std=c++NN`` flag libclang ACTUALLY accepts, verified by
    a real probe parse of libc++ ``<ranges>`` + ``<format>``.

    Candidates are tried highest-first. A candidate is REJECTED only when libclang
    emits a *driver-level* std diagnostic for it — ``invalid value '<v>' in
    '-std=<v>'`` or ``unknown argument '<v>'`` — i.e. the toolchain does not know
    that standard; we then step DOWN to the next candidate. (A missing-header or
    template error is NOT a std-flag rejection — stepping down would not fix it —
    so it never triggers a step-down.) The first accepted candidate wins and is
    logged with its probe evidence. If NO candidate is accepted we RAISE (RULE #1:
    this is an EXPLICIT probe, never a silent except — and we never drop below the
    configured candidate set)."""
    if not candidates:
        raise RuntimeError(f"[{lib}] cxx_std probe requested with no candidates")
    ip._configure_libclang()
    probe_index = ip.Index.create()
    parse_opts = int(getattr(ip.TranslationUnit, "PARSE_INCOMPLETE", 0) or 0)
    rejections: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"gsi_stdprobe_{lib}_") as td:
        probe_src = os.path.join(td, "cxx_std_probe.cpp")
        with open(probe_src, "w") as fh:
            fh.write("#include <ranges>\n#include <format>\nint main() { return 0; }\n")
        for cand in candidates:
            value = cand.split("=", 1)[1] if "=" in cand else cand
            compile_args = ["-x", "c++", cand]
            if include_root:
                compile_args += ["-I", include_root]
            compile_args += list(sysroot_args)
            # An UNKNOWN -std makes libclang's driver fail so hard it produces no
            # TU and raises TranslationUnitLoadError. That raised parse IS the
            # rejection signal for this candidate: record it and step DOWN. This
            # is the probe's explicit detection mechanism ("if the std flag errors,
            # step down"), NOT a silent fallback — if NO candidate parses we RAISE
            # below with every collected reason (RULE #1: fail loud).
            try:
                tu = probe_index.parse(probe_src, args=compile_args,
                                       options=parse_opts)
            except Exception as exc:  # noqa: BLE001 - per-candidate probe rejection
                rejections.append(f"{cand} (parse raised: {type(exc).__name__})")
                print(f"  [{lib}] cxx_std probe: {cand} REJECTED by libclang "
                      f"({type(exc).__name__}) -> stepping down",
                      file=sys.stderr, flush=True)
                continue
            diags = list(tu.diagnostics)
            std_rejected = any(
                value in d.spelling
                and ("invalid value" in d.spelling or "unknown argument" in d.spelling)
                for d in diags
            )
            if std_rejected:
                rejections.append(f"{cand} (driver diagnostic)")
                print(f"  [{lib}] cxx_std probe: {cand} REJECTED by libclang "
                      f"(driver diagnostic) -> stepping down",
                      file=sys.stderr, flush=True)
                continue
            n_fatal = sum(1 for d in diags if d.severity >= 4)
            headers_parsed = bool(include_root) and n_fatal == 0
            print(f"  [{lib}] cxx_std probe: chose {cand} "
                  f"(libclang accepts; diagnostics={len(diags)} fatal={n_fatal} "
                  f"libcxx_ranges_format="
                  f"{'parsed' if headers_parsed else 'flag-accepted'})",
                  file=sys.stderr, flush=True)
            return cand
    raise RuntimeError(
        f"[{lib}] no C++ standard from {tuple(candidates)} was accepted by "
        f"libclang on a libc++ <ranges>/<format> probe parse (tried: "
        f"{rejections}). RULE #1: refusing to index std with an unknown -std flag."
    )


# --------------------------------------------------------------------------- #
# Per-file libclang parse worker (isolated subprocess so a segfault is local).
# --------------------------------------------------------------------------- #
def _parse_file_worker(args_tuple):
    filepath, project_dir, std_arg, lang, include_dirs, lib_env = args_tuple
    lib_env = lib_env or {}
    sys.setrecursionlimit(50000)
    ip._configure_libclang()
    clang_index = ip.Index.create()
    ext = os.path.splitext(filepath)[1].lower()
    p = filepath.replace("\\", "/")
    # C++ std level is per-lib: the std umbrella headers need >= c++20; other
    # libs use the run-wide --std-arg (default c++17).
    cxx_std = lib_env.get("cxx_std") or std_arg
    # Language is decided PER LIB (lang), not per extension: header-only C++
    # template libs (boost/eigen/fmt) MUST parse as C++ or their public API
    # yields zero symbols. A genuine .c file in a C++ lib is still parsed as C.
    if ext == ".c" or (lang == "c" and ext in {".h", ".inc", ".inl"}):
        compile_args = ["-x", "c", "-std=c11"]
    else:
        compile_args = ["-x", "c++", cxx_std]
    # Include roots. For std (stdlib_include_markers set) pin a SINGLE per-file
    # include root (the tree the header belongs to) to avoid libc++/libstdc++/
    # MS-STL cross-tree contamination. All other libs use the discovered
    # self-include dirs so a header resolves its own siblings (cuts parse errors
    # + time for header-heavy libs). PARSE_INCOMPLETE keeps going past unresolved
    # system includes.
    markers = lib_env.get("stdlib_include_markers")
    std_root = _stdlib_include_root_for(p, markers) if markers else None
    if std_root:
        compile_args += ["-I", std_root]
    else:
        for inc in include_dirs:
            compile_args += ["-I", inc]
    # MS-STL needs MSVC compatibility/intrinsics; enable best-effort for its
    # headers. Files that still need genuine MSVC intrinsics fail PER FILE (a
    # counted parse error in the report) — NOT a silent wholesale drop (RULE #1).
    msvc_marker = lib_env.get("msvc_path_marker")
    if msvc_marker and msvc_marker in p:
        compile_args += ["-fms-compatibility", "-fms-extensions",
                         "-fdelayed-template-parsing"]
    # clang builtin headers + C sysroot (required by std umbrella headers).
    compile_args += lib_env.get("sysroot_args") or []
    funcs, types = ip.parse_translation_unit(
        filepath, clang_index, compile_args, project_dir,
        allow_system_types=bool(lib_env.get("allow_system_types")),
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

    # Per-lib compile environment (only std sets these). Computed ONCE in the
    # parent and shipped to every worker so the C++ stdlib umbrella headers
    # resolve their builtin/C-library includes; allow_system_types lets the
    # parser keep std:: TYPE definitions (std::vector/std::map/...). For non-std
    # libs every field is absent -> the worker behaves exactly as before.
    sysroot_args: list[str] = []
    if spec.get("needs_cxx_stdlib_env"):
        sysroot_args = _cxx_stdlib_sysroot_args()  # RAISES if no builtins found

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

    # ---- choose the C++ standard for this lib ----
    # When the spec configures cxx_std_probe_candidates (only std does), pick the
    # HIGHEST -std the installed libclang actually accepts on a real libc++
    # <ranges>/<format> probe parse — a SUPERSET std so the c++23/26 standard
    # surface (ranges/concepts/format/expected/flat_map) is captured. The static
    # spec["cxx_std"] is only the documented FLOOR; the probe (explicit, logged,
    # RAISES on total failure) replaces it. Non-std libs keep their static cxx_std.
    probe_candidates = spec.get("cxx_std_probe_candidates")
    if probe_candidates:
        libcxx_root = _find_libcxx_include_root(cpp_files)
        chosen_cxx_std = _probe_highest_cxx_std(
            probe_candidates, libcxx_root, sysroot_args, lib=lib,
        )
    else:
        chosen_cxx_std = spec.get("cxx_std")

    lib_env = {
        "cxx_std": chosen_cxx_std,
        "sysroot_args": sysroot_args,
        "stdlib_include_markers": spec.get("stdlib_include_markers"),
        "msvc_path_marker": spec.get("msvc_path_marker"),
        "allow_system_types": bool(spec.get("allow_system_types")),
    }

    print(f"  [{lib}] lang={lang} cxx_std={lib_env['cxx_std'] or std_arg} "
          f"include_dirs={len(include_dirs)} "
          f"stdlib_env={'on' if sysroot_args else 'off'} "
          f"allow_system_types={lib_env['allow_system_types']} "
          f"files_to_parse={res.n_files} ({workers} workers)",
          file=sys.stderr, flush=True)
    if sysroot_args:
        print(f"  [{lib}] stdlib sysroot_args: {' '.join(sysroot_args)}",
              file=sys.stderr, flush=True)

    batch: list[tuple] = []
    BATCH_COMMIT = 5000
    parsed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(_parse_file_worker,
                      (fp, project_dir, std_arg, lang, include_dirs, lib_env)): fp
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

    # Fail-loud guard for the std-type path (CLAUDE.md RULE #1): when a lib is
    # configured to capture system (std::) TYPES, parsing files but extracting
    # ZERO class types means the C++ stdlib compile-env is broken (missing
    # builtins / wrong include root / unparsable umbrellas) — exactly the old
    # "std=0 types" defect. Do NOT silently mark such a lib done; RAISE so the
    # broken env is surfaced and fixed.
    if spec.get("allow_system_types") and res.n_files > 0 and res.n_types == 0:
        raise RuntimeError(
            f"[{lib}] parsed {res.n_files} files but captured 0 class TYPES with "
            f"allow_system_types=on ({res.n_funcs} funcs, {res.n_errors} errors). "
            f"The C++ stdlib compile-env is broken (clang builtins / include root "
            f"/ umbrella parse). RULE #1: refusing to record a broken std index."
        )

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
    p.add_argument("--extract-cache-dir", default=None,
                   help="Persistent source extraction cache. When set, the "
                        "tarball is streamed only for libs missing from the "
                        "cache; parser/indexer code still re-runs every build.")
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
    if args.extract_cache_dir and args.work_dir:
        raise SystemExit("--extract-cache-dir and --work-dir are mutually exclusive")

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

    # Persistent cache: extract source once, re-index many times. This is the
    # path for full A1/A2 rebuilds while std/C++20/23/26 parser work evolves.
    if args.extract_cache_dir:
        extract_root = Path(args.extract_cache_dir)
        todo_specs = {lib: BASE_LIBS[lib] for lib in todo}
        print(f"\n=== extraction cache for {todo} -> {extract_root} ===",
              file=sys.stderr, flush=True)
        t_ext = time.time()
        dirs, counts = prepare_extraction_cache(
            tarball, todo_specs, extract_root,
            max_files=args.max_files_per_lib, member_cap=args.member_cap,
        )
        print(f"  extraction cache ready in {time.time()-t_ext:.1f}s: {counts}",
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
                lib, spec, dirs[lib], store,
                workers=args.workers, std_arg=args.std_arg,
            )
            _finish(lib, spec, res)
    # ONE tarball pass extracts ALL todo libs to a shared staging root (the
    # 235 GiB tarball is streamed ONCE, not once per lib), unless --per-lib-pass.
    elif len(todo) > 1 and not args.per_lib_pass:
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
