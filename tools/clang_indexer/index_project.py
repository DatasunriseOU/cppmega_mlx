#!/usr/bin/env python3
# ruff: noqa: E402
"""
Clang-based cross-file dependency indexer for C++ training data preparation.

Uses libclang to parse C++ translation units with full semantic analysis,
building a cross-file call graph and generating bottom-up training documents.

Architecture:
  1. Walk project directory, find all .cpp/.cc/.cxx/.c files
  2. Parse each with libclang (optionally using compile_commands.json)
  3. Extract functions, classes, and cross-file call references
  4. Build global call graph
  5. Topological sort: HAL/system → drivers → subsystems → API
  6. Generate 16K-token training documents with bottom-up dependency ordering

Usage:
  # With compile_commands.json (best quality):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl

  # Without build system (fallback mode):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl --no-compile-db

  # Process multiple projects in parallel:
  python index_project.py --projects-list projects.txt --output chunks.jsonl --workers 48
"""

import argparse
import ctypes.util
import glob
import importlib
import json
import os
import sys
import hashlib

# Increase recursion limit for deeply nested ASTs (gcc-mirror, llvm-project, boost)
sys.setrecursionlimit(50000)
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Protocol, TypeAlias, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.nanochat_pipeline.language_info import detect_language_info
from cppmega_mlx.data.nanochat_pipeline.build_context import (
    detect_build_context,
    load_compile_commands_file,
)

if TYPE_CHECKING:
    import clang.cindex as clang_cindex  # pyright: ignore[reportMissingImports]
    from clang.cindex import Config as ClangConfig  # pyright: ignore[reportMissingImports]
    from clang.cindex import Cursor, CursorKind, Index, TranslationUnit  # pyright: ignore[reportMissingImports]
else:
    clang_cindex = None
    ClangConfig = object
    Cursor = object
    CursorKind = object
    Index = object
    TranslationUnit = object


class _MissingCursorKind:
    def __getattr__(self, name: str) -> str:
        return f"<missing-clang-cursorkind:{name}>"


class _MissingIndex:
    @staticmethod
    def create():
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        )


class _ClangCIndexModule(Protocol):
    __file__: str


class _ClangConfig(Protocol):
    @staticmethod
    def set_library_file(filename: str) -> None: ...


class _IndexFactory(Protocol):
    @staticmethod
    def create() -> "Index": ...


class PlatformDetectFn(Protocol):
    def __call__(self, text: str) -> dict[str, object] | None: ...


DetectLanguageInfoFn = Callable[
    [str, str | None, dict[str, object] | None, list[str] | None, dict | None],
    dict[str, object] | None,
]


clang_cindex_module: _ClangCIndexModule | None
clang_config_cls: _ClangConfig
index_cls: _IndexFactory
_CLANG_IMPORT_ERROR: ImportError | None = None

try:
    clang_cindex_runtime = importlib.import_module("clang.cindex")
    clang_cindex_module = cast(_ClangCIndexModule, clang_cindex_runtime)
    clang_config_cls = cast(_ClangConfig, getattr(clang_cindex_runtime, "Config"))
    index_cls = cast(_IndexFactory, getattr(clang_cindex_runtime, "Index"))
    if not TYPE_CHECKING:
        clang_cindex = clang_cindex_runtime  # type: ignore[assignment]
        ClangConfig = getattr(clang_cindex_runtime, "Config")  # type: ignore[assignment]
        Cursor = getattr(clang_cindex_runtime, "Cursor")  # type: ignore[assignment]
        CursorKind = getattr(clang_cindex_runtime, "CursorKind")  # type: ignore[assignment]
        Index = getattr(clang_cindex_runtime, "Index")  # type: ignore[assignment]
        TranslationUnit = getattr(clang_cindex_runtime, "TranslationUnit")  # type: ignore[assignment]
except ImportError as _clang_import_err:
    _CLANG_IMPORT_ERROR = _clang_import_err
    clang_cindex_module = None
    clang_config_cls = cast(_ClangConfig, object)
    index_cls = cast(_IndexFactory, _MissingIndex)
    if not TYPE_CHECKING:
        CursorKind = _MissingCursorKind()  # type: ignore[assignment]
        Index = _MissingIndex  # type: ignore[assignment]


_LIBCLANG_CONFIGURED = False
_COMMON_LIBCLANG_CANDIDATES = [
    "/usr/lib/llvm-21/lib/libclang-21.so.1",
    "/usr/lib/llvm-21/lib/libclang.so.1",
    "/usr/lib/llvm-20/lib/libclang-20.so.1",
    "/usr/lib/llvm-20/lib/libclang.so.1",
    "/usr/lib/llvm-19/lib/libclang-19.so.1",
    "/usr/lib/llvm-19/lib/libclang.so.1",
    "/usr/lib/llvm-18/lib/libclang-18.so.1",
    "/usr/lib/llvm-18/lib/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang-18.so.18",
    "/lib/x86_64-linux-gnu/libclang-18.so",
    "/lib/x86_64-linux-gnu/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang.so",
    "/usr/lib/libclang.so.1",
    "/usr/lib/libclang.so",
]


def _iter_bundled_libclang_candidates() -> Iterator[str]:
    if clang_cindex_module is None:
        return
    module_path = getattr(clang_cindex_module, "__file__", "")
    if not module_path:
        return
    native_dir = Path(module_path).resolve().parent / "native"
    if not native_dir.is_dir():
        return
    for pattern in ("libclang.so*", "libclang.dylib*", "libclang.dll*"):
        for candidate in sorted(native_dir.glob(pattern)):
            if candidate.is_file():
                yield str(candidate)


def _iter_libclang_candidates(explicit_path: str | None = None) -> Iterator[str]:
    seen: set[str] = set()

    def add(value: str | None) -> str | None:
        if not value:
            return None
        candidate = value.strip()
        if not candidate or candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    candidate = add(explicit_path)
    if candidate:
        yield candidate
    for env_name in (
        "NANOCHAT_LIBCLANG_PATH",
        "LIBCLANG_PATH",
        "LIBCLANG_FILE",
        "CLANG_LIBRARY_FILE",
    ):
        candidate = add(os.environ.get(env_name))
        if candidate:
            yield candidate
    for candidate in _iter_bundled_libclang_candidates():
        normalized = add(candidate)
        if normalized:
            yield normalized
    for candidate in _COMMON_LIBCLANG_CANDIDATES:
        if os.path.exists(candidate):
            normalized = add(candidate)
            if normalized:
                yield normalized
    detected = add(ctypes.util.find_library("clang"))
    if detected:
        yield detected
    for pattern in (
        "/usr/lib*/**/libclang*.so*",
        "/usr/local/lib*/**/libclang*.so*",
        "/lib*/**/libclang*.so*",
    ):
        for candidate in glob.iglob(pattern, recursive=True):
            if os.path.isfile(candidate):
                normalized = add(candidate)
                if normalized:
                    yield normalized


def _configure_libclang(libclang_path: str | None = None) -> str | None:
    global _LIBCLANG_CONFIGURED
    if _LIBCLANG_CONFIGURED:
        return os.environ.get("NANOCHAT_LIBCLANG_PATH") or libclang_path
    if _CLANG_IMPORT_ERROR is not None:
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        ) from _CLANG_IMPORT_ERROR

    last_error: Exception | None = None
    for candidate in _iter_libclang_candidates(libclang_path):
        try:
            clang_config_cls.set_library_file(candidate)
            index_cls.create()
        except Exception as exc:  # pragma: no cover - runtime-specific
            last_error = exc
            continue
        os.environ["NANOCHAT_LIBCLANG_PATH"] = candidate
        _LIBCLANG_CONFIGURED = True
        return candidate

    try:
        index_cls.create()
    except Exception as exc:
        last_error = exc
    else:
        _LIBCLANG_CONFIGURED = True
        return None

    message = (
        "Unable to load libclang. "
        "Set --libclang-path or NANOCHAT_LIBCLANG_PATH to an absolute libclang.so path."
    )
    if last_error is not None:
        raise RuntimeError(f"{message} Last error: {last_error}") from last_error
    raise RuntimeError(message)

# tree-sitter for AST metadata extraction (ast_depth, sibling_index, ast_node_type)
_HAS_TREE_SITTER = False
_TS_LANG = None
try:
    _ts = importlib.import_module("tree_sitter")
    _ts_cpp = importlib.import_module("tree_sitter_cpp")
    _TS_LANG = _ts.Language(_ts_cpp.language())
    _HAS_TREE_SITTER = True
except ImportError:
    pass


PartInfo: TypeAlias = tuple[str, int, int, str, str | None]


# C++ source file extensions
CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c', '.c++', '.cp'}
HEADER_EXTENSIONS = {'.h', '.hpp', '.hxx', '.hh', '.h++', '.inl', '.inc'}

# System/stdlib function prefixes (skip for dependency tracking)
SYSTEM_PREFIXES = (
    'std::', 'boost::', '__builtin', '__', 'operator', 'printf', 'fprintf',
    'sprintf', 'snprintf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat',
    'strcmp', 'fopen', 'fclose', 'fread', 'fwrite', 'exit', 'abort',
    'assert', 'pthread_', 'EXPECT_', 'ASSERT_', 'TEST',
)


class FunctionDef:
    """A function definition with its source location and call references."""
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'end_line', 'text',
                 'callees', 'dep_level', 'is_definition']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, callees: list, is_definition: bool = True,
                 end_line: int = 0):
        self.name = name
        self.qualified_name = qualified_name
        self.file = file
        self.line = line
        self.end_line = end_line or (line + text.count('\n'))
        self.text = text
        self.callees = callees  # list of qualified names called
        self.dep_level = 0
        self.is_definition = is_definition

    def to_dict(self) -> dict:
        """Serialize for multiprocessing IPC."""
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'text': self.text,
            'callees': self.callees, 'is_definition': self.is_definition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FunctionDef':
        return cls(**d)


class ProjectIndex:
    """Cross-file function index for a single project."""

    def __init__(self):
        # qualified_name -> FunctionDef (definitions only)
        self.functions: dict[str, FunctionDef] = {}
        # file -> list of function qualified_names defined there
        self.file_functions: dict[str, list[str]] = defaultdict(list)
        # file -> preamble text (includes, typedefs, forward decls)
        self.file_preambles: dict[str, str] = {}
        # qualified_name -> list of qualified_names that call it
        self.callers: dict[str, list[str]] = defaultdict(list)

    def add_function(self, func: FunctionDef):
        """Add a function definition to the index."""
        key = func.qualified_name
        if key in self.functions and self.functions[key].is_definition:
            return  # don't overwrite definitions with declarations
        self.functions[key] = func
        if func.is_definition:
            self.file_functions[func.file].append(key)

    def build_reverse_edges(self):
        """Build caller -> callee reverse edges for dep level computation."""
        self.callers.clear()
        for qname, func in self.functions.items():
            for callee in func.callees:
                if callee in self.functions:
                    self.callers[callee].append(qname)

    def compute_dep_levels(self):
        """Compute dependency levels via BFS from leaves."""
        # Find leaves: functions with no callees in the index
        in_degree = {}
        for qname, func in self.functions.items():
            local_callees = [c for c in func.callees if c in self.functions and c != qname]
            in_degree[qname] = len(local_callees)

        queue = deque()
        for qname, deg in in_degree.items():
            if deg == 0:
                self.functions[qname].dep_level = 0
                queue.append(qname)

        self.build_reverse_edges()

        while queue:
            qname = queue.popleft()
            level = self.functions[qname].dep_level
            for caller_name in self.callers.get(qname, []):
                new_level = level + 1
                if new_level > self.functions[caller_name].dep_level:
                    self.functions[caller_name].dep_level = new_level
                in_degree[caller_name] -= 1
                if in_degree[caller_name] == 0:
                    queue.append(caller_name)

        # Handle cycles
        max_level = max((f.dep_level for f in self.functions.values()), default=0)
        for qname, deg in in_degree.items():
            if deg > 0:
                self.functions[qname].dep_level = max_level + 1

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            'total_functions': len(self.functions),
            'total_files': len(self.file_functions),
            'definitions': sum(1 for f in self.functions.values() if f.is_definition),
            'max_dep_level': max((f.dep_level for f in self.functions.values()), default=0),
        }


def is_system_function(name: str | None) -> bool:
    """Check if a function name looks like a system/stdlib function."""
    if not isinstance(name, str) or not name:
        return False
    return any(name.startswith(p) for p in SYSTEM_PREFIXES)


_DROP_CLANG_ARG_PREFIXES = (
    "-fsanitize=",
    "-fno-sanitize=",
    "-fsanitize-recover=",
    "-fno-sanitize-recover=",
    "-fsanitize-trap=",
    "-fsanitize-ignorelist=",
    "-fsanitize-blacklist=",
    "-fsanitize-coverage=",
)

_DROP_CLANG_ARGS = {
    "-fsanitize-address-use-after-scope",
    "-fno-sanitize-address-use-after-scope",
    "-shared-libasan",
    "-static-libasan",
    "-shared-libsan",
    "-static-libsan",
}


def _is_runtime_only_sanitizer_flag(text: str) -> bool:
    if text in _DROP_CLANG_ARGS:
        return True
    if text.startswith(("-shared-lib", "-static-lib")) and "san" in text:
        return True
    return False


def _sanitize_compile_args_for_clang(args: list[str] | None) -> list[str]:
    """Drop nullish and runtime-only compile args that break libclang parsing."""
    if not args:
        return []
    sanitized: list[str] = []
    for arg in args:
        if arg is None:
            continue
        text = str(arg).strip()
        if not text:
            continue
        if _is_runtime_only_sanitizer_flag(text):
            continue
        if any(text.startswith(prefix) for prefix in _DROP_CLANG_ARG_PREFIXES):
            continue
        sanitized.append(text)
    return sanitized


def get_function_text(cursor: Cursor, tu: TranslationUnit) -> str:
    """Extract the source text for a cursor's extent."""
    extent = cursor.extent
    start = extent.start
    end = extent.end

    try:
        filename = start.file.name if start.file else None
        if not filename or not os.path.exists(filename):
            return ""

        with open(filename, 'r', errors='replace') as f:
            content = f.read()

        # Convert offsets
        start_offset = start.offset
        end_offset = end.offset
        if start_offset < len(content) and end_offset <= len(content):
            return content[start_offset:end_offset]
    except Exception:
        pass
    return ""


def extract_callees(cursor: Cursor) -> list[str]:
    """Extract all function call references from a cursor's children."""
    callees = set()

    def walk(node: Cursor):
        if node.kind == CursorKind.CALL_EXPR:
            ref = node.referenced
            if ref and ref.spelling:
                # Get fully qualified name
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    callees.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(callees)


def get_qualified_name(cursor: Cursor) -> str:
    """Get the fully qualified name of a cursor (namespace::class::func)."""
    parts = []
    c = cursor
    while c and c.kind != CursorKind.TRANSLATION_UNIT:
        if c.spelling:
            parts.append(c.spelling)
        c = c.semantic_parent
    parts.reverse()
    return '::'.join(parts)


def extract_preamble(tu: TranslationUnit, filename: str) -> str:
    """Extract #include directives and forward declarations from a file."""
    preamble_parts = []
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name != filename:
            continue
        if cursor.kind in (CursorKind.INCLUSION_DIRECTIVE,
                           CursorKind.USING_DIRECTIVE,
                           CursorKind.USING_DECLARATION,
                           CursorKind.TYPEDEF_DECL,
                           CursorKind.TYPE_ALIAS_DECL,
                           CursorKind.NAMESPACE_ALIAS):
            text = get_function_text(cursor, tu)
            if text:
                preamble_parts.append(text)
    return '\n'.join(preamble_parts)


FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
}

CONTAINER_KINDS = {
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}


def parse_translation_unit(
    filepath: str,
    index: Index,
    compile_args: list[str],
    project_dir: str,
) -> list[FunctionDef]:
    """Parse a single C++ file and extract function definitions with callees."""
    functions: list[FunctionDef] = []

    try:
        tu = index.parse(
            filepath,
            args=compile_args,
            options=(
                TranslationUnit.PARSE_INCOMPLETE |
                TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
            ),
        )
    except Exception as e:
        print(f"  WARN: Failed to parse {filepath}: {e}", file=sys.stderr)
        return functions

    rel_path = os.path.relpath(filepath, project_dir)

    def visit(cursor):
        """Recursively visit cursors, descending into namespaces and classes."""
        if not cursor.location.file:
            return
        if cursor.location.file.name != filepath:
            return

        if cursor.kind in FUNCTION_KINDS and cursor.is_definition():
            text = get_function_text(cursor, tu)
            if text and len(text) >= 20:
                callees = extract_callees(cursor)
                qname = get_qualified_name(cursor)
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=rel_path,
                    line=cursor.location.line,
                    text=text,
                    callees=callees,
                    is_definition=True,
                ))

        elif cursor.kind in CONTAINER_KINDS:
            # Recurse into namespaces, classes, structs
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return functions


_DEFAULT_SKIP_DIRS = frozenset({
    '.git', 'build', 'cmake-build', '__pycache__', 'node_modules',
    '.vs', '.vscode', 'third_party', 'external', 'deps', 'vendor',
    'test', 'tests', 'unittests', 'benchmarks',
    # Align with Rust cpp-chunker skip list
    'testing', 'examples', 'example', 'samples', 'sample', 'docs', 'doc',
    # Additional noise dirs for large repos
    'fuzzers', 'fuzzing', 'regression', 'fixtures',
})

C_EXTENSIONS = {'.c'}


def find_cpp_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
) -> list[str]:
    """Find all C/C++ source files in a directory."""
    skip_dirs = _DEFAULT_SKIP_DIRS | (extra_exclude_dirs or set())
    files = []
    for root, dirs, filenames in os.walk(project_dir):
        # Prune recursion in-place — os.walk won't descend into skipped dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in CPP_EXTENSIONS:
                filepath = os.path.join(root, fname)
                # Skip very large files
                try:
                    if os.path.getsize(filepath) > 500_000:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


def load_compile_commands(project_dir: str) -> Optional[dict]:
    """Load compile_commands.json if available."""
    cc_path = os.path.join(project_dir, 'compile_commands.json')
    compile_index = load_compile_commands_file(cc_path)
    if compile_index is None:
        return None
    file_args = {}
    for entry in compile_index.entries:
        file_args[entry.filepath] = {
            "compile_args": list(entry.compile_args),
            "build_info": dict(entry.build_info),
        }
    return file_args
    return None


def _discover_include_dirs(project_dir: str) -> list[str]:
    """Recursively find directories named 'include' for -I flags."""
    include_dirs = set()
    for candidate in ['include', 'src', 'lib', 'source', '.']:
        d = os.path.join(project_dir, candidate)
        if os.path.isdir(d):
            include_dirs.add(d)
    # Walk up to 3 levels deep for nested include/ dirs
    for root, dirs, _ in os.walk(project_dir):
        depth = root[len(project_dir):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in {'.git', 'build', 'third_party', 'external', 'test', 'tests'}]
        if os.path.basename(root) == 'include':
            include_dirs.add(root)
    return sorted(include_dirs)


def _is_sane_compile_args(args: list[str]) -> bool:
    """Check if build_context returned usable args (not shell script garbage)."""
    for arg in args:
        if arg is None:
            return False
        # Garbage indicators: semicolons, unresolved $vars, paths after -x
        if ';' in arg or '$' in arg:
            return False
        # Empty -I flags
        if arg == '-I':
            return False
        # Compound flags (multiple args jammed into one string)
        # e.g., "-std=c89 -Werror -O0" — legitimate args never have spaces
        # except -x <lang> which is handled as separate args
        if ' ' in arg and not arg.startswith('-D'):
            return False
    return True


_SIMPLE_FALLBACK_ARGS = ["-fsyntax-only", "-Wno-everything"]


def get_default_compile_args(project_dir: str) -> list[str]:
    """Generate default compile args for projects without compile_commands.json."""
    _platform_info, args, _compile_index = detect_build_context(project_dir)

    if _is_sane_compile_args(args):
        result = list(args)
    else:
        # build_context returned garbage (e.g., shell fragments from configure)
        result = list(_SIMPLE_FALLBACK_ARGS)

    if not any(arg.startswith('-I') for arg in result):
        result.append(f'-I{project_dir}')
        for d in _discover_include_dirs(project_dir):
            result.append(f'-I{d}')
    return result


def _adapt_args_for_file(args: list[str], filepath: str) -> list[str]:
    """Adapt compile args based on file extension — .c files need C mode, not C++."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in C_EXTENSIONS:
        adapted = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == '-x':
                # Skip -x <lang> pair
                skip_next = True
                continue
            if arg.startswith('-std=c++'):
                adapted.append('-std=c11')
                continue
            if arg.startswith('-xc') or arg.startswith('-x '):
                # -xc++ joined form — replace with -xc
                continue
            adapted.append(arg)
        # Ensure C mode is set
        adapted = ['-x', 'c', '-std=c11'] + adapted
        return adapted
    return args


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 bytes per token for code)."""
    return max(1, len(text) // 4)


def collect_transitive_deps(
    root_qname: str,
    index: ProjectIndex,
    max_depth: int = 5,
) -> list[str]:
    """BFS to collect transitive dependencies of a function."""
    visited = {root_qname}
    queue = deque([(root_qname, 0)])
    deps = []

    while queue:
        qname, depth = queue.popleft()
        if depth >= max_depth:
            continue
        func = index.functions.get(qname)
        if not func:
            continue
        for callee in func.callees:
            if callee not in visited and callee in index.functions:
                visited.add(callee)
                deps.append(callee)
                queue.append((callee, depth + 1))

    return deps


def bucket_node_type(kind: str) -> int:
    """Map a tree-sitter node type string to a bucket in 0-255.

    Matches the Rust ``bucket_node_type`` in cpp_chunker/enriched.rs and
    v4_context_graph/enriched.rs so that the Python clang_indexer produces
    identical node-type buckets.

    Buckets:
      0: unknown/unnamed
      1-9: declarations
      10-19: statements
      20-29: expressions
      30-39: types
      40-49: literals
      50-59: operators and punctuation
      60-69: parameters and arguments
      70-79: access and scope
      80-89: miscellaneous
      255: catch-all for unrecognised named nodes
    """
    _MAP: dict[str, int] = {
        # 0: unknown
        "": 0,
        # 1-9: declarations
        "function_definition": 1,
        "class_specifier": 2,
        "struct_specifier": 3,
        "enum_specifier": 4,
        "namespace_definition": 5,
        "declaration": 6,
        "field_declaration": 7,
        "template_declaration": 8,
        "alias_declaration": 9, "type_definition": 9, "using_declaration": 9,
        # 10-19: statements
        "compound_statement": 10,
        "if_statement": 11,
        "for_statement": 12, "for_range_loop": 12,
        "while_statement": 13, "do_statement": 13,
        "return_statement": 14,
        "switch_statement": 15, "case_statement": 15,
        "expression_statement": 16,
        "try_statement": 17, "catch_clause": 17, "throw_statement": 17,
        "break_statement": 18, "continue_statement": 18, "goto_statement": 18,
        "labeled_statement": 19,
        # 20-29: expressions
        "call_expression": 20,
        "binary_expression": 21,
        "unary_expression": 22,
        "assignment_expression": 23, "augmented_assignment_expression": 23,
        "conditional_expression": 24,
        "field_expression": 25,
        "subscript_expression": 26,
        "cast_expression": 27, "static_cast_expression": 27,
        "dynamic_cast_expression": 27, "reinterpret_cast_expression": 27,
        "const_cast_expression": 27,
        "new_expression": 28, "delete_expression": 28,
        "lambda_expression": 29, "parenthesized_expression": 29,
        # 30-39: types
        "type_identifier": 30,
        "primitive_type": 31, "sized_type_specifier": 31,
        "template_type": 32, "template_argument_list": 32,
        "qualified_identifier": 33, "scoped_identifier": 33,
        "scoped_type_identifier": 33,
        "pointer_declarator": 34, "reference_declarator": 34,
        "abstract_pointer_declarator": 34, "abstract_reference_declarator": 34,
        "function_declarator": 35, "abstract_function_declarator": 35,
        "array_declarator": 36, "abstract_array_declarator": 36,
        "auto": 37, "decltype": 37,
        # 40-49: literals
        "number_literal": 40,
        "string_literal": 41, "raw_string_literal": 41, "string_content": 41,
        "char_literal": 42,
        "true": 43, "false": 43,
        "null": 44, "nullptr": 44,
        "user_defined_literal": 45,
        "concatenated_string": 46,
        # 50-59: operators and punctuation
        ",": 50, ";": 50, ":": 50, "::": 50,
        "{": 51, "}": 51,
        "(": 52, ")": 52,
        "[": 53, "]": 53,
        "<": 54, ">": 54, "<=": 54, ">=": 54, "==": 54, "!=": 54,
        "+": 55, "-": 55, "*": 55, "/": 55, "%": 55,
        "&": 56, "|": 56, "^": 56, "~": 56, "!": 56,
        "=": 57, "+=": 57, "-=": 57, "*=": 57, "/=": 57,
        "%=": 57, "&=": 57, "|=": 57, "^=": 57, "<<=": 57, ">>=": 57,
        "&&": 58, "||": 58,
        "->": 59, ".": 59, "->*": 59, ".*": 59,
        # 60-69: parameters and arguments
        "parameter_declaration": 60, "optional_parameter_declaration": 60,
        "variadic_parameter_declaration": 60,
        "parameter_list": 61,
        "argument_list": 62, "initializer_list": 62,
        "init_declarator": 63,
        # 70-79: access and scope
        "access_specifier": 70,
        "base_class_clause": 71,
        "field_initializer_list": 72, "field_initializer": 72,
        "namespace_identifier": 73,
        # 80-89: miscellaneous
        "comment": 80,
        "preproc_include": 81, "preproc_def": 81, "preproc_ifdef": 81,
        "preproc_ifndef": 81, "preproc_if": 81, "preproc_else": 81,
        "preproc_elif": 81, "preproc_call": 81, "preproc_function_def": 81,
        "translation_unit": 82,
        "identifier": 83,
        "ERROR": 84,
    }
    return _MAP.get(kind, 255)


def extract_ast_metadata(source: str) -> tuple[list[int], list[int], list[int]]:
    """Extract per-character AST metadata using tree-sitter.

    Returns ``(ast_depth, sibling_index, ast_node_type)`` each of length
    ``len(source)``.  Values match the Rust ``extract_ast_metadata`` in the
    cpp_chunker and v4_context_graph tools.

    If tree-sitter is not available, returns zero-filled arrays (graceful
    degradation).
    """
    text_len = len(source)
    if text_len == 0 or not _HAS_TREE_SITTER:
        return ([0] * text_len, [0] * text_len, [0] * text_len)

    source_bytes = source.encode("utf-8")
    byte_len = len(source_bytes)

    # Byte-indexed arrays
    b_depth = [0] * byte_len
    b_sib = [0] * byte_len
    b_ntype = [0] * byte_len

    parser = _ts.Parser(_TS_LANG)
    tree = parser.parse(source_bytes)
    if tree is None:
        return ([0] * text_len, [0] * text_len, [0] * text_len)

    # Iterative DFS (pre-order) — parents paint first, children overwrite.
    # Stack entries: (node, depth, sibling_idx)
    root = tree.root_node
    stack: list[tuple] = [(root, 0, 0)]

    while stack:
        node, depth, sib_idx = stack.pop()
        start = node.start_byte
        end = min(node.end_byte, byte_len)

        if start < end:
            bucket = bucket_node_type(node.type)
            for bi in range(start, end):
                b_depth[bi] = depth
                b_sib[bi] = sib_idx
                b_ntype[bi] = bucket

        # Push children in reverse order so first child pops first
        child_depth = min(depth + 1, 63)
        children = node.children
        for i in range(len(children) - 1, -1, -1):
            child_sib = min(i, 63)
            stack.append((children[i], child_depth, child_sib))

    # Convert byte-indexed to char-indexed.
    # For pure ASCII (typical C++ code), byte len == char len.
    if byte_len == text_len:
        return (b_depth, b_sib, b_ntype)

    # Non-ASCII: map byte positions to char positions
    c_depth = [0] * text_len
    c_sib = [0] * text_len
    c_ntype = [0] * text_len
    byte_offset = 0
    for char_idx, ch in enumerate(source):
        if byte_offset < byte_len:
            c_depth[char_idx] = b_depth[byte_offset]
            c_sib[char_idx] = b_sib[byte_offset]
            c_ntype[char_idx] = b_ntype[byte_offset]
        byte_offset += len(ch.encode("utf-8"))
    return (c_depth, c_sib, c_ntype)


# -------------------------------------------------------------------------
# Semantic metadata extraction (symbol_ids, call_targets, type_refs, def_use)
# -------------------------------------------------------------------------

# Def-use constants
DEF_USE_NONE = 0
DEF_USE_DEF = 1
DEF_USE_USE = 2

# CursorKind sets for classification
_DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.VAR_DECL,
    CursorKind.FIELD_DECL,
    CursorKind.PARM_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.UNION_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL,
    CursorKind.TYPEDEF_DECL,
    CursorKind.TYPE_ALIAS_DECL,
    CursorKind.NAMESPACE,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}

_CALL_KINDS = {
    CursorKind.CALL_EXPR,
}

_TYPE_REF_KINDS = {
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
    CursorKind.NAMESPACE_REF,
}

_REFERENCE_KINDS = {
    CursorKind.DECL_REF_EXPR,
    CursorKind.MEMBER_REF_EXPR,
    CursorKind.MEMBER_REF,
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
    CursorKind.NAMESPACE_REF,
    CursorKind.OVERLOADED_DECL_REF,
    CursorKind.VARIABLE_REF,
}


def _compute_symbol_id(qname: str) -> int:
    """Deterministic 32-bit symbol ID from a qualified name.

    Uses the lower 31 bits of md5, leaving 0 as the sentinel for 'no symbol'.
    """
    if not qname:
        return 0
    h = int(hashlib.md5(qname.encode("utf-8", errors="replace")).hexdigest(), 16)
    # Mask to 31 bits (positive int32), reserve 0 for no-symbol
    result = (h & 0x7FFFFFFF) or 1
    return result


def extract_semantic_metadata(
    source: str,
    tu: 'TranslationUnit',
    filename: str,
) -> dict:
    """Extract per-character semantic metadata from a translation unit.

    Walks the clang AST and produces four char-level arrays:

    - symbol_ids:   per-char deterministic hash of the symbol's qualified name
                    (0 = no symbol)
    - call_targets: per-char symbol ID of the call target for CALL_EXPR tokens
                    (0 = not a call)
    - type_refs:    per-char symbol ID of the referenced type for TYPE_REF tokens
                    (0 = not a type reference)
    - def_use:      per-char def/use marker (0=none, 1=def, 2=use)

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(source).
    """
    text_len = len(source)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use = [0] * text_len

    if text_len == 0 or tu is None:
        return {
            "symbol_ids": symbol_ids,
            "call_targets": call_targets,
            "type_refs": type_refs,
            "def_use": def_use,
        }

    source_bytes = source.encode("utf-8")
    byte_len = len(source_bytes)

    # Build byte-to-char mapping for non-ASCII sources
    if byte_len != text_len:
        byte_to_char = [0] * byte_len
        byte_offset = 0
        for char_idx, ch in enumerate(source):
            ch_bytes = len(ch.encode("utf-8"))
            for b in range(ch_bytes):
                if byte_offset + b < byte_len:
                    byte_to_char[byte_offset + b] = char_idx
            byte_offset += ch_bytes
    else:
        byte_to_char = None  # identity mapping

    def _byte_to_char_offset(byte_off: int) -> int:
        if byte_to_char is None:
            return byte_off
        if byte_off >= byte_len:
            return text_len
        return byte_to_char[byte_off]

    def _visit(cursor):
        """Walk the AST and annotate char ranges."""
        loc = cursor.location
        if not loc or not loc.file:
            return
        if loc.file.name != filename:
            return

        extent = cursor.extent
        if not extent:
            return

        start_offset = extent.start.offset
        end_offset = extent.end.offset
        if start_offset >= end_offset:
            return

        # Convert to char offsets
        char_start = _byte_to_char_offset(start_offset)
        char_end = _byte_to_char_offset(end_offset)
        if char_start >= text_len or char_start >= char_end:
            return

        char_end = min(char_end, text_len)
        kind = cursor.kind

        # Symbol identification: get qualified name for definitions and refs
        qname = ""
        if kind in _DEFINITION_KINDS and cursor.is_definition():
            qname = get_qualified_name(cursor)
            if qname:
                sym_id = _compute_symbol_id(qname)
                for ci in range(char_start, char_end):
                    symbol_ids[ci] = sym_id
                    def_use[ci] = DEF_USE_DEF

        elif kind in _REFERENCE_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                qname = get_qualified_name(ref)
                if qname:
                    sym_id = _compute_symbol_id(qname)
                    for ci in range(char_start, char_end):
                        symbol_ids[ci] = sym_id
                        def_use[ci] = DEF_USE_USE

        # Call target annotation
        if kind in _CALL_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                target_qname = get_qualified_name(ref)
                if target_qname:
                    target_id = _compute_symbol_id(target_qname)
                    for ci in range(char_start, char_end):
                        call_targets[ci] = target_id

        # Type reference annotation
        if kind in _TYPE_REF_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                ref_qname = get_qualified_name(ref)
                if ref_qname:
                    ref_id = _compute_symbol_id(ref_qname)
                    for ci in range(char_start, char_end):
                        type_refs[ci] = ref_id

        # Recurse into children
        for child in cursor.get_children():
            _visit(child)

    try:
        for child in tu.cursor.get_children():
            _visit(child)
    except (RecursionError, Exception):
        # Graceful degradation: return whatever we have so far
        pass

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use,
    }


def extract_semantic_metadata_from_parts(
    full_text: str,
    parts_info: list[PartInfo],
    index: ProjectIndex,
) -> dict:
    """Produce semantic metadata from pre-extracted parts without running libclang.

    This is the offline path used when the enriched doc is built from
    already-parsed FunctionDef objects. It assigns symbol_ids and def_use
    based on the parts_info metadata (qname, kind) and the call graph in
    the ProjectIndex.

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(full_text).
    """
    text_len = len(full_text)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use_arr = [0] * text_len

    offset = 0
    for i, part in enumerate(parts_info):
        part_text = part[0]
        kind = part[1]
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        qname = part[4] if len(part) > 4 else None

        if qname:
            sym_id = _compute_symbol_id(qname)
            # Function parts are definition sites
            for ci in range(offset, offset + part_len):
                symbol_ids[ci] = sym_id
                def_use_arr[ci] = DEF_USE_DEF

            # Annotate call targets from the index's callee list
            func_def = index.functions.get(qname)
            if func_def:
                for callee_qname in func_def.callees:
                    callee_id = _compute_symbol_id(callee_qname)
                    if callee_id:
                        # Mark the call target on the function body
                        # (we don't have exact call-site offsets without
                        # re-parsing, so mark at chunk granularity)
                        for ci in range(offset, offset + part_len):
                            if call_targets[ci] == 0:
                                call_targets[ci] = callee_id
                                break  # one call target per callee

        elif kind == 1:  # PREAMBLE
            # Preamble: mark as use sites (references to included entities)
            for ci in range(offset, offset + part_len):
                def_use_arr[ci] = DEF_USE_USE

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n" separator

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use_arr,
    }


def build_enriched_doc(
    parts_info: list[PartInfo],
    index: ProjectIndex,
    filepath: str | None = None,
    compile_args: list[str] | None = None,
    build_info: dict | None = None,
) -> dict:
    """Build enriched document from parts with metadata.

    Args:
        parts_info: list of (text, kind, dep_level, name, qname_or_none) tuples
            kind: 0=other, 1=preamble, 2=func_signature, 3=func_body,
                  4=class_decl, 5=class_member, 6=comment, 7=typedef, 8=namespace
        index: ProjectIndex for computing edges

    Returns:
        dict with text, structure_ids, chunk_boundaries, call_edges, type_edges,
        ast_depth, sibling_index, ast_node_type
    """
    texts = [p[0] for p in parts_info]
    full_text = '\n\n'.join(texts)
    text_len = len(full_text)

    structure_ids = [0] * text_len
    chunk_boundaries = []
    offset = 0

    # Map chunk_idx -> qname for edge computation
    chunk_qnames = {}
    chunk_callees = {}

    for i, (part_text, kind, dep_level, name, qname) in enumerate(parts_info):
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        # Fill structure_ids
        for j in range(offset, offset + part_len):
            structure_ids[j] = kind

        chunk_boundaries.append({
            'start': offset,
            'end': offset + part_len,
            'kind': kind,
            'dep_level': dep_level,
            'name': name,
        })

        # Track function qnames for edge computation
        if qname and qname in index.functions:
            chunk_qnames[i] = qname
            chunk_callees[i] = index.functions[qname].callees

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n"

    # Compute call_edges
    call_edges = []
    for ci, caller_qname in chunk_qnames.items():
        callees = chunk_callees.get(ci, [])
        for callee_qname in callees:
            for cj, target_qname in chunk_qnames.items():
                if ci != cj and target_qname == callee_qname:
                    call_edges.append({'from': ci, 'to': cj})

    # type_edges: empty for now (clang_indexer focuses on functions, not types)
    type_edges: list[dict[str, object]] = []

    # Extract per-character AST metadata via tree-sitter
    ast_depth, sibling_index, ast_node_type = extract_ast_metadata(full_text)

    # v4 enrichment: per-file platform detection
    import sys as _sys
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
    if _v5_dir not in _sys.path:
        _sys.path.insert(0, _v5_dir)
    platform_info: dict[str, object] | None
    try:
        _platform_detect = importlib.import_module("platform_detect")
        _detect_plat = cast(PlatformDetectFn, getattr(_platform_detect, "detect_platforms"))
        platform_info = _detect_plat(full_text)
    except ImportError:
        platform_info = None

    # Extract semantic metadata (symbol IDs, call targets, type refs, def-use)
    semantic_meta = extract_semantic_metadata_from_parts(full_text, parts_info, index)

    result = {
        'text': full_text,
        'structure_ids': structure_ids,
        'chunk_boundaries': chunk_boundaries,
        'call_edges': call_edges,
        'type_edges': type_edges,
        'ast_depth': ast_depth,
        'sibling_index': sibling_index,
        'ast_node_type': ast_node_type,
        'symbol_ids': semantic_meta['symbol_ids'],
        'call_targets': semantic_meta['call_targets'],
        'type_refs': semantic_meta['type_refs'],
        'def_use': semantic_meta['def_use'],
    }
    language_info = detect_language_info(
        full_text,
        filepath,
        platform_info,
        compile_args=compile_args,
        build_info=build_info,
    )

    if platform_info:
        result['platform_info'] = platform_info
    if build_info:
        result['build_info'] = build_info
    if language_info:
        result['language_info'] = language_info
    return result


def build_training_documents(
    index: ProjectIndex,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    enriched: bool = False,
    *,
    project_dir: str | None = None,
    compile_db: dict | None = None,
    default_args: list[str] | None = None,
    default_build_info: dict | None = None,
) -> list:
    """Build training documents with bottom-up dependency ordering.

    Returns list[str] when enriched=False, list[dict] when enriched=True.
    """
    documents: list[str | dict[str, object]] = []
    seen_hashes: set[str] = set()

    index.compute_dep_levels()

    for qname, func in index.functions.items():
        if not func.is_definition:
            continue

        # Collect transitive deps
        dep_qnames = collect_transitive_deps(qname, index, max_dep_depth)

        # Sort by dep_level (leaves/most foundational first)
        dep_funcs: list[FunctionDef] = []
        for dq in dep_qnames:
            df = index.functions.get(dq)
            if df and df.is_definition and df.text:
                dep_funcs.append(df)
        dep_funcs.sort(key=lambda f: f.dep_level)

        # Build document
        preamble = index.file_preambles.get(func.file, '')

        def _assemble(dep_funcs_list: list[FunctionDef]) -> str:
            parts: list[str] = []
            if preamble:
                parts.append(preamble)
            for df in dep_funcs_list:
                parts.append(df.text)
            parts.append(func.text)
            return '\n\n'.join(parts)

        doc = _assemble(dep_funcs)
        tokens = estimate_tokens(doc)

        # Token budget management
        if tokens > max_tokens * 2 and dep_funcs:
            # Too big: trim deps from highest dep_level first
            while tokens > max_tokens * 2 and dep_funcs:
                dep_funcs.pop()  # remove highest-level dep
                doc = _assemble(dep_funcs)
                tokens = estimate_tokens(doc)

        if tokens < 20:
            continue

        # Deduplicate
        doc_hash = hashlib.md5(doc.encode()).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)

        if enriched:
            # Build parts_info for enriched output. Thread repo-level build
            # context through fallback lanes so build-file-derived args keep
            # their authoritative provenance in language_info/build_info.
            parts_info: list[PartInfo] = []
            if preamble:
                parts_info.append((preamble, 1, 0, '', None))  # kind=1 PREAMBLE
            for df in dep_funcs:
                parts_info.append((df.text, 2, df.dep_level, df.name,
                                   df.qualified_name))  # kind=2 FUNC
            parts_info.append((func.text, 2, func.dep_level, func.name,
                               func.qualified_name))  # kind=2 FUNC (root)
            compile_args = list(default_args or [])
            build_info = dict(default_build_info) if default_build_info else None
            if project_dir is not None:
                abs_func_path = os.path.normpath(os.path.join(project_dir, func.file))
                file_build = compile_db.get(abs_func_path) if compile_db else None
                if file_build:
                    compile_args = file_build.get("compile_args", compile_args)
                    build_info = file_build.get("build_info") or build_info
            documents.append(
                build_enriched_doc(
                    parts_info,
                    index,
                    filepath=func.file,
                    compile_args=compile_args,
                    build_info=build_info,
                )
            )
        else:
            documents.append(doc)

    return documents


def _resolve_file_args(filepath, compile_db, default_args):
    """Get per-file compile args, adapting C vs C++ flags."""
    if compile_db and filepath in compile_db:
        build_entry = compile_db[filepath]
        args = _sanitize_compile_args_for_clang(build_entry.get("compile_args", default_args))
    else:
        args = _sanitize_compile_args_for_clang(default_args)
    return _adapt_args_for_file(args, filepath)


def _parse_file_batch(args_tuple):
    """Worker function for parallel parsing. Each worker creates its own Index."""
    filepaths, compile_db, default_args, project_dir = args_tuple
    sys.setrecursionlimit(50000)  # Set in each worker process too
    _configure_libclang()
    clang_index = Index.create()
    results = []
    errors = 0
    for filepath in filepaths:
        args = _resolve_file_args(filepath, compile_db, default_args)
        try:
            functions = parse_translation_unit(filepath, clang_index, args, project_dir)
            results.extend(f.to_dict() for f in functions)
        except (Exception, RecursionError):
            errors += 1
    return results, len(filepaths), errors


def _parse_single_file_worker(args_tuple):
    """Parse one file in a fresh subprocess so a segfault is file-local."""
    filepath, compile_db, default_args, project_dir = args_tuple
    sys.setrecursionlimit(50000)
    _configure_libclang()
    clang_index = Index.create()
    args = _resolve_file_args(filepath, compile_db, default_args)
    functions = parse_translation_unit(filepath, clang_index, args, project_dir)
    return [f.to_dict() for f in functions]


def _parse_files_isolated(
    cpp_files: list[str],
    compile_db: dict | None,
    default_args: list[str],
    project_dir: str,
    *,
    max_workers: int,
) -> tuple[ProjectIndex, int, int]:
    """Parse files via one-file subprocess tasks so crashes don't kill the whole repo."""
    index_obj = ProjectIndex()
    parsed = 0
    errors = 0
    worker_count = max(1, max_workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _parse_single_file_worker,
                (filepath, compile_db, default_args, project_dir),
            ): filepath
            for filepath in cpp_files
        }
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                func_dicts = future.result(timeout=120)  # 2 min per file
                for d in func_dicts:
                    index_obj.add_function(FunctionDef.from_dict(d))
                parsed += 1
            except TimeoutError:
                errors += 1
                if errors <= 5:
                    print(f"  TIMEOUT parsing {filepath} (>120s)", file=sys.stderr)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
            if parsed % 500 == 0 and parsed > 0:
                print(
                    f"  Parsed {parsed}/{len(cpp_files)} files, {len(index_obj.functions)} functions",
                    file=sys.stderr,
                )
    return index_obj, parsed, errors


def process_project(
    project_dir: str,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    parse_workers: int = 1,
    enriched: bool = False,
    extra_exclude_dirs: set[str] | None = None,
) -> list:
    """Process a single project: parse all files, build index, generate docs."""
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)
    _configure_libclang()

    print(f"\n--- Processing project: {project_name} ---", file=sys.stderr)

    # Find source files
    cpp_files = find_cpp_files(project_dir, extra_exclude_dirs=extra_exclude_dirs)
    print(f"  Found {len(cpp_files)} C/C++ source files", file=sys.stderr)

    if not cpp_files:
        return []

    # Load or derive build context.
    compile_db = load_compile_commands(project_dir)
    _platform_info, _raw_args, _compile_index = detect_build_context(project_dir)
    default_args = _sanitize_compile_args_for_clang(
        get_default_compile_args(project_dir)
    )
    default_build_info = {
        key: value
        for key, value in _platform_info.items()
        if key in {"build_system", "source", "compiler", "standard"} and value is not None
    }

    # Parse all files and build index
    index_obj = ProjectIndex()

    # Use parallel parsing for large projects
    effective_workers = min(parse_workers, max(1, len(cpp_files) // 100))
    if effective_workers > 1 and len(cpp_files) > 200:
        print(f"  Using {effective_workers} parse workers", file=sys.stderr)
        chunk_size = max(50, len(cpp_files) // effective_workers)
        batches = []
        for i in range(0, len(cpp_files), chunk_size):
            batch = cpp_files[i:i + chunk_size]
            batches.append((batch, compile_db, default_args, project_dir))

        total_parsed = 0
        total_errors = 0
        try:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                for func_dicts, parsed_count, error_count in executor.map(_parse_file_batch, batches):
                    for d in func_dicts:
                        index_obj.add_function(FunctionDef.from_dict(d))
                    total_parsed += parsed_count
                    total_errors += error_count
                    print(f"  Parsed {total_parsed}/{len(cpp_files)} files, "
                          f"{len(index_obj.functions)} functions", file=sys.stderr)

            print(f"  Parsed {total_parsed} files ({total_errors} errors), "
                  f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
        except Exception as exc:
            print(
                f"  WARN: parallel parse failed ({exc}); retrying isolated per-file workers",
                file=sys.stderr,
            )
            index_obj, parsed, errors = _parse_files_isolated(
                cpp_files,
                compile_db,
                default_args,
                project_dir,
                max_workers=min(4, effective_workers),
            )
            print(f"  Parsed {parsed} files ({errors} errors), "
                  f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
    else:
        # Sequential for small projects
        clang_index = Index.create()
        parsed = 0
        errors = 0
        for filepath in cpp_files:
            args = _resolve_file_args(filepath, compile_db, default_args)
            try:
                functions = parse_translation_unit(filepath, clang_index, args, project_dir)
                for func in functions:
                    index_obj.add_function(func)
                parsed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
            if parsed % 500 == 0 and parsed > 0:
                print(f"  Parsed {parsed}/{len(cpp_files)} files, "
                      f"{len(index_obj.functions)} functions", file=sys.stderr)
        print(f"  Parsed {parsed} files ({errors} errors), "
              f"{len(index_obj.functions)} functions indexed", file=sys.stderr)

    # Build training documents
    documents = build_training_documents(
        index_obj,
        max_tokens,
        max_dep_depth,
        enriched=enriched,
        project_dir=project_dir,
        compile_db=compile_db,
        default_args=default_args,
        default_build_info=default_build_info,
    )
    print(f"  Generated {len(documents)} training documents", file=sys.stderr)

    stats = index_obj.stats()
    print(f"  Index stats: {stats}", file=sys.stderr)

    return documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Clang-based cross-file C++ dependency indexer')
    parser.add_argument('--project-dir', type=str,
                        help='Single project directory to process')
    parser.add_argument('--projects-list', type=str,
                        help='File listing project directories (one per line)')
    parser.add_argument('--projects-dir', type=str,
                        help='Directory containing multiple project subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='Max tokens per training document (default: 16384)')
    parser.add_argument('--max-dep-depth', type=int, default=5,
                        help='Max dependency resolution depth (default: 5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for multi-project mode')
    parser.add_argument('--parse-workers', type=int, default=8,
                        help='Number of parallel parse workers within each project (default: 8)')
    parser.add_argument('--libclang-path', type=str, default=None,
                        help='Path to libclang.so (auto-detected if not set)')
    parser.add_argument('--append', action='store_true',
                        help='Append to output file instead of overwriting')
    parser.add_argument('--enriched', action='store_true',
                        help='Emit enriched output with structure_ids, chunk_boundaries, '
                             'call_edges, type_edges alongside text')
    parser.add_argument('--exclude-dirs', type=str, default=None,
                        help='Comma-separated extra directory names to exclude from file discovery')

    args = parser.parse_args()

    try:
        _configure_libclang(args.libclang_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Collect project directories
    project_dirs = []
    if args.project_dir:
        project_dirs.append(args.project_dir)
    elif args.projects_list:
        with open(args.projects_list) as f:
            project_dirs = [line.strip() for line in f if line.strip()]
    elif args.projects_dir:
        for entry in sorted(os.listdir(args.projects_dir)):
            full = os.path.join(args.projects_dir, entry)
            if os.path.isdir(full):
                project_dirs.append(full)
    else:
        parser.error("Provide --project-dir, --projects-list, or --projects-dir")

    print(f"Processing {len(project_dirs)} project(s)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Max tokens: {args.max_tokens}", file=sys.stderr)

    extra_exclude = set(args.exclude_dirs.split(',')) if args.exclude_dirs else None

    total_docs = 0
    seen_hashes = set()

    append_mode = getattr(args, 'append', False)
    enriched = args.enriched
    with open(args.output, 'a' if append_mode else 'w') as out:
        if args.workers > 1 and len(project_dirs) > 1:
            # Multi-project parallel mode
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_project, pd, args.max_tokens, args.max_dep_depth,
                        args.parse_workers, enriched, extra_exclude
                    ): pd
                    for pd in project_dirs
                }
                for future in as_completed(futures):
                    pd = futures[future]
                    try:
                        docs = future.result()
                        for doc in docs:
                            if enriched:
                                doc_text = doc['text']
                            else:
                                doc_text = doc
                            doc_hash = hashlib.md5(doc_text.encode()).hexdigest()
                            if doc_hash in seen_hashes:
                                continue
                            seen_hashes.add(doc_hash)
                            if enriched:
                                json.dump(doc, out)
                            else:
                                json.dump({'text': doc}, out)
                            out.write('\n')
                            total_docs += 1
                    except Exception as e:
                        print(f"ERROR processing {pd}: {e}", file=sys.stderr)
        else:
            # Sequential mode
            for pd in project_dirs:
                try:
                    docs = process_project(pd, args.max_tokens, args.max_dep_depth,
                                           args.parse_workers, enriched, extra_exclude)
                    for doc in docs:
                        if enriched:
                            doc_text = doc['text']
                        else:
                            doc_text = doc
                        doc_hash = hashlib.md5(doc_text.encode()).hexdigest()
                        if doc_hash in seen_hashes:
                            continue
                        seen_hashes.add(doc_hash)
                        if enriched:
                            json.dump(doc, out)
                        else:
                            json.dump({'text': doc}, out)
                        out.write('\n')
                        total_docs += 1
                    out.flush()
                except Exception as e:
                    print(f"ERROR processing {pd}: {e}", file=sys.stderr)
                    if len(project_dirs) == 1:
                        return 1
                    print("  Skipping project, continuing...", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total documents: {total_docs}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
