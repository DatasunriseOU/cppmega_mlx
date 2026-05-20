"""Helpers for build-system-aware compile context detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
from typing import Callable


CPP_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cp",
        ".cpp",
        ".cxx",
        ".c++",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".h++",
        ".inl",
        ".ipp",
        ".tpp",
        ".cu",
        ".cuh",
        ".hip",
        ".cl",
    }
)

DEFAULT_PLATFORM = {
    "platform": "x86_64-linux-gnu",
    "compiler": "g++",
    "standard": "c++17",
    "arch": "x86_64",
    "mode": "user",
}

_STD_FLAG_RE = re.compile(r"(?P<flag>(?:-std=|--std=|/std:|-cl-std=)(?P<value>[^\s\"']+))")
_TARGET_FLAG_RE = re.compile(r"(?P<flag>(?:--target=|-march=|-mcpu=)(?P<value>[^\s\"']+))")
_LANG_TOKEN_RE = re.compile(r"^(?:-x(?P<joined>.+)|-x)$")
_VAR_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Z_][A-Z0-9_]*)\s*(?:[:+?]?=)\s*(?P<value>.*?)\s*$")
_STRING_LITERAL_RE = re.compile(r"['\"]([^'\"]+)['\"]", re.S)
_CMAKE_STD_RE = re.compile(r"CMAKE_(?P<lang>C|CXX)_STANDARD\s+(\d+)", re.IGNORECASE)
_CMAKE_COMPILER_RE = re.compile(
    r"CMAKE_(?P<lang>C|CXX)_COMPILER\s+\"?([^\"\s)]+)\"?",
    re.IGNORECASE,
)
_CMAKE_PROCESSOR_RE = re.compile(
    r"CMAKE_SYSTEM_PROCESSOR\s+\"?([^\"\s)]+)\"?",
    re.IGNORECASE,
)
_MESON_STD_RE = re.compile(r"(?P<kind>cpp|c)_std\s*[:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_MAKEFILE_NAMES = ("GNUmakefile", "Makefile", "makefile")
_BAZEL_BUILD_NAMES = ("BUILD.bazel", "BUILD", "WORKSPACE.bazel", "WORKSPACE", "MODULE.bazel", ".bazelrc")
_GN_BUILD_NAMES = ("BUILD.gn", "args.gn", ".gn")
_SCONS_BUILD_NAMES = ("SConstruct", "SConscript")
_XMAKE_BUILD_NAMES = ("xmake.lua",)
_FLAG_PREFIXES = ("-I", "-D", "-U", "-std=", "--std=", "/std:", "-cl-std=", "--target=", "-march=", "-mcpu=")
_VERBATIM_FLAGS = {"-m32", "-m64", "--hip-link"}


def is_cpp_path(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in CPP_EXTENSIONS


def split_shell_command(command: str | None) -> list[str]:
    if not command:
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


@dataclass(slots=True)
class CompileCommandEntry:
    filepath: str
    compile_args: list[str]
    build_info: dict
    candidates: set[str] = field(default_factory=set)


@dataclass(slots=True)
class CompileCommandsIndex:
    entries: list[CompileCommandEntry]
    by_key: dict[str, CompileCommandEntry]

    def lookup(self, file_path: str) -> tuple[list[str] | None, dict | None]:
        normalized = os.path.normpath(file_path)
        for key in _path_candidates(normalized):
            entry = self.by_key.get(key)
            if entry is not None:
                return list(entry.compile_args), dict(entry.build_info)
        for entry in self.entries:
            if normalized == entry.filepath or normalized.endswith(entry.filepath):
                return list(entry.compile_args), dict(entry.build_info)
            for candidate in entry.candidates:
                if normalized.endswith(candidate):
                    return list(entry.compile_args), dict(entry.build_info)
        return None, None


@dataclass(slots=True)
class BuildDetection:
    build_system: str
    source: str
    compiler: str | None = None
    standard: str | None = None
    language: str | None = None
    flags: list[str] = field(default_factory=list)
    arch: str | None = None


def _path_candidates(path: str, directory: str | None = None) -> set[str]:
    candidates: set[str] = set()
    if not path:
        return candidates
    normalized = os.path.normpath(path)
    candidates.add(normalized)
    candidates.add(os.path.basename(normalized))
    if directory and not os.path.isabs(path):
        joined = os.path.normpath(os.path.join(directory, path))
        candidates.add(joined)
        candidates.add(os.path.basename(joined))
    return candidates


def _sanitize_compile_args(argv: list[str], filepath: str) -> tuple[list[str], str | None]:
    compiler = argv[0] if argv else None
    normalized_file = os.path.normpath(filepath)
    flags: list[str] = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-o", "-MF", "-MQ", "-MT", "/Fo", "/Fd", "/Fi"}:
            skip_next = True
            continue
        if arg in {"-c", "-S"}:
            continue
        normalized_arg = os.path.normpath(arg)
        if normalized_arg == normalized_file or normalized_arg.endswith(normalized_file):
            continue
        if arg.endswith((".o", ".obj", ".pcm")):
            continue
        flags.append(arg)
    return flags, compiler


def parse_compile_commands_entries(entries: list[dict]) -> CompileCommandsIndex:
    normalized_entries: list[CompileCommandEntry] = []
    by_key: dict[str, CompileCommandEntry] = {}

    for raw in entries:
        filepath = raw.get("file", "")
        directory = raw.get("directory", "") or None
        args = raw.get("arguments")
        if isinstance(args, list) and args:
            argv = [str(arg) for arg in args]
        else:
            argv = split_shell_command(raw.get("command", ""))
        if not filepath or not argv:
            continue
        candidates = _path_candidates(filepath, directory)
        preferred = os.path.normpath(os.path.join(directory, filepath)) if directory and not os.path.isabs(filepath) else os.path.normpath(filepath)
        compile_args, compiler = _sanitize_compile_args(argv, preferred)
        entry = CompileCommandEntry(
            filepath=preferred,
            compile_args=compile_args,
            build_info={
                "build_system": "compile_commands",
                "source": "compile_commands",
                **({"compiler": compiler} if compiler else {}),
            },
            candidates=candidates,
        )
        normalized_entries.append(entry)
        for candidate in candidates:
            by_key.setdefault(candidate, entry)

    return CompileCommandsIndex(entries=normalized_entries, by_key=by_key)


def load_compile_commands_text(text: str | None) -> CompileCommandsIndex | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    index = parse_compile_commands_entries(data)
    return index if index.entries else None


def load_compile_commands_file(path: str) -> CompileCommandsIndex | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return load_compile_commands_text(f.read())


def _extract_string_literals(text: str) -> list[str]:
    return [match.group(1) for match in _STRING_LITERAL_RE.finditer(text)]


def _extract_flag_tokens(tokens: list[str]) -> list[str]:
    selected: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        stripped = token.strip().rstrip(",)")
        if stripped in _VERBATIM_FLAGS or stripped.startswith(_FLAG_PREFIXES):
            selected.append(stripped)
        elif stripped == "-x" and i + 1 < len(tokens):
            selected.extend(["-x", tokens[i + 1].strip().rstrip(",)")])
            i += 1
        elif stripped.startswith("-x") and len(stripped) > 2:
            selected.append(stripped)
        i += 1
    return selected


def _normalize_standard(raw_value: str | None) -> str | None:
    if not isinstance(raw_value, str):
        return None
    value = raw_value.strip().lower()
    if value.startswith("gnu++"):
        return f"c++{value.removeprefix('gnu++')}"
    if value.startswith("c++"):
        return value
    if value.startswith("gnu"):
        return f"c{value.removeprefix('gnu')}"
    if value.startswith("c"):
        return value
    if value.startswith("cl"):
        return value
    if value.startswith("cxx"):
        return f"c++{value.removeprefix('cxx')}"
    return None


def _extract_standard_from_flags(flags: list[str]) -> tuple[str | None, str | None]:
    for flag in flags:
        match = _STD_FLAG_RE.match(flag)
        if match:
            return _normalize_standard(match.group("value")), match.group("flag")
    return None, None


def _infer_arch_from_flags(flags: list[str]) -> str | None:
    for flag in flags:
        if flag == "-m32":
            return "i686"
        if flag == "-m64" or flag == "-march=x86-64":
            return "x86_64"
        lowered = flag.lower()
        if "aarch64" in lowered or "arm64" in lowered:
            return "aarch64"
        if "riscv64" in lowered:
            return "riscv64"
    return None


def _arch_to_platform(arch: str | None) -> str | None:
    if arch == "i686":
        return "i686-linux-gnu"
    if arch == "x86_64":
        return "x86_64-linux-gnu"
    if arch == "aarch64":
        return "aarch64-linux-gnu"
    if arch == "riscv64":
        return "riscv64-linux-gnu"
    return None


def _infer_language(standard: str | None, compiler: str | None, flags: list[str], default: str | None = None) -> str | None:
    compiler_name = Path(compiler or "").name.lower()
    if compiler_name == "nvcc":
        return "cuda"
    if compiler_name == "hipcc":
        return "hip"
    for idx, flag in enumerate(flags):
        lowered = flag.lower()
        match = _LANG_TOKEN_RE.match(lowered)
        if match:
            joined = match.group("joined")
            if joined is not None:
                if joined == "cuda":
                    return "cuda"
                if joined == "hip":
                    return "hip"
                if joined in {"cl", "opencl", "opencl-c"}:
                    return "opencl"
                if joined in {"c", "c-header"}:
                    return "c"
                if joined.startswith("c++"):
                    return "c++"
        if lowered == "-x" and idx + 1 < len(flags):
            next_flag = flags[idx + 1].lower()
            if next_flag in {"cuda", "hip", "cl", "opencl", "opencl-c", "c"}:
                return {"cl": "opencl", "opencl": "opencl", "opencl-c": "opencl"}.get(next_flag, next_flag)
            if next_flag.startswith("c++"):
                return "c++"
        if lowered in {"/tc"}:
            return "c"
        if lowered in {"/tp"}:
            return "c++"
        if lowered.startswith("--cuda-gpu-arch") or lowered.startswith("--generate-code") or lowered.startswith("-gencode"):
            return "cuda"
        if lowered.startswith("--offload-arch") or lowered.startswith("--hip-path"):
            return "hip"
    if standard:
        if standard.startswith("c++"):
            return "c++"
        if standard.startswith("cl"):
            return "opencl"
        if standard.startswith("c"):
            return "c"
    return default


def _fallback_compile_args(detection: BuildDetection) -> list[str]:
    args: list[str] = []
    flags = _extract_flag_tokens(detection.flags)
    language = _infer_language(detection.standard, detection.compiler, flags, default=detection.language) or "c++"
    if not any(flag == "-x" or flag.startswith("-x") or flag.lower() in {"/tc", "/tp"} for flag in flags):
        if language == "opencl":
            args.extend(["-x", "cl"])
        else:
            args.extend(["-x", language])
    std_flag = _extract_standard_from_flags(flags)[1]
    if std_flag is None and detection.standard:
        normalized = detection.standard
        if normalized.startswith("c++"):
            std_flag = f"-std={normalized}"
        elif normalized.startswith("c"):
            std_flag = f"-std={normalized}"
        elif normalized.startswith("cl"):
            std_flag = f"-cl-std={normalized}"
    if std_flag:
        args.append(std_flag)
    args.extend(flags)
    args.extend(["-fsyntax-only", "-Wno-everything"])
    return args


def _detection_to_context(detection: BuildDetection) -> tuple[dict, list[str]]:
    result = dict(DEFAULT_PLATFORM)
    compiler = detection.compiler
    if compiler:
        compiler_name = Path(compiler).name
        result["compiler"] = compiler_name
    standard = detection.standard or _extract_standard_from_flags(_extract_flag_tokens(detection.flags))[0]
    if standard:
        result["standard"] = standard
    arch = detection.arch or _infer_arch_from_flags(_extract_flag_tokens(detection.flags))
    if arch:
        result["arch"] = arch
        platform = _arch_to_platform(arch)
        if platform:
            result["platform"] = platform
    result["build_system"] = detection.build_system
    result["source"] = detection.source
    compile_args = _fallback_compile_args(detection)
    return result, compile_args


def _build_detection_from_compile_commands(text: str | None) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    index = load_compile_commands_text(text)
    if index is None or not index.entries:
        return {}, [], None
    first = index.entries[0]
    flags = list(first.compile_args)
    standard = _extract_standard_from_flags(flags)[0]
    context, compile_args = _detection_to_context(
        BuildDetection(
            build_system="compile_commands",
            source="compile_commands",
            compiler=first.build_info.get("compiler"),
            standard=standard,
            flags=flags,
        )
    )
    return context, compile_args or flags, index


def _parse_cmake(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    compiler = None
    cxx_standard = None
    c_standard = None
    for match in _CMAKE_STD_RE.finditer(text):
        lang = match.group("lang").upper()
        normalized = _normalize_standard(f"{'c++' if lang == 'CXX' else 'c'}{match.group(2)}")
        if lang == "CXX" and normalized:
            cxx_standard = normalized
        elif lang == "C" and normalized:
            c_standard = normalized
    for match in _CMAKE_COMPILER_RE.finditer(text):
        lang = match.group("lang").upper()
        candidate = match.group(2)
        if lang == "CXX":
            compiler = candidate
        elif compiler is None:
            compiler = candidate
    arch = None
    processor_match = _CMAKE_PROCESSOR_RE.search(text)
    if processor_match:
        proc = processor_match.group(1).lower()
        if "arm" in proc or "aarch64" in proc:
            arch = "aarch64"
        elif "riscv" in proc:
            arch = "riscv64"
        elif "86" in proc:
            arch = "x86_64"
    standard = cxx_standard or c_standard
    language = "c++" if cxx_standard or compiler and "++" in compiler else ("c" if c_standard else None)
    if not (flags or compiler or standard or arch):
        return None
    return BuildDetection(
        build_system="cmake",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
        arch=arch,
    )


def _parse_autoconf(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(split_shell_command(" ".join(_extract_string_literals(text))))
    compiler = None
    if "ac_prog_cxx" in text.lower():
        compiler = "g++"
    elif "ac_prog_cc" in text.lower():
        compiler = "gcc"
    standard = _extract_standard_from_flags(flags)[0]
    language = "c++" if compiler == "g++" or (standard and standard.startswith("c++")) else ("c" if standard else None)
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="autoconf",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_meson(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    cpp_standard = None
    c_standard = None
    for match in _MESON_STD_RE.finditer(text):
        kind = match.group("kind").lower()
        normalized = _normalize_standard(match.group(2))
        if kind == "cpp" and normalized:
            cpp_standard = normalized
        elif kind == "c" and normalized:
            c_standard = normalized
    standard = cpp_standard or c_standard
    language = "c++" if cpp_standard else ("c" if c_standard else None)
    if not (flags or standard):
        return None
    return BuildDetection(
        build_system="meson",
        source="build_files",
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_make(text: str) -> BuildDetection | None:
    variables: dict[str, list[str]] = {}
    compiler = None
    language: str | None = None
    recipe_flags: list[str] = []
    for line in text.splitlines():
        match = _VAR_ASSIGN_RE.match(line)
        if match:
            name = match.group("name")
            value = match.group("value")
            if name in {"CC", "CXX"}:
                parts = split_shell_command(value)
                if parts:
                    compiler = parts[0]
            elif name in {"CFLAGS", "CXXFLAGS", "CPPFLAGS"}:
                variables.setdefault(name, []).extend(split_shell_command(value))
            continue
        stripped = line.lstrip()
        if stripped.startswith(("gcc ", "g++ ", "clang ", "clang++ ", "cc ", "c++ ", "nvcc ", "hipcc ")):
            recipe_flags.extend(split_shell_command(stripped))
    flags: list[str] = []
    flags.extend(variables.get("CPPFLAGS", []))
    if variables.get("CXXFLAGS"):
        flags.extend(variables["CXXFLAGS"])
        language = "c++"
    else:
        flags.extend(variables.get("CFLAGS", []))
        language = "c" if variables.get("CFLAGS") else None
    flags.extend(recipe_flags)
    flags = _extract_flag_tokens(flags)
    standard = _extract_standard_from_flags(flags)[0]
    if not (compiler or flags or standard):
        return None
    if compiler is None and any("c++" in flag or standard and standard.startswith("c++") for flag in flags):
        compiler = "g++"
    return BuildDetection(
        build_system="make",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_bazel(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for name, text in texts.items():
        literals = _extract_string_literals(text)
        if name == ".bazelrc":
            for token in literals + text.split():
                if token.startswith("--cxxopt="):
                    flags.append(token.removeprefix("--cxxopt="))
                elif token.startswith("--conlyopt="):
                    flags.append(token.removeprefix("--conlyopt="))
        else:
            flags.extend(_extract_flag_tokens(literals))
    if "nvcc" in " ".join(flags).lower():
        compiler = "nvcc"
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or standard):
        return None
    return BuildDetection(
        build_system="bazel",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language="c++",
        flags=flags,
    )


def _parse_gn(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for text in texts.values():
        literals = _extract_string_literals(text)
        flags.extend(_extract_flag_tokens(literals))
        if "is_clang = true" in text.lower():
            compiler = "clang++"
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="gn",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language="c++",
        flags=flags,
    )


def _parse_scons(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for text in texts.values():
        for line in text.splitlines():
            match = _VAR_ASSIGN_RE.match(line)
            if match and match.group("name") in {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CCFLAGS", "CPPFLAGS"}:
                name = match.group("name")
                value = split_shell_command(match.group("value"))
                if name in {"CC", "CXX"} and value:
                    compiler = value[0]
                else:
                    flags.extend(value)
        flags.extend(_extract_flag_tokens(_extract_string_literals(text)))
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or compiler or standard):
        return None
    language = "c++" if compiler and ("++" in compiler or compiler == "clang++") else None
    if language is None and standard:
        language = "c++" if standard.startswith("c++") else "c"
    return BuildDetection(
        build_system="scons",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_xmake(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    compiler = None
    for literal in _extract_string_literals(text):
        lowered = literal.lower()
        if lowered in {"clang", "clang++", "gcc", "g++", "nvcc", "hipcc"} and compiler is None:
            compiler = literal
        if lowered.startswith("cxx"):
            flags.append(f"-std=c++{lowered.removeprefix('cxx')}")
        elif lowered.startswith("c") and lowered[1:].isdigit():
            flags.append(f"-std=c{lowered.removeprefix('c')}")
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    language = "c++" if standard and standard.startswith("c++") else None
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="xmake",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def detect_build_context_from_loader(read_text: Callable[[str], str | None]) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    compile_commands = _build_detection_from_compile_commands(read_text("compile_commands.json"))
    if compile_commands[2] is not None:
        return compile_commands

    cmake_text = read_text("CMakeLists.txt")
    if cmake_text:
        detection = _parse_cmake(cmake_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    configure_text = read_text("configure.ac")
    if configure_text:
        detection = _parse_autoconf(configure_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    meson_text = read_text("meson.build")
    if meson_text:
        detection = _parse_meson(meson_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    bazel_texts = {name: text for name in _BAZEL_BUILD_NAMES if (text := read_text(name)) is not None}
    if bazel_texts:
        detection = _parse_bazel(bazel_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    gn_texts = {name: text for name in _GN_BUILD_NAMES if (text := read_text(name)) is not None}
    if gn_texts:
        detection = _parse_gn(gn_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    scons_texts = {name: text for name in _SCONS_BUILD_NAMES if (text := read_text(name)) is not None}
    if scons_texts:
        detection = _parse_scons(scons_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    xmake_text = next((read_text(name) for name in _XMAKE_BUILD_NAMES if read_text(name)), None)
    if xmake_text:
        detection = _parse_xmake(xmake_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    make_text = next((read_text(name) for name in _MAKEFILE_NAMES if read_text(name)), None)
    if make_text:
        detection = _parse_make(make_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    return {
        **DEFAULT_PLATFORM,
        "build_system": "default",
        "source": "default",
    }, ["-std=c++17", "-fsyntax-only", "-Wno-everything"], None


def detect_build_context(repo_dir: str) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    def read_text(name: str) -> str | None:
        path = os.path.join(repo_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    return detect_build_context_from_loader(read_text)
