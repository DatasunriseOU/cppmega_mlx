"""Deterministic build-system domain parsers with lazy public exports."""

from importlib import import_module


_EXPORT_MODULES = {
    "parse_autoconf": "cppmega_mlx.data.build_parsers.autotools",
    "parse_automake": "cppmega_mlx.data.build_parsers.autotools",
    "parse_configure": "cppmega_mlx.data.build_parsers.autotools",
    "parse_bazel": "cppmega_mlx.data.build_parsers.bazel",
    "parse_cmake": "cppmega_mlx.data.build_parsers.cmake",
    "parse_compile_commands": "cppmega_mlx.data.build_parsers.compile_commands",
    "parse_dockerfile": "cppmega_mlx.data.build_parsers.dockerfile",
    "parse_make": "cppmega_mlx.data.build_parsers.make",
    "parse_meson": "cppmega_mlx.data.build_parsers.meson",
    "parse_ninja": "cppmega_mlx.data.build_parsers.ninja",
}

__all__ = [
    "parse_autoconf",
    "parse_automake",
    "parse_configure",
    "parse_bazel",
    "parse_cmake",
    "parse_compile_commands",
    "parse_dockerfile",
    "parse_make",
    "parse_meson",
    "parse_ninja",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
