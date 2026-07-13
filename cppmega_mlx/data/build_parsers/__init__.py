"""Deterministic build-system domain parsers."""

from cppmega_mlx.data.build_parsers.autotools import (
    parse_autoconf,
    parse_automake,
    parse_configure,
)
from cppmega_mlx.data.build_parsers.bazel import parse_bazel
from cppmega_mlx.data.build_parsers.cmake import parse_cmake
from cppmega_mlx.data.build_parsers.make import parse_make
from cppmega_mlx.data.build_parsers.ninja import parse_ninja

__all__ = [
    "parse_autoconf",
    "parse_automake",
    "parse_configure",
    "parse_bazel",
    "parse_cmake",
    "parse_make",
    "parse_ninja",
]
