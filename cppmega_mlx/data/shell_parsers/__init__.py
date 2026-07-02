"""Shell-domain parser entrypoints."""

from cppmega_mlx.data.shell_parsers.bash import parse_bash
from cppmega_mlx.data.shell_parsers.posix import parse_sh
from cppmega_mlx.data.shell_parsers.tcsh import parse_tcsh
from cppmega_mlx.data.shell_parsers.zsh import parse_zsh

__all__ = ["parse_bash", "parse_sh", "parse_tcsh", "parse_zsh"]
