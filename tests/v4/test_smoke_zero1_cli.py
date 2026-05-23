"""V7-Q05: regression for cppmega_mlx.cli.smoke_zero1.

Pin the argparse + command-construction path. We don't actually launch
mlx.launch in CI (multi-process, slow); --dry-run mode exposes the
argv that *would* be invoked and the smoke receipt regeneration is
covered separately in scripts/bench_zero1_loopback.py's own test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from cppmega_mlx.cli import smoke_zero1


def test_resolve_world_size_from_hosts() -> None:
    assert smoke_zero1._resolve_world_size("a,b,c", 0) == 3
    assert smoke_zero1._resolve_world_size("a,b", 0) == 2
    assert smoke_zero1._resolve_world_size("solo", 0) == 1


def test_resolve_world_size_explicit_override() -> None:
    # Explicit --world-size wins over derived count.
    assert smoke_zero1._resolve_world_size("a,b", 4) == 4


def test_build_launch_argv_threads_hosts_and_steps() -> None:
    argv = smoke_zero1._build_launch_argv(
        mlx_launch="mlx.launch",
        hosts_csv="127.0.0.1,127.0.0.1",
        world_size=2, num_steps=11, lr=2e-4,
        out=Path("/tmp/out.json"),
    )
    # Order matters: mlx.launch -n W --hosts H python script ...
    assert argv[0] == "mlx.launch"
    assert "-n" in argv and argv[argv.index("-n") + 1] == "2"
    assert "--hosts" in argv
    assert argv[argv.index("--hosts") + 1] == "127.0.0.1,127.0.0.1"
    assert "--steps" in argv and argv[argv.index("--steps") + 1] == "11"
    assert "--lr" in argv and argv[argv.index("--lr") + 1] == "0.0002"
    assert "--out" in argv
    assert argv[argv.index("--out") + 1] == "/tmp/out.json"


def test_build_simulate_argv_inserts_simulate_flag() -> None:
    argv = smoke_zero1._build_simulate_argv(
        num_steps=7, lr=1e-3, out=Path("/tmp/sim.json"),
    )
    assert "--simulate" in argv
    assert argv[0] == sys.executable
    assert "--steps" in argv and argv[argv.index("--steps") + 1] == "7"


def test_dry_run_returns_zero_without_subprocess(capsys) -> None:
    code = smoke_zero1.main([
        "--hosts", "127.0.0.1,127.0.0.1",
        "--num-steps", "2",
        "--dry-run",
    ])
    assert code == 0
    err = capsys.readouterr().err
    # The dry-run path prints the would-be command on stderr.
    assert "smoke_zero1" in err
    assert "127.0.0.1,127.0.0.1" in err


def test_dry_run_simulate_returns_zero(capsys) -> None:
    code = smoke_zero1.main(["--simulate", "--dry-run"])
    assert code == 0
    err = capsys.readouterr().err
    assert "--simulate" in err


def test_missing_mlx_launch_returns_ex_usage(capsys) -> None:
    """When mlx.launch is not on PATH (non --simulate path), exit 64."""
    with mock.patch("shutil.which", return_value=None):
        code = smoke_zero1.main([
            "--hosts", "127.0.0.1,127.0.0.1",
            "--mlx-launch", "/nonexistent/mlx.launch",
        ])
    assert code == 64
    err = capsys.readouterr().err
    assert "not found" in err or "PATH" in err


def test_out_path_default_loopback() -> None:
    """--out default for non-simulate is bench/baselines/zero1_smoke.json."""
    assert smoke_zero1.DEFAULT_OUT.name == "zero1_smoke.json"
    assert smoke_zero1.DEFAULT_SIMULATE_OUT.name == \
        "zero1_simulated_2proc_m4.json"
