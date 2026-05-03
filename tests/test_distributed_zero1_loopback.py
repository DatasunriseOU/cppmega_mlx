"""End-to-end smoke for the ZeRO-1 single-host loopback receipt.

Spawns ``scripts/bench_zero1_loopback.py`` via subprocess in two
configurations and asserts that the resulting receipt JSON satisfies the
1% parity criterion against the single-process W=1 control:

* ``--simulate`` -- in-process simulation that mirrors the helper used
  in :mod:`tests.test_distributed_zero1`. Fast (<10 s) and exercised on
  every CI run that includes the slow marker.
* ``mlx.launch -n 2 --hosts 127.0.0.1`` loopback -- spins up two
  processes that talk to each other via the ring TCP backend on
  ``127.0.0.1``. Tests the real :func:`mx.distributed.all_sum` /
  :func:`mx.distributed.init` runtime; gated behind ``mlx.launch``
  being importable + the ring backend being available.

Both tests are marked :mod:`pytest.mark.slow` because they each spawn
new Python processes and load the smoke model; the default test run
(``pytest -m 'not slow'``) skips them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_zero1_loopback.py"


def _venv_python() -> str:
    """Return the project venv's python interpreter.

    Callers in CI must run via ``.venv/bin/python -m pytest``; we re-use
    :data:`sys.executable` so the spawned subprocess also has access to
    the project's installed dependencies. Tests do not fall back to the
    system python because cppmega_mlx + mlx are venv-installed.
    """

    return sys.executable


def _mlx_launch_available() -> bool:
    """Best-effort detection of ``mlx.launch`` availability.

    We require both the launcher script (installed by mlx-pip) and the
    ring backend (ring is the default macOS / non-CUDA backend; absence
    means the loopback receipt would fall back to a no-op group).
    """

    try:
        import mlx.core as mx  # noqa: WPS433 -- runtime probe

        if not mx.distributed.is_available("ring"):
            return False
    except Exception:
        return False
    return shutil.which("mlx.launch") is not None or (
        Path(_venv_python()).parent / "mlx.launch"
    ).exists()


def _venv_mlx_launch() -> str:
    """Resolve the venv-bundled ``mlx.launch`` shim, falling back to PATH."""

    candidate = Path(_venv_python()).parent / "mlx.launch"
    if candidate.exists():
        return str(candidate)
    fallback = shutil.which("mlx.launch")
    if fallback is None:
        raise RuntimeError("mlx.launch is not installed; cannot run loopback test")
    return fallback


@pytest.mark.slow
def test_zero1_simulated_2proc_receipt(tmp_path: Path) -> None:
    """In-process simulation must produce a receipt with parity_passed.

    This test does not invoke ``mlx.launch`` -- it runs the in-process
    simulated W=2 path that mirrors the in-test helper. Useful for CI
    environments where ``mlx.launch`` is unavailable; still validates the
    wrapper's selection / shard / merge helpers end-to-end.
    """

    out_path = tmp_path / "zero1_simulated_2proc.json"
    proc = subprocess.run(
        [
            _venv_python(),
            str(SCRIPT),
            "--simulate",
            "--steps",
            "20",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        f"bench script exit={proc.returncode}\n"
        f"stdout=\n{proc.stdout}\n"
        f"stderr=\n{proc.stderr}"
    )
    assert out_path.exists(), "receipt JSON was not written"
    receipt = json.loads(out_path.read_text())

    assert receipt["primitive"] == "multiprocessing-simulation"
    assert receipt["world_size"] == 2
    assert receipt["host_count"] == 1
    assert receipt["production_multi_node_receipt"] is False
    assert receipt["parity_passed"] is True
    assert receipt["loss_diff_w2_vs_w1_relative"] < receipt["parity_tolerance_relative"]
    assert len(receipt["ranks"]) == 2
    assert {row["rank"] for row in receipt["ranks"]} == {0, 1}
    for row in receipt["ranks"]:
        # Each rank must hold strictly less optimizer state than the W=1
        # full-replica control (round-robin sharding cannot send the
        # entire state to a single rank with W=2 + multiple leaves).
        assert row["opt_state_bytes"] < receipt["control_run_w1"]["opt_state_bytes"]


@pytest.mark.slow
def test_zero1_loopback_2proc_receipt(tmp_path: Path) -> None:
    """Real ``mlx.launch -n 2 --hosts 127.0.0.1`` loopback receipt.

    Skips when ``mlx.launch`` or the ring backend is not available --
    that is the documented fallback path in
    :mod:`docs/distributed_zero1_smoke_procedure.md` and the simulated
    test above is the substitute coverage. When the loopback launcher
    is available the test asserts the same parity contract plus the
    additional invariant that the receipt was tagged
    ``primitive: "mx.distributed"`` (i.e. the real distributed runtime
    actually fired).
    """

    if not _mlx_launch_available():
        pytest.skip("mlx.launch / ring backend unavailable; loopback receipt skipped")

    out_path = tmp_path / "zero1_loopback_2proc.json"
    receipt_dir = tmp_path / "zero1_loopback_2proc.ranks"
    proc = subprocess.run(
        [
            _venv_mlx_launch(),
            "-n",
            "2",
            "--hosts",
            "127.0.0.1",
            "--backend",
            "ring",
            "--python",
            _venv_python(),
            "--",
            str(SCRIPT),
            "--steps",
            "10",  # 10 steps keeps the test under the 5 min default budget.
            "--out",
            str(out_path),
            "--receipt-dir",
            str(receipt_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"mlx.launch exit={proc.returncode}\n"
        f"stdout=\n{proc.stdout}\n"
        f"stderr=\n{proc.stderr}"
    )
    assert out_path.exists(), "loopback receipt JSON was not written"
    receipt = json.loads(out_path.read_text())

    assert receipt["primitive"] == "mx.distributed"
    assert receipt["world_size"] == 2
    assert receipt["host_count"] == 1
    assert receipt["production_multi_node_receipt"] is False
    assert receipt["parity_passed"] is True
    assert receipt["loss_diff_w2_vs_w1_relative"] < receipt["parity_tolerance_relative"]

    assert len(receipt["ranks"]) == 2
    assert {row["rank"] for row in receipt["ranks"]} == {0, 1}
    for row in receipt["ranks"]:
        assert row["is_sharded"] is True
        # Each rank's opt-state is strictly less than the W=1 control's.
        assert row["opt_state_bytes"] < receipt["control_run_w1"]["opt_state_bytes"]
