"""Unit tests for :mod:`cppmega_mlx.runtime.kernel_policy`."""

from __future__ import annotations

import pytest

from cppmega_mlx.runtime.kernel_policy import (
    KernelPath,
    clear_dispatch_log,
    get_dispatch_log,
    record_dispatch,
    selected_path,
)


@pytest.fixture(autouse=True)
def _reset_log() -> None:
    clear_dispatch_log()
    yield
    clear_dispatch_log()


def test_selected_path_defaults_to_auto_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    assert selected_path("sparse_mla") is KernelPath.AUTO


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("auto", KernelPath.AUTO),
        ("AUTO", KernelPath.AUTO),
        ("ref", KernelPath.REFERENCE),
        ("reference", KernelPath.REFERENCE),
        ("path_a", KernelPath.REFERENCE),
        ("a", KernelPath.REFERENCE),
        ("path_b", KernelPath.PATH_B),
        ("PATH_B", KernelPath.PATH_B),
        ("b", KernelPath.PATH_B),
        ("path_c", KernelPath.PATH_C),
        ("c", KernelPath.PATH_C),
        ("  c  ", KernelPath.PATH_C),
        ("", KernelPath.AUTO),
        ("garbage", KernelPath.AUTO),  # unknown -> AUTO
    ],
)
def test_selected_path_parses_global_env(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: KernelPath
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", raw)
    assert selected_path("sparse_mla") is expected


def test_per_op_override_wins_over_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_b")
    assert selected_path("sparse_mla") is KernelPath.PATH_B
    # Other ops still see the global ref.
    assert selected_path("mamba3_mimo") is KernelPath.REFERENCE


def test_unknown_op_name_returns_global_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    assert selected_path("never_seen_op_zzz") is KernelPath.REFERENCE
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    assert selected_path("never_seen_op_zzz") is KernelPath.AUTO


def test_selected_path_rejects_non_string_op_name() -> None:
    with pytest.raises(TypeError, match="op_name must be str"):
        selected_path(123)  # type: ignore[arg-type]


def test_record_dispatch_appends_and_get_returns_copy() -> None:
    record_dispatch("sparse_mla", KernelPath.PATH_B, "metal_kernel_fwd_v1")
    record_dispatch("mamba3_mimo", KernelPath.REFERENCE, "reference_pure_mlx")
    log = get_dispatch_log()
    assert log == [
        {
            "op_name": "sparse_mla",
            "path": "path_b",
            "kernel_used": "metal_kernel_fwd_v1",
        },
        {
            "op_name": "mamba3_mimo",
            "path": "ref",
            "kernel_used": "reference_pure_mlx",
        },
    ]
    # Mutating the snapshot must not affect the buffer.
    log.clear()
    assert len(get_dispatch_log()) == 2


def test_record_dispatch_validates_arguments() -> None:
    with pytest.raises(ValueError, match="op_name"):
        record_dispatch("", KernelPath.PATH_B, "kernel")
    with pytest.raises(TypeError, match="path must be KernelPath"):
        record_dispatch("op", "path_b", "kernel")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kernel_used"):
        record_dispatch("op", KernelPath.PATH_B, "")


def test_clear_dispatch_log_resets_buffer() -> None:
    record_dispatch("sparse_mla", KernelPath.AUTO, "metal_kernel_fwd_v1")
    assert get_dispatch_log()
    clear_dispatch_log()
    assert get_dispatch_log() == []


def test_dispatch_log_is_bounded_to_ring_buffer_capacity() -> None:
    capacity = 256
    for i in range(capacity + 50):
        record_dispatch("sparse_mla", KernelPath.AUTO, f"k{i}")
    log = get_dispatch_log()
    assert len(log) == capacity
    # Oldest entries dropped — first entry should be k50.
    assert log[0]["kernel_used"] == "k50"
    assert log[-1]["kernel_used"] == f"k{capacity + 49}"
