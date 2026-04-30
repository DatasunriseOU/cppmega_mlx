from __future__ import annotations

import inspect

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.kernels import metal_ops
from cppmega_mlx.kernels.metal_ops import MetalKernelUnsupported, squared_relu


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
        mx.eval(x)
    return np.asarray(x)


def _sample() -> mx.array:
    values = np.array(
        [
            [-3.0, -0.25, 0.0, 0.5],
            [1.25, 2.0, -7.0, 4.0],
        ],
        dtype=np.float32,
    )
    return mx.array(values)


def _assert_reference(actual: mx.array, x: mx.array) -> None:
    expected = np.maximum(_to_numpy(x), 0.0) ** 2
    np.testing.assert_allclose(_to_numpy(actual), expected, rtol=1e-6, atol=1e-6)


class _MetalProbe:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_can_run_metal_requires_default_gpu_and_available_metal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mx, "metal", _MetalProbe(True))
    monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)

    assert not metal_ops.can_run_metal()

    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx, "metal", _MetalProbe(False))

    assert not metal_ops.can_run_metal()

    monkeypatch.setattr(mx, "metal", None)

    assert not metal_ops.can_run_metal()


def test_pure_mlx_squared_relu_is_reference_path() -> None:
    x = _sample()

    out = squared_relu(x, backend="mlx")

    _assert_reference(out, x)


def test_auto_backend_falls_back_when_metal_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    x = _sample()
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: False)

    out = squared_relu(x, backend="auto")

    _assert_reference(out, x)


def test_status_reports_missing_metal_kernel_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: True)
    monkeypatch.setattr(metal_ops, "_metal_kernel_constructor", lambda: None)

    status = metal_ops.metal_kernel_status(_sample())

    assert not status.available
    assert "mx.fast.metal_kernel API is not available" in status.reason


def test_status_reports_unconstructed_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: True)
    monkeypatch.setattr(metal_ops, "_metal_kernel_constructor", lambda: object())
    monkeypatch.setattr(metal_ops, "_squared_relu_kernel", None)

    status = metal_ops.metal_kernel_status(_sample())

    assert not status.available
    assert "not constructed" in status.reason


def test_explicit_metal_backend_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _sample()
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: False)

    with pytest.raises(MetalKernelUnsupported, match="Metal backend"):
        squared_relu(x, backend="metal")


def test_empty_tensor_uses_fallback_and_explicit_metal_fails_closed() -> None:
    x = mx.array([], dtype=mx.float32)

    out = squared_relu(x, backend="auto")

    assert _to_numpy(out).shape == (0,)
    with pytest.raises(MetalKernelUnsupported, match="empty tensors"):
        squared_relu(x, backend="metal")


def test_explicit_metal_rejects_unsupported_dtype() -> None:
    x = mx.array([1, -2, 3], dtype=mx.int32)

    with pytest.raises(MetalKernelUnsupported, match="unsupported dtype"):
        squared_relu(x, backend="metal")


def test_auto_backend_falls_back_for_unsupported_dtype() -> None:
    x = mx.array([1, -2, 3], dtype=mx.int32)

    out = squared_relu(x, backend="auto")

    _assert_reference(out, x)


def test_training_status_requires_pure_mlx_fallback() -> None:
    status = metal_ops.squared_relu_training_status()

    assert status.differentiable is False
    assert status.fallback_backend == "mlx"
    assert "VJP/JVP" in status.reason


def test_training_auto_uses_pure_mlx_even_when_metal_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _sample()
    monkeypatch.setattr(
        metal_ops,
        "metal_kernel_status",
        lambda _: metal_ops.MetalKernelStatus(True, "test-only eligible metal"),
    )

    def fail_if_called(_: mx.array) -> mx.array:
        raise AssertionError("training=True must not call a forward-only Metal kernel")

    monkeypatch.setattr(metal_ops, "_squared_relu_metal", fail_if_called)

    out = squared_relu(x, backend="auto", training=True)

    _assert_reference(out, x)


def test_training_mlx_fallback_preserves_dtype() -> None:
    x = mx.array([-2.0, 0.5, 3.0], dtype=mx.float16)

    out = squared_relu(x, backend="mlx", training=True)

    assert out.dtype == mx.float16
    _assert_reference(out, x)


def test_training_metal_backend_rejects_forward_only_kernel() -> None:
    with pytest.raises(MetalKernelUnsupported, match="forward-only.*VJP/JVP"):
        squared_relu(_sample(), backend="metal", training=True)


def test_training_fallback_is_differentiable_with_mlx_grad() -> None:
    def loss_fn(x: mx.array) -> mx.array:
        return mx.sum(squared_relu(x, training=True))

    x = _sample()
    grad = mx.grad(loss_fn)(x)

    expected = np.where(_to_numpy(x) > 0.0, 2.0 * _to_numpy(x), 0.0)
    np.testing.assert_allclose(_to_numpy(grad), expected, rtol=1e-6, atol=1e-6)


def test_training_fallback_supports_mlx_jvp() -> None:
    x = _sample()
    tangent = mx.ones_like(x)

    primals, tangents = mx.jvp(lambda y: squared_relu(y, training=True), [x], [tangent])

    assert len(primals) == 1
    assert len(tangents) == 1
    _assert_reference(primals[0], x)
    expected_tangent = np.where(_to_numpy(x) > 0.0, 2.0 * _to_numpy(x), 0.0)
    np.testing.assert_allclose(_to_numpy(tangents[0]), expected_tangent, rtol=1e-6, atol=1e-6)


def test_bad_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        squared_relu(_sample(), backend="cuda")  # type: ignore[arg-type]


def test_metal_ops_module_has_no_cuda_runtime_branch() -> None:
    source = inspect.getsource(metal_ops).lower()

    assert "cuda" not in source


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (mx.float32, 1e-6, 1e-6),
        (mx.float16, 1e-3, 1e-3),
        (mx.bfloat16, 1e-2, 1e-2),
    ],
)
def test_metal_squared_relu_supported_dtype_parity_if_available(
    dtype: mx.Dtype,
    rtol: float,
    atol: float,
) -> None:
    x = _sample().astype(dtype)
    status = metal_ops.metal_kernel_status(x)
    if not status.available:
        pytest.skip(status.reason)

    out = squared_relu(x, backend="metal")

    assert out.dtype == dtype
    expected = np.maximum(_to_numpy(x), 0.0) ** 2
    np.testing.assert_allclose(_to_numpy(out), expected, rtol=rtol, atol=atol)
