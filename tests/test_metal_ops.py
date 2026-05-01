from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import mlx.core as mx

import cppmega_mlx.kernels as kernels
from cppmega_mlx.kernels import metal_ops
from cppmega_mlx.kernels.metal_ops import (
    MetalKernelUnsupported,
    TrainingKernelStatus,
    squared_relu,
)


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

    assert isinstance(status, TrainingKernelStatus)
    assert status.in_tree is True
    assert status.source_pinned is True
    assert status.license_covered is True
    assert status.fallback_covered is True
    assert status.parity_covered is True
    assert status.hotspot_evidence is False
    assert status.vjp_covered is False
    assert status.jvp_covered is False
    assert status.training_safe is False
    assert status.differentiable is False
    assert status.fallback_backend == "mlx"
    assert "in-tree" in status.reason
    assert "parity remains covered" in status.reason
    assert "hotspot evidence" in status.reason
    assert "VJP/JVP" in status.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"in_tree": False},
        {"source_pinned": False},
        {"license_covered": False},
        {"fallback_covered": False},
        {"parity_covered": False},
        {"hotspot_evidence": False},
        {"differentiable": False},
        {"vjp_covered": False},
        {"fallback_backend": "metal"},
    ],
)
def test_training_status_cannot_claim_training_safe_without_required_gates(
    overrides: dict[str, object],
) -> None:
    fields = {
        "in_tree": True,
        "source_pinned": True,
        "license_covered": True,
        "fallback_covered": True,
        "parity_covered": True,
        "hotspot_evidence": True,
        "vjp_covered": True,
        "jvp_covered": False,
        "training_safe": True,
        "differentiable": True,
        "reason": "test-only complete gate",
        "fallback_backend": "mlx",
    }
    fields.update(overrides)

    with pytest.raises(ValueError, match="training-safe Metal kernels require"):
        TrainingKernelStatus(**fields)  # type: ignore[arg-type]


def test_training_status_allows_forward_mode_jvp_gap_to_be_explicit() -> None:
    status = TrainingKernelStatus(
        in_tree=True,
        source_pinned=True,
        license_covered=True,
        fallback_covered=True,
        parity_covered=True,
        hotspot_evidence=True,
        vjp_covered=True,
        jvp_covered=False,
        training_safe=True,
        differentiable=True,
        reason="test-only VJP-backed training kernel without forward-mode use",
        fallback_backend="mlx",
    )

    assert status.training_safe is True
    assert status.jvp_covered is False


def test_training_status_rejects_production_claim_without_hotspot_evidence() -> None:
    with pytest.raises(ValueError, match="profiled hotspot evidence"):
        TrainingKernelStatus(
            in_tree=True,
            source_pinned=True,
            license_covered=True,
            fallback_covered=True,
            parity_covered=True,
            hotspot_evidence=False,
            vjp_covered=True,
            jvp_covered=True,
            training_safe=True,
            differentiable=True,
            reason="test-only missing hotspot",
            fallback_backend="mlx",
        )


def test_kernel_package_exports_training_gate() -> None:
    status = kernels.squared_relu_training_status()

    assert kernels.TrainingKernelStatus is TrainingKernelStatus
    assert isinstance(status, kernels.TrainingKernelStatus)
    assert status.fallback_backend == "mlx"
    assert not status.training_safe


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


@pytest.mark.parametrize("backend", ["auto", "mlx"])
def test_training_fallback_does_not_query_metal_status(
    monkeypatch: pytest.MonkeyPatch,
    backend: metal_ops.Backend,
) -> None:
    x = _sample()

    def fail_status(_: mx.array | None = None) -> metal_ops.MetalKernelStatus:
        raise AssertionError("training=True fallback must not query Metal eligibility")

    def fail_if_called(_: mx.array) -> mx.array:
        raise AssertionError("training=True fallback must not dispatch Metal")

    monkeypatch.setattr(metal_ops, "metal_kernel_status", fail_status)
    monkeypatch.setattr(metal_ops, "_squared_relu_metal", fail_if_called)

    out = squared_relu(x, backend=backend, training=True)

    _assert_reference(out, x)


def test_training_mlx_fallback_preserves_dtype() -> None:
    x = mx.array([-2.0, 0.5, 3.0], dtype=mx.float16)

    out = squared_relu(x, backend="mlx", training=True)

    assert out.dtype == mx.float16
    _assert_reference(out, x)


def test_training_metal_backend_rejects_forward_only_kernel() -> None:
    with pytest.raises(MetalKernelUnsupported, match="forward-only.*VJP/JVP"):
        squared_relu(_sample(), backend="metal", training=True)


def test_training_metal_backend_rejects_before_metal_status_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_status(_: mx.array | None = None) -> metal_ops.MetalKernelStatus:
        raise AssertionError("explicit training rejection must precede Metal eligibility")

    def fail_if_called(_: mx.array) -> mx.array:
        raise AssertionError("backend='metal' training must reject before kernel dispatch")

    monkeypatch.setattr(metal_ops, "metal_kernel_status", fail_status)
    monkeypatch.setattr(metal_ops, "_squared_relu_metal", fail_if_called)

    with pytest.raises(MetalKernelUnsupported, match="forward-only.*VJP/JVP"):
        squared_relu(_sample(), backend="metal", training=True)


def test_forward_only_kernel_has_no_custom_vjp_or_jvp() -> None:
    source = inspect.getsource(metal_ops)

    assert "@mx.custom_function" not in source
    assert not re.search(r"(?m)^\s*@.*\.(vjp|jvp)\b", source)


def test_training_metal_rejects_even_if_metal_status_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metal_ops,
        "metal_kernel_status",
        lambda _: metal_ops.MetalKernelStatus(True, "test-only eligible metal"),
    )

    def fail_if_called(_: mx.array) -> mx.array:
        raise AssertionError("backend='metal' training must reject before kernel dispatch")

    monkeypatch.setattr(metal_ops, "_squared_relu_metal", fail_if_called)

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


def test_auto_backend_falls_back_when_kernel_object_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _sample()
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: True)
    monkeypatch.setattr(metal_ops, "_metal_kernel_constructor", lambda: object())
    monkeypatch.setattr(metal_ops, "_squared_relu_kernel", None)

    out = squared_relu(x, backend="auto")

    _assert_reference(out, x)


def test_auto_backend_falls_back_when_constructor_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _sample()
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: True)
    monkeypatch.setattr(metal_ops, "_metal_kernel_constructor", lambda: None)

    out = squared_relu(x, backend="auto")

    _assert_reference(out, x)


def test_explicit_metal_fails_closed_when_constructor_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metal_ops, "can_run_metal", lambda: True)
    monkeypatch.setattr(metal_ops, "_metal_kernel_constructor", lambda: None)

    with pytest.raises(MetalKernelUnsupported, match="mx.fast.metal_kernel"):
        squared_relu(_sample(), backend="metal")


def test_metal_kernel_factory_keeps_contiguity_gate_explicit() -> None:
    source = inspect.getsource(metal_ops._make_squared_relu_kernel)

    assert "ensure_row_contiguous=True" in source


def test_metal_ops_module_has_no_cuda_runtime_branch() -> None:
    source = inspect.getsource(metal_ops).lower()

    assert "cuda" not in source


def test_metal_ops_module_has_no_remote_kernel_dependency_strings() -> None:
    source = inspect.getsource(metal_ops).lower()

    assert "huggingface" not in source
    assert "hf.co" not in source
    assert "kernels-community" not in source
    assert "http://" not in source
    assert "https://" not in source


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


def test_metal_squared_relu_non_contiguous_input_parity_if_available() -> None:
    base = mx.array(np.arange(-12, 12, dtype=np.float32).reshape(4, 6))
    x = base[:, ::2]
    status = metal_ops.metal_kernel_status(x)
    if not status.available:
        pytest.skip(status.reason)

    out = squared_relu(x, backend="metal")

    _assert_reference(out, x)
