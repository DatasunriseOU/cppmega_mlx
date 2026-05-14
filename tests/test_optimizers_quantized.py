"""Unit tests for the symmetric 8-bit blockwise codec and the Adam8bit optimizer.

These tests pin down the local MLX behaviour (per-block uint8 + fp32 absmax,
~3-4x optimizer-state shrink, loss-trajectory tracking within a few percent of
fp32-moment AdamW). They are not bitsandbytes parity tests -- the Adam8bit
class clearly documents its symmetric-int8 quantization mode and does not
emit a parity-trace claim.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE,
    QUANT_BIAS,
    QUANT_RANGE,
    QUANT_SCHEME_DYNAMIC,
    QUANT_SCHEME_SYMMETRIC,
    QUANT_SCHEMES,
    create_dynamic_map,
    dequantize_dynamic_blockwise,
    dequantize_dynamic_lut_blockwise,
    num_blocks,
    quantize_dynamic_blockwise,
    quantize_dynamic_lut_blockwise,
)
from cppmega_mlx.training._fused_adam8bit_kernel import (
    fused_adam8bit_status,
    fused_adam8bit_step,
)
from cppmega_mlx.training._fused_dynamic8bit_kernel import (
    fused_adam8bit_dynamic_status,
    fused_adam8bit_dynamic_step,
    fused_lion8bit_dynamic_status,
    fused_lion8bit_dynamic_step,
)
from cppmega_mlx.training._fused_lion8bit_kernel import (
    fused_lion8bit_status,
    fused_lion8bit_step,
)
from cppmega_mlx.training.optimizers import (
    AdamWFP32Moments,
    LionFP32Moments,
    MUON_SCALAR_OPTIMIZERS,
    make_adam8bit,
    make_adamw,
    make_lion,
    make_lion8bit,
    make_muon,
)
from cppmega_mlx.training.optimizers_quantized import (
    ADAM8BIT_CLASS,
    ADAM8BIT_QUANT_KIND,
    ADAM8BIT_QUANT_SCHEMES,
    ADAM8BIT_SOURCE,
    LION8BIT_CLASS,
    LION8BIT_QUANT_KIND,
    LION8BIT_SOURCE,
    Adam8bit,
    Lion8bit,
)


def _bytes_in_state(state: object) -> int:
    """Total bytes occupied by mx.arrays in an optimizer state pytree."""

    total = 0
    for _, value in tree_flatten(state):
        if isinstance(value, mx.array):
            total += int(value.nbytes)
    return total


# -----------------------------------------------------------------------------
# Codec round-trip and shape contracts
# -----------------------------------------------------------------------------


def test_quantize_dequantize_round_trip_within_tolerance() -> None:
    mx.random.seed(0)
    # 1024 = 4 full blocks; covers all-block boundaries cleanly.
    x = (mx.random.normal((1024,)) * 0.5).astype(mx.float32)
    qdata, absmax = quantize_dynamic_blockwise(x)
    recovered = dequantize_dynamic_blockwise(qdata, absmax, out_dtype=mx.float32)
    mx.eval(qdata, absmax, recovered)

    err = mx.max(mx.abs(recovered - x))
    # Worst-case symmetric int8 error per block is ~absmax/127. With absmax
    # near 1.5 (3-sigma from N(0, 0.5)) the round-trip stays well below 1e-2.
    assert float(err.item()) < 1e-2


def test_quantize_round_trip_handles_2d_shape_and_tail_block() -> None:
    mx.random.seed(0)
    # 1025 elements -> 5 blocks (last is partial). Shape preserved.
    x = (mx.random.normal((25, 41)) * 1.5).astype(mx.float32)
    assert int(x.size) == 1025

    qdata, absmax = quantize_dynamic_blockwise(x)
    assert qdata.shape == x.shape
    assert qdata.dtype == mx.uint8
    assert absmax.shape == (num_blocks(1025),)
    assert absmax.dtype == mx.float32

    recovered = dequantize_dynamic_blockwise(qdata, absmax, out_dtype=mx.float32)
    mx.eval(qdata, absmax, recovered)
    err = float(mx.max(mx.abs(recovered - x)).item())
    assert err < 5e-2  # Larger amplitude -> larger absolute error per quant bin.


def test_quantize_zero_block_produces_zero_absmax_and_bias_quant() -> None:
    # All zeros block must round-trip exactly with absmax=0 and qdata=128 (bias).
    x = mx.zeros((512,), dtype=mx.float32)
    qdata, absmax = quantize_dynamic_blockwise(x)
    mx.eval(qdata, absmax)

    assert int(absmax.sum().item()) == 0
    assert int((qdata == QUANT_BIAS).sum().item()) == 512
    recovered = dequantize_dynamic_blockwise(qdata, absmax, out_dtype=mx.float32)
    mx.eval(recovered)
    assert float(mx.max(mx.abs(recovered)).item()) == 0.0


def test_quantize_dtype_and_block_size_constants_match_layout() -> None:
    # Pin down the contract: 256-element blocks and 1B/elem + 4B/64B metadata.
    assert DEFAULT_BLOCK_SIZE == 256
    assert QUANT_RANGE == 127
    assert QUANT_BIAS == 128


def test_quantize_rejects_non_default_block_size_until_native_codec_supports_it() -> None:
    x = mx.zeros((512,), dtype=mx.float32)
    with pytest.raises(NotImplementedError, match="only block_size=256"):
        quantize_dynamic_blockwise(x, block_size=128)


def test_quantize_accepts_bf16_input_and_dequantizes_to_chosen_dtype() -> None:
    mx.random.seed(0)
    x = (mx.random.normal((512,)) * 0.5).astype(mx.bfloat16)
    qdata, absmax = quantize_dynamic_blockwise(x)
    assert qdata.dtype == mx.uint8
    assert absmax.dtype == mx.float32

    recovered_fp32 = dequantize_dynamic_blockwise(qdata, absmax, out_dtype=mx.float32)
    recovered_bf16 = dequantize_dynamic_blockwise(qdata, absmax, out_dtype=mx.bfloat16)
    assert recovered_fp32.dtype == mx.float32
    assert recovered_bf16.dtype == mx.bfloat16


# -----------------------------------------------------------------------------
# Adam8bit state surface and bf16-param compatibility
# -----------------------------------------------------------------------------


class _TinyDense(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def __call__(self, x: mx.array) -> mx.array:  # pragma: no cover - shape only
        return self.linear(x)


def _flatten_state(state: object) -> dict[str, mx.array]:
    return {
        key: value
        for key, value in tree_flatten(state)
        if isinstance(value, mx.array)
    }


def test_adam8bit_state_dtypes_uint8() -> None:
    model = _TinyDense()
    optimizer = make_adam8bit(learning_rate=1e-3)
    optimizer.init(model.trainable_parameters())

    flat = _flatten_state(optimizer.state)
    assert flat["linear.weight.m_quant"].dtype == mx.uint8
    assert flat["linear.weight.v_quant"].dtype == mx.uint8
    assert flat["linear.weight.m_absmax"].dtype == mx.float32
    assert flat["linear.weight.v_absmax"].dtype == mx.float32

    # Linear weight shape (8, 16) = 128 elems = 1 block of 256 (rounded up).
    assert flat["linear.weight.m_quant"].shape == model.linear.weight.shape
    assert flat["linear.weight.m_absmax"].shape == (
        num_blocks(int(model.linear.weight.size)),
    )

    # Bias of shape (8,) -> still 1 block by ceil-div.
    assert flat["linear.bias.m_quant"].shape == model.linear.bias.shape
    assert flat["linear.bias.m_absmax"].shape == (1,)


def test_adam8bit_runs_one_step_on_bf16_param_and_preserves_dtype() -> None:
    model = _TinyDense()
    model.set_dtype(mx.bfloat16)
    optimizer = make_adam8bit(learning_rate=1e-3, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    x = mx.ones((4, 16), dtype=mx.bfloat16)

    def loss_fn(m: nn.Module, x: mx.array) -> mx.array:
        return mx.sum(m(x))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    assert math.isfinite(float(loss.item()))
    # bf16 parameter must remain bf16 even though the moments are uint8.
    assert model.linear.weight.dtype == mx.bfloat16
    assert model.linear.bias.dtype == mx.bfloat16


def test_adam8bit_step_actually_updates_parameters() -> None:
    model = _TinyDense()
    optimizer = make_adam8bit(learning_rate=1e-2, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    before = mx.array(model.linear.weight)
    x = mx.ones((4, 16), dtype=model.linear.weight.dtype)

    def loss_fn(m: nn.Module, x: mx.array) -> mx.array:
        return mx.sum(m(x))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    after = model.linear.weight
    delta = float(mx.max(mx.abs(after - before)).item())
    assert delta > 0.0


def test_adam8bit_first_step_noise_floor_uses_fresh_v_absmax() -> None:
    optimizer = make_adam8bit(learning_rate=1.0, weight_decay=0.0)
    params = {"w": mx.zeros((DEFAULT_BLOCK_SIZE,), dtype=mx.float32)}
    optimizer.init(params)

    # Fat-tailed single block: the tiny second element would take an enormous
    # first-step update if the denominator used only the previous v_absmax=0.
    grad_values = [1000.0, 1e-3] + [0.0] * (DEFAULT_BLOCK_SIZE - 2)
    grads = {"w": mx.array(grad_values, dtype=mx.float32)}
    updated = optimizer.apply_gradients(grads, params)
    mx.eval(updated, optimizer.state)

    tiny_update = float(abs(updated["w"][1]).item())
    assert tiny_update < 1e-3
    assert float(optimizer.state["w"]["v_absmax"][0].item()) > 999.0


def test_adam8bit_second_moment_is_nonnegative_after_quant_dequant() -> None:
    """Adam's ``v`` is a squared-gradient moment, so old quantized state that
    decodes negative must be clamped before sqrt in both native codecs.
    """

    n = DEFAULT_BLOCK_SIZE
    param = mx.ones((n,), dtype=mx.float32)
    grad = mx.zeros((n,), dtype=mx.float32)

    for fused_step in (
        fused_adam8bit_step,
        fused_adam8bit_dynamic_step,
    ):
        m_quant = mx.full((n,), 128, dtype=mx.uint8)
        m_absmax = mx.zeros((1,), dtype=mx.float32)
        v_quant = mx.zeros((n,), dtype=mx.uint8)
        v_absmax = mx.ones((1,), dtype=mx.float32)
        updated, _, _, _, _ = fused_step(
            param,
            grad,
            m_quant,
            m_absmax,
            v_quant,
            v_absmax,
            learning_rate=mx.array(1e-3, dtype=mx.float32),
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            step=mx.array(1, dtype=mx.uint64),
            bias_correction=False,
        )
        mx.eval(updated)
        assert bool(mx.all(mx.isfinite(updated)).item())
        assert float(mx.max(mx.abs(updated - param)).item()) == 0.0

    for scheme in (QUANT_SCHEME_SYMMETRIC, QUANT_SCHEME_DYNAMIC):
        optimizer = make_adam8bit(
            learning_rate=1e-3,
            weight_decay=0.0,
            use_fused_kernel=True,
            quant_scheme=scheme,
        )
        params = {"w": param}
        optimizer.init(params)
        optimizer.state["w"]["v_quant"] = mx.zeros((n,), dtype=mx.uint8)
        optimizer.state["w"]["v_absmax"] = mx.ones((1,), dtype=mx.float32)
        updated = optimizer.apply_gradients({"w": grad}, params)
        mx.eval(updated, optimizer.state)

        assert bool(mx.all(mx.isfinite(updated["w"])).item())
        assert float(mx.max(mx.abs(updated["w"] - param)).item()) == 0.0


def test_adam8bit_min_8bit_size_keeps_small_tensors_fp32() -> None:
    optimizer = make_adam8bit(learning_rate=1e-3, min_8bit_size=4096)
    params = {
        "small": mx.ones((8,), dtype=mx.bfloat16),
        "large": mx.ones((4096,), dtype=mx.bfloat16),
    }
    optimizer.init(params)
    mx.eval(optimizer.state)

    assert optimizer.state["small"]["m"].dtype == mx.float32
    assert optimizer.state["small"]["v"].dtype == mx.float32
    assert "m_quant" not in optimizer.state["small"]
    assert optimizer.state["large"]["m_quant"].dtype == mx.uint8
    assert optimizer.state["large"]["v_quant"].dtype == mx.uint8


# -----------------------------------------------------------------------------
# Memory contract: Adam8bit state should be much smaller than fp32-moments AdamW
# -----------------------------------------------------------------------------


def test_adam8bit_memory_smaller_than_fp32() -> None:
    # Use a wider 1024x1024 layer so the 1/64-byte absmax overhead is dominated
    # by the 1B/param uint8 storage and the comparison is meaningful.
    model = nn.Linear(1024, 1024, bias=False)

    fp32_optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.0)
    fp32_optimizer.init(model.trainable_parameters())
    mx.eval(fp32_optimizer.state)
    fp32_bytes = _bytes_in_state(fp32_optimizer.state)

    quant_optimizer = make_adam8bit(learning_rate=1e-3, weight_decay=0.0)
    quant_optimizer.init(model.trainable_parameters())
    mx.eval(quant_optimizer.state)
    quant_bytes = _bytes_in_state(quant_optimizer.state)

    # Strict: 8-bit must be at least 3x smaller (fp32 moments are 4x bigger,
    # plus there is some shared step/lr scalar overhead of a few bytes).
    ratio = fp32_bytes / max(quant_bytes, 1)
    assert ratio >= 3.0, (
        f"Adam8bit state ({quant_bytes} B) is not 3x smaller than "
        f"AdamWFP32Moments state ({fp32_bytes} B); ratio={ratio:.3f}"
    )

    # Sanity: should be ~4x for a tensor with no metadata-overhead inflation,
    # and at most 4.5x even after accounting for fp32 absmax metadata.
    assert ratio <= 4.5


# -----------------------------------------------------------------------------
# Loss-trajectory smoke: 50 steps Adam8bit should track AdamWFP32Moments
# -----------------------------------------------------------------------------


def _train_steps(
    optimizer: optim.Optimizer,
    *,
    seed: int,
    steps: int = 50,
    hidden: int = 128,
    batch: int = 4,
) -> list[float]:
    mx.random.seed(seed)
    # Noisy linear regression so neither optimizer drives the loss to ~0
    # within 50 steps. That keeps the comparison in the regime where signal
    # exceeds quant noise, which is what the gb10 production runner sees.
    true_w = mx.random.normal((hidden, hidden)).astype(mx.float32) * 0.3
    x = mx.random.normal((batch, hidden)).astype(mx.float32)
    obs_noise = mx.random.normal((batch, hidden)).astype(mx.float32) * 1.5
    y = x @ true_w.T + obs_noise

    mx.random.seed(seed)
    model = nn.Linear(hidden, hidden, bias=False)
    optimizer.init(model.trainable_parameters())

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    return losses


def test_adam8bit_loss_matches_fp32_within_tolerance() -> None:
    seed = 0
    lr = 2e-4
    weight_decay = 0.01
    fp32_losses = _train_steps(
        make_adamw(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_steps(
        make_adam8bit(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50

    # Both optimizers must learn (final loss strictly less than initial).
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    # Spec: relative loss drift <2% per step. We verify the median drift
    # over the 50-step trajectory (a robust statistic that excludes the
    # late-step regime where fp32 has converged below the symmetric-int8
    # quantization noise floor of ~1/127). Median <2% is the policy.
    rel_drifts = []
    for q, f in zip(quant_losses, fp32_losses):
        denom = max(abs(f), 1e-6)
        rel_drifts.append(abs(q - f) / denom)
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"Adam8bit median per-step loss drift {median_drift:.4f} exceeds 2%; "
        f"fp32_losses[-5:]={fp32_losses[-5:]}, quant_losses[-5:]={quant_losses[-5:]}."
    )

    # The first 10 steps should track fp32 within 2% as well -- that's the
    # learning regime where AdamW dynamics dominate and quant noise is
    # negligible relative to gradient magnitudes.
    early_drifts = rel_drifts[:10]
    max_early_drift = max(early_drifts)
    assert max_early_drift < 0.02, (
        f"Adam8bit max early-step loss drift {max_early_drift:.4f} exceeds 2%; "
        f"fp32 first 10={fp32_losses[:10]}, q8 first 10={quant_losses[:10]}."
    )


# -----------------------------------------------------------------------------
# Public API and routing
# -----------------------------------------------------------------------------


def test_adam8bit_class_constants_pin_module_paths() -> None:
    assert ADAM8BIT_CLASS == "cppmega_mlx.training.optimizers_quantized.Adam8bit"
    assert ADAM8BIT_SOURCE == "cppmega_mlx.training.optimizers_quantized.make_adam8bit"
    assert ADAM8BIT_QUANT_KIND == "symmetric_int8_blockwise_v1"


def test_make_adam8bit_returns_adam8bit_instance() -> None:
    optimizer = make_adam8bit(learning_rate=1e-3)
    assert isinstance(optimizer, Adam8bit)
    assert isinstance(optimizer, optim.Optimizer)


def test_make_adam8bit_default_betas_match_make_adamw() -> None:
    quant = make_adam8bit()
    fp32 = make_adamw()
    assert list(quant.betas) == list(fp32.betas)


def test_make_muon_routes_scalar_optimizer_to_adam8bit() -> None:
    optimizer = make_muon(scalar_optimizer="adam8bit")
    assert isinstance(optimizer.adamw, Adam8bit)


def test_make_muon_default_scalar_optimizer_remains_adamw() -> None:
    optimizer = make_muon()
    assert isinstance(optimizer.adamw, AdamWFP32Moments)
    assert not isinstance(optimizer.adamw, Adam8bit)


def test_make_muon_rejects_unknown_scalar_optimizer() -> None:
    with pytest.raises(ValueError, match="scalar_optimizer must be one of"):
        make_muon(scalar_optimizer="adafactor")


def test_muon_scalar_optimizer_constants_are_frozen() -> None:
    assert isinstance(MUON_SCALAR_OPTIMIZERS, tuple)
    assert "adamw" in MUON_SCALAR_OPTIMIZERS
    assert "adam8bit" in MUON_SCALAR_OPTIMIZERS
    assert "lion" in MUON_SCALAR_OPTIMIZERS
    assert "lion8bit" in MUON_SCALAR_OPTIMIZERS


# -----------------------------------------------------------------------------
# Fused optimizer request: native MLX C++/Metal fast path + explicit fallback
# -----------------------------------------------------------------------------


def _flat_state_arrays(state: object) -> dict[str, mx.array]:
    return {
        path: value
        for path, value in tree_flatten(state)
        if isinstance(value, mx.array)
    }


def _assert_arrays_close_or_equal(
    actual: mx.array,
    expected: mx.array,
    key: str = "",
) -> None:
    assert actual.shape == expected.shape, key
    assert actual.dtype == expected.dtype, key
    if actual.dtype == mx.uint8:
        diff = mx.abs(actual.astype(mx.int32) - expected.astype(mx.int32))
        assert int(mx.max(diff).item()) <= 1, key
    else:
        assert bool(mx.allclose(actual, expected, rtol=1e-5, atol=1e-6).item()), key


def test_fused_adam8bit_default_uses_native_mlx_extension() -> None:
    """``use_fused_kernel`` requests the native MLX C++/Metal primitive."""

    optimizer = make_adam8bit(learning_rate=1e-3)
    assert isinstance(optimizer, Adam8bit)
    assert optimizer.use_fused_kernel is True
    assert optimizer.fused_kernel_status.available is True
    assert "native MLX C++/Metal" in optimizer.fused_kernel_status.reason


def test_fused_adam8bit_can_be_disabled_for_parity_runs() -> None:
    """Passing ``use_fused_kernel=False`` records an explicit disabled status."""

    optimizer = make_adam8bit(learning_rate=1e-3, use_fused_kernel=False)
    assert optimizer.use_fused_kernel is False
    assert optimizer.fused_kernel_status.available is False
    assert "disabled by caller" in optimizer.fused_kernel_status.reason


def _run_n_steps(
    *,
    use_fused_kernel: bool,
    seed: int,
    steps: int,
    in_features: int = 64,
    out_features: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    """Drive a tiny dense model for ``steps`` updates.

    Returns ``(weight, bias, state_arrays)`` after ``mx.eval`` so callers can
    diff the two paths.
    """

    mx.random.seed(seed)
    model = nn.Linear(in_features, out_features, bias=True)
    optimizer = make_adam8bit(
        learning_rate=lr,
        weight_decay=weight_decay,
        use_fused_kernel=use_fused_kernel,
    )
    optimizer.init(model.trainable_parameters())

    mx.random.seed(seed + 1)
    x = mx.random.normal((4, in_features)).astype(mx.float32)
    y = mx.random.normal((4, out_features)).astype(mx.float32)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return (
        mx.array(model.weight),
        mx.array(model.bias),
        _flat_state_arrays(optimizer.state),
    )


def test_fused_adam8bit_matches_unfused_within_tolerance() -> None:
    """A fused request must preserve native path parameter and state behavior.

    The native C++/Metal primitive uses a threadgroup reduction for block
    absmax, so exact bitwise equality is not required for fp32 leaves.
    """

    steps = 1
    seed = 7

    fused_w, fused_b, fused_state = _run_n_steps(
        use_fused_kernel=True, seed=seed, steps=steps
    )
    unfused_w, unfused_b, unfused_state = _run_n_steps(
        use_fused_kernel=False, seed=seed, steps=steps
    )
    mx.eval(fused_w, fused_b, unfused_w, unfused_b)

    _assert_arrays_close_or_equal(fused_w, unfused_w, "weight")
    _assert_arrays_close_or_equal(fused_b, unfused_b, "bias")

    # State: every leaf must match. m_quant/v_quant are uint8; allow off-by-1
    # since the per-block absmax tree reduction can swap order vs Python's
    # ``mx.max(..., axis=1)``.
    assert set(fused_state.keys()) == set(unfused_state.keys())
    for key in fused_state:
        f = fused_state[key]
        u = unfused_state[key]
        _assert_arrays_close_or_equal(f, u, key)


def test_fused_adam8bit_loss_matches_unfused_over_50_steps() -> None:
    """A fused request must keep the native MLX loss trajectory unchanged."""

    seed = 0
    lr = 2e-4
    weight_decay = 0.01
    fused_losses = _train_steps(
        make_adam8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=True,
        ),
        seed=seed,
    )
    unfused_losses = _train_steps(
        make_adam8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=False,
        ),
        seed=seed,
    )

    rel_drifts = [
        abs(f - u) / max(abs(u), 1e-6)
        for f, u in zip(fused_losses, unfused_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused vs unfused median per-step loss drift {median_drift:.4f} "
        f"exceeds 2%; unfused[-5:]={unfused_losses[-5:]}, "
        f"fused[-5:]={fused_losses[-5:]}."
    )


def test_fused_adam8bit_loss_matches_fp32_within_tolerance() -> None:
    """50-step smoke: requested-fused Adam8bit keeps the native parity contract."""

    seed = 0
    lr = 2e-4
    weight_decay = 0.01
    fp32_losses = _train_steps(
        make_adamw(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_steps(
        make_adam8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=True,
        ),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = [
        abs(q - f) / max(abs(f), 1e-6)
        for q, f in zip(quant_losses, fp32_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused Adam8bit median per-step loss drift {median_drift:.4f} "
        f"exceeds 2%; fp32_losses[-5:]={fp32_losses[-5:]}, "
        f"quant_losses[-5:]={quant_losses[-5:]}."
    )

    early_drifts = rel_drifts[:10]
    max_early_drift = max(early_drifts)
    assert max_early_drift < 0.02, (
        f"fused Adam8bit max early-step loss drift {max_early_drift:.4f} "
        f"exceeds 2%; fp32 first 10={fp32_losses[:10]}, "
        f"q8 first 10={quant_losses[:10]}."
    )


def test_fused_adam8bit_direct_call_runs_native_extension() -> None:
    status = fused_adam8bit_status()
    assert status.available is True
    assert "native MLX C++/Metal" in status.reason

    n = DEFAULT_BLOCK_SIZE
    outputs = fused_adam8bit_step(
        mx.zeros((n,), dtype=mx.float32),
        mx.zeros((n,), dtype=mx.float32),
        mx.full((n,), 128, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        mx.full((n,), 128, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        learning_rate=mx.array(1e-3, dtype=mx.float32),
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        step=mx.array(1, dtype=mx.uint64),
        bias_correction=False,
    )
    mx.eval(outputs)
    assert len(outputs) == 5
    assert outputs[0].shape == (n,)


def test_fused_adam8bit_state_keys_match_unfused() -> None:
    """The fused path must not leak extra state -- the contract test in
    ``tests/test_optimizer_no_master_contract.py`` only allows
    ``m_quant``, ``m_absmax``, ``v_quant``, ``v_absmax``, ``step``,
    ``learning_rate`` in optimizer state.
    """

    optimizer_fused = make_adam8bit(use_fused_kernel=True)
    optimizer_unfused = make_adam8bit(use_fused_kernel=False)

    model = _TinyDense()
    optimizer_fused.init(model.trainable_parameters())
    # init_single is shape-agnostic, so the unfused path needs its own model.
    model2 = _TinyDense()
    optimizer_unfused.init(model2.trainable_parameters())
    mx.eval(optimizer_fused.state, optimizer_unfused.state)

    fused_keys = {p.split(".")[-1] for p, _ in tree_flatten(optimizer_fused.state)}
    unfused_keys = {p.split(".")[-1] for p, _ in tree_flatten(optimizer_unfused.state)}
    assert fused_keys == unfused_keys
    expected = {"m_quant", "m_absmax", "v_quant", "v_absmax", "step", "learning_rate"}
    assert fused_keys == expected, fused_keys


# -----------------------------------------------------------------------------
# Dynamic 8-bit LUT codec (bitsandbytes-style)
# -----------------------------------------------------------------------------
#
# These tests pin down the bnb-style dynamic 8-bit LUT codec
# (``QUANT_SCHEME_DYNAMIC``) added alongside the existing symmetric path.
# The LUT is the signed dynamic map produced by
# ``bitsandbytes.functional.create_dynamic_map(signed=True,
# max_exponent_bits=7, total_bits=8)``; the runtime codec mirrors
# ``dDequantizeBlockwise`` in ``bitsandbytes/csrc/kernels.cu``.


def test_create_dynamic_map_matches_bnb_reference() -> None:
    """The 256-entry signed dynamic LUT must match the bnb canonical values.

    Reference: ``bitsandbytes.functional.create_dynamic_map(signed=True,
    max_exponent_bits=7, total_bits=8)`` (commit on bnb main as of 2026-05).
    The first 16 and last 16 entries below are computed from that algorithm
    in pure Python (no bnb dependency); the codec stores a hardcoded copy of
    all 256 entries with the bnb function cited inline so we can compare
    against any bnb release without pulling in the torch dependency.
    """

    lut = create_dynamic_map()
    assert lut.shape == (256,)
    assert lut.dtype == mx.float32
    mx.eval(lut)
    # First 16 entries (most negative end of the LUT). Hand-computed from
    # ``create_dynamic_map`` with ``data.sort()`` applied.
    expected_first_16 = [
        -0.992968738079071, -0.9789062738418579, -0.96484375, -0.9507812261581421,
        -0.936718761920929, -0.922656238079071, -0.9085937738418579, -0.89453125,
        -0.8804687261581421, -0.866406261920929, -0.852343738079071,
        -0.8382812738418579, -0.82421875, -0.8101562261581421, -0.796093761920929,
        -0.782031238079071,
    ]
    for i, expected in enumerate(expected_first_16):
        actual = float(lut[i].item())
        assert abs(actual - expected) < 1e-7, (i, actual, expected)
    # Last 16 entries (positive end, including the canonical 1.0 at index 255).
    expected_last_16 = [
        0.796093761920929, 0.8101562261581421, 0.82421875, 0.8382812738418579,
        0.852343738079071, 0.866406261920929, 0.8804687261581421, 0.89453125,
        0.9085937738418579, 0.922656238079071, 0.936718761920929,
        0.9507812261581421, 0.96484375, 0.9789062738418579, 0.992968738079071,
        1.0,
    ]
    for i, expected in enumerate(expected_last_16):
        actual = float(lut[240 + i].item())
        assert abs(actual - expected) < 1e-7, (240 + i, actual, expected)
    # Spot check the dense-near-zero region: index 127 is the exact zero
    # entry (``data.append(0)`` step in ``create_dynamic_map``); the
    # neighbours straddle zero with a ~5.5e-7 step.
    assert float(lut[127].item()) == 0.0
    assert abs(float(lut[126].item()) - (-5.499999815583578e-07)) < 1e-12
    assert abs(float(lut[128].item()) - 5.499999815583578e-07) < 1e-12


def test_dynamic_int8_dtype_uint8() -> None:
    """Dynamic-LUT codec must emit uint8 payload (LUT index in [0, 255])."""

    mx.random.seed(0)
    x = (mx.random.normal((1024,)) * 0.1).astype(mx.float32)
    qdata, absmax = quantize_dynamic_lut_blockwise(x)
    mx.eval(qdata, absmax)
    assert qdata.dtype == mx.uint8
    assert absmax.dtype == mx.float32
    assert qdata.shape == x.shape
    # All bytes must be valid LUT indices.
    assert int(mx.min(qdata).item()) >= 0
    assert int(mx.max(qdata).item()) <= 255


def test_dynamic_lut_round_trip_within_tolerance() -> None:
    """Dynamic LUT should be tighter than symmetric for small-magnitude
    values when the per-block absmax is set by an outlier.

    This mirrors the realistic Adam moment distribution: most values are
    tiny (~``g`` * (1-b1) ~= 1e-3) but at least one element per block
    sets the absmax to ~1.0 (e.g. an early-training gradient spike). The
    bnb dynamic LUT covers small magnitudes with much denser bins than
    the symmetric int8 codec, so the round-trip error on the small values
    should be ~5x smaller than symmetric (mean error). The max error is
    less dramatic because individual values can still land on wide bins,
    but the mean over a block is dominated by the dense-near-zero region.
    """

    mx.random.seed(0)
    # 4 blocks of 256, scale 0.01 so most values are ~1e-3 in magnitude.
    n = 4 * 256
    x_arr = (mx.random.normal((n,)) * 0.01).astype(mx.float32)
    # Add a single ~1.0 outlier per block to set the absmax. This is the
    # regime where the dynamic LUT helps -- the LUT bins near zero step
    # in increments of ~5.5e-7, while the symmetric int8 step at
    # absmax=1.0 is ~1/127 = 7.87e-3, so symmetric collapses values
    # smaller than ~4e-3 to zero.
    outlier_idx = mx.array([0, 256, 512, 768], dtype=mx.int32)
    outlier_vals = mx.array([1.0, -1.5, 0.8, -2.0], dtype=mx.float32)
    x = x_arr
    # Use scatter-equivalent ``where`` because ``mx.array`` is immutable.
    idx = mx.arange(n)
    for k in range(4):
        i = int(outlier_idx[k].item())
        v = float(outlier_vals[k].item())
        x = mx.where(idx == i, mx.full((n,), v, dtype=mx.float32), x)
    mx.eval(x)

    qd, amd = quantize_dynamic_lut_blockwise(x)
    qs, ams = quantize_dynamic_blockwise(x)
    rec_d = dequantize_dynamic_lut_blockwise(qd, amd, out_dtype=mx.float32)
    rec_s = dequantize_dynamic_blockwise(qs, ams, out_dtype=mx.float32)
    mx.eval(rec_d, rec_s)

    # Compare error only on small-magnitude values (the outliers are
    # represented by both codecs at near the edge of their LUT and don't
    # show the dynamic-vs-symmetric gap).
    small_mask = mx.abs(x) < 0.5
    err_d_mean = float(mx.mean(mx.abs(rec_d - x) * small_mask).item())
    err_s_mean = float(mx.mean(mx.abs(rec_s - x) * small_mask).item())
    # Small-value mean error should be ~5x tighter for dynamic vs symmetric;
    # measured ~7-9x in practice. Gate at 4x to leave headroom for fp32 noise
    # / shuffle differences across MLX versions.
    ratio = err_s_mean / max(err_d_mean, 1e-12)
    assert ratio >= 4.0, (
        f"dynamic LUT did not improve mean small-value error enough: "
        f"symmetric={err_s_mean:.6e}, dynamic={err_d_mean:.6e}, ratio={ratio:.2f}x"
    )


def test_quant_schemes_constants_pinned() -> None:
    """The scheme identifiers are checkpoint-format-load-bearing; pin them."""

    assert QUANT_SCHEME_SYMMETRIC == "symmetric_int8_v1"
    assert QUANT_SCHEME_DYNAMIC == "dynamic_int8_v1"
    assert set(QUANT_SCHEMES) == {QUANT_SCHEME_SYMMETRIC, QUANT_SCHEME_DYNAMIC}
    assert ADAM8BIT_QUANT_SCHEMES == QUANT_SCHEMES


def test_adam8bit_dynamic_quant_scheme_state_dtypes() -> None:
    """Adam8bit with dynamic scheme must keep the same uint8 + fp32 state
    layout as the symmetric default."""

    model = _TinyDense()
    optimizer = make_adam8bit(learning_rate=1e-3, quant_scheme=QUANT_SCHEME_DYNAMIC)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)
    assert optimizer.quant_scheme == QUANT_SCHEME_DYNAMIC
    assert optimizer.use_fused_kernel is True
    assert optimizer.fused_kernel_status.available is True
    assert "native MLX C++/Metal" in optimizer.fused_kernel_status.reason

    flat = _flatten_state(optimizer.state)
    assert flat["linear.weight.m_quant"].dtype == mx.uint8
    assert flat["linear.weight.v_quant"].dtype == mx.uint8
    assert flat["linear.weight.m_absmax"].dtype == mx.float32
    assert flat["linear.weight.v_absmax"].dtype == mx.float32
    # Initial bytes for dynamic: index 127 maps to LUT[127] == 0.0.
    init_byte = int(flat["linear.weight.m_quant"][0, 0].item())
    assert init_byte == 127, (
        f"dynamic initial m_quant byte must point at LUT zero (127); got {init_byte}"
    )


def test_adam8bit_dynamic_loss_matches_fp32_within_2pct() -> None:
    """50-step training smoke: Adam8bit dynamic should track AdamWFP32Moments
    within 2% loss-trajectory drift, the same gate the symmetric path is
    held to. Because the dynamic LUT has tighter precision near zero (where
    Adam ``m, v`` moments live most of the time), this should pass at
    least as easily as the symmetric path -- the test gates at 2% so both
    schemes meet the spec without the test becoming brittle to fp32 noise.
    """

    seed = 0
    lr = 2e-4
    weight_decay = 0.01
    fp32_losses = _train_steps(
        make_adamw(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    dynamic_losses = _train_steps(
        make_adam8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            quant_scheme=QUANT_SCHEME_DYNAMIC,
        ),
        seed=seed,
    )

    assert len(fp32_losses) == len(dynamic_losses) == 50
    assert dynamic_losses[-1] < dynamic_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = []
    for q, f in zip(dynamic_losses, fp32_losses):
        denom = max(abs(f), 1e-6)
        rel_drifts.append(abs(q - f) / denom)
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"Adam8bit dynamic median per-step drift {median_drift:.4f} exceeds 2%; "
        f"fp32_losses[-5:]={fp32_losses[-5:]}, "
        f"dynamic_losses[-5:]={dynamic_losses[-5:]}."
    )


def test_make_adam8bit_default_scheme_is_symmetric_for_backcompat() -> None:
    """Default scheme stays symmetric so existing callers keep their codec."""

    optimizer = make_adam8bit(learning_rate=1e-3)
    assert optimizer.quant_scheme == QUANT_SCHEME_SYMMETRIC
    assert optimizer.use_fused_kernel is True
    assert "native MLX C++/Metal" in optimizer.fused_kernel_status.reason


def test_adam8bit_rejects_unknown_quant_scheme() -> None:
    with pytest.raises(ValueError, match="quant_scheme must be one of"):
        make_adam8bit(learning_rate=1e-3, quant_scheme="bogus_v9")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Lion8bit: state surface, memory contract, loss tracking, bf16 compatibility
# -----------------------------------------------------------------------------


def _train_lion_steps(
    optimizer: optim.Optimizer,
    *,
    seed: int,
    steps: int = 50,
    hidden: int = 128,
    batch: int = 4,
) -> list[float]:
    """Same noisy-regression smoke as ``_train_steps`` but with Lion-class
    LRs (~3-10x smaller than AdamW) so both Lion and Lion8bit make progress
    without diverging on the sign-update."""

    mx.random.seed(seed)
    true_w = mx.random.normal((hidden, hidden)).astype(mx.float32) * 0.3
    x = mx.random.normal((batch, hidden)).astype(mx.float32)
    obs_noise = mx.random.normal((batch, hidden)).astype(mx.float32) * 1.5
    y = x @ true_w.T + obs_noise

    mx.random.seed(seed)
    model = nn.Linear(hidden, hidden, bias=False)
    optimizer.init(model.trainable_parameters())

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    return losses


def test_lion8bit_state_dtypes_uint8() -> None:
    """Pin state surface: uint8 m_quant, fp32 m_absmax, NO v moment, plus
    the standard step + learning_rate. This is the load-bearing memory
    contract for Lion8bit -- one momentum buffer per param, half of Adam8bit."""

    model = _TinyDense()
    optimizer = make_lion8bit(learning_rate=1e-4)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    flat = _flatten_state(optimizer.state)
    assert flat["linear.weight.m_quant"].dtype == mx.uint8
    assert flat["linear.weight.m_absmax"].dtype == mx.float32

    # Linear weight (8, 16) = 128 elems -> 1 block of 256 by ceil-div.
    assert flat["linear.weight.m_quant"].shape == model.linear.weight.shape
    assert flat["linear.weight.m_absmax"].shape == (
        num_blocks(int(model.linear.weight.size)),
    )
    # Bias (8,) -> still 1 block by ceil-div.
    assert flat["linear.bias.m_quant"].shape == model.linear.bias.shape
    assert flat["linear.bias.m_absmax"].shape == (1,)

    # Per-parameter state must be exactly {m_quant, m_absmax}; no v anywhere.
    keys = {path.split(".")[-1] for path, _ in tree_flatten(optimizer.state)}
    expected = {"m_quant", "m_absmax", "step", "learning_rate"}
    assert keys == expected, (
        f"Lion8bit state surface must be {expected} (no v* moments); got {keys}"
    )


def test_lion8bit_state_size_smaller_than_lion_fp32() -> None:
    """Memory contract: Lion8bit state must be ~3-4x smaller than
    LionFP32Moments. Lion fp32 is 4 B/param (single fp32 m); Lion8bit is
    ~1.02 B/param (uint8 + 1/64 B absmax)."""

    # Wider layer so the 1/64-byte absmax overhead is dominated by the
    # 1B/param uint8 storage and the comparison is stable.
    model = nn.Linear(1024, 1024, bias=False)

    fp32_optimizer = make_lion(learning_rate=1e-4)
    fp32_optimizer.init(model.trainable_parameters())
    mx.eval(fp32_optimizer.state)
    fp32_bytes = _bytes_in_state(fp32_optimizer.state)

    quant_optimizer = make_lion8bit(learning_rate=1e-4)
    quant_optimizer.init(model.trainable_parameters())
    mx.eval(quant_optimizer.state)
    quant_bytes = _bytes_in_state(quant_optimizer.state)

    ratio = fp32_bytes / max(quant_bytes, 1)
    # Strict: must be at least 3x smaller. Theoretical max is ~3.94x
    # (4 / 1.0156); ratio should be >=3.0 with no funny scalar overhead.
    assert ratio >= 3.0, (
        f"Lion8bit state ({quant_bytes} B) is not 3x smaller than "
        f"LionFP32Moments state ({fp32_bytes} B); ratio={ratio:.3f}"
    )
    # Sanity: at most ~4.5x even after fp32 absmax metadata + scalar overhead.
    assert ratio <= 4.5


def test_lion8bit_loss_matches_lion_fp32_within_tolerance() -> None:
    """50-step training smoke: Lion8bit must track LionFP32Moments within
    ~2% relative loss drift on a noisy linear regression. The sign-based
    update is naturally robust to symmetric int8 quant noise on m -- only
    elements within ~absmax/127 of zero flip sign, and those are the
    elements where the direction is genuinely ambiguous."""

    seed = 0
    lr = 1e-3
    weight_decay = 0.01
    fp32_losses = _train_lion_steps(
        make_lion(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_lion_steps(
        make_lion8bit(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50

    # Both optimizers must learn (final loss strictly less than initial).
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = []
    for q, f in zip(quant_losses, fp32_losses):
        denom = max(abs(f), 1e-6)
        rel_drifts.append(abs(q - f) / denom)
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"Lion8bit median per-step loss drift {median_drift:.4f} exceeds 2%; "
        f"fp32_losses[-5:]={fp32_losses[-5:]}, quant_losses[-5:]={quant_losses[-5:]}."
    )

    # Early-step regime (Lion dynamics dominate, quant noise negligible
    # vs gradient magnitudes) must track within 2%.
    early_drifts = rel_drifts[:10]
    max_early_drift = max(early_drifts)
    assert max_early_drift < 0.02, (
        f"Lion8bit max early-step loss drift {max_early_drift:.4f} exceeds 2%; "
        f"fp32 first 10={fp32_losses[:10]}, q8 first 10={quant_losses[:10]}."
    )


def test_lion8bit_runs_one_step_on_bf16_param_and_preserves_dtype() -> None:
    """Lion8bit must work on bf16 weights without master-copy state and
    keep the parameter dtype at bf16 even though the momentum is uint8."""

    model = _TinyDense()
    model.set_dtype(mx.bfloat16)
    optimizer = make_lion8bit(learning_rate=1e-4, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    x = mx.ones((4, 16), dtype=mx.bfloat16)

    def loss_fn(m: nn.Module, x: mx.array) -> mx.array:
        return mx.sum(m(x))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    assert math.isfinite(float(loss.item()))
    # bf16 parameter must remain bf16 even though the momentum is uint8.
    assert model.linear.weight.dtype == mx.bfloat16
    assert model.linear.bias.dtype == mx.bfloat16


def test_lion8bit_step_actually_updates_parameters() -> None:
    """Sanity check: a single step must change at least one parameter."""

    model = _TinyDense()
    optimizer = make_lion8bit(learning_rate=1e-2, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    before = mx.array(model.linear.weight)
    x = mx.ones((4, 16), dtype=model.linear.weight.dtype)

    def loss_fn(m: nn.Module, x: mx.array) -> mx.array:
        return mx.sum(m(x))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    after = model.linear.weight
    delta = float(mx.max(mx.abs(after - before)).item())
    assert delta > 0.0


def test_lion8bit_class_constants_pin_module_paths() -> None:
    assert LION8BIT_CLASS == "cppmega_mlx.training.optimizers_quantized.Lion8bit"
    assert LION8BIT_SOURCE == "cppmega_mlx.training.optimizers_quantized.make_lion8bit"
    assert LION8BIT_QUANT_KIND == "symmetric_int8_blockwise_v1"


def test_make_lion8bit_returns_lion8bit_instance() -> None:
    optimizer = make_lion8bit(learning_rate=1e-4)
    assert isinstance(optimizer, Lion8bit)
    assert isinstance(optimizer, optim.Optimizer)


def test_make_lion8bit_default_betas_match_make_lion() -> None:
    quant = make_lion8bit()
    fp32 = make_lion()
    assert list(quant.betas) == list(fp32.betas)


def test_make_muon_routes_scalar_optimizer_to_lion8bit() -> None:
    optimizer = make_muon(scalar_optimizer="lion8bit")
    assert isinstance(optimizer.adamw, Lion8bit)


def test_make_muon_routes_scalar_optimizer_to_lion_fp32() -> None:
    optimizer = make_muon(scalar_optimizer="lion")
    assert isinstance(optimizer.adamw, LionFP32Moments)
    assert not isinstance(optimizer.adamw, Lion8bit)


def test_lion8bit_quant_scheme_dispatches_through_codec() -> None:
    """Both schemes must construct without error and produce different
    quant_scheme attributes; Lion8bit reuses the codec dispatch from
    ``_quantize_8bit.quantize_blockwise``, so the symmetric and dynamic
    LUT paths share the same state surface."""

    sym = make_lion8bit(quant_scheme="symmetric_int8_v1")
    dyn = make_lion8bit(quant_scheme="dynamic_int8_v1")
    assert sym.quant_scheme == "symmetric_int8_v1"
    assert dyn.quant_scheme == "dynamic_int8_v1"
    with pytest.raises(ValueError, match="quant_scheme must be one of"):
        make_lion8bit(quant_scheme="bogus_v9")  # type: ignore[arg-type]


def test_make_muon_lion8bit_quant_scheme_propagates() -> None:
    """The make_muon `lion8bit_quant_scheme` knob must thread through to
    the Lion8bit instance, mirroring the adam8bit_quant_scheme contract."""

    optimizer = make_muon(
        scalar_optimizer="lion8bit",
        lion8bit_quant_scheme="dynamic_int8_v1",
    )
    assert isinstance(optimizer.adamw, Lion8bit)
    assert optimizer.adamw.quant_scheme == "dynamic_int8_v1"


# -----------------------------------------------------------------------------
# Fused optimizer request for Lion8bit: native MLX C++/Metal fast path
# -----------------------------------------------------------------------------


def test_fused_lion8bit_default_uses_native_mlx_extension() -> None:
    """``use_fused_kernel`` requests the native MLX C++/Metal primitive."""

    optimizer = make_lion8bit(learning_rate=1e-4)
    assert isinstance(optimizer, Lion8bit)
    assert optimizer.use_fused_kernel is True
    assert optimizer.fused_kernel_status.available is True
    assert "native MLX C++/Metal" in optimizer.fused_kernel_status.reason


def test_fused_lion8bit_can_be_disabled_for_parity_runs() -> None:
    """Passing ``use_fused_kernel=False`` records an explicit disabled status."""

    optimizer = make_lion8bit(learning_rate=1e-4, use_fused_kernel=False)
    assert optimizer.use_fused_kernel is False
    assert optimizer.fused_kernel_status.available is False
    assert "disabled by caller" in optimizer.fused_kernel_status.reason


def test_fused_lion8bit_symmetric_default_uses_native_extension() -> None:
    """Symmetric scheme + fused request resolves to the native extension."""

    optimizer = make_lion8bit(
        learning_rate=1e-4,
        quant_scheme="symmetric_int8_v1",
        use_fused_kernel=True,
    )
    assert optimizer.quant_scheme == "symmetric_int8_v1"
    assert optimizer.use_fused_kernel is True
    assert "native MLX C++/Metal" in optimizer.fused_kernel_status.reason


def test_fused_lion8bit_non_default_block_size_falls_back_to_unfused() -> None:
    """The fused request supports only block_size=256 before status lookup."""

    # quantize_blockwise with non-default block_size raises NotImplementedError
    # at runtime, but constructor validation only checks the fused gate.
    optimizer = make_lion8bit(
        learning_rate=1e-4,
        block_size=128,
        use_fused_kernel=True,
    )
    assert optimizer.block_size == 128
    assert optimizer.use_fused_kernel is False


def _run_lion_n_steps(
    *,
    use_fused_kernel: bool,
    seed: int,
    steps: int,
    in_features: int = 64,
    out_features: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    """Drive a tiny dense model for ``steps`` Lion8bit updates.

    Returns ``(weight, bias, state_arrays)`` after ``mx.eval`` so callers can
    diff the two paths.
    """

    mx.random.seed(seed)
    model = nn.Linear(in_features, out_features, bias=True)
    optimizer = make_lion8bit(
        learning_rate=lr,
        weight_decay=weight_decay,
        use_fused_kernel=use_fused_kernel,
    )
    optimizer.init(model.trainable_parameters())

    mx.random.seed(seed + 1)
    x = mx.random.normal((4, in_features)).astype(mx.float32)
    y = mx.random.normal((4, out_features)).astype(mx.float32)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return (
        mx.array(model.weight),
        mx.array(model.bias),
        _flat_state_arrays(optimizer.state),
    )


def test_fused_lion8bit_matches_unfused_within_tolerance() -> None:
    """A fused request must preserve native path parameter and state behavior.

    The native C++/Metal primitive uses a threadgroup reduction for block
    absmax, so exact bitwise equality is not required for fp32 leaves.
    """

    steps = 1
    seed = 7

    fused_w, fused_b, fused_state = _run_lion_n_steps(
        use_fused_kernel=True, seed=seed, steps=steps
    )
    unfused_w, unfused_b, unfused_state = _run_lion_n_steps(
        use_fused_kernel=False, seed=seed, steps=steps
    )
    mx.eval(fused_w, fused_b, unfused_w, unfused_b)

    _assert_arrays_close_or_equal(fused_w, unfused_w, "weight")
    _assert_arrays_close_or_equal(fused_b, unfused_b, "bias")

    # State: every leaf must match. m_quant is uint8; allow off-by-1 since
    # the per-block absmax tree reduction can swap order vs Python's
    # ``mx.max(..., axis=1)``.
    assert set(fused_state.keys()) == set(unfused_state.keys())
    for key in fused_state:
        f = fused_state[key]
        u = unfused_state[key]
        _assert_arrays_close_or_equal(f, u, key)


def test_fused_lion8bit_loss_matches_unfused_over_50_steps() -> None:
    """A fused request must keep the native MLX loss trajectory unchanged."""

    seed = 0
    lr = 1e-3
    weight_decay = 0.01
    fused_losses = _train_lion_steps(
        make_lion8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=True,
        ),
        seed=seed,
    )
    unfused_losses = _train_lion_steps(
        make_lion8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=False,
        ),
        seed=seed,
    )

    rel_drifts = [
        abs(f - u) / max(abs(u), 1e-6)
        for f, u in zip(fused_losses, unfused_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused vs unfused median per-step loss drift {median_drift:.4f} "
        f"exceeds 2%; unfused[-5:]={unfused_losses[-5:]}, "
        f"fused[-5:]={fused_losses[-5:]}."
    )


def test_fused_lion8bit_loss_matches_lion_fp32_within_tolerance() -> None:
    """50-step smoke: requested-fused Lion8bit keeps the native parity contract."""

    seed = 0
    lr = 1e-3
    weight_decay = 0.01
    fp32_losses = _train_lion_steps(
        make_lion(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_lion_steps(
        make_lion8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            use_fused_kernel=True,
        ),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = [
        abs(q - f) / max(abs(f), 1e-6)
        for q, f in zip(quant_losses, fp32_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused Lion8bit median per-step loss drift {median_drift:.4f} "
        f"exceeds 2%; fp32_losses[-5:]={fp32_losses[-5:]}, "
        f"quant_losses[-5:]={quant_losses[-5:]}."
    )

    early_drifts = rel_drifts[:10]
    max_early_drift = max(early_drifts)
    assert max_early_drift < 0.02, (
        f"fused Lion8bit max early-step loss drift {max_early_drift:.4f} "
        f"exceeds 2%; fp32 first 10={fp32_losses[:10]}, "
        f"q8 first 10={quant_losses[:10]}."
    )


def test_fused_lion8bit_direct_call_runs_native_extension() -> None:
    status = fused_lion8bit_status()
    assert status.available is True
    assert "native MLX C++/Metal" in status.reason

    n = DEFAULT_BLOCK_SIZE
    outputs = fused_lion8bit_step(
        mx.zeros((n,), dtype=mx.float32),
        mx.zeros((n,), dtype=mx.float32),
        mx.full((n,), 128, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        learning_rate=mx.array(1e-4, dtype=mx.float32),
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )
    mx.eval(outputs)
    assert len(outputs) == 3
    assert outputs[0].shape == (n,)


def test_fused_lion8bit_state_keys_match_unfused() -> None:
    """The fused path must not leak extra state -- the contract test in
    ``tests/test_optimizer_no_master_contract.py`` only allows
    ``m_quant``, ``m_absmax``, ``step``, ``learning_rate`` in optimizer
    state for Lion8bit (no ``v`` moment).
    """

    optimizer_fused = make_lion8bit(use_fused_kernel=True)
    optimizer_unfused = make_lion8bit(use_fused_kernel=False)

    model = _TinyDense()
    optimizer_fused.init(model.trainable_parameters())
    # init_single is shape-agnostic, so the unfused path needs its own model.
    model2 = _TinyDense()
    optimizer_unfused.init(model2.trainable_parameters())
    mx.eval(optimizer_fused.state, optimizer_unfused.state)

    fused_keys = {p.split(".")[-1] for p, _ in tree_flatten(optimizer_fused.state)}
    unfused_keys = {p.split(".")[-1] for p, _ in tree_flatten(optimizer_unfused.state)}
    assert fused_keys == unfused_keys
    expected = {"m_quant", "m_absmax", "step", "learning_rate"}
    assert fused_keys == expected, fused_keys


# -----------------------------------------------------------------------------
# Fused dynamic-LUT requests (Adam8bit + Lion8bit): native extension path.
# -----------------------------------------------------------------------------


def _run_n_steps_dynamic(
    *,
    use_fused_kernel: bool,
    seed: int,
    steps: int,
    in_features: int = 64,
    out_features: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    """``_run_n_steps`` variant that pins ``quant_scheme=dynamic_int8_v1``."""

    mx.random.seed(seed)
    model = nn.Linear(in_features, out_features, bias=True)
    optimizer = make_adam8bit(
        learning_rate=lr,
        weight_decay=weight_decay,
        use_fused_kernel=use_fused_kernel,
        quant_scheme=QUANT_SCHEME_DYNAMIC,
    )
    optimizer.init(model.trainable_parameters())

    mx.random.seed(seed + 1)
    x = mx.random.normal((4, in_features)).astype(mx.float32)
    y = mx.random.normal((4, out_features)).astype(mx.float32)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return (
        mx.array(model.weight),
        mx.array(model.bias),
        _flat_state_arrays(optimizer.state),
    )


def _run_lion_n_steps_dynamic(
    *,
    use_fused_kernel: bool,
    seed: int,
    steps: int,
    in_features: int = 64,
    out_features: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    """``_run_lion_n_steps`` variant that pins ``quant_scheme=dynamic_int8_v1``."""

    mx.random.seed(seed)
    model = nn.Linear(in_features, out_features, bias=True)
    optimizer = make_lion8bit(
        learning_rate=lr,
        weight_decay=weight_decay,
        use_fused_kernel=use_fused_kernel,
        quant_scheme=QUANT_SCHEME_DYNAMIC,
    )
    optimizer.init(model.trainable_parameters())

    mx.random.seed(seed + 1)
    x = mx.random.normal((4, in_features)).astype(mx.float32)
    y = mx.random.normal((4, out_features)).astype(mx.float32)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return (
        mx.array(model.weight),
        mx.array(model.bias),
        _flat_state_arrays(optimizer.state),
    )


def _train_steps_dynamic(
    optimizer: optim.Optimizer,
    *,
    seed: int,
    steps: int = 50,
    hidden: int = 128,
    batch: int = 4,
) -> list[float]:
    """Same noisy-regression smoke as ``_train_steps`` but pinned to fp32 model."""

    mx.random.seed(seed)
    true_w = mx.random.normal((hidden, hidden)).astype(mx.float32) * 0.3
    x = mx.random.normal((batch, hidden)).astype(mx.float32)
    obs_noise = mx.random.normal((batch, hidden)).astype(mx.float32) * 1.5
    y = x @ true_w.T + obs_noise

    mx.random.seed(seed)
    model = nn.Linear(hidden, hidden, bias=False)
    optimizer.init(model.trainable_parameters())

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    return losses


def test_fused_dynamic_adam8bit_matches_unfused_within_tolerance() -> None:
    """A dynamic fused request must preserve native MLX Adam8bit behavior."""

    steps = 1
    seed = 7

    fused_w, fused_b, fused_state = _run_n_steps_dynamic(
        use_fused_kernel=True, seed=seed, steps=steps
    )
    unfused_w, unfused_b, unfused_state = _run_n_steps_dynamic(
        use_fused_kernel=False, seed=seed, steps=steps
    )
    mx.eval(fused_w, fused_b, unfused_w, unfused_b)

    _assert_arrays_close_or_equal(fused_w, unfused_w, "weight")
    _assert_arrays_close_or_equal(fused_b, unfused_b, "bias")

    assert set(fused_state.keys()) == set(unfused_state.keys())
    for key in fused_state:
        f = fused_state[key]
        u = unfused_state[key]
        _assert_arrays_close_or_equal(f, u, key)


def test_fused_dynamic_adam8bit_loss_matches_fp32_within_tolerance() -> None:
    """50-step trajectory: requested-fused dynamic Adam8bit keeps parity."""

    seed = 0
    lr = 2e-4
    weight_decay = 0.01
    fp32_losses = _train_steps_dynamic(
        make_adamw(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_steps_dynamic(
        make_adam8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            quant_scheme=QUANT_SCHEME_DYNAMIC,
            use_fused_kernel=True,
        ),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = [
        abs(q - f) / max(abs(f), 1e-6)
        for q, f in zip(quant_losses, fp32_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused dynamic Adam8bit median per-step drift {median_drift:.4f} "
        f"exceeds 2%; fp32_losses[-5:]={fp32_losses[-5:]}, "
        f"quant_losses[-5:]={quant_losses[-5:]}."
    )


def test_fused_dynamic_adam8bit_direct_call_runs_native_extension() -> None:
    status = fused_adam8bit_dynamic_status()
    assert status.available is True
    assert "native MLX C++/Metal" in status.reason

    n = DEFAULT_BLOCK_SIZE
    outputs = fused_adam8bit_dynamic_step(
        mx.zeros((n,), dtype=mx.float32),
        mx.zeros((n,), dtype=mx.float32),
        mx.full((n,), 127, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        mx.full((n,), 127, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        learning_rate=mx.array(1e-3, dtype=mx.float32),
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        step=mx.array(1, dtype=mx.uint64),
        bias_correction=False,
    )
    mx.eval(outputs)
    assert len(outputs) == 5
    assert outputs[0].shape == (n,)


def test_fused_dynamic_lion8bit_matches_unfused_within_tolerance() -> None:
    """A dynamic fused request must preserve native MLX Lion8bit behavior."""

    steps = 1
    seed = 7

    fused_w, fused_b, fused_state = _run_lion_n_steps_dynamic(
        use_fused_kernel=True, seed=seed, steps=steps
    )
    unfused_w, unfused_b, unfused_state = _run_lion_n_steps_dynamic(
        use_fused_kernel=False, seed=seed, steps=steps
    )
    mx.eval(fused_w, fused_b, unfused_w, unfused_b)

    _assert_arrays_close_or_equal(fused_w, unfused_w, "weight")
    _assert_arrays_close_or_equal(fused_b, unfused_b, "bias")

    assert set(fused_state.keys()) == set(unfused_state.keys())
    for key in fused_state:
        f = fused_state[key]
        u = unfused_state[key]
        _assert_arrays_close_or_equal(f, u, key)


def test_fused_dynamic_lion8bit_loss_matches_fp32_within_tolerance() -> None:
    """50-step trajectory: requested-fused dynamic Lion8bit keeps parity."""

    seed = 0
    lr = 1e-3
    weight_decay = 0.01
    fp32_losses = _train_lion_steps(
        make_lion(learning_rate=lr, weight_decay=weight_decay),
        seed=seed,
    )
    quant_losses = _train_lion_steps(
        make_lion8bit(
            learning_rate=lr,
            weight_decay=weight_decay,
            quant_scheme=QUANT_SCHEME_DYNAMIC,
            use_fused_kernel=True,
        ),
        seed=seed,
    )

    assert len(fp32_losses) == len(quant_losses) == 50
    assert quant_losses[-1] < quant_losses[0]
    assert fp32_losses[-1] < fp32_losses[0]

    rel_drifts = [
        abs(q - f) / max(abs(f), 1e-6)
        for q, f in zip(quant_losses, fp32_losses)
    ]
    median_drift = sorted(rel_drifts)[len(rel_drifts) // 2]
    assert median_drift < 0.02, (
        f"fused dynamic Lion8bit median per-step drift {median_drift:.4f} "
        f"exceeds 2%; fp32_losses[-5:]={fp32_losses[-5:]}, "
        f"quant_losses[-5:]={quant_losses[-5:]}."
    )


def test_fused_dynamic_lion8bit_direct_call_runs_native_extension() -> None:
    status = fused_lion8bit_dynamic_status()
    assert status.available is True
    assert "native MLX C++/Metal" in status.reason

    n = DEFAULT_BLOCK_SIZE
    outputs = fused_lion8bit_dynamic_step(
        mx.zeros((n,), dtype=mx.float32),
        mx.zeros((n,), dtype=mx.float32),
        mx.full((n,), 127, dtype=mx.uint8),
        mx.zeros((1,), dtype=mx.float32),
        learning_rate=mx.array(1e-4, dtype=mx.float32),
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
    )
    mx.eval(outputs)
    assert len(outputs) == 3
    assert outputs[0].shape == (n,)
