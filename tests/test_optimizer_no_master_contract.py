"""Contract tests: cppmega.mlx optimizers must mirror cppmega CUDA's
``Float16NoMasterOptimizer`` pattern.

The CUDA reference keeps optimizer state strictly to the moments and a tiny
header (step + learning_rate) — there is no master copy of weights, no
fp32 grad buffer, no aliased per-parameter side-channel. The bf16 weights
are the source of truth and gradients/parameters are cast to fp32 inline
inside ``apply_single``.

These opt-in tests walk ``tree_flatten(opt.state)`` for the production-shape
``local_gb10_quarter()`` model and assert:

* The exact set of leaf keys is the documented allow-list.
* ``m`` and ``v`` (when present) are fp32 with the same shape as the
  parameter; ``step`` is uint64; ``learning_rate`` is fp32 scalar.
* The total state byte count equals the analytic expectation derived from
  ``sum(p.size for p in trainable_parameters())`` — any excess would be an
  unaccounted master/grad/aliased buffer.

The audit script ``/tmp/audit_optimizer_state.py`` produced these anchor
numbers (3 May 2026):

* AdamW   : 14_372_170_844 bytes (1.797B params * 2 fp32 moments + 12 B header)
* Lion    :  7_186_085_428 bytes (1.797B params * 1 fp32 moment + 12 B header)
* Muon+AW :  9_125_624_936 bytes (Muon group ``v`` + AdamW group ``m`` + ``v``)
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.recipes.model_factory import local_gb10_quarter
from cppmega_mlx.training.optimizers import (
    AdamWFP32Moments,
    LionFP32Moments,
    MuonAdamWMulti,
    is_muon_compatible,
    make_adamw,
    make_lion,
    make_muon,
)


# Header byte cost shared by every optimizer subclass: one fp32 ``learning_rate``
# scalar (4 bytes) plus one uint64 ``step`` counter (8 bytes).
_HEADER_BYTES = 12


def _flatten_arrays(state: object) -> list[tuple[str, mx.array]]:
    return [(k, v) for k, v in tree_flatten(state) if isinstance(v, mx.array)]


def _leaf_key(path: str) -> str:
    return path.rsplit(".", 1)[-1] if "." in path else path


def _param_summary(params: object) -> tuple[int, int, int]:
    """Return (n_unique_arrays, n_elements, n_param_bytes) for a param pytree."""
    arrays = [v for _, v in tree_flatten(params) if isinstance(v, mx.array)]
    n_unique = len({id(a) for a in arrays})
    n_elements = sum(a.size for a in arrays)
    n_bytes = sum(a.nbytes for a in arrays)
    return n_unique, n_elements, n_bytes


@pytest.fixture(scope="module")
def production_model() -> object:
    """Build the bf16 1.797B-param ``local_gb10_quarter`` model once.

    The audit needs production shapes — toy models cannot detect a hidden
    master buffer because their state is dominated by the header. Tests
    that can afford ~17 GiB peak (3.3 GiB params + 13.4 GiB AdamW) must
    set ``CPPMEGA_OPTIMIZER_CONTRACT_PRODUCTION_SHAPE=1`` to opt in.
    """
    if os.environ.get("CPPMEGA_OPTIMIZER_CONTRACT_PRODUCTION_SHAPE") != "1":
        pytest.skip(
            "set CPPMEGA_OPTIMIZER_CONTRACT_PRODUCTION_SHAPE=1 to run the "
            "production-shape optimizer-state audit"
        )
    if os.environ.get("CPPMEGA_OPTIMIZER_CONTRACT_SKIP") == "1":
        pytest.skip("CPPMEGA_OPTIMIZER_CONTRACT_SKIP=1 set; skipping production-shape audit")
    return local_gb10_quarter()


@pytest.fixture(scope="module")
def production_params(production_model: object) -> object:
    return production_model.trainable_parameters()


@pytest.fixture(scope="module")
def param_stats(production_params: object) -> tuple[int, int, int]:
    return _param_summary(production_params)


@pytest.mark.training
def test_adamw_fp32_moments_holds_only_m_v_step_lr_no_master(
    production_params: object,
    param_stats: tuple[int, int, int],
) -> None:
    """Float16NoMasterOptimizer-equivalent: AdamW must hold per-param ``m``,
    ``v`` in fp32; ``step`` (uint64); ``learning_rate`` (fp32). Nothing else.
    No master copy of weights, no fp32 grad buffer, no aliased state.
    """
    n_unique, n_elements, _ = param_stats
    optimizer = make_adamw()
    assert isinstance(optimizer, AdamWFP32Moments)
    optimizer.init(production_params)
    mx.eval(optimizer.state)

    leaves = _flatten_arrays(optimizer.state)
    leaf_keys = {_leaf_key(name) for name, _ in leaves}
    assert leaf_keys == {"m", "v", "step", "learning_rate"}, leaf_keys

    counts: dict[str, int] = {}
    total_bytes = 0
    for name, value in leaves:
        counts[_leaf_key(name)] = counts.get(_leaf_key(name), 0) + 1
        total_bytes += value.nbytes
        key = _leaf_key(name)
        if key in {"m", "v"}:
            assert value.dtype == mx.float32, (name, value.dtype)
        elif key == "step":
            assert value.dtype == mx.uint64, (name, value.dtype)
        elif key == "learning_rate":
            assert value.dtype == mx.float32, (name, value.dtype)

    assert counts["m"] == n_unique
    assert counts["v"] == n_unique
    assert counts["step"] == 1
    assert counts["learning_rate"] == 1

    expected_bytes = n_elements * 2 * 4 + _HEADER_BYTES
    assert total_bytes == expected_bytes, (
        f"AdamW state {total_bytes} bytes != expected {expected_bytes} "
        f"(delta {total_bytes - expected_bytes}). Excess implies a master "
        f"copy, fp32 grad buffer, or aliased state leak."
    )


@pytest.mark.training
def test_lion_fp32_moments_holds_only_m_step_lr_no_master(
    production_params: object,
    param_stats: tuple[int, int, int],
) -> None:
    """Lion only carries ``m`` (fp32) per parameter; ``step`` (uint64);
    ``learning_rate`` (fp32). No ``v``, no master, no fp32 grad cache.
    """
    n_unique, n_elements, _ = param_stats
    optimizer = make_lion()
    assert isinstance(optimizer, LionFP32Moments)
    optimizer.init(production_params)
    mx.eval(optimizer.state)

    leaves = _flatten_arrays(optimizer.state)
    leaf_keys = {_leaf_key(name) for name, _ in leaves}
    assert leaf_keys == {"m", "step", "learning_rate"}, leaf_keys

    counts: dict[str, int] = {}
    total_bytes = 0
    for name, value in leaves:
        counts[_leaf_key(name)] = counts.get(_leaf_key(name), 0) + 1
        total_bytes += value.nbytes
        key = _leaf_key(name)
        if key == "m":
            assert value.dtype == mx.float32, (name, value.dtype)
        elif key == "step":
            assert value.dtype == mx.uint64, (name, value.dtype)
        elif key == "learning_rate":
            assert value.dtype == mx.float32, (name, value.dtype)

    assert counts["m"] == n_unique
    assert counts["step"] == 1
    assert counts["learning_rate"] == 1

    expected_bytes = n_elements * 1 * 4 + _HEADER_BYTES
    assert total_bytes == expected_bytes, (
        f"Lion state {total_bytes} bytes != expected {expected_bytes} "
        f"(delta {total_bytes - expected_bytes}). Excess implies a master "
        f"copy, second moment, or aliased state leak."
    )


@pytest.mark.training
def test_muon_adamw_multi_holds_only_m_v_step_lr_no_master(
    production_params: object,
) -> None:
    """``MuonAdamWMulti`` must keep two parallel buckets, each obeying the
    Float16NoMasterOptimizer contract:

    * Muon bucket: per-param ``v`` (fp32); ``step`` (uint64); ``learning_rate``
      (fp32). No ``m`` (Muon has no first moment), no Newton-Schulz scratch
      buffer cached in state.
    * AdamW bucket: per-param ``m``/``v`` (fp32); ``step`` (uint64);
      ``learning_rate`` (fp32). Same allow-list as the standalone AdamW test.

    The two buckets together must cover every parameter exactly once
    (Megatron emerging_optimizers' ``_is_nonlinear_or_embedding`` partition).
    """
    optimizer = make_muon()
    assert isinstance(optimizer, MuonAdamWMulti)
    optimizer.init(production_params)
    mx.eval(optimizer.state)

    state = optimizer.state
    assert set(state.keys()) == {"muon", "adamw"}, set(state.keys())

    # Partition the parameter pytree using the same predicate the optimizer
    # uses internally so that we can pin the per-bucket leaf counts and byte
    # totals exactly.
    muon_count = 0
    muon_elements = 0
    adamw_count = 0
    adamw_elements = 0
    for name, value in tree_flatten(production_params):
        if not isinstance(value, mx.array):
            continue
        if is_muon_compatible(name, value):
            muon_count += 1
            muon_elements += value.size
        else:
            adamw_count += 1
            adamw_elements += value.size

    # --- Muon bucket --------------------------------------------------
    muon_leaves = _flatten_arrays(state["muon"])
    muon_leaf_keys = {_leaf_key(name) for name, _ in muon_leaves}
    assert muon_leaf_keys == {"v", "step", "learning_rate"}, muon_leaf_keys

    muon_counts: dict[str, int] = {}
    muon_bytes = 0
    for name, value in muon_leaves:
        muon_counts[_leaf_key(name)] = muon_counts.get(_leaf_key(name), 0) + 1
        muon_bytes += value.nbytes
        key = _leaf_key(name)
        if key == "v":
            assert value.dtype == mx.float32, (name, value.dtype)
        elif key == "step":
            assert value.dtype == mx.uint64, (name, value.dtype)
        elif key == "learning_rate":
            assert value.dtype == mx.float32, (name, value.dtype)

    assert muon_counts["v"] == muon_count
    assert muon_counts["step"] == 1
    assert muon_counts["learning_rate"] == 1

    expected_muon_bytes = muon_elements * 1 * 4 + _HEADER_BYTES
    assert muon_bytes == expected_muon_bytes, (
        f"Muon bucket {muon_bytes} bytes != expected {expected_muon_bytes} "
        f"(delta {muon_bytes - expected_muon_bytes}). Excess implies an NS "
        f"scratch buffer, master copy, or per-param momentum aliasing."
    )

    # --- AdamW bucket ------------------------------------------------
    adamw_leaves = _flatten_arrays(state["adamw"])
    adamw_leaf_keys = {_leaf_key(name) for name, _ in adamw_leaves}
    assert adamw_leaf_keys == {"m", "v", "step", "learning_rate"}, adamw_leaf_keys

    adamw_counts: dict[str, int] = {}
    adamw_bytes = 0
    for name, value in adamw_leaves:
        adamw_counts[_leaf_key(name)] = adamw_counts.get(_leaf_key(name), 0) + 1
        adamw_bytes += value.nbytes
        key = _leaf_key(name)
        if key in {"m", "v"}:
            assert value.dtype == mx.float32, (name, value.dtype)
        elif key == "step":
            assert value.dtype == mx.uint64, (name, value.dtype)
        elif key == "learning_rate":
            assert value.dtype == mx.float32, (name, value.dtype)

    assert adamw_counts["m"] == adamw_count
    assert adamw_counts["v"] == adamw_count
    assert adamw_counts["step"] == 1
    assert adamw_counts["learning_rate"] == 1

    expected_adamw_bytes = adamw_elements * 2 * 4 + _HEADER_BYTES
    assert adamw_bytes == expected_adamw_bytes, (
        f"Muon-AdamW fallback bucket {adamw_bytes} != expected "
        f"{expected_adamw_bytes} (delta {adamw_bytes - expected_adamw_bytes})."
    )


@pytest.mark.training
def test_no_optimizer_leaf_dtype_matches_bf16_param_dtype(
    production_params: object,
) -> None:
    """A defensive sweep: no leaf in any optimizer's state may share the
    parameter dtype (bf16). Any bf16 array in ``opt.state`` would imply a
    master/secondary copy of the weights stored alongside the moments.
    """
    for factory in (make_adamw, make_lion, make_muon):
        optimizer = factory()
        optimizer.init(production_params)
        mx.eval(optimizer.state)
        leaves = _flatten_arrays(optimizer.state)
        bf16_leaves = [name for name, value in leaves if value.dtype == mx.bfloat16]
        assert bf16_leaves == [], (
            f"{factory.__name__} state contains bf16 leaves (likely a master "
            f"copy of weights): {bf16_leaves}"
        )
