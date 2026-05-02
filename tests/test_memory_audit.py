"""Regression tests for the runtime memory audit."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.runtime.memory_audit import audit_memory, format_report


class _AliasedModel(nn.Module):
    """Two attributes pointing at the same submodule — should dedupe."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 16)
        self.alias = self.fc  # same arrays under two names


class _UntiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(64, 16)
        self.layer = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 64, bias=False)


def test_audit_dedupes_aliased_attribute_paths() -> None:
    model = _AliasedModel()
    mx.eval(model.parameters())
    report = audit_memory(model, tag="aliased")

    # fc.weight + fc.bias are reachable as fc.* AND alias.* — only counted once.
    assert report["model_param_unique_arrays"] == 2  # weight + bias
    aliased = report["model_param_aliased_arrays"]
    assert len(aliased) == 2
    # Either {alias.* primary, fc.* alias} or vice versa — order depends on
    # the underlying tree walk; what matters is that for each unique array
    # exactly one path is the primary and the other is recorded as an alias.
    pairs = {
        tuple(sorted([entry["primary_name"], *entry["alias_names"]]))
        for entry in aliased
    }
    assert pairs == {
        ("alias.bias", "fc.bias"),
        ("alias.weight", "fc.weight"),
    }

    fc_total = 16 * 8 + 16  # weight + bias
    by_top = report["model_params_by_top_module"]
    assert len(by_top) == 1
    only_top = next(iter(by_top))
    assert only_top in {"alias", "fc"}
    assert by_top[only_top]["numel"] == fc_total


def test_audit_buckets_categories_correctly() -> None:
    model = _UntiedModel()
    mx.eval(model.parameters())
    report = audit_memory(model, tag="untied")

    by_category = report["model_params_by_category_dtype"]
    embed_keys = [k for k in by_category if k.startswith("scalar_fallback_embedding_or_output|")]
    assert embed_keys, by_category
    matrix_keys = [k for k in by_category if k.startswith("muon_matrix|")]
    assert matrix_keys, by_category

    embed_numel = sum(by_category[k]["numel"] for k in embed_keys)
    # token_embedding (64x16) + lm_head (16x64) — both flagged as embedding-like
    assert embed_numel == 64 * 16 + 16 * 64


def test_audit_report_format_returns_string_with_totals() -> None:
    model = _UntiedModel()
    mx.eval(model.parameters())
    report = audit_memory(model, tag="format")

    text = format_report(report, top_n=5)
    assert "format" in text
    assert "total params" in text
    assert "muon_matrix" in text or "scalar_fallback" in text


def test_audit_includes_optimizer_state_when_supplied() -> None:
    import mlx.optimizers as optim

    model = _UntiedModel()
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3)
    opt.init(model.trainable_parameters())
    mx.eval(opt.state)

    report = audit_memory(model, optimizer=opt, tag="opt")
    assert report["optimizer_state_by_dtype"]
    assert report["optimizer_state_by_key_dtype"]
    # Adam tracks m, v as float32 by default.
    fp32_total = report["optimizer_state_by_dtype"].get("float32", {}).get("numel", 0)
    assert fp32_total >= 2 * report["model_param_count"]
