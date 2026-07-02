from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpp_jsonl_generation_compile_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cpp_jsonl_generation_compile_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_text_source_prefix_and_docstring():
    mod = _load_module()
    case = {
        "task_id": "x",
        "prompt": "complete it",
        "source_prefix": "int f(){\n",
    }
    assert mod.prompt_text(case, "source-prefix") == "int f(){\n"
    assert mod.prompt_text(case, "docstring") == "int f(){\n"


def test_default_side_channels_are_token_aligned_zero_arrays():
    mod = _load_module()
    side = mod.default_side_channels(7)
    assert set(side) == set(mod.SIDE_CHANNEL_NAMES)
    for tensor in side.values():
        assert tuple(tensor.shape) == (1, 7)
        assert tensor.dtype == mx.int32
        assert int(mx.sum(tensor).item()) == 0


class _FakeTokenizer:
    code_start_id = 7

    def encode(self, text: str, prepend: int | None = None):
        ids = [101, 102, len(text)]
        return ([prepend] if prepend is not None else []) + ids


def test_build_prompt_context_zero_sidecars_align_with_prepend():
    mod = _load_module()
    ids, side, provenance = mod.build_prompt_context(
        _FakeTokenizer(),
        "int f() {",
        prompt_sidecars="zero",
        prepend_code_start=True,
    )

    assert ids[0] == _FakeTokenizer.code_start_id
    assert provenance == {"prompt_sidecars": "zero"}
    assert set(side) == set(mod.SIDE_CHANNEL_NAMES)
    assert all(len(values) == len(ids) for values in side.values())
    assert all(sum(values) == 0 for values in side.values())


def test_build_prompt_context_rejects_clang_sidecars_with_prepended_code_start():
    mod = _load_module()
    with pytest.raises(ValueError, match="prepend-code-start"):
        mod.build_prompt_context(
            _FakeTokenizer(),
            "int f() {",
            prompt_sidecars="clang",
            prepend_code_start=True,
        )


def test_build_prompt_context_clang_sidecars_are_fail_closed_when_available():
    mod = _load_module()
    tokenizer = pytest.importorskip("cppmega_mlx.tokenizer.cpp_tokenizer")
    adapter = mod.get_builtin_code_metadata_adapter("cpp")
    capabilities = adapter.probe({"language": "cpp"})
    if not capabilities.available:
        pytest.skip(f"clang adapter unavailable: {capabilities.reason}")

    ids, side, provenance = mod.build_prompt_context(
        tokenizer.load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER),
        "int f() {\n  return 1;\n}\n",
        prompt_sidecars="clang",
        prepend_code_start=False,
    )

    assert ids
    assert set(side) == set(mod.SIDE_CHANNEL_NAMES)
    assert all(len(values) == len(ids) for values in side.values())
    assert provenance.get("adapter") == "cpp:clang-ast-v1"
    assert not any("dropped" in str(value) for value in provenance.values())


def test_body_decode_constraints_ban_specials_and_degenerate_token_run():
    mod = _load_module()

    class Tok:
        bos_token_id = 2
        eos_token_id = 3
        fim_prefix_id = 4
        fim_middle_id = 5
        fim_suffix_id = 6
        code_start_id = 7
        code_end_id = 8
        think_start_id = 9
        think_end_id = 10
        query_tool_id = 11
        tool_result_id = 19

    constraints = mod.BodyDecodeConstraints(Tok(), prompt_len=2, max_token_run=4)
    logits = mx.zeros((1, 64), dtype=mx.float32)
    tokens = mx.array([[100, 101, 42, 42, 42, 42]], dtype=mx.int32)

    masked = constraints(logits, tokens)

    assert float(masked[0, 42].item()) == float("-inf")
    assert float(masked[0, Tok.code_start_id].item()) == float("-inf")
    assert float(masked[0, Tok.fim_prefix_id].item()) == float("-inf")


def test_script_help_bootstraps_repo_root_from_sibling_cwd():
    sibling_cppmega = ROOT.parent / "cppmega"
    if not sibling_cppmega.is_dir():
        pytest.skip("sibling cppmega checkout not present")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=sibling_cppmega,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--prompt-sidecars" in proc.stdout


def test_rejects_megatron_distcp_file(tmp_path: Path):
    mod = _load_module()
    ckpt = tmp_path / "__0_0.distcp"
    ckpt.write_bytes(b"not used")
    with pytest.raises(ValueError, match="Megatron torch_dist"):
        mod.reject_unsupported_checkpoint(ckpt)


def test_rejects_megatron_checkpoint_directory(tmp_path: Path):
    mod = _load_module()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5000\n")
    with pytest.raises(ValueError, match="Megatron torch_dist"):
        mod.reject_unsupported_checkpoint(tmp_path)


def test_checkpoint_model_config_loads_safetensors_sidecar(tmp_path: Path):
    mod = _load_module()
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a real checkpoint")
    (tmp_path / "model.json").write_text(
        json.dumps({"config": {"structure_components": "core", "hidden_size": 1280}}),
        encoding="utf-8",
    )

    assert mod.checkpoint_model_config(ckpt) == {
        "structure_components": "core",
        "hidden_size": 1280,
    }


def test_compile_gate_env_prepends_tool_dirs(tmp_path: Path):
    mod = _load_module()
    clang_format = tmp_path / "clang-format"
    clang_format.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    clang_format.chmod(0o755)

    env = mod.compile_gate_env({"PATH": ""}, path_dirs=(tmp_path,))

    assert env["PATH"].split(":")[0] == str(tmp_path)
