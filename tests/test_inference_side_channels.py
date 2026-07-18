from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    AdapterCapabilities,
    CodeMetadataAdapter,
    GoCodeMetadataAdapter,
    InferenceSideChannelBuilder,
    PythonCodeMetadataAdapter,
    RustCodeMetadataAdapter,
    TokenMetadata,
    builtin_code_metadata_adapters,
    generate_tokens,
    get_builtin_code_metadata_adapter,
    normalize_code_language,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM


class TinyTokenizer:
    name_or_path = "tiny-tokenizer"

    def encode(self, text: str) -> list[int]:
        return [max(1, (ord(char) % 31) + 1) for char in text]


class FakeCppAdapter:
    language = "cpp"
    version = "fake-clang-v1"

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities:
        return AdapterCapabilities(
            language=self.language,
            version=self.version,
            families=("structure", "syntax"),
        )

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> str:
        assert options["language"] == "cpp"
        return source_or_project

    def map_to_tokens(
        self,
        metadata: str,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        del metadata, tokenizer
        n = len(tokens)
        return TokenMetadata(
            structure_ids=[1] * n,
            dep_levels=list(range(n)),
            ast_depth_ids=[2] * n,
            sibling_index_ids=[0] * n,
            node_type_ids=[3] * n,
            provenance={"adapter": "fake-clang-v1"},
        )


class FailingAdapter(FakeCppAdapter):
    version = "failing-v1"

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> str:
        del source_or_project, options
        raise ValueError("no compile database")


class EmptyAdapter(FakeCppAdapter):
    version = "empty-v1"

    def map_to_tokens(
        self,
        metadata: str,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        del metadata, tokens, tokenizer
        return TokenMetadata()


class SemanticIdentityAdapter(FakeCppAdapter):
    version = "semantic-identity-v1"

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities:
        del context
        return AdapterCapabilities(
            language=self.language,
            version=self.version,
            families=("semantic_graph",),
        )

    def map_to_tokens(
        self,
        metadata: str,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        del metadata, tokenizer
        identity = (1 << 63) + 12345
        return TokenMetadata(
            side_channels={
                "semantic_graph": {
                    "token_symbol_ids": [identity] * len(tokens),
                }
            },
            provenance={"adapter": self.version},
        )


def _tiny_model() -> HybridTinyLM:
    return HybridTinyLM(
        HybridTinyConfig(
            vocab_size=64,
            hidden_size=16,
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            num_attention_heads=4,
            max_seq_length=64,
            structure_components="all",
            structure_num_categories=16,
        )
    )


def test_inference_side_channel_builder_fails_closed_without_adapter() -> None:
    with pytest.raises(RuntimeError, match="adapter_missing"):
        InferenceSideChannelBuilder(TinyTokenizer()).build("int x;")


def test_inference_side_channel_builder_defaults_to_error_on_adapter_failure() -> None:
    with pytest.raises(RuntimeError, match="adapter_error:ValueError"):
        InferenceSideChannelBuilder(
            TinyTokenizer(),
            adapter=FailingAdapter(),
        ).build("int x;", language="cpp")


def test_inference_side_channel_builder_rejects_empty_adapter_metadata() -> None:
    with pytest.raises(RuntimeError, match="adapter_empty_metadata"):
        InferenceSideChannelBuilder(
            TinyTokenizer(),
            adapter=EmptyAdapter(),
        ).build("int x;", language="cpp")

    dropped = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=EmptyAdapter(),
        fail_policy="drop_family",
    ).build("int x;", language="cpp")
    assert dropped.side_channels == {}
    assert dropped.model_kwargs == {}
    assert dropped.provenance["inference_failure_reason"] == (
        "adapter_empty_metadata"
    )
    assert dropped.provenance["inference_degraded"] == "true"


def test_inference_side_channel_builder_text_only_prompt_is_explicit() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        fail_policy="text_only",
    )

    result = builder.build("int x;")

    assert result.prompt_ids.shape == (1, 6)
    assert result.model_kwargs == {}
    assert result.side_channels == {}
    assert result.cache_components.content_sha256 == sha256(b"int x;").hexdigest()
    assert result.cache_components.tokenizer_id == "tiny-tokenizer"
    assert result.provenance == {
        "fallback": "text_only",
        "inference_fail_policy": "text_only",
        "inference_enrichment_status": "explicit_text_only",
        "inference_degraded": "true",
    }


def test_inference_side_channel_builder_platform_context_smoke() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        fail_policy="drop_family",
    )

    result = builder.build(
        "int x;",
        platform_context={
            "os": "macos",
            "arch": "arm64",
            "compiler": "clang",
            "accelerator": "metal",
        },
    )
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    platform_ids = result.model_kwargs["platform_ids"]
    assert platform_ids.shape[0] == 1
    assert int(np.count_nonzero(np.array(platform_ids))) > 0
    assert "os=macos" in result.rendered_platform_context
    assert result.side_channels["platform"]["platform_ids"] is platform_ids
    assert result.provenance["fallback"] == "drop_family"
    assert result.provenance["inference_failure_reason"] == "adapter_missing"
    assert result.provenance["inference_degraded"] == "true"
    assert logits.shape[:2] == result.prompt_ids.shape


def test_inference_side_channel_builder_adapter_metadata_smoke() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )

    result = builder.build("int x;", language="cpp")
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    assert {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    } <= set(result.model_kwargs)
    assert result.model_kwargs["structure_ids"].shape == result.prompt_ids.shape
    assert result.side_channels["syntax"]["ast_depth_ids"].shape == result.prompt_ids.shape
    assert result.provenance["adapter"] == "fake-clang-v1"
    assert result.provenance["inference_fail_policy"] == "error"
    assert result.provenance["inference_enrichment_status"] == "enriched"
    assert result.provenance["inference_degraded"] == "false"
    assert result.cache_components.adapter_language == "cpp"
    assert result.cache_components.adapter_version == "fake-clang-v1"
    assert logits.shape[:2] == result.prompt_ids.shape


def test_inference_side_channels_preserve_opaque_uint64_identities() -> None:
    result = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=SemanticIdentityAdapter(),
    ).build("int x;", language="cpp")

    identities = result.side_channels["semantic_graph"]["token_symbol_ids"]
    values = np.array(identities)
    assert values.dtype == np.dtype(np.uint64)
    assert set(values.reshape(-1).tolist()) == {(1 << 63) + 12345}


def test_inference_side_channel_builder_result_feeds_generation_loop() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )
    result = builder.build("int x;", language="cpp")

    generated = generate_tokens(
        _tiny_model(),
        result.prompt_ids,
        max_new_tokens=1,
        temperature=0.0,
        model_kwargs=result.model_kwargs,
    )
    mx.eval(generated)

    assert generated.shape == (1, result.prompt_ids.shape[1] + 1)


def test_inference_side_channel_builder_fallback_policies_are_explicit() -> None:
    platform_context = {"os": "linux", "compiler": "gcc"}
    dropped = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FailingAdapter(),
        fail_policy="drop_family",
    ).build("int x;", platform_context=platform_context, language="cpp")
    text_only = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FailingAdapter(),
        fail_policy="text_only",
    ).build("int x;", platform_context=platform_context, language="cpp")

    assert dropped.provenance["adapter"].startswith("dropped:adapter_error")
    assert dropped.provenance["fallback"] == "drop_family"
    assert dropped.provenance["inference_fail_policy"] == "drop_family"
    assert dropped.provenance["inference_enrichment_status"] == (
        "degraded_drop_family"
    )
    assert dropped.provenance["inference_degraded"] == "true"
    assert dropped.provenance["inference_failure_reason"] == (
        "adapter_error:ValueError"
    )
    assert "platform_ids" in dropped.model_kwargs
    assert text_only.model_kwargs == {}
    assert text_only.side_channels == {}
    assert text_only.provenance == {
        "fallback": "text_only",
        "inference_fail_policy": "text_only",
        "inference_enrichment_status": "explicit_text_only",
        "inference_degraded": "true",
    }
    with pytest.raises(RuntimeError, match="adapter_error"):
        InferenceSideChannelBuilder(
            TinyTokenizer(),
            adapter=FailingAdapter(),
            fail_policy="error",
        ).build("int x;", platform_context=platform_context, language="cpp")


def test_inference_side_channel_builder_rejects_empty_policy_override() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )

    with pytest.raises(ValueError, match="fail_policy"):
        builder.build("int x;", language="cpp", fail_policy="")  # type: ignore[arg-type]


def test_inference_side_channel_builder_cache_key_uses_required_components() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )

    macos = builder.build("int x;", platform_context={"os": "macos"}, language="cpp")
    linux = builder.build("int x;", platform_context={"os": "linux"}, language="cpp")

    assert macos.cache_key != linux.cache_key
    assert macos.cache_components.content_sha256 == linux.cache_components.content_sha256
    assert macos.cache_components.tokenizer_id == "tiny-tokenizer"
    assert macos.cache_components.adapter_language == "cpp"
    assert macos.cache_components.adapter_version == "fake-clang-v1"
    assert "os=macos" in macos.cache_components.platform_context
    assert "os=linux" in linux.cache_components.platform_context


def test_inference_root_exports_side_channel_builder() -> None:
    assert inference.InferenceSideChannelBuilder is InferenceSideChannelBuilder
    assert "InferenceSideChannelBuilder" in inference.__all__
    assert "InferenceFailPolicy" in inference.__all__
    assert "get_builtin_code_metadata_adapter" in inference.__all__
    assert "PythonCodeMetadataAdapter" in inference.__all__
    assert "RustCodeMetadataAdapter" in inference.__all__
    assert "GoCodeMetadataAdapter" in inference.__all__


def test_builtin_language_adapter_registry_is_generic() -> None:
    registry = builtin_code_metadata_adapters()

    assert set(registry) == {"cpp", "rust", "go", "python"}
    assert normalize_code_language("c++") == "cpp"
    assert normalize_code_language("py") == "python"
    assert isinstance(registry["python"], PythonCodeMetadataAdapter)
    assert isinstance(registry["rust"], RustCodeMetadataAdapter)
    assert isinstance(registry["go"], GoCodeMetadataAdapter)
    for language, adapter in registry.items():
        assert isinstance(adapter, CodeMetadataAdapter)
        caps = adapter.probe({})
        assert caps.language == language
        assert caps.version
        assert caps.families
        assert isinstance(caps.available, bool)


def test_unknown_language_adapter_fails_explicitly() -> None:
    builder = InferenceSideChannelBuilder(TinyTokenizer())

    adapter = get_builtin_code_metadata_adapter("zig")
    caps = adapter.probe({})
    assert caps.available is False
    assert caps.reason == "adapter_unknown_language"

    with pytest.raises(RuntimeError, match="adapter_unavailable"):
        builder.build("const x = 1;", language="zig")

    dropped = InferenceSideChannelBuilder(
        TinyTokenizer(),
        fail_policy="drop_family",
    ).build("const x = 1;", language="zig")
    assert dropped.model_kwargs == {}
    assert dropped.side_channels == {}
    assert dropped.provenance["adapter"].startswith("dropped:adapter_unavailable")
    assert dropped.provenance["fallback"] == "drop_family"
    assert dropped.provenance["inference_degraded"] == "true"

    with pytest.raises(RuntimeError, match="adapter_unavailable"):
        InferenceSideChannelBuilder(
            TinyTokenizer(),
            fail_policy="error",
        ).build("const x = 1;", language="zig")


def _assert_builtin_language_adapter_conforms(
    language: str,
    source: str,
    expected_version: str,
) -> None:
    adapter = get_builtin_code_metadata_adapter(language)
    caps = adapter.probe({})

    assert caps.language == language
    assert caps.families == ("syntax", "structure")
    if not caps.available:
        result = InferenceSideChannelBuilder(
            TinyTokenizer(),
            fail_policy="drop_family",
        ).build(source, language=language)
        assert result.provenance["adapter"].startswith("dropped:adapter_unavailable")
        return

    result = InferenceSideChannelBuilder(TinyTokenizer()).build(
        source,
        language=language,
    )
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    assert result.cache_components.adapter_language == language
    assert result.cache_components.adapter_version == expected_version
    assert result.model_kwargs["structure_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["dep_levels"].shape == result.prompt_ids.shape
    assert result.model_kwargs["ast_depth_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["sibling_index_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["node_type_ids"].shape == result.prompt_ids.shape
    assert "structure" in result.side_channels
    assert "syntax" in result.side_channels
    assert result.provenance["adapter"] == f"{language}:{expected_version}"


def test_builtin_python_adapter_produces_token_metadata() -> None:
    _assert_builtin_language_adapter_conforms(
        "python",
        "def add(x: int) -> int:\n    return x + 1\n",
        "python-ast-v1",
    )


def test_builtin_python_adapter_maps_utf8_ast_offsets_to_char_positions() -> None:
    source = "éé = 1\nx = 2\n"
    metadata = PythonCodeMetadataAdapter().extract(source, {})

    assert metadata.node_type[0] != 0
    assert metadata.node_type[source.index("\n")] == 0


def test_builtin_rust_adapter_conforms_or_reports_unavailable() -> None:
    _assert_builtin_language_adapter_conforms(
        "rust",
        "pub fn add(x: i32) -> i32 { x + 1 }\n",
        "rust-syntax-v1",
    )


def test_builtin_go_adapter_conforms_or_reports_unavailable() -> None:
    _assert_builtin_language_adapter_conforms(
        "go",
        "package main\nfunc add(x int) int { return x + 1 }\n",
        "go-syntax-v1",
    )


def test_builtin_cpp_adapter_conforms_or_reports_unavailable() -> None:
    adapter = get_builtin_code_metadata_adapter("cpp")
    caps = adapter.probe({})

    assert caps.language == "cpp"
    assert caps.families == ("syntax", "structure")
    if not caps.available:
        result = InferenceSideChannelBuilder(
            TinyTokenizer(),
            fail_policy="drop_family",
        ).build(
            "int main() { return 0; }\n",
            language="cpp",
        )
        assert result.provenance["adapter"].startswith("dropped:adapter_unavailable")
        return

    result = InferenceSideChannelBuilder(TinyTokenizer()).build(
        "int add(int x) { return x + 1; }\n",
        language="cpp",
    )
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    assert result.cache_components.adapter_language == "cpp"
    assert result.cache_components.adapter_version == "clang-ast-v1"
    assert result.model_kwargs["structure_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["ast_depth_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["sibling_index_ids"].shape == result.prompt_ids.shape
    assert result.model_kwargs["node_type_ids"].shape == result.prompt_ids.shape
    assert "structure" in result.side_channels
    assert "syntax" in result.side_channels
    assert result.provenance["adapter"] == "cpp:clang-ast-v1"
