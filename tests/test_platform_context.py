from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from cppmega_mlx.data.platform_context import (
    MAX_PLATFORM_IDS,
    PLATFORM_VOCAB,
    PLATFORM_VOCAB_SIZE,
    PlatformContext,
    encode_platform_context,
    platform_ids_array,
    parse_platform_context,
    render_platform_context,
)


def _load_nanochat_platform_vocab() -> ModuleType:
    candidates = (
        Path("/Users/dave/sources/nanochat/nanochat/platform_vocab.py"),
        Path(__file__).resolve().parents[1].parent
        / "nanochat"
        / "nanochat"
        / "platform_vocab.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        pytest.skip("nanochat platform_vocab.py checkout is not available")
    spec = importlib.util.spec_from_file_location("_nanochat_platform_vocab", source)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load nanochat platform vocab from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_context_matches_nanochat_platform_vocab_source_of_truth() -> None:
    nanochat_vocab = _load_nanochat_platform_vocab()
    platform_info = {
        "os": ["macos"],
        "gpu": ["metal"],
        "arch": ["arm64"],
        "compiler": ["clang"],
        "cpp_std": "c++20",
    }

    assert PLATFORM_VOCAB == nanochat_vocab.PLATFORM_VOCAB
    assert PLATFORM_VOCAB_SIZE == nanochat_vocab.PLATFORM_VOCAB_SIZE
    assert MAX_PLATFORM_IDS == nanochat_vocab.MAX_PLATFORM_IDS
    assert encode_platform_context(platform_info) == tuple(
        nanochat_vocab.platform_info_to_ids(platform_info)
    )


def test_platform_context_parse_render_encode_round_trip() -> None:
    context = parse_platform_context(
        {
            "language": "C++",
            "os": "Darwin",
            "arch": "aarch64",
            "compiler": "AppleClang",
            "cpp_standard": "c++20",
            "stdlib": "libc++",
            "accelerator": "Metal",
            "backend": "MLX",
            "target_triple": "arm64-apple-darwin",
        }
    )

    assert context == PlatformContext(
        language="cpp",
        os=("macos",),
        arch=("arm64",),
        compiler=("clang",),
        cpp_std="c++20",
        stdlib=("libc++",),
        gpu=("metal",),
        backend=("mlx",),
        target_triple="arm64-apple-darwin",
    )

    rendered = render_platform_context(context)
    assert "os=macos" in rendered
    assert "arch=arm64" in rendered
    assert "compiler=clang" in rendered
    assert "cpp_std=c++20" in rendered
    assert "stdlib=libc++" in rendered
    assert "gpu=metal" in rendered
    assert "backend=mlx" in rendered

    assert parse_platform_context(rendered) == context

    ids = set(encode_platform_context(context))
    assert PLATFORM_VOCAB["os"]["macos"] in ids
    assert PLATFORM_VOCAB["arch"]["arm64"] in ids
    assert PLATFORM_VOCAB["compiler"]["clang"] in ids
    assert PLATFORM_VOCAB["cpp_std"]["c++20"] in ids
    assert PLATFORM_VOCAB["gpu"]["metal"] in ids
    assert max(ids) < PLATFORM_VOCAB_SIZE
    assert "stdlib" not in PLATFORM_VOCAB
    assert "backend" not in PLATFORM_VOCAB


def test_platform_ids_array_pads_unknown_or_missing_context() -> None:
    ids = platform_ids_array(
        [
            parse_platform_context({"os": "linux", "compiler": "gcc"}),
            parse_platform_context(None),
        ],
        width=5,
    )

    assert ids.dtype == np.int32
    assert ids.shape == (2, 5)
    assert ids[0, 0] != 0
    assert np.count_nonzero(ids[1]) == 0
