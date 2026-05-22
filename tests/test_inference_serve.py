from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

import cppmega_mlx.inference as inference

TestClient: Any = import_module("fastapi.testclient").TestClient


def _as_numpy(tokens: mx.array) -> np.ndarray:
    mx.eval(tokens)
    return np.array(tokens)


class _ScriptedLogitsModel:
    def __init__(
        self,
        next_ids_by_step: Sequence[Sequence[int]],
        *,
        vocab_size: int = 16,
        max_seq_length: int | None = None,
    ) -> None:
        self.next_ids_by_step = next_ids_by_step
        self.vocab_size = vocab_size
        self.calls = 0
        self.seen_shapes: list[tuple[int, int]] = []
        self.seen_model_kwargs: list[dict[str, np.ndarray]] = []
        if max_seq_length is not None:
            self.config = SimpleNamespace(max_seq_length=max_seq_length)

    def __call__(self, tokens: mx.array, **model_kwargs: mx.array) -> mx.array:
        batch_size, sequence_length = tokens.shape
        self.seen_shapes.append((batch_size, sequence_length))
        self.seen_model_kwargs.append(
            {key: _as_numpy(value) for key, value in sorted(model_kwargs.items())}
        )
        step = min(self.calls, len(self.next_ids_by_step) - 1)
        self.calls += 1

        next_ids = self.next_ids_by_step[step]
        assert len(next_ids) == batch_size
        logits = np.full(
            (batch_size, sequence_length, self.vocab_size),
            -1000.0,
            dtype=np.float32,
        )
        for row, token_id in enumerate(next_ids):
            logits[row, -1, token_id] = 1000.0
        return mx.array(logits)


def test_local_generation_app_health_and_token_id_generation() -> None:
    app = inference.create_local_generation_app(
        _ScriptedLogitsModel([[4], [5]]),
        model_id="tiny-local",
        decode_token=lambda token_id: chr(ord("a") + token_id),
    )
    client = TestClient(app)

    assert client.get("/health").json() == {
        "status": "ok",
        "model_id": "tiny-local",
    }

    response = client.post(
        "/generate",
        json={
            "input_ids": [1, 2],
            "max_new_tokens": 2,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "tiny-local",
        "input_ids": [[1, 2]],
        "output_ids": [[1, 2, 4, 5]],
        "generated_ids": [[4, 5]],
        "finish_reason": "length",
        "generated_text": ["ef"],
    }


def test_local_generation_app_accepts_batched_rectangular_token_ids() -> None:
    app = inference.create_local_generation_app(_ScriptedLogitsModel([[4, 5]]))
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "input_ids": [[1, 2], [3, 4]],
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["output_ids"] == [[1, 2, 4], [3, 4, 5]]


def test_local_generation_app_threads_model_kwargs_builder() -> None:
    model = _ScriptedLogitsModel([[4]])
    seen_rows: list[list[list[int]]] = []

    def build_model_kwargs(input_rows: list[list[int]]) -> dict[str, mx.array]:
        seen_rows.append(input_rows)
        return {"platform_ids": mx.array([[3, 64, 0]], dtype=mx.int32)}

    app = inference.create_local_generation_app(
        model,
        model_kwargs_builder=build_model_kwargs,
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "input_ids": [1, 2],
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    assert seen_rows == [[[1, 2]]]
    np.testing.assert_array_equal(
        model.seen_model_kwargs[0]["platform_ids"],
        np.array([[3, 64, 0]], dtype=np.int32),
    )


def test_local_generation_app_returns_eos_finish_reason() -> None:
    app = inference.create_local_generation_app(
        _ScriptedLogitsModel([[2]]),
        eos_token_id=2,
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "input_ids": [1],
            "max_new_tokens": 4,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["generated_ids"] == [[2]]
    assert response.json()["finish_reason"] == "eos"


def test_local_generation_app_rejects_invalid_requests() -> None:
    client = TestClient(inference.create_local_generation_app(_ScriptedLogitsModel([[4]])))

    bad_requests = [
        {"input_ids": []},
        {"input_ids": [[1], [2, 3]]},
        {"input_ids": [1, True]},
        {"input_ids": [1], "max_new_tokens": -1},
        {"input_ids": [1], "temperature": -0.5},
        {"input_ids": [1], "top_p": 1.5},
    ]

    for request_json in bad_requests:
        response = client.post("/generate", json=request_json)
        assert response.status_code == 400, request_json
