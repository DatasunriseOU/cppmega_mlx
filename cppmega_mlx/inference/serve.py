"""Local FastAPI adapter for token-id generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, NoReturn

import mlx.core as mx

from cppmega_mlx.inference.generation import generate_tokens

ModelKwargsBuilder = Callable[[list[list[int]]], Mapping[str, mx.array]]
TokenDecoder = Callable[[int], str]


def create_local_generation_app(
    model: Any,
    *,
    model_id: str = "local",
    eos_token_id: int | None = None,
    decode_token: TokenDecoder | None = None,
    model_kwargs_builder: ModelKwargsBuilder | None = None,
) -> Any:
    """Create the Mac-local token-id serving app.

    This is intentionally not an OpenAI-compatible chat/completions API. It is
    the local Stream I adapter around the eager token-id generation loop, so
    callers keep ownership of tokenization and optional side-channel metadata.
    """

    FastAPI, HTTPException = _load_fastapi()
    app = FastAPI(title="cppmega.mlx local generation", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_id": model_id}

    @app.post("/generate")
    async def generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            input_rows = _parse_input_ids(payload.get("input_ids"))
            max_new_tokens = _parse_int_option(
                payload,
                "max_new_tokens",
                default=16,
                minimum=0,
            )
            temperature = _parse_float_option(
                payload,
                "temperature",
                default=1.0,
                minimum=0.0,
            )
            top_k = _parse_optional_int_option(
                payload,
                "top_k",
                minimum=1,
            )
            top_p = _parse_optional_float_option(
                payload,
                "top_p",
                default=1.0,
                minimum_exclusive=0.0,
                maximum=1.0,
            )
            request_eos_token_id = _parse_optional_int_option(
                payload,
                "eos_token_id",
                default=eos_token_id,
                minimum=0,
            )
            model_kwargs = _build_model_kwargs(model_kwargs_builder, input_rows)
            prompt = mx.array(input_rows, dtype=mx.int32)
            output = generate_tokens(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                model_kwargs=model_kwargs,
                eos_token_id=request_eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        except _BadGenerationRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        mx.eval(output)
        output_rows = _coerce_nested_int_rows(output.tolist())
        prompt_width = len(input_rows[0])
        generated_rows = [row[prompt_width:] for row in output_rows]
        response: dict[str, Any] = {
            "model": model_id,
            "input_ids": input_rows,
            "output_ids": output_rows,
            "generated_ids": generated_rows,
            "finish_reason": _finish_reason(generated_rows, request_eos_token_id),
        }
        if decode_token is not None:
            response["generated_text"] = [
                "".join(decode_token(token_id) for token_id in row)
                for row in generated_rows
            ]
        return response

    return app


class _BadGenerationRequest(ValueError):
    pass


def _load_fastapi() -> tuple[Any, Any]:
    try:
        fastapi = import_module("fastapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "fastapi is required for create_local_generation_app; install the "
            "cppmega-mlx[gui] extra or provide FastAPI in the serving environment"
        ) from exc
    return fastapi.FastAPI, fastapi.HTTPException


def _parse_input_ids(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        _raise_bad_request("input_ids must be list[int] or list[list[int]]")
    if not value:
        _raise_bad_request("input_ids must not be empty")

    if all(_is_token_id(token_id) for token_id in value):
        rows = [list(value)]
    elif all(isinstance(row, list) for row in value):
        rows = [list(row) for row in value]
    else:
        _raise_bad_request("input_ids must be list[int] or list[list[int]]")

    if not rows or not rows[0]:
        _raise_bad_request("input_ids rows must not be empty")
    row_width = len(rows[0])
    for row in rows:
        if len(row) != row_width:
            _raise_bad_request("input_ids rows must have the same length")
        if not all(_is_token_id(token_id) for token_id in row):
            _raise_bad_request("input_ids must contain non-negative integer token ids")
    return rows


def _build_model_kwargs(
    builder: ModelKwargsBuilder | None,
    input_rows: list[list[int]],
) -> Mapping[str, mx.array] | None:
    if builder is None:
        return None
    built = builder([row.copy() for row in input_rows])
    if not isinstance(built, Mapping):
        raise ValueError("model_kwargs_builder must return a mapping")
    for key, value in built.items():
        if not isinstance(key, str):
            raise ValueError("model_kwargs_builder keys must be strings")
        if not isinstance(value, mx.array):
            raise ValueError("model_kwargs_builder values must be mlx arrays")
    return built


def _parse_int_option(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = payload.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        _raise_bad_request(f"{name} must be an integer")
    if value < minimum:
        _raise_bad_request(f"{name} must be >= {minimum}")
    return value


def _parse_optional_int_option(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: int | None = None,
    minimum: int,
) -> int | None:
    value = payload.get(name, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _raise_bad_request(f"{name} must be an integer or null")
    if value < minimum:
        _raise_bad_request(f"{name} must be >= {minimum}")
    return value


def _parse_float_option(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: float,
    minimum: float,
) -> float:
    value = payload.get(name, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        _raise_bad_request(f"{name} must be numeric")
    result = float(value)
    if result < minimum:
        _raise_bad_request(f"{name} must be >= {minimum}")
    return result


def _parse_optional_float_option(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: float | None = None,
    minimum_exclusive: float,
    maximum: float,
) -> float | None:
    value = payload.get(name, default)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        _raise_bad_request(f"{name} must be numeric or null")
    result = float(value)
    if not minimum_exclusive < result <= maximum:
        _raise_bad_request(f"{name} must be in ({minimum_exclusive}, {maximum}]")
    return result


def _finish_reason(
    generated_rows: list[list[int]],
    eos_token_id: int | None,
) -> str:
    if eos_token_id is not None and generated_rows:
        if all(row and row[-1] == eos_token_id for row in generated_rows):
            return "eos"
    return "length"


def _coerce_nested_int_rows(rows: Any) -> list[list[int]]:
    if not isinstance(rows, list):
        raise TypeError("generated tokens must serialize to nested rows")
    return [[int(token_id) for token_id in row] for row in rows]


def _is_token_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _raise_bad_request(detail: str) -> NoReturn:
    raise _BadGenerationRequest(detail)
