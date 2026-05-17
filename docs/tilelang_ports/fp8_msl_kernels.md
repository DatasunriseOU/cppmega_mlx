# FP8 Reference Helpers (retired direct-MSL)

- Module: `cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py`
- Tests: `tests/test_fp8_msl_kernels.py`
- Bench: `scripts/bench_fp8_msl_kernels.py` -> `bench/tilelang_ports/fp8_msl_kernels.json`

## Current Status

The old vendored FP8 direct-MSL Path B surface is retired. The module no longer
embeds Metal source, constructs raw kernels, or dispatches through the legacy
MSL transform. `fp8_msl_status()` returns `available=False` with the retired
Path B reason even on Metal-capable hosts.

The public helper names remain because tests and receipt harnesses need a stable
math oracle:

| Helper | Current implementation | Purpose |
| --- | --- | --- |
| `fp8_to_half` | `mx.from_fp8(..., dtype=mx.float16)` | decode reference |
| `half_to_fp8` | `mx.to_fp8(fp16.astype(mx.float32))` | encode reference |
| `fp8_scaled_matmul_raw` | dequantize + `mx.matmul` + scales | reference baseline |
| `fp8_scaled_matmul` | custom-function wrapper over the reference baseline | VJP coverage |
| `fp8_scaled_vecmat` | dequantize + `mx.matmul` + scales | M=1 vecmat oracle |

The production framework route is the owner-output TileLang/tvm-ffi surface in
`fp8_matmul_path_c.py` and `fp8_vecmat_path_c.py`. The helpers here are not an
acceleration path and must not be treated as an AUTO production fallback.

## Source Attribution

Historical versions of this module carried direct ports of AppMana
`mps-fp8-for-torch-and-comfyui-python-package` (Apache 2.0, commit `a902571e`)
and `audiohacking/fp8-mps-metal` (MIT, commit `d4fbd40c`). The current module
does not embed those sources, but the `__license_notice__` remains so old binary
attribution stays visible where needed.

## Receipt

The checked receipt now records the retired direct-MSL status and pure-MLX
reference timings. Use it as a cleanup/audit receipt, not as proof of a fast FP8
kernel.
