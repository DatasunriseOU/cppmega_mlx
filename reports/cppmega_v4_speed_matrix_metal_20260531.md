# cppmega v4 Speed Matrix (op-level GDN/KDA) — Local Metal (Apple M4 Max)

- Date: 20260531
- Host: macOS-26.5-arm64-arm-64bit-Mach-O
- cppmega SHA: `c9b5aac`
- Python: 3.13.12
- Op measured: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), NOT full m04 1B training (the v4 benchmark harness is op-level).
- Shape: B=1 T=512 H=8 Dk=64 Dv=64 (Path-E-eligible: Dk%32==0, Dv%4==0)
- Sweep: block {gdn,kda} x path {a,b,c,d,e} x dtype {f32,bf16,fp16,fp16d}. warmup=2, iters=5, median of 5. allow_fallback=DISABLED (fail-loud).
- dtype axis: f32=harness reference; bf16=REAL cast of q/k/v/beta/g. fp8/nvfp4 are NOT representable as MLX op-input array dtypes (no float8 array dtype; only to_fp8/from_fp8 packers) so they do not apply at the op level and are omitted (the v3 fp8 axis was an m04-training GEMM storage scheme, a different layer).
- Per-cell bound: 1800s OS timeout, fresh subprocess (GPU cleared on exit).

| block | dtype | path | status | measured_path | median fwd ms | throughput Melem/s | reason |
| ----- | ----- | ---- | ------ | ------------- | ------------: | -----------------: | ------ |
| gdn | f32 | a | ok | a | 24.358 | 10.76 | pure-MLX reference |
| gdn | f32 | b | ok | b | 2.149 | 121.97 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | f32 | c | ok | c | 3.210 | 81.68 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | f32 | d | failed | d | — | — | RuntimeError: GDN dispatch: selected path_d crashed at runtime (PathDRuntimeUnavailable: GDN Path D runtime adapter only supports q.dtype=mlx.core.float16; got  |
| gdn | f32 | e | ok | e | 1.133 | 231.47 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | bf16 | a | ok | a | 25.958 | 10.10 | pure-MLX reference |
| gdn | bf16 | b | ok | b | 2.051 | 127.80 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | bf16 | c | ok | c | 4.477 | 58.56 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | bf16 | d | failed | d | — | — | RuntimeError: GDN dispatch: selected path_d crashed at runtime (PathDRuntimeUnavailable: GDN Path D runtime adapter only supports q.dtype=mlx.core.float16; got  |
| gdn | bf16 | e | ok | e | 1.381 | 189.89 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16 | a | ok | a | 26.653 | 9.84 | pure-MLX reference |
| gdn | fp16 | b | ok | b | 3.594 | 72.94 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | fp16 | c | ok | c | 4.867 | 53.86 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. Path D nativ |
| gdn | fp16 | e | ok | e | 1.286 | 203.88 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16d | a | ok | a | 23.023 | 11.39 | pure-MLX reference |
| gdn | fp16d | b | ok | b | 2.466 | 106.31 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | fp16d | c | ok | c | 2.300 | 113.99 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16d | d | ok | d | 567.397 | 0.46 | GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. GDN Path D runtime adapter available for shape-specialized fp16 prefill  |
| gdn | fp16d | e | ok | e | 1.333 | 196.65 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| kda | f32 | a | ok | a | 21.217 | 12.36 | pure-MLX KDA reference |
| kda | f32 | b | ok | b | 3.011 | 87.07 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | f32 | c | ok | c | 3.668 | 71.46 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | f32 | d | failed | d | — | — | RuntimeError: KDA dispatch: selected path_d crashed at runtime (PathDRuntimeUnavailable: KDA Path D runtime adapter only supports q.dtype=mlx.core.float16; got  |
| kda | f32 | e | ok | e | 1.078 | 243.23 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | bf16 | a | ok | a | 49.608 | 5.28 | pure-MLX KDA reference |
| kda | bf16 | b | ok | b | 2.765 | 94.81 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | bf16 | c | ok | c | 11.944 | 21.95 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | bf16 | d | failed | d | — | — | RuntimeError: KDA dispatch: selected path_d crashed at runtime (PathDRuntimeUnavailable: KDA Path D runtime adapter only supports q.dtype=mlx.core.float16; got  |
| kda | bf16 | e | ok | e | 1.045 | 250.81 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | fp16 | a | ok | a | 21.750 | 12.05 | pure-MLX KDA reference |
| kda | fp16 | b | ok | b | 2.811 | 93.26 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | fp16 | c | ok | c | 2.764 | 94.86 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16 | d | failed | d | — | — | RuntimeError: KDA dispatch: selected path_d crashed at runtime (PathDRuntimeUnavailable: KDA Path D runtime adapter only supports g.dtype=mlx.core.float32; got  |
| kda | fp16 | e | ok | e | 1.378 | 190.24 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | fp16d | a | ok | a | 25.201 | 10.40 | pure-MLX KDA reference |
| kda | fp16d | b | ok | b | 2.796 | 93.75 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | fp16d | c | ok | c | 3.404 | 77.02 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16d | d | failed | d | — | — | RuntimeError: KDA dispatch: selected path_d crashed at runtime (RuntimeError: TVM-FFI function call failed inside MLX graph eval: Traceback (most recent call la |
| kda | fp16d | e | ok | e | 1.021 | 256.64 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |

## Per-Cell Commands

- `gdn_f32_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
