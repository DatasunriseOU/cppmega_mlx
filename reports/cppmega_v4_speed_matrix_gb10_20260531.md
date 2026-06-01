# cppmega v4 Speed Matrix (op-level GDN/KDA) — gb10 (CUDA sm_121, GB10)

- Date: 20260531
- Host: Linux-6.17.0-1021-nvidia-aarch64-with-glibc2.39
- cppmega SHA: `3ef4894`
- Python: 3.13.11
- Op measured: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), NOT full m04 1B training (the v4 benchmark harness is op-level).
- Shape: B=1 T=512 H=8 Dk=64 Dv=64 (Path-E-eligible: Dk%32==0, Dv%4==0)
- Sweep: block {gdn,kda} x path {a,b,c,d,e} x dtype {f32,bf16,fp16,fp16d}. warmup=2, iters=5, median of 5. allow_fallback=DISABLED (fail-loud).
- dtype axis: f32=harness reference; bf16=REAL cast of q/k/v/beta/g. fp8/nvfp4 are NOT representable as MLX op-input array dtypes (no float8 array dtype; only to_fp8/from_fp8 packers) so they do not apply at the op level and are omitted (the v3 fp8 axis was an m04-training GEMM storage scheme, a different layer).
- Per-cell bound: 1800s OS timeout, fresh subprocess (GPU cleared on exit).

| block | dtype | path | status | measured_path | median fwd ms | throughput Melem/s | reason |
| ----- | ----- | ---- | ------ | ------------- | ------------: | -----------------: | ------ |
| gdn | f32 | a | ok | a | 32.214 | 8.14 | pure-MLX reference |
| gdn | f32 | b | ok | b | 1.204 | 217.73 | GDN Path B forward via TileLang-CUDA EAGER bridge (gdn_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| gdn | f32 | c | ok | c | 1.204 | 217.80 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | f32 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | f32 | e | ok | e | 16.385 | 16.00 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | bf16 | a | ok | a | 29.878 | 8.77 | pure-MLX reference |
| gdn | bf16 | b | ok | b | 1.292 | 202.93 | GDN Path B forward via TileLang-CUDA EAGER bridge (gdn_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| gdn | bf16 | c | ok | c | 1.573 | 166.69 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | bf16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | bf16 | e | ok | e | 19.809 | 13.23 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16 | a | ok | a | 24.987 | 10.49 | pure-MLX reference |
| gdn | fp16 | b | ok | b | 2.107 | 124.42 | GDN Path B forward via TileLang-CUDA EAGER bridge (gdn_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| gdn | fp16 | c | ok | c | 2.006 | 130.68 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16 | e | ok | e | 20.357 | 12.88 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16d | a | ok | a | 33.197 | 7.90 | pure-MLX reference |
| gdn | fp16d | b | ok | b | 1.431 | 183.14 | GDN Path B forward via TileLang-CUDA EAGER bridge (gdn_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| gdn | fp16d | c | ok | c | 1.435 | 182.71 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16d | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16d | e | ok | e | 20.063 | 13.07 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| kda | f32 | a | ok | a | 33.155 | 7.91 | pure-MLX KDA reference |
| kda | f32 | b | ok | b | 1.563 | 167.77 | KDA Path B forward via TileLang-CUDA EAGER bridge (kda_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| kda | f32 | c | ok | c | 1.629 | 160.97 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | f32 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | f32 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | bf16 | a | ok | a | 25.621 | 10.23 | pure-MLX KDA reference |
| kda | bf16 | b | ok | b | 2.819 | 92.99 | KDA Path B forward via TileLang-CUDA EAGER bridge (kda_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| kda | bf16 | c | ok | c | 1.827 | 143.48 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | bf16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | bf16 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | fp16 | a | ok | a | 29.804 | 8.80 | pure-MLX KDA reference |
| kda | fp16 | b | ok | b | 2.346 | 111.76 | KDA Path B forward via TileLang-CUDA EAGER bridge (kda_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| kda | fp16 | c | ok | c | 1.965 | 133.38 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | fp16d | a | ok | a | 25.661 | 10.22 | pure-MLX KDA reference |
| kda | fp16d | b | ok | b | 1.854 | 141.39 | KDA Path B forward via TileLang-CUDA EAGER bridge (kda_fwd_cuda_eager; Metal unavailable on this CUDA host). TileLang-CUDA EAGER path ready |
| kda | fp16d | c | ok | c | 1.852 | 141.55 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16d | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16d | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |

## Per-Cell Commands

- `gdn_f32_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
