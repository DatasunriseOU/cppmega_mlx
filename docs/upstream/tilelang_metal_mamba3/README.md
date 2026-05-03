# TileLang Metal Mamba3 Path C Profile

Scope: local M4 Max validation of Mamba3 Path C, the TileLang
`@T.prim_func` form lowered to Metal and dispatched through MLX
`mx.fast.metal_kernel`, against Path B, the hand-written MSL kernel.

Environment:
- Host: `Davids-Mac-Studio.local`
- Device: Apple M4 Max, `applegpu_g16s`
- Python: 3.13.12
- MLX: 0.31.1
- TileLang: 0.1.9+gita69d6df7

## Commands

Correctness:

```bash
.venv/bin/python -m pytest tests/test_tilelang_mamba3_path_c.py -q
```

Benchmarks:

```bash
.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --seq 128 --warmup 10 --iters 50 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_seq128_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_seq128.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_seq128.diff

.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --warmup 10 --iters 50 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_spec_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_spec.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_spec.diff

.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --seq 1024 --warmup 10 --iters 50 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_seq1024_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_seq1024.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_seq1024.diff
```

Profiler captures require Metal capture to be enabled:

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python <capture script>
```

Markers are the separate capture filenames:
- `captures/mamba3_spec_path_b_fwd.gputrace`
- `captures/mamba3_spec_path_c_fwd.gputrace`
- `captures/mamba3_spec_path_b_bwd.gputrace`
- `captures/mamba3_spec_path_c_bwd.gputrace`

## Correctness

`tests/test_tilelang_mamba3_path_c.py` passed: 11 passed, 2 TileLang
builder deprecation warnings.

All benchmark receipts reported bit-identical forward parity against Path B:

| Shape | `y_max_abs` | `h_max_abs` |
| --- | ---: | ---: |
| B=2 T=128 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |
| B=2 T=512 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |
| B=2 T=1024 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |

## Timings

Median timings from same-process Path B/Path C matched runs:

| Shape | Metric | Path B | Path C | C/B |
| --- | --- | ---: | ---: | ---: |
| T=128 | fwd | 0.555 ms | 0.470 ms | 0.847 |
| T=128 | bwd | 1.317 ms | 1.059 ms | 0.805 |
| T=128 | fwd+bwd | 1.871 ms | 1.529 ms | 0.817 |
| T=512 | fwd | 1.122 ms | 0.990 ms | 0.882 |
| T=512 | bwd | 6.416 ms | 5.146 ms | 0.802 |
| T=512 | fwd+bwd | 7.538 ms | 6.136 ms | 0.814 |
| T=1024 | fwd | 1.869 ms | 1.907 ms | 1.021 |
| T=1024 | bwd | 12.214 ms | 10.032 ms | 0.821 |
| T=1024 | fwd+bwd | 14.083 ms | 11.939 ms | 0.848 |

Peak memory matched Path B at every measured shape:

| Shape | fwd peak | fwd+bwd peak |
| --- | ---: | ---: |
| T=128 | 1.38 MB | 26.83 MB |
| T=512 | 4.78 MB | 106.38 MB |
| T=1024 | 9.31 MB | 212.44 MB |

Interpretation: the original table claim is now stale for Mamba3. With the
checked-in 32-thread Path C policy, Path C is bit-exact and faster than Path B
on fwd+bwd across the checked M4 Max shapes, with unchanged peak memory.

## Profiler Captures

Programmatic MLX capture only emitted files when `MTL_CAPTURE_ENABLED=1` was set.
The initial attempt without that environment variable produced no trace files.

Existing pre-32-thread traces:

| Marker filename | Size | Scope |
| --- | ---: | --- |
| `captures/mamba3_spec_path_b_fwd.gputrace` | 136184234 bytes | spec-shape Path B forward |
| `captures/mamba3_spec_path_c_fwd.gputrace` | 136183768 bytes | spec-shape Path C forward |
| `captures/mamba3_spec_path_b_bwd.gputrace` | 352162953 bytes | spec-shape Path B fwd+bwd |
| `captures/mamba3_spec_path_c_bwd.gputrace` | 352161968 bytes | spec-shape Path C fwd+bwd |

## Bottlenecks

Path C is not a separate runtime path. It lowers TileLang DSL to MSL and then
uses the same MLX `mx.fast.metal_kernel` dispatcher as Path B.

The lowered MSL is structurally the same per-lane scan:
- One thread owns each `(batch, head, headdim)` lane.
- Time is serial inside the thread.
- State dimension is serial inside the thread with `thread float h_state[64]`.
- Forward uses per-timestep `exp` for decay and another `exp` for the sigmoid.
- Backward rematerializes `h_steps` into global scratch, then performs a serial
  reverse scan and emits per-lane partial gradients.
- Host-side MLX reductions over the `P` axis are still needed for `dB`, `dC`,
  `dA`, `ddt`, and `dD`.

The dominant cost is therefore the same in Path B and Path C: serial recurrent
work per lane plus large backward scratch traffic. The current Path C win comes
from avoiding the register-pressure/occupancy cliff in the backward replay and
reverse-scan kernel, not from a different algorithm.

## Threadgroup Tuning

A pre-patch Python process monkeypatched `_threads_for` in memory and rebuilt
the Path C shape-specialized kernels for the spec shape.

| Path C threads | fwd median | fwd+bwd median | Notes |
| ---: | ---: | ---: | --- |
| 64 | 1.012 ms | 6.452 ms | Best fwd+bwd in this probe |
| 128 | 0.980 ms | 6.681 ms | Best fwd-only in this probe |
| 256 | 1.065 ms | 7.828 ms | Old checked-in policy |

The production patch now caps Path C at 32 threads. The full three-shape matrix
above is the acceptance evidence: 32 threads keeps one Apple SIMD group active
while reducing the per-thread register-state occupancy cliff relative to the
old 256-thread launch.

## Verdict

Path C is now correct, stable, and faster than Path B for Mamba3 fwd+bwd on the
checked M4 Max FP32 shapes. The minimal patch is the 32-thread launch cap plus
the benchmark verdict fix; no TileLang Metal codegen change is required for this
lane.
