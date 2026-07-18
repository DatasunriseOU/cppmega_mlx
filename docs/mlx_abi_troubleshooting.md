# MLX venv-vs-brew ABI mismatch — troubleshooting

## Symptom

`cppmega_mlx` engine-path tests silently skip on a Mac host that has both:
- a venv-installed `mlx` (via `pip install mlx`)
- a brew-installed `mlx` (via `brew install mlx`)

The venv's `mlx.core.so` is built against one `libmlx.dylib` but DYLD's
search order picks up a different version from
`/opt/homebrew/Cellar/mlx/<X.Y.Z>/lib/libmlx.dylib`. The result is an
`ImportError`, an `OSError: dyld: Symbol not found`, or — most insidiously —
an apparent successful import that fails at first kernel-launch with
`AbortError`.

Test runners catch this via `pytest.importorskip("mlx.core")` and silently
skip — empirical test matrices then look "GREEN" while validating nothing.

## Diagnosis

Run `scripts/check_mlx_abi.sh`. It defaults to the dedicated environment and
does not read the checkout `.venv` symlink:

    $ ./scripts/check_mlx_abi.sh
    mode: installed_wheel
    python: /Volumes/external/sources/.venvs/cppmega.mlx/bin/python
    mlx package: 0.32.0
    mlx-metal: 0.32.0
    libmlx: /Volumes/external/sources/.venvs/cppmega.mlx/.../mlx/lib/libmlx.dylib
    MLX runtime smoke: mx.eval([1, 2] + 1) produced [2, 3]
    mlx-metal release contract: 0.32.0

Exit code 0 means the package contract is valid; 1 means the environment must
be repaired. A workspace MLX checkout is reported separately and is compared
to its `mlx-metal` release package, not to a Homebrew version string.
`--json` always emits a `cppmega_mlx_abi_receipt` with schema version 1,
including early interpreter/import failures. The selected interpreter and env
root must match the runtime `sys.executable` and `sys.prefix` exactly.

On Homebrew systems the loaded `libmlx.dylib` is authoritative. Its resolved
Cellar keg is checked before the currently linked `opt/mlx` keg; a stale loaded
keg cannot be hidden by relinking Homebrew after process startup. A Homebrew
keg that is installed but not loaded is informational only.

Ambient MLX source paths are ignored. Source-mode diagnosis is explicit:

    ./scripts/check_mlx_abi.sh --source-root /Volumes/external/sources/mlx

## Path-C source environment

Use the dedicated launcher when Path-C must run against the sibling source
TileLang stack:

    $ ./scripts/run_mlx_path_c.py --verify-only
    contract: path_c_source
    python: /Volumes/external/sources/.venvs/cppmega.mlx/bin/python
    mlx_mode: installed_wheel
    tilelang_mode: workspace_source
    PASS

The default command above is expected to exit with `1` when any selected
TileLang/TVM/TVM-FFI checkout is dirty. Do not remove that gate to make a local
run green; use the explicit development opt-in shown below only when the dirty
source tree is the object being tested.

The launcher starts from a clean environment contract. It replaces ambient
`PYTHONPATH`, `VIRTUAL_ENV`, `TL_*`, `TILELANG_*`, `TVM_*`, `DYLD_*`,
`MLX_*`, `MTL_*`, and include/library paths with values derived from:

- `/Volumes/external/sources/.venvs/cppmega.mlx`
- `/Volumes/external/sources/tilelang`
- `/Volumes/external/sources/tilelang/3rdparty/tvm`
- the vendored TVM-FFI source/build tree

The schema-versioned receipt records the active TileLang, TVM, and TVM-FFI Git
revisions, dirty state, and a SHA-256 worktree digest that covers tracked diffs
and untracked content. Dirty or unreadable source state fails closed before the
launcher imports that source code. A local development run may opt in
explicitly with `--allow-dirty-source-stack`; the receipt records that exception
and the exact dirty digest. It also fails closed unless MLX and
`libmlx.dylib` come from the dedicated wheel environment while TileLang, TVM,
TVM-FFI Python, its native extension, and the loaded runtime libraries come
from the selected source tree. The active TileLang source release must also
match the pinned TileLang wheel release.

Launch a Path-C command after the verifier with `--`:

    ./scripts/run_mlx_path_c.py -- \
      python scripts/bench_tilelang_fp8_path_c.py --help

    ./scripts/run_mlx_path_c.py -- \
      python -m pytest -q tests/test_tilelang_bench_harness.py

For an intentionally modified local TileLang/TVM checkout:

    ./scripts/run_mlx_path_c.py --allow-dirty-source-stack --verify-only

Without `run_mlx_path_c.py`, the dedicated environment remains ordinary wheel
mode: `mlx==0.32.0`, `mlx-metal==0.32.0`, and the locked TileLang wheel. The
launcher overlays only the TileLang/TVM/TVM-FFI source stack; it never adds the
workspace MLX checkout to `PYTHONPATH`.

## Fix recipes (in order of preference)

### 1. Provision a dedicated environment

The checkout `.venv` is a shared symlink on the development machine. Never run
`pip install`, `uv sync`, or the old repair recipe through that path. Create or
sync an environment outside the checkout, then point the probe at it:

    env -u PYTHONPATH \
      UV_PROJECT_ENVIRONMENT=/Volumes/external/sources/.venvs/cppmega.mlx \
      uv sync --project . --locked --extra parquet --extra gui --extra widget \
        --extra path-c

The `dev` group installs `pytest`; the `path-c` extra records the released
TileLang wheel and its test dependencies (`torch`, `einops`,
`apache-tvm-ffi`) in the lockfile. `scripts/run_mlx_path_c.py` selects the
sibling source checkout explicitly and reports its origin instead of relying on
ambient `PYTHONPATH`.

### 2. Explicit repair, only for an isolated environment

`scripts/fix_mlx_abi.sh` is retained for compatibility, but it is diagnostic
by default and refuses every environment except the dedicated
`../.venvs/cppmega.mlx` target. `--apply` runs the complete locked sync; it does
not pin packages from a live Homebrew directory:

    CPPMEGA_MLX_PYTHON=/Volumes/external/sources/.venvs/cppmega.mlx/bin/python \
      ./scripts/fix_mlx_abi.sh --apply

The project lockfile remains the single source of truth. Override the target
only with an explicit `CPPMEGA_MLX_ENV_ROOT`; targets inside any Git checkout
are rejected. `--apply` accepts no additional `uv` arguments, validates that
the selected Python belongs to the requested venv, runs
`uv lock --check --no-sources`, and then syncs with both `--locked` and
`--no-sources`.

### 3. Manual: pin DYLD search order

Tells DYLD to look in brew first so the venv's `mlx.core.so` finds the
matching `libmlx.dylib`:

    export DYLD_LIBRARY_PATH=/opt/homebrew/lib
    ./.venv/bin/python -m pytest tests/test_engine_path_switch.py -v

Note: `DYLD_LIBRARY_PATH` is wiped by macOS SIP for system binaries; you
must invoke the venv's python directly (not via a shebang script).

### 4. Manual: unlink brew mlx

If the venv's `mlx` is the canonical install, remove the brew copy from
the search path:

    brew unlink mlx
    ./.venv/bin/python -c "import mlx.core as mx; print(mx.__version__)"

You can re-link later with `brew link mlx`.

### 5. Manual: recreate the isolated venv

Nuclear option — start fresh:

    rm -rf /Volumes/external/sources/.venvs/cppmega.mlx
    UV_PROJECT_ENVIRONMENT=/Volumes/external/sources/.venvs/cppmega.mlx \
      uv sync --project . --locked --extra parquet --extra gui --extra widget \
        --extra path-c

The fresh environment resolves the exact `uv.lock` contract. Verify it with
`scripts/check_mlx_abi.sh` before running wheel-mode tests, then use
`scripts/run_mlx_path_c.py --verify-only` before source Path-C tests. For an
intentionally modified source checkout, append
`--allow-dirty-source-stack` and preserve that flag in the receipt.

## Why this matters for the wave-7/8 test matrices

`docs/research/numerical_parity_metal.md` and
`engine_vs_shim_parity.md` (in
[`DatasunriseOU/tilelang`](https://github.com/DatasunriseOU/tilelang/tree/main/docs/research))
both reported large `pytest.importorskip` blocks for cppmega.mlx
engine-path tests — every cell traced back to this single ABI
mismatch. After provisioning an isolated environment and running the
fail-closed probe, re-run the test matrices to get real pass/fail data
instead of skip-noise.

## Wave-8 status

The current probe and repair wrapper deliberately do not mutate a shared
`.venv`. A successful ABI receipt must name the exact Python executable and
show matching `mlx`/`mlx-metal` release contracts.
