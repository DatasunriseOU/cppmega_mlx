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

Run `scripts/check_mlx_abi.sh`:

    $ ./scripts/check_mlx_abi.sh
    venv (./.venv/bin/python): 0.31.2
    brew (/opt/homebrew/Cellar/mlx): 0.31.1
    FAIL: venv mlx==0.31.2 vs brew mlx==0.31.1 (ABI may diverge)
    FIX: run scripts/fix_mlx_abi.sh

Exit code 0 means no mismatch; 1 means a fix is needed.

## Fix recipes (in order of preference)

### 1. Run the auto-fix script

    ./scripts/fix_mlx_abi.sh

This pins the venv's `mlx` package to the brew-installed version with
`pip install --force-reinstall --no-cache-dir mlx==<brew-version>`.
Falls back to plain `pip install` if `--force-reinstall` is rejected.

### 2. Manual: pin DYLD search order

Tells DYLD to look in brew first so the venv's `mlx.core.so` finds the
matching `libmlx.dylib`:

    export DYLD_LIBRARY_PATH=/opt/homebrew/lib
    ./.venv/bin/python -m pytest tests/test_engine_path_switch.py -v

Note: `DYLD_LIBRARY_PATH` is wiped by macOS SIP for system binaries; you
must invoke the venv's python directly (not via a shebang script).

### 3. Manual: unlink brew mlx

If the venv's `mlx` is the canonical install, remove the brew copy from
the search path:

    brew unlink mlx
    ./.venv/bin/python -c "import mlx.core as mx; print(mx.__version__)"

You can re-link later with `brew link mlx`.

### 4. Manual: recreate the venv

Nuclear option — start fresh:

    rm -rf .venv
    python3 -m venv .venv
    ./.venv/bin/pip install -e .

The fresh venv's `mlx` will resolve to whatever pip ships at install
time; if brew's version is older, run option 1 to pin.

## Why this matters for the wave-7/8 test matrices

`docs/research/numerical_parity_metal.md` and
`engine_vs_shim_parity.md` (in
[`DatasunriseOU/tilelang`](https://github.com/DatasunriseOU/tilelang/tree/main/docs/research))
both reported large `pytest.importorskip` blocks for cppmega.mlx
engine-path tests — every cell traced back to this single ABI
mismatch. After running `fix_mlx_abi.sh`, re-run the test matrices
to get real pass/fail data instead of skip-noise.

## Wave-8 status

Scripts and this doc landed as part of wave-8 #6
(`chore: scripts/check_mlx_abi + fix_mlx_abi`). Auto-fix works on the
common venv-newer-than-brew case; if the brew version is newer than the
latest pip release, manual recipe 3 (`brew unlink mlx`) is the easiest
path forward.
