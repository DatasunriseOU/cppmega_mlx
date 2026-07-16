# Hardening cppmega.mlx: ABI, Path C, packaging и полный локальный suite

Дата: 2026-07-16
Репозиторий: [DatasunriseOU/cppmega_mlx](https://github.com/DatasunriseOU/cppmega_mlx/tree/codex/canonical-complete-20260715)

## Что именно исправлено

### 1. MLX ABI определяется по реально загруженной библиотеке

- Probe получает путь к загруженному `libmlx.dylib` через dyld image list.
- Если библиотека загружена из Homebrew Cellar, версия извлекается из её
  фактического path. Linked `opt/mlx` не может скрыть старый loaded keg.
- Если MLX загружен из dedicated wheel, Homebrew версия считается только
  информационной.
- Проверяются selected Python, environment prefix, module/distribution origin,
  `mlx-metal` release и реальный `mx.eval` smoke.

Live receipt:

- MLX: `0.32.0`
- mlx-metal: `0.32.0`
- loaded dylib:
  `/Volumes/external/sources/.venvs/cppmega.mlx/lib/python3.13/site-packages/mlx/lib/libmlx.dylib`
- linked Homebrew keg: `/opt/homebrew/Cellar/mlx/0.32.0` (informational)

Исходники:

- [scripts/check_mlx_abi.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/check_mlx_abi.py)
- [scripts/check_mlx_abi.sh](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/check_mlx_abi.sh)
- [scripts/fix_mlx_abi.sh](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/fix_mlx_abi.sh)

### 2. Dirty TileLang/TVM stack теперь формальный fail-closed contract

- Path-C launcher загружает repository-owned environment module по точному path,
  до любых импортов из ambient `PYTHONPATH`.
- По умолчанию dirty TileLang или TVM дают exit code `1`, и probe даже не
  запускает runtime.
- Разрешение dirty stack возможно только через
  `--allow-dirty-source-stack`.
- Receipt сохраняет revision, dirty flag и SHA-256 digest каждого worktree:
  TileLang, TVM и TVM-FFI.
- Shell launcher очищает Python/TVM/TileLang/DYLD environment до старта Python.

Live verification:

- default dirty run: FAIL, exit `1`;
- explicit opt-in run: PASS, exit `0`;
- runtime origin: MLX wheel + source TileLang/TVM/TVM-FFI;
- TileLang revision: `6ca0b3e76c0e9189ff6681d01710cba2f262089a`;
- TVM revision: `7405db750bff895183ce8ad434ccef4324a53ef5`;
- TVM-FFI revision: `9a71637fbb8d35364f30535cab51e89dd61f1765`.

Старый build с чужими CMake roots сохранён как
`/Volumes/external/sources/tilelang/build.stale-20260716`. Новый
`tilelang/build` собран с vendored TVM, Metal ON, CUDA/ROCm OFF и
`TILELANG_BUILD_MLX_TVM_FFI=OFF`; source Cython wrapper собирается из
явно объявленного `cython` dependency.

Исходники:

- [scripts/mlx_env_contract.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/mlx_env_contract.py)
- [scripts/run_mlx_path_c.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/run_mlx_path_c.py)
- [scripts/run_path_c.sh](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/run_path_c.sh)

### 3. Model/runtime fixes

- DSA Metal row reductions используют явные совместимые layouts и fail-closed
  validation вместо неявного reducer fallback.
- Mamba F2 принимает `prev_states` в fp32, как их реально производит F1;
  full-scale repro callers больше не downcast-ят handoff в fp16.
- Mamba state contraction выбирает tile по threadgroup memory limit.
- Cut cross entropy production default снижен до `chunk_rows=8`. Это даёт
  устойчивый запас над acceptance floor 4x по forward+backward peak memory.
- Pure-MLX quantization CLI очищает dirty TileLang/TVM loader variables до
  `import mlx`; regression подтверждает wheel `mlx==0.32.0` даже при подложенном
  `DYLD_LIBRARY_PATH/PYTHONPATH`.
- Native graph-output bridge остаётся явным opt-in: source генерируется TileLang,
  owner-output путь продолжает идти через TVM-FFI, MLX graph path не маскируется
  как автоматический fallback.

Исходники:

- [cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py)
- [cppmega_mlx/nn/_tilelang/mamba3_chunked_scan_core.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/cppmega_mlx/nn/_tilelang/mamba3_chunked_scan_core.py)
- [cppmega_mlx/training/cut_cross_entropy.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/cppmega_mlx/training/cut_cross_entropy.py)
- [scripts/quantize_for_inference.py](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/scripts/quantize_for_inference.py)

### 4. uv.lock, Path-C extras и CUDA/self-hosted workflow

- `path-c` extra включает TileLang/TVM-FFI для Darwin arm64 и Linux x86_64.
- Locked export содержит:
  `apache-tvm-ffi==0.1.7`, `cython==3.2.8`, `tilelang==0.1.9`, `torch==2.13.0`,
  `triton==3.7.1` на Linux.
- CUDA workflow использует `uv pip check --python`, а не отсутствующий
  `python -m pip` в managed environment.
- Linux CUDA dry-run проходит для `manylinux_2_35` и намеренно блокируется для
  `manylinux_2_34`: locked MLX wheel требует glibc 2.35 или новее.
- CUDA environment receipt публикуется отдельным artifact.
- E2E path filters включают MLX sources, parquet fixtures и tokenizer fixtures.

Исходники:

- [pyproject.toml](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/pyproject.toml)
- [uv.lock](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/uv.lock)
- [.github/workflows/cuda-lane.yml](https://github.com/DatasunriseOU/cppmega_mlx/blob/codex/canonical-complete-20260715/.github/workflows/cuda-lane.yml)

## Parquet fixtures: данные не менялись

Все 16 dirty parquet fixtures были прочитаны PyArrow и сравнены с `git show
HEAD:<path>`:

- одинаковое число строк и колонок;
- одинаковая Arrow field schema;
- одинаковые decoded values;
- различаются writer metadata (`parquet-cpp-arrow version 21.0.0` -> `24.0.0`)
  и физические compressed/uncompressed sizes column chunks;
  это изменение представления файла, а не decoded data или schema.

Поэтому parquet binaries признаны случайным metadata churn и не включаются в
production commit.

## Test harness: strict safe-path и длинный процесс

- Полный suite запускается с `PYTHONSAFEPATH=1`.
- Autouse fixture удаляет foreign `PYTHONPATH`, но возвращает ровно текущий
  checkout, чтобы дочерние `python -m scripts...` не зависели от cwd semantics.
- Parser/converter fixtures явно ставят `memory_limit_gb=0.0`, потому что они
  проверяют данные и schema, а не memory guard. Production default `10 GiB` и
  fail-closed watchdog не менялись.

## Проверка

| Gate | Результат |
|---|---|
| Полный pytest в dedicated MLX env | `7385 passed, 57 skipped, 2 xfailed, 19 warnings` за `22:31` |
| Safe-path subprocess regression group | `64 passed` |
| Parser/provenance fixture groups | `100 passed` |
| ABI/Path-C/CUDA workflow contract suite | `69 passed, 1 skipped` |
| ABI live receipt | PASS, loaded wheel dylib |
| Dirty Path-C default | expected FAIL, exit `1` |
| Dirty Path-C explicit opt-in | PASS, exit `0` |
| `uv lock --check` | PASS |
| `uv lock --check --no-sources` | PASS |
| `uv pip check` | 95 packages compatible |
| Changed-file Ruff | `All checks passed` (`ruff 0.15.22`) |
| Python compileall | PASS для `cppmega_mlx cppmega_v4 scripts tests` |
| Tracked/new shell `bash -n` | PASS |
| `git diff --check` | PASS |
| HTML changelog desktop/mobile | PASS: 1440px и 390px без page-level horizontal overflow |

Команда полного теста:

```bash
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  /Volumes/external/sources/.venvs/cppmega.mlx/bin/python -m pytest -q
```

## Что сознательно не включено в production commit

| Dirty group | Причина |
|---|---|
| 16 `tests/fixtures/parquet/*.parquet` | Decoded data/schema identical; только PyArrow writer metadata churn |

## Что осталось

- Реальный CUDA kernel compile/run здесь не выполнялся: Mac не является CUDA
  proof. Hardware test остаётся skipped до self-hosted CUDA runner.
- TileLang и TVM source trees сейчас dirty. Их использование разрешено только
  явным opt-in и всегда сопровождается digest receipt.
- Два ожидаемых `xfail` относятся к известной Mamba chained-backward parity
  работе и не скрывают неожиданные failures.
- Полный repo-wide Ruff остаётся legacy debt (`411` нарушения в нетронутых
  файлах); все Python-файлы production diff проходят pinned changed-file Ruff.
