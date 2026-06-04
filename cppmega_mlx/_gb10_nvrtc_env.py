"""gb10 NVRTC loader-path guard — persist the CUDA 13.3 builtins dir for EVERY run.

WHY (the measured gb10 sm_121 bug, 2026-06-04). TransformerEngine 2.16 (source
build at /home/dave/TransformerEngine) links ``libnvrtc.so.13`` built against
CUDA 13.3. At JIT-compile time that NVRTC dlopen's a companion
``libnvrtc-builtins.so.<MAJOR>.<MINOR>`` whose MINOR must match EXACTLY — i.e.
``libnvrtc-builtins.so.13.3``. That file exists
(``/usr/local/cuda-13.3/targets/sbsa-linux/lib/libnvrtc-builtins.so.13.3``) but
its directory is not on the login-shell ``LD_LIBRARY_PATH`` (empty) and was
historically absent from the ldconfig cache, so the TE tensorwise (Float8
delayed-scaling) route failed with::

    failed to open libnvrtc-builtins.so.13.3

This is a TOOLCHAIN/loader-path problem, NOT a code bug — fixed purely by making
that directory findable. The system-level real fix is a one-time
``sudo -n ldconfig`` (the cache now resolves 13.3). This module is the COMMITTED,
self-healing, per-run belt-and-suspenders that makes the fix survive a future
ldconfig regression WITHOUT relying on a transient one-shell ``export`` (RULE #1:
no silent dependence on a shell var that a future operator might forget).

HOW. ``LD_LIBRARY_PATH`` is parsed by glibc ONCE at process startup; a late
``os.environ`` mutation is NOT honored by later ``dlopen`` calls for the dynamic
search path. So this helper, when the 13.3 dir is missing from the *startup*
``LD_LIBRARY_PATH``, prepends it and RE-EXECS the interpreter so the loader picks
it up at startup. The re-exec is idempotent (guarded by an env sentinel) and runs
at most once. RULE #1: if the dir does not exist or the re-exec cannot be
performed, this RAISES with the precise where+what — it never silently proceeds
with a half-applied path that would later fail deep inside an NVRTC dlopen.

NON-gb10 hosts (no such directory) are a no-op: the helper only acts when the
13.3 builtins dir is actually present, so importing it on a Mac/M4 is harmless.

This module is deliberately a TOP-LEVEL ``cppmega_mlx`` module (NOT under
``cppmega_mlx.runtime``) and imports ONLY the stdlib, so importing it does not
pull torch / mlx / TransformerEngine — the re-exec MUST happen before any of
those build their loader link map.
"""

from __future__ import annotations

import os
import sys

# The directory that holds libnvrtc-builtins.so.13.3 on gb10 (CUDA 13.3 toolkit,
# sbsa-linux / aarch64). This is the loader path TE's NVRTC needs.
GB10_CUDA133_LIB = "/usr/local/cuda-13.3/targets/sbsa-linux/lib"

# Sentinel so the re-exec happens at most once (guards against an exec loop).
_REEXEC_SENTINEL = "CPPMEGA_GB10_NVRTC_PATH_APPLIED"


def _dir_has_builtins(path: str) -> bool:
    """True iff *path* exists and contains the libnvrtc-builtins.so.13.3 SONAME."""
    soname = os.path.join(path, "libnvrtc-builtins.so.13.3")
    return os.path.isdir(path) and os.path.exists(soname)


def ensure_nvrtc_builtins_path(*, lib_dir: str = GB10_CUDA133_LIB) -> str:
    """Guarantee *lib_dir* is on the STARTUP LD_LIBRARY_PATH for this process.

    Returns one of: ``"noop-not-gb10"`` (dir absent — not this host, harmless),
    ``"already-present"`` (startup env already had it), or it RE-EXECS the
    interpreter (does not return in that case). RULE #1: raises RuntimeError if
    the 13.3 builtins dir exists but the re-exec cannot be performed — never a
    silent half-fix.

    MUST be called BEFORE importing torch / transformer_engine (i.e. at the very
    top of any gb10 run entrypoint) so the corrected env is in place when NVRTC
    is later dlopen'd.
    """
    if not _dir_has_builtins(lib_dir):
        # Not gb10 (or the 13.3 toolkit is not installed here). Nothing to do —
        # this is the correct no-op on a Mac/M4 dev box. We do NOT fabricate the
        # path on a host that lacks it.
        return "noop-not-gb10"

    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [p for p in current.split(os.pathsep) if p]
    if lib_dir in entries:
        # Startup env already carries it (e.g. the run command's LD_LIBRARY_PATH
        # prefix, or a prior re-exec). The loader honors it — done.
        return "already-present"

    if os.environ.get(_REEXEC_SENTINEL) == "1":
        # We already re-exec'd once yet the dir is still not on LD_LIBRARY_PATH.
        # That should be impossible; fail loud rather than loop or limp on.
        raise RuntimeError(
            "gb10_nvrtc_env: re-exec sentinel set but "
            f"{lib_dir!r} still absent from LD_LIBRARY_PATH={current!r}; refusing "
            "to proceed — TE's NVRTC would later fail to open "
            "libnvrtc-builtins.so.13.3 (RULE #1: no silent half-applied path)."
        )

    # Prepend the 13.3 dir and re-exec so the loader parses it at startup.
    new_ldlp = os.pathsep.join([lib_dir, *entries]) if entries else lib_dir
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = new_ldlp
    new_env[_REEXEC_SENTINEL] = "1"
    try:
        os.execve(sys.executable, [sys.executable, *sys.argv], new_env)
    except OSError as exc:  # pragma: no cover - exec almost never fails
        raise RuntimeError(
            "gb10_nvrtc_env: failed to re-exec the interpreter to apply "
            f"LD_LIBRARY_PATH={new_ldlp!r} (needed so TE's NVRTC can dlopen "
            f"libnvrtc-builtins.so.13.3): {exc!r}. RULE #1: not falling through "
            "with a path that would crash inside NVRTC later — fix the run env."
        ) from exc
    # os.execve does not return on success.
    raise AssertionError("unreachable: os.execve returned without raising")  # pragma: no cover
