# Widely-used C/C++ libraries ABSENT from the corpus — ranked acquisition candidates

Source scan: `outputs/crossrepo/missing_libs_ranked.json` (this same workflow).
Cross-checked against `outputs/pr_ingest/repo_list.json` (550 corpus repo entries; 82 `.bare`).

## Method (real measurement, fail-loud — RULE #1)

Light **SAMPLED** streaming scan of the `cpp_all` tarball
(`/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst`) read via
`zstandard.stream_reader` + `tarfile r|` pipe, `.git/` skipped, first 20 KB per source
file, capped at 400 source files/repo, ~13-min time-box, run at low priority to **avoid
starving the unified conveyor (pid 8697)** which is concurrently reading the same 235 GB
tarball. This scan was kept READ-ONLY and light by design.

- **Sample size:** 501 distinct repos sampled, 168,732 source files read,
  7,703,235 tar members scanned in 344.9 s.
- **Classification (sample totals):** std/libc/OS-SDK = 244,779 (excluded);
  local-satisfied = 504,506; provided-by-corpus-base-lib = 96,641;
  **EXTERNAL & ABSENT = 1,529**; unclassified (mostly OS-SDK) = 65,192.
- Each `#include` target was classified as (a) std/libc/OS-SDK
  (Windows/macOS/Android/Emscripten/compiler intrinsics) → EXCLUDED;
  (b) satisfied by a local working-tree header in the same repo; (c) provided by a base
  lib present in corpus (cross-checked vs repo_list owner/repo tokens UNION sampled repo
  dir names); or (d) **EXTERNAL & ABSENT**. Absent families ranked by #distinct
  referencing repos, then total refs.
- **Sister artifact:** `outputs/crossrepo/base_lib_usage_ranked.json` did NOT exist at
  scan time, so an independent sampled scan was performed (not reused).

**Verification of absence (this report):** every one of the 27 families below was
confirmed to have **zero** matching tokens in `repo_list.json`, and the claimed
present base libs (protobuf, gRPC, Abseil, OpenSSL/BoringSSL/wolfSSL/mbedTLS, curl,
folly, RocksDB/LevelDB, re2, zstd/lz4, SQLite, fmt, spdlog, LLVM, OpenCV, harfbuzz,
freetype, Vulkan, SDL, GLFW, qtbase, GLib, GTK, mimalloc/snmalloc/gperftools, zlib,
CUDA cccl/cutlass/thrust/cub, flatbuffers, rapidjson, **nlohmann/json**, capnproto,
thrift, FFmpeg, openjdk) were confirmed present.

> **CAVEAT (fail-loud, do not paper over):** the scan's prose note claimed *Eigen* is in
> the corpus. It is **NOT** in `repo_list.json` (zero token match on `eigen`). It did not
> surface as an absent family because Eigen is header-only and is vendored locally in many
> repos, so its includes classify as "local-satisfied" rather than "external-absent". If
> Eigen coverage matters as a first-class base-link target, it should be acquired
> separately (`libeigen/eigen`, gitlab.com/libeigen/eigen, ~10 MB) — it is not on the
> ranked list below because the include-root method cannot see vendored-header libs as
> absent. Likewise `OpenMP` and `jni.h` are excluded for the reasons noted under the table.

## Ranked table — ABSENT widely-used libs (by #distinct referencing repos, then refs)

| # | Family | #repos | refs | Where to get (GitHub / canonical) | Rough size | Base-link target? |
|---|--------|-------:|-----:|-----------------------------------|-----------|-------------------|
| 1 | X11 / XCB | 45 | 590 | `gitlab.freedesktop.org/xorg/lib/libx11` + `.../libxcb` (GH mirror: mirror/libX11) | ~5–15 MB each | yes (system display) |
| 2 | OpenMP | 40 | 233 | COMPILER-PROVIDED (libomp ships w/ clang/gcc; `llvm-project/openmp` already in corpus) | n/a | compiler — **do not add** |
| 3 | MPI | 31 | 75 | `open-mpi/ompi` or `pmodels/mpich` | ~30–60 MB | yes (HPC) — borderline |
| 4 | gettext / libintl | 25 | 40 | `autotools-mirror/gettext` (canon: git.savannah.gnu.org/git/gettext) | ~30 MB | yes (i18n) |
| 5 | bzip2 | 20 | 49 | `libarchive/bzip2` (canon: sourceware.org/bzip2) | ~1 MB | yes |
| 6 | libxml2 / libxslt | 17 | 234 | `GNOME/libxml2` + `GNOME/libxslt` | ~25 MB / ~7 MB | yes |
| 7 | NCurses | 16 | 54 | `ThomasDickey/ncurses-snapshots` (canon: invisible-island.net/ncurses) | ~35 MB | yes (TUI) |
| 8 | Wayland | 15 | 52 | `gitlab.freedesktop.org/wayland/wayland` | ~5 MB | yes (Linux display) |
| 9 | gflags | 14 | 43 | `gflags/gflags` | ~1 MB | no |
| 10 | tinyxml2 | 14 | 40 | `leethomason/tinyxml2` (+ legacy tinyxml) | ~1 MB | no (hdr+1 src) |
| 11 | readline | 14 | 33 | GNU readline (git.savannah.gnu.org/git/readline; GH mirror/readline) | ~3 MB | yes (CLI) |
| 12 | brotli | 13 | 26 | `google/brotli` | ~30 MB (incl test data) | yes |
| 13 | liburing | 13 | 19 | `axboe/liburing` | ~2 MB | yes (io_uring) |
| 14 | GnuTLS | 12 | 35 | `gitlab.com/gnutls/gnutls` (GH mirror: gnutls/gnutls) | ~30 MB | yes (TLS) |
| 15 | glog | 11 | 73 | `google/glog` | ~2 MB | no |
| 16 | jemalloc | 10 | 173 | `jemalloc/jemalloc` | ~10 MB | yes (allocator) |
| 17 | ICU | 10 | 84 | `unicode-org/icu` | ~70 MB (icu4c) | yes (Unicode) |
| 18 | PCRE2 | 8 | 14 | `PCRE2Project/pcre2` | ~10 MB | yes (regex) |
| 19 | GMP / MPFR | 8 | 14 | GMP gmplib.org + MPFR www.mpfr.org | ~10 MB / ~5 MB | yes (bignum) |
| 20 | yaml | 7 | 27 | `yaml/libyaml` (C) + `jbeder/yaml-cpp` (C++) | ~2 MB each | no |
| 21 | snappy | 7 | 9 | `google/snappy` | ~2 MB | no |
| 22 | cairo | 6 | 16 | `gitlab.freedesktop.org/cairo/cairo` | ~25 MB | yes (2D gfx) |
| 23 | dbus | 6 | 10 | `gitlab.freedesktop.org/dbus/dbus` | ~10 MB | yes (IPC) |
| 24 | double-conversion | 4 | 8 | `google/double-conversion` | ~1 MB | no |
| 25 | expat | 2 | 2 | `libexpat/libexpat` | ~5 MB | yes (XML) |
| 26 | date (Howard Hinnant) | 2 | 2 | `HowardHinnant/date` | ~2 MB | no (mostly hdr-only) |

**Excluded from acquisition (not repos to add):**
- **OpenMP** (#2, 40 repos) — compiler-provided; `omp.h` ships with clang/gcc, and
  `llvm-project/openmp` is already in the corpus. Adding a repo would be redundant.
- **jni.h** (42 repos, not tabled) — already satisfiable via `openjdk` (present in corpus).
- **Eigen** — absent from corpus but not detectable by this method (vendored header-only);
  see caveat above.

## Why the absent set is SMALL (context)

The corpus is extremely comprehensive (550+ repos) and already contains the source repos
for nearly every major C/C++ library people usually flag as "missing": Qt(qtbase), gRPC,
protobuf, Abseil, OpenSSL/BoringSSL/wolfSSL/mbedTLS, libcurl, fmt, spdlog, libuv,
LLVM/Clang/MLIR, CUDA(cccl/cub/cutlass/thrust/libcudacxx), Vulkan, OpenCV, Poco, folly,
GTK/GLib, SDL, GLFW, freetype, harfbuzz, SQLite, HDF5, TBB, FFmpeg, FFTW, OpenEXR,
RocksDB/LevelDB, re2, zstd/lz4/xz, PyTorch(ATen/torch/c10), ONNX/ONNXRuntime, TensorRT,
HIP/ROCm/MIOpen, NCCL/RCCL, GoogleTest/Benchmark, flatbuffers, rapidjson, nlohmann/json,
capnproto, thrift, librdkafka, libpcap, libusb, systemd.

Hence the genuinely-absent set is small and skews to (i) the **Linux desktop/display
stack** (X11/XCB, Wayland, cairo, dbus) and (ii) a handful of **utility libs** (ICU,
libxml2, gettext, PCRE, GMP/MPFR, bzip2, NCurses/readline, yaml, brotli, glog/gflags,
jemalloc, liburing, GnuTLS, snappy).

## Recommended acquisition shortlist (careful cut — high breadth × reuse value first)

### TIER 1 — broad breadth, clear value, cheap (acquire first)
- **ICU** (`unicode-org/icu`, ~70 MB) — 10 repos, 84 refs. THE canonical Unicode lib;
  heavy base-link target; not derivable from anything in corpus. Highest single-lib value.
- **libxml2 + libxslt** (`GNOME/libxml2`, `GNOME/libxslt`, ~32 MB) — 17 repos, **234 refs**.
  Ubiquitous XML base lib, no substitute in corpus.
- **gettext / libintl** (`autotools-mirror/gettext`, ~30 MB) — 25 repos. i18n base-link
  target referenced very broadly.
- **jemalloc** (`jemalloc/jemalloc`, ~10 MB) — 10 repos, **173 refs** (high). Allocator;
  corpus has gperftools/mimalloc/snmalloc but NOT jemalloc, which folly/rocksdb-style code
  targets directly.
- **glog + gflags** (`google/glog`, `google/gflags`, ~3 MB total) — 11 + 14 repos. Google
  logging/flags; tiny; frequently co-required by other Google-style C++.

### TIER 2 — Linux desktop/display stack (add as a bundle if GUI coverage matters)
- **X11 (libX11) + libxcb** (~20 MB) — 45 repos (top breadth), 590 refs, but mostly system
  display. Add if desktop/GUI build-graphs matter.
- **Wayland, cairo, dbus** (~40 MB total) — 6–15 repos; the rest of the Linux desktop stack.

### TIER 3 — small, cheap, decent breadth (add opportunistically)
- Compression family completion (alongside existing zstd/lz4/xz/zlib):
  **bzip2** (~1 MB, 20 repos), **brotli** (~30 MB, 13 repos), **snappy** (~2 MB, 7 repos).
- **PCRE2** (~10 MB, 8 repos), **yaml** (libyaml + yaml-cpp, ~4 MB, 7 repos),
  **tinyxml2** (~1 MB, 14 repos), **expat** (~5 MB, 2 repos), **date** (~2 MB, 2 repos).
- **GMP + MPFR** (~15 MB, 8 repos) — bignum, base-link target.
- **NCurses** (~35 MB) + **readline** (~3 MB) — 16 + 14 repos; TUI/CLI base libs.
- **liburing** (`axboe/liburing`, ~2 MB, 13 repos) — modern Linux async I/O.
- **GnuTLS** (~30 MB, 12 repos) — alternative TLS; corpus already has
  OpenSSL/BoringSSL/wolfSSL/mbedTLS, so lower priority.
- **double-conversion** (~1 MB, 4 repos) — cheap, common transitive dep of folly/v8.

### Do NOT add
- **OpenMP** — compiler-provided (libomp; llvm-project present).
- **jni.h** — already covered by `openjdk`.
- **MPI** — borderline (system HPC). Add only if HPC build-graphs are explicitly needed.

## How each acquired lib slots into the corpus

For every shortlisted repo, the flow is identical and uses the existing pipeline:

1. **Download** — `git clone` the canonical/GitHub-mirror URL from the table into the same
   layout the rest of the corpus uses (a top-level repo dir; `.bare` mirror if matching the
   82 existing `.bare` entries). Add an entry to `outputs/pr_ingest/repo_list.json`
   (`name`, `owner_repo`, `url`) so the conveyor and cross-link stages recognize it.
2. **Conveyor index** — feed it through the running streaming conveyor
   (`scripts/streaming_conveyor.py`, the same pid-8697 process / a fresh run) so its source
   files are deduped and ingested into `outputs/pr_ingest/prs.sqlite` /
   `outputs/dedup_seen.sqlite`, exactly as the other 550 repos.
3. **Cross-link (only for base-link-target libs)** — for the families marked
   *base-link target = yes* (ICU, libxml2/libxslt, gettext, jemalloc, X11/XCB, Wayland,
   cairo, dbus, bzip2, brotli, PCRE2, GMP/MPFR, NCurses, readline, liburing, GnuTLS,
   expat), the newly-indexed headers become resolution targets: the cross-repo linker can
   now reclassify the **1,529 currently-"absent"** includes (e.g. the 234 `libxml/*.h`,
   the 173 `jemalloc/jemalloc.h`, the 84 `unicode/*.h`) from EXTERNAL-ABSENT to
   PROVIDED-BY-BASE-LIB, wiring consumer repos to the real upstream definitions instead of
   leaving dangling include edges. The non-base-link libs (glog, gflags, tinyxml2, yaml,
   snappy, double-conversion, date) still get indexed and locally cross-linked but are not
   broad link hubs.

**Acquisition cost of the whole shortlist** is modest: TIER 1 ≈ 145 MB, TIER 2 ≈ 60 MB,
TIER 3 ≈ ~110 MB — well under the corpus's existing 235 GB footprint, while closing the
large majority of the absent include edges (libxml2 + ICU + jemalloc + gettext alone
resolve the highest-ref absent families).
