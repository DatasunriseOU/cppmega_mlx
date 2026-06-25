# Base-Library Cross-Link Candidates — Ranked Usage-Breadth Census

**Goal.** We already infer platform from cmake/make/ninja. We have full-history base-lib repos
(linux, glibc, boost, ACE, libstdc++/gcc, openssl, zlib, Windows base libs, etc.). We want to
*selectively* cross-repo link **only the base libraries that are used in many places**, so a
function in repo X that calls e.g. a boost/glibc/std symbol can pull that definition cross-repo.
This report ranks base-lib usage breadth across the entire corpus and recommends a careful cut.

**RULE #1 (fail-loud, real measurement).** The ranking below is a **full census**, not a sample
and not a guess. Honest classifier-coverage gaps are called out explicitly rather than papered over.

## Method (read-only, real measurement)

Single read-only streaming pass over the entire corpus tarball:

```
nice -15 zstd -dc /Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst \
  | nice -15 python3 /tmp/rank_usage2.py     # tarfile mode 'r|' — stream, no extraction
```

For each source file (ext c/cc/cpp/cxx/h/hpp/hh/cu/cuh/m/mm/tcc/inl/ipp/...) we read only the
top 49152 bytes (includes live at file top) and run:
1. ONE `#include` regex capturing every include path; each path is classified to a base-lib family
   by a dispatch function (first-path-segment prefix + exact-basename tables for STL/libc/Windows
   headers) — O(includes), not O(families×bytes).
2. ONE combined symbol-prefix regex (`std:: boost:: absl:: Eigen:: folly:: fmt::
   google::protobuf:: gsl:: thrust::`) for the prefixed families.

**Self-references excluded** via per-family provider subtree sets (a file inside `boost/` does not
count toward boost). **Breadth metric** = count of DISTINCT referencing repos (top-level
`cpp_all/<repo>`); **volume** = total includes + total symbol refs. Classifier unit-tested 28/28
before the run. v1 (per-family findall over decoded text) ran at 73 files/s and was discarded for
v2 (include-centric) at ~3,100 files/s — a 42× speedup.

**Coverage / completeness (this is a complete census, not a sample):**

| metric | value |
|---|---|
| run mode | FULL — entire ~235 GiB tarball streamed end-to-end |
| elapsed | 884.9 s (vs 14,400 s budget) |
| tar members streamed | 7,703,235 |
| subtrees seen | 594 / 594 (100% of manifest) |
| source files scanned | 2,749,108 |
| file-content bytes scanned | 25,069,556,276 (top-48 KiB-per-file cap) |
| files skipped (>16 MiB) | 15 |
| rate | 3,106.7 files/sec |

Concurrency safety: the unified conveyor (pid 8697) reads the same tarball + repo_list.json and was
confirmed alive and unharmed before/after; the tarball is read-only; no writes to
`outputs/reindexed*`. Membership manifest:
`/Volumes/external/sources/cppmega/outputs/pr_ingest/repo_list.json`. Source artifact (ranked JSON):
`/Volumes/external/sources/cppmega/outputs/crossrepo/base_lib_usage_ranked.json` (27,500 bytes).

## Presence in corpus

**All base-lib families measured were confirmed present in the corpus (`present_in_corpus=true` for
every family).** Providers are drawn from the 594 subtrees in the manifest, all members of the
tarball under layout `cpp_all/<repo>/...`. Key providers → what they provide:

- **libstdcxx_STL** (C++ std lib): gcc-mirror (libstdc++), llvm-project (libc++), STL/stl
  (microsoft/STL), stlport, sgi-stl, libcudacxx, cccl(+.bare), rocm-llvm, intel-llvm-dpcpp.
- **libc_posix** (C stdlib/POSIX/syscalls): glibc, musl, apple-libc, gcc-mirror, llvm-project, plus
  kernels linux / freebsd-src / openbsd / netbsd / src (NetBSD) / xnu / illumos-gate / 4.4bsd-lite2.
- **boost**: boost (boostorg). **abseil**: abseil-cpp. **folly**: folly. **eigen**: eigen (gitlab
  libeigen). **fmt**: fmt (+ spdlog bundles it). **spdlog**: spdlog.
- **openssl_crypto**: openssl, boringssl, portable (libressl). **zlib**: zlib. **lz4/zstd/xz**:
  lz4, zstd, xz. mbedtls/wolfssl/libsodium present.
- **protobuf, flatbuffers, nlohmann_json (json), rapidjson, msgpack-c, capnproto, simdjson, thrift**
  (serialization).
- **googletest, Catch2, benchmark** (test). **gsl_microsoft (GSL/gsl), range-v3, wil** (utility hdrs).
- **windows_base** (CRT/SDK/MFC/ATL/DDK): windows_2000_source_code, windows_nt_4_source_code,
  nt5src, windows-research-kernel, ddk_wdk, windows_10_shared_source_kit, STL/stl.
- **libuv, libevent, asio, curl, c-ares, nghttp2** (event/io/net). **sqlite, leveldb, lmdb**
  (embedded db). **glib, cpython, mimalloc** (runtime/alloc). **libpng, libjpeg-turbo, freetype,
  harfbuzz, openexr** (image/text codec).

**RULE #1 caveats found:**
- `bionic` subtree = nickg/nvc (VHDL compiler), **NOT** Android libc; AOSP libc is present as
  aosp-system-core / aosp-frameworks-*.
- 82 `.bare` subtrees are bare git mirrors (no working-tree source to scan, so they show 0 scannable
  files but carry full history). The 512 non-bare subtrees provided the working-tree source scanned.

## Ranked usage table (by breadth = distinct referencing repos; self-refs excluded)

Columns: family | distinct_repos (union of include+symbol) | inc_repos | sym_repos |
total_includes | total_symbol_refs | provider subtree(s).

| # | family | repos∪ | inc_repos | sym_repos | includes | symbol_refs | provider(s) |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | libc_posix | 481 | 481 | 0 | 864,640 | 0 | glibc, musl, apple-libc, gcc-mirror, llvm-project, linux, freebsd-src, openbsd, netbsd, src, xnu, illumos-gate, 4.4bsd-lite2 |
| 2 | libstdcxx_STL | 423 | 414 | 418 | 978,244 | 8,310,086 | gcc-mirror, llvm-project, STL, stl, stlport, sgi-stl, libcudacxx, cccl(.bare), rocm-llvm, intel-llvm-dpcpp |
| 3 | windows_base | 416 | 416 | 0 | 63,111 | 0 | windows_2000_source_code, windows_nt_4_source_code, nt5src, windows-research-kernel, ddk_wdk, windows_10_shared_source_kit, STL/stl |
| 4 | zlib | 200 | 200 | 0 | 3,031 | 0 | zlib |
| 5 | googletest | 191 | 191 | 0 | 62,365 | 0 | googletest |
| 6 | boost | 166 | 107 | 165 | 238,917 | 338,402 | boost |
| 7 | openssl_crypto | 152 | 152 | 0 | 78,494 | 0 | openssl, boringssl, portable (libressl) |
| 8 | abseil | 105 | 78 | 105 | 81,923 | 377,062 | abseil-cpp |
| 9 | cpython | 99 | 99 | 0 | 1,021 | 0 | cpython |
| 10 | zstd | 95 | 95 | 0 | 766 | 0 | zstd |
| 11 | fmt | 75 | 58 | 73 | 3,637 | 59,163 | fmt, spdlog |
| 12 | cuda_cccl | 72 | 72 | 49 | 18,774 | 46,217 | libcudacxx, cccl(.bare) |
| 13 | libevent | 71 | 71 | 0 | 4,427 | 0 | libevent |
| 13 | curl | 71 | 71 | 0 | 1,119 | 0 | curl |
| 15 | eigen | 68 | 58 | 68 | 3,773 | 55,944 | eigen |
| 16 | libpng | 66 | 66 | 0 | 525 | 0 | libpng |
| 17 | libjpeg | 65 | 65 | 0 | 2,211 | 0 | libjpeg-turbo |
| 18 | protobuf | 61 | 51 | 56 | 27,449 | 23,217 | protobuf |
| 19 | glib | 60 | 60 | 0 | 4,242 | 0 | glib |
| 19 | nlohmann_json | 60 | 60 | 0 | 3,369 | 0 | json |
| 19 | xz | 60 | 60 | 0 | 220 | 0 | xz |
| 22 | mbedtls | 54 | 54 | 0 | 16,881 | 0 | mbedtls |
| 22 | lz4 | 54 | 54 | 0 | 394 | 0 | lz4 |
| 24 | sqlite | 53 | 53 | 0 | 285 | 0 | sqlite |
| 25 | freetype | 52 | 52 | 0 | 6,641 | 0 | freetype |
| 26 | catch2 | 49 | 49 | 0 | 1,396 | 0 | Catch2 |
| 27 | gsl_microsoft | 45 | 27 | 32 | 902 | 13,388 | GSL, gsl |
| 28 | rapidjson | 42 | 42 | 0 | 1,464 | 0 | rapidjson |
| 29 | wolfssl | 35 | 35 | 0 | 6,548 | 0 | wolfssl |
| 30 | folly | 28 | 17 | 28 | 2,124 | 12,038 | folly |
| 30 | libuv | 28 | 28 | 0 | 1,766 | 0 | libuv |
| 32 | asio | 26 | 26 | 0 | 20,339 | 0 | asio, boost |
| 33 | flatbuffers | 23 | 23 | 0 | 1,495 | 0 | flatbuffers |
| 34 | benchmark | 21 | 21 | 0 | 77 | 0 | benchmark |
| 35 | spdlog | 19 | 19 | 0 | 1,614 | 0 | spdlog |
| 35 | c_ares | 19 | 19 | 0 | 90 | 0 | c-ares |
| 37 | harfbuzz | 18 | 18 | 0 | 193 | 0 | harfbuzz |
| 38 | wil_microsoft | 14 | 14 | 0 | 775 | 0 | wil |
| 39 | mimalloc | 11 | 11 | 0 | 192 | 0 | mimalloc |
| 40 | lmdb | 9 | 9 | 0 | 51 | 0 | lmdb |
| 41 | simdjson | 4 | 4 | 0 | 13 | 0 | simdjson |
| 42 | range_v3 | 3 | 3 | 0 | 4,078 | 0 | range-v3 |
| 42 | libsodium | 3 | 3 | 0 | 181 | 0 | libsodium |
| 42 | capnproto | 3 | 3 | 0 | 25 | 0 | capnproto |
| 45 | leveldb | 2 | 2 | 0 | 214 | 0 | leveldb |
| 46 | msgpack | 1 | 1 | 0 | 1 | 0 | msgpack-c |

**Honest coverage gaps (RULE #1 — these are NOT zero real usage):**
- `thrift` (uses `thrift/` dir prefix — unmapped by classifier), `openexr` (`OpenEXR/` prefix —
  unmapped), `nghttp2` (`nghttp2/` dir form — only the bare `.h` was mapped) all show 0 in this run
  purely from classifier-coverage gaps. All three are long-tail and **below the recommended cut**, so
  the gap does not change the recommendation; flagged here for transparency.

## Recommended cross-link cut

Selection criterion: **distinct-repos-using breadth × value / cost**. Cost is dominated by (a)
provider-tree size, (b) provider multiplicity (multiply-defined symbols), and (c) symbol-namespace
cleanliness (prefixed → safe symbol cross-link; unprefixed → high false-link risk).

### TIER 1 — cross-link (highest value, used almost everywhere)

- **libstdcxx_STL** (423 repos, 978k includes, 8.31M `std::` refs) — **HIGHEST VALUE.**
  Cost MEDIUM: pick **ONE canonical provider** (gcc-mirror libstdc++ is the cleanest single tree)
  rather than linking all 11 STL providers; do NOT cross-link every STL impl or you multiply
  definitions. Risk: header-only templates resolve to many candidate defs — **canonicalize to one
  provider**.
- **boost** (166 repos, 239k includes, 338k `boost::` refs) — HIGH VALUE, header-heavy, single clean
  provider (boost). Cost MEDIUM (large tree). Recommended.
- **libc_posix** (481 repos — widest of all, 865k includes) — cross-link but **CAREFULLY.**
  Cost HIGH: providers include linux+glibc (huge) and libc symbols are unprefixed, so symbol-level
  linking is ambiguous across 13 providers. Recommendation: cross-link **DECLARATIONS** from ONE
  canonical libc (glibc headers) for include→header resolution; do NOT attempt full cross-repo symbol
  resolution into the linux kernel tree (definitions are unprefixed, multiply-defined across
  kernels/libcs — high false-link risk).

### TIER 2 — cross-link (high breadth, single clean provider, low cost)

- **googletest** (191), **zlib** (200), **openssl_crypto** (152), **abseil** (105), **fmt** (75),
  **eigen** (68), **protobuf** (61). Each has 1–3 well-namespaced providers; low ambiguity; strong
  value. abseil / eigen / fmt / protobuf carry clear symbol prefixes (`absl:: Eigen:: fmt::
  google::protobuf::`) making **symbol** cross-link safe (not just include-resolution).

### TIER 3 — optional (60–99 breadth, cheap, include-resolution only)

cpython (99), zstd (95), cuda_cccl (72), libevent (71), curl (71), libpng (66), libjpeg (65),
glib (60), nlohmann_json (60), xz (60).

### SKIP (long tail < 55 repos, or ambiguous)

mbedtls / lz4 / sqlite / freetype / catch2 / gsl / rapidjson / wolfssl and everything below
(folly 28 … msgpack 1).

Also **SKIP windows_base for symbol-linking** despite its 416-repo breadth: it is almost entirely
`<windows.h>` / `<atlbase.h>` includes against leaked SDK kits with no clean symbol namespace —
include-resolution only, **no symbol cross-link**.

### Net recommendation

Cross-link **~10 high-value libs** for the bulk of cross-repo benefit:
**STL, boost, libc (decls-only), googletest, zlib, openssl, abseil, protobuf, eigen, fmt** —
and **skip the ~30-lib long tail**. Biggest cost/risk items are **linux+glibc** (huge, unprefixed)
and the **11-way STL provider multiplicity** — mitigate by canonicalizing each family to a **SINGLE
provider tree**.

| decision | families | rationale |
|---|---|---|
| TIER 1 cross-link | libstdcxx_STL, boost, libc_posix (decls-only) | top-3 breadth; canonicalize to one provider; libc symbol-link off |
| TIER 2 cross-link | googletest, zlib, openssl_crypto, abseil, fmt, eigen, protobuf | high breadth, clean single provider, prefixed symbols safe |
| TIER 3 optional | cpython, zstd, cuda_cccl, libevent, curl, libpng, libjpeg, glib, nlohmann_json, xz | 60–99 breadth, cheap, include-resolution only |
| SKIP | mbedtls, lz4, sqlite, freetype, catch2, gsl_microsoft, rapidjson, wolfssl, folly, libuv, asio, flatbuffers, benchmark, spdlog, c_ares, harfbuzz, wil_microsoft, mimalloc, lmdb, simdjson, range_v3, libsodium, capnproto, leveldb, msgpack | long tail < 55 repos or ambiguous namespace |
| SKIP (symbol) — include-only | windows_base | 416 breadth but no clean symbol namespace |

## Artifacts

- Ranked JSON (authoritative): `/Volumes/external/sources/cppmega/outputs/crossrepo/base_lib_usage_ranked.json`
- Membership manifest: `/Volumes/external/sources/cppmega/outputs/pr_ingest/repo_list.json`
- This report: `/Volumes/external/sources/cppmega.mlx/outputs/crossrepo/base_lib_crosslink_candidates.md`
