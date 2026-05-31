# cppmega.mlx — Grounded-KB + Formal-Verification Capability + Plan Report

Target: `/Volumes/external/sources/cppmega.mlx`
Date: 2026-05-31
Scope: HONEST capability assessment. READ-ONLY on cppmega.mlx except `.gkb/`.
Reuses the IMPLEMENTED `grounded_kb.formal` layer at `/Users/dave/sources/datagraph` (143 formal
eval tests collected here; ~149 across cloud_love incl. provenance). NEVER touches the nanochat venv.
Run env: `PYTHONPATH=/Users/dave/sources/datagraph /Users/dave/sources/datagraph/.venv-formal/bin/python`
(verified: z3 4.15.4 imports; all 5 detectors + `verify_consistency` MODE_A/MODE_B + `select_mode` present).

Grounded against: `/tmp/cppmega-recon.md`, `/tmp/cppmega-detect.md`,
`FORMAL-VERIFICATION-PLAN.md` §2.8 / §2.9 / §3a-bis, and the real module source
(`grounded_kb/formal/source_manifest.py:221` `detect_codebase`; `z3_verifier.py:70`
`verify_consistency`; `logic_router.py:31` `select_mode`). The source-capability manifest was
already produced read-only: `.gkb/source-manifest.json` (schema `gkb.source_capability_manifest/v1`).

---

## 0. TL;DR

cppmega.mlx is a C++/MLX + Metal + Python ML-kernel/training project — NOT a Cloudflare TS web app.
The formal layer's **source-capability gate (M0.5), git-provenance (E6), bd-provenance (E7), docs→facts
ingestion (M-I), and Z3 MODE-B doc/doc-consistency (E5, §3a-bis)** apply **as-built today**. The
layer's **code-as-axioms tier (E3 AST extract / E4 reachability / MODE-A entailment / TS routes /
drizzle)** does **NOT** apply as-built — it is hardcoded to a TS `src/` layout and TS/JS/Py-only
extensions. The **live-HTTP-API tier (VERIFIED_PROD)** does NOT apply (no served HTTP truth oracle);
the project's real runtime oracle is the **pytest/bench parity+kernel suite**, which the as-built
layer does not yet call. Two NEW adapters give full parity: (1) a **C++/Python AST extractor**
(clang AST or tree-sitter-cpp for the native core; stdlib `ast` for the Python side) feeding
`compile_axioms`; (2) a **"run-the-test" oracle** replacing the HTTP probe as the runtime arbiter.
The recommended **first-pass that works TODAY**: docs KB + git provenance + bd provenance + **MODE-B
doc-consistency** over cppmega's contract docs (ContractProbe.md, parity_anchors.md,
kernel_coverage_matrix.md, metal_kernel_policy.md, AGENTS.md), flagging internal contradictions
among the specs — with the one scoped, non-adapter prerequisite that the canonicalizer be seeded
with cppmega's domains (see §3).

---

## 1. Sources present on cppmega.mlx → WHICH parts of the formal layer apply

From the produced manifest (`.gkb/source-manifest.json`) and `/tmp/cppmega-detect.md`:

| source   | manifest reading | reality | formal-layer tier it gates |
|----------|------------------|---------|----------------------------|
| **docs** | PRESENT (91 files: docs=63, root=28) | PRESENT | docs→facts ingestion (M-I), MODE-B consistency |
| **git**  | PRESENT (depth 769, full provenance, head `930748a`) | PRESENT | git-provenance E6 / drift timeline |
| **bd**   | PRESENT (bd 1.0.3, 601 issues, 1191 jsonl rows) | PRESENT | bd-provenance E7 / status→verdict |
| **codebase** | **FALSE — false-negative** (`test -d src` → "src/ not found") | PRESENT (1382 .py, 44 .metal, 18 .cpp, 13 .hpp, 29 .h, 1+ .cu) | code-as-axioms E3/E4/MODE-A — DETECTOR GAP |
| **live_api** | ABSENT (correctly recorded; "no base_url … tier dropped") | ABSENT (no served HTTP truth oracle) | VERIFIED_PROD / live-e2e — N/A by design |

Mapping each requested capability to "applies as-built?":

### docs → facts  — **YES (applies as-built)**
`detect_docs` (`source_manifest.py:664`) reused `cli._iter_corpus` and found 91 markdown files via
the corpus globs already written into `.gkb/config.json`
(`["docs/**/*.md","*.md","cppmega_mlx/**/*.md","cppmega_v4/**/*.md"]`). The incremental ingestion
(`blocks.py` MarkdownHeaderTextSplitter + `incremental.py` RecordManager port) is markdown-native and
language-agnostic, so it ingests cppmega's specs into atomic Facts with stable `block_id`s and 0-cost
skip on unchanged files. No code change needed.

### git provenance  — **YES (applies as-built)**
`detect_git` (`:526`) validated full-depth provenance (769 commits, not shallow, head `930748a`).
The E6 index (`provenance/git_provenance.py`: `build_git_index`, `check_git` arbiter via
`git show`/`blame --porcelain`/`log -S` pickaxe) reads committed git artifacts only — repo-shape
agnostic. The commit↔bd join chain (`commit.linked_bd_ids → bd-index`) works because both git and bd
are present. The per-symbol drift timeline (`drift-scan`) will work once a symbol set exists to track
(today that symbol set comes from docs/bd, not yet from code — see the code-axiom gap).

### bd provenance  — **YES (applies as-built)** *(bonus beyond the requested set)*
`detect_bd` (`:570`) found a committed beads workspace (601 issues: 81 open / 1 in-progress / 518
closed; 1191 jsonl rows). E7 (`provenance/bd_provenance.py`: `build_bd_index`, `check_bd`) reads
`.beads/*.jsonl` directly. bd attaches at DOC rank (intent, never out-ranks code/live). This is
directly useful here: ContractProbe.md already references bd memories
(`after-2026-05-02-space-nl-tokenizer-redesign`), so bd corroborates/dates doc claims.

### Z3 MODE-B doc/doc-consistency (no-code-axiom path)  — **YES (applies as-built, with a seed prereq)**
`logic_router.select_mode` (`:31`) deterministically returns **MODE B** whenever the manifest shows
neither code nor api present+capable for a key — which is exactly cppmega.mlx's current state
(codebase recorded absent, live_api absent). `z3_verifier.verify_consistency(..., z3_mode=MODE_B)`
(`:70`) asserts NO code axiom (post-checked, raises if violated at `:119`) and reports `INCONSISTENT`
+ `unsat_core` on a doc-vs-doc or policy-vs-policy contradiction over ONE canonical key. The Z3 engine
itself runs end-to-end (verified import + call). **Honest caveat (load-bearing):** binding goes
through `TierQuotaContext.bind(f)` (`z3_verifier.py:97`) — the canonicalizer/domain context is
**seeded for cloud_love's domains** (tier/quota/pricing/RBAC). A smoke with a cppmega-shaped fact
(`tokenizer.vocabSize` 65536 vs 131072) returned `NOT_FORMALIZABLE / no_binding` because cppmega's
subjects do not bind to cloud_love canonical vars. The MODE-B *machinery* applies as-built; making it
*bind cppmega's contract numbers* needs a cppmega domain context + a seeded
`.gkb/canonical-keys.json` (NN-layout constants, tokenizer vocab/special-ids, kernel
shapes/dtypes/tolerances). This is normal per-repo canonicalizer seeding (the §4.0 "bridge" artifact
is explicitly per-repo and reviewed), **not** a new adapter.

### Incremental ingest of docs  — **YES (applies as-built)**
M-I (`blocks.py` + `incremental.py`, RecordManager port) is markdown-first and torch-free; it indexes
exactly the 91 docs the manifest already found. Stable `block_id = sha256(path::header)` → a 1-section
edit re-extracts 1 block (1 Gemini call). No code change.

### What does NOT apply as-built
- **TS AST / route / drizzle / reachability axioms (E3/E4) — NO.** `compile_axioms`/`ast_extract`/
  `callgraph` target `@babel/parser` (TS) + stdlib `ast` (Py) over a `src/` tree; cppmega has no
  worker routes, no drizzle, no SPA router. The C++/Metal native core is invisible to it.
- **MODE-A entailment (code-as-axioms) — NO,** because there are no compiled code axioms (consequence
  of the above). Every key correctly falls to MODE B.
- **live-api runtime grounding / VERIFIED_PROD — NO.** No served HTTP truth oracle. The recon notes a
  FastAPI builder UI on :8765 with 25 JSON-RPC `*method*` modules, but that is a build/inspect/ablation
  control surface, not a model-serving correctness oracle. The REAL runtime oracle is the pytest/bench
  **parity + kernel** suite (numerical-correctness contracts) — which the as-built HTTP-probe tier does
  not call.

---

## 2. NEW adapters needed for full parity

Exactly two, matching the two missing tiers. Both are additive to the `grounded_kb.formal` package
(not to cppmega.mlx).

### Adapter A — C++/Python AST extractor for code axioms (replaces the TS `src/` assumption)
Three concrete defects in the as-built detector/extractor (confirmed in source):
1. `detect_codebase` (`source_manifest.py:223`) hardcodes `src = os.path.join(repo, "src")` and
   IGNORES `cfg` for the scan root — so cppmega's package dirs (`cppmega_mlx/`, `cppmega_v4/`) are
   never scanned ⇒ false `codebase.present=false`.
2. `_CODE_EXTS = (".ts",".tsx",".js",".jsx",".py")` (`:218`) — excludes `.cpp/.h/.hpp/.cu/.mm/.metal`,
   so even with the root fixed the native core is uncounted and `primary_lang` mis-selects.
3. `ast_extract`/`callgraph` use `@babel/parser` — no C++/CUDA/Metal parser.

Adapter A:
- Make `detect_codebase` take scan roots + extra extensions from `cfg` (`code.roots`,
  `code.extensions`) — small, config-driven; already recommended in `/tmp/cppmega-detect.md`.
- Add a **C++/Metal/CUDA AST extractor** (clang AST via `libclang`, or `tree-sitter-cpp`) and reuse
  the **stdlib `ast`** path (already present for Python) for the `cppmega_mlx`/`cppmega_v4` Python.
  Emit the same provenanced (`file:line`) axiom nodes `compile_axioms` consumes.
- Code axioms worth extracting (from §6 of recon): NN-layout constants (NAM56R depth/pattern/layer
  splits, MoE expert/top_k), tokenizer vocab/special-id counts, kernel shape/dtype/backward-coverage
  constraints, fusion rules. These let MODE-A entailment + authority reconciliation (E5) judge a doc
  claim against the code literal — the highest-value upgrade.
- Reachability (E4 `wired`/`NOT_WIRED`) is **lower value here**: an ML-kernel lib has no
  declared-but-404 route class. A meaningful analogue would be "kernel registered in the
  dispatch/routing table but never reached by the build graph" — possible but secondary.

### Adapter B — a "run-the-test" runtime oracle (replaces the HTTP probe arbiter)
The project's source-of-truth IS its tests (recon §5: `testpaths=["tests"]`, markers `parity`,
`kernel`, `training`, `bench`, `distributed`; bench harnesses `m03_cuda_logits_parity_harness.py`,
`m05_fastmtp_parity_harness.py`). A claim like a kernel tolerance or a parity anchor is grounded by
**RUNNING the bench/test**, exactly the project's `source-of-truth=tests` principle.

Adapter B design — a **test-pass oracle** that substitutes for the live-api tier:
- New `ApiProv`-analogue `TestProv{test_id, marker, assertion, expected}` whose `check_test` arbiter
  (mirroring `provenance.check_api`, `provenance.py`) **runs** `pytest -m parity <nodeid>` /
  `-m kernel` or invokes a `bench/*parity*` harness in a subprocess, parses pass/fail + the asserted
  numeric tolerance, and returns `OK(observed=…)` / `MISMATCH(...)`. Fail-CLOSED + fail-LOUD
  (Rule #1): a test that errors/can't-run RAISES (operational), a definite FAIL is `CONTRADICTED`, a
  PASS is `VERIFIED` at a new `source_type=test` rank slotting where `live_api` sits (the runtime/
  executed-truth tier). This is a true runtime oracle — the test ACTUALLY EXECUTES the kernel.
- It plugs into the same E2 bounded verify-loop (`verify_atom_provenance`) and the same
  `compose_verdict` state machine; only the arbiter changes (run-test instead of curl).
- The JSON-RPC `compile_trace`/`ckpt_inspect`/`catalog` methods are a SECONDARY introspection oracle
  (a softer arbiter), not the numerical truth.

Both adapters preserve Rule #1: every outcome is a DEFINITE verdict, explicit `NOT_FORMALIZABLE`, or a
RAISE with WHERE+WHAT — no silent fallback, no guessed binding, no coerce-unknown-to-green.

---

## 3. Recommended first-pass that works TODAY (as-built layer)

Build a **docs KB + git provenance + bd provenance + MODE-B doc-consistency** over cppmega's contract
docs, flagging internal contradictions among the specs. Nothing above needs Adapter A or B.

Confirmed contract docs present on disk:
`AGENTS.md`, `ContractProbe.md` (+ `docs/contract_probe_schema.json`), `docs/parity_anchors.md`,
`docs/kernel_coverage_matrix.md`, `docs/metal_kernel_policy.md`, `Auto-FusionLayerBricks.md`,
`ModelBuildSpec.md`, `ParallelismSpec.md`, `E2EMatrix.md`, `CLAUDE.md`, plus the
`VisualBuilderSpec-v2..v9` family (treat *Spec* as contracts, *Plan* as notes).

Steps (all under the existing run env; only writes under `.gkb/`):
1. **`gkb build --incremental`** — ingest the 91 docs into atomic Facts (markdown splitter + ledger;
   the corpus globs in `.gkb/config.json` are already correct). 0-cost skip on re-run.
2. **`gkb detect-sources`** — already done; manifest persisted. Re-run folds into build.
3. **`gkb index-git` + `gkb index-bd`** — build the provenance indexes (E6/E7) so each doc fact can be
   anchored to its introducing commit and any tracking bd issue (e.g. ContractProbe.md's referenced
   bd memory). Drift timeline available for any doc-asserted symbol.
4. **Seed the cppmega canonicalizer (the one prerequisite, NOT an adapter):** author a per-repo
   `.gkb/canonical-keys.json` + a cppmega `DomainContext` so contract numbers bind to canonical vars —
   e.g. `tokenizer.vocabSize`, `tokenizer.specialIdCount`, `model.depth`, `moe.numExperts`,
   `moe.topK`, `kernel.gemm.kAlign`. This is the §4.0 "bridge" artifact, explicitly per-repo and
   reviewed/tested. Without it, MODE-B binds nothing (verified: synthetic cppmega facts returned
   `NOT_FORMALIZABLE/no_binding`).
5. **`gkb verify-logic --z3-mode auto`** — `select_mode` returns **MODE B** for every key (no code/api
   present), running pure doc/doc + doc/policy consistency. Target contradictions to hunt:
   - tokenizer vocab: ContractProbe.md says local/profile vocab **65536** while megacpp tokenizer is
     **131072** — verify these are stated on the SAME canonical key only where they should agree
     (the canonicalizer must NOT conflate `local` vs `megacpp` — they are different keys; a real
     contradiction is two docs disagreeing on the SAME tokenizer's vocab).
   - parity anchors vs CHANGELOG receipts: e.g. "all 10 TileLang→Metal kernels shipped" /
     "407/407 kernel tests PASS" claims vs kernel_coverage_matrix.md's per-op shipped/partial/must-write
     status — flag where a receipt asserts done but the coverage matrix says partial/must-write.
   - metal_kernel_policy.md NO-silent-fallback contract vs any doc that describes an auto-fallback path.
   - NAM56R layout constants stated in more than one doc (parity_anchors vs ModelBuildSpec) — must agree.
6. **VERIFY.md scoreboard + `formal-findings.jsonl`** — every finding records `z3_mode="B"`, the
   `source_type` pair, `unsat_core` fact ids, and (E6/E7) the anchoring commit + bd issue. Ceiling is
   `CONSISTENT_UNVERIFIED` (no code/api oracle present) — **zero false code/api verdicts** by design.

Deliverable of the first-pass: a verified, non-contradictory docs KB over cppmega's specs, with every
spec claim dated to a commit and (where applicable) a bd issue, and any internal spec-vs-spec
contradiction surfaced with an unsat-core citing the two conflicting doc lines. This is genuinely
useful and entirely honest about its ceiling.

---

## 4. Honest statement of what would be OVER-CLAIMING

- **"We verified the kernels / parity / tolerances."** FALSE today. With docs-only + MODE B we verify
  the SPECS are mutually consistent and dated — NOT that the code or kernels honor them. Numerical
  correctness requires Adapter B (run-the-test oracle). Saying otherwise violates Rule #1.
- **"The KB grounds claims against the code."** FALSE today. `codebase` is recorded ABSENT
  (detector gap) and there are no compiled code axioms; every key is MODE B. There is NO MODE-A
  entailment, NO `wired`/`NOT_WIRED`, NO drizzle/route axiom. A doc claim is only checked against
  OTHER docs/policies, never against the C++/Metal/Python source.
- **"Reachability / declared-but-404 detection works here."** N/A — that verdict class is a TS-web
  artifact; an ML-kernel lib has no route dispatch table. Reporting `NOT_WIRED` here would be
  meaningless.
- **"VERIFIED_PROD / live grounding."** FALSE — no served HTTP truth oracle; the live_api tier is
  correctly recorded absent. The builder JSON-RPC surface is introspection, not a numerical oracle.
- **"MODE B already catches cppmega contradictions out of the box."** OVER-CLAIM until the per-repo
  canonicalizer is seeded — verified that an unseeded cppmega fact yields `NOT_FORMALIZABLE/no_binding`.
  The machinery applies; the binding for cppmega's domains is a (small, scoped) prerequisite.
- **"detect_codebase=false means cppmega has no code."** FALSE — it is a detector false-negative from
  the hardcoded `src/`-only, TS-only assumption. cppmega is overwhelmingly source code.
- **Authority ladder shape.** With code/api absent, the present ladder is `['doc','policy']`
  (manifest-confirmed). bd attaches at doc rank; git is a provenance qualifier, not a truth peer. No
  fact can reach `code(reachable)` or `live_api` authority until Adapter A/B land.

---

## 5. Bottom line

- **Sources present:** docs (91 md), git (full, 769 commits), bd (601 issues). codebase = real but
  detector-false-negative; live_api = genuinely absent.
- **Applies as-built:** docs→facts ingestion (M-I), git provenance (E6), bd provenance (E7), Z3
  **MODE-B** doc/doc + doc/policy consistency (E5, §3a-bis) — with a per-repo canonicalizer seed as
  the one scoped prerequisite (not an adapter).
- **Needs NEW adapters for full parity:** (A) a config-driven C++/Metal/CUDA + Python AST extractor
  (clang AST / tree-sitter-cpp + stdlib `ast`) feeding `compile_axioms` for MODE-A entailment +
  authority reconciliation; (B) a **"run-the-test" oracle** (pytest `-m parity`/`-m kernel` + bench
  parity harnesses) as the runtime arbiter replacing the HTTP probe — the project's
  source-of-truth=tests principle made into the executed-truth tier.
- **Recommended first-pass (today):** docs KB + git + bd + MODE-B spec-consistency over ContractProbe.md,
  parity_anchors.md, kernel_coverage_matrix.md, metal_kernel_policy.md, AGENTS.md (+ the Spec family),
  flagging internal contradictions among the specs, every claim dated to a commit/bd issue. Ceiling:
  `CONSISTENT_UNVERIFIED`, zero false code/api verdicts.
