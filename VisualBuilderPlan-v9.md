# VisualBuilderPlan-v9 — UX Polish & Visual System Closure

**Status**: narrowed & in progress 2026-08-01  
**Decision**: U01 + U02 ship as the V9 scope; U03–U10 are deferred to V10
because release blockers (P025–P040 data pipeline, P085–P096 hardware parity)
are consuming all P1/P2 capacity. U01/U02 were chosen because they deliver the
highest operator value per unit of work (discoverable drafts + persistent
canvas splicing toolbar).

**Driver**: post-V8 operator-feedback cycle. V8 closed the
gallery→train round trip (R01-R12). V9 closes the *visual surface*
the operator interacts with every minute: canvas chrome, sidebar
ergonomics, theme consistency, and discoverability.

> "Цвета, фонты, ux — куча компонентов и непонятно зачем. mtp это не
> компонент? зачем оно тут банером? memory bar длиннюща, как ты это
> покажешь на h100:8? ckpt path виден когда idle. parallel/insert/
> transplant — ugly full-width strips. draft 1 + new draft — для
> чего? TopBar 18 элементов, цвета непоследовательны."

## bd ticket ID mapping

| U#   | scope                                                      | priority | depends on |
|------|------------------------------------------------------------|----------|------------|
| U01  | Draft tabs strip (browser-tab style) in TopBar left group  | P2       | -          |
| U02  | Floating canvas toolbar (vertical 32px icon-strip)         | P2       | -          |
| U03  | Mass hex-literal migration — 592 → ≤ 150 across `vbgui/src/components` | P3 | UX#8 ratchet |
| U04  | Theme tokens v2 — layered surfaces + elevation scale       | P3       | U03        |
| U05  | Sidebar tab strip — collapse 10 tabs into grouped accordion | P2      | -          |
| U06  | Empty-state contracts for Loss/Optim/Memory/Gotchas tabs   | P3       | -          |
| U07  | BrickNode unified styling — Palette item == canvas chip    | P3       | U04        |
| U08  | TopBar responsive collapse — groups stack < 1024px         | P3       | UX#7       |
| U09  | Help icon catalogue audit — every topic has content or 404 | P2       | -          |
| U10  | Keyboard shortcuts overlay (? key)                         | P3       | -          |

## 1. Already done (do NOT re-open)

The operator-feedback redesign sprint closed these between
2026-05-23 and 2026-05-24 (commits `23bb694`, `7908932`, `6fe5518`,
`7bc0b43`, `5a4e314`, `7e1a031`):

| # | Title | Commit |
|---|---|---|
| #0 | HelpModal Portal — escape transformed ancestor (`position:fixed` containing-block trap) | `S3615` |
| #1 | FeatureInjectionBar dedupe → chip with `×N` count + per-chip remove | `6fe5518` |
| #2 | DimEnvEditor relocated to right Sidebar `Dimensions` tab | `7bc0b43` |
| #3 | TrainOptionsPanel + RunHistoryPicker → new `TrainOps` Sidebar tab | `7bc0b43` |
| #4 | TrainLiveControls conditional `{trainInFlight && ...}` | `7bc0b43` |
| #5 (interim) | Parallel / Insert / Transplant bars removed from canvas, hosted in TrainOps tab | `7bc0b43` |
| #6 | MemoryBar compact (100×18 pill) + topology-aware N-rank mini-bars | `23bb694` |
| #7 | TopBar 3 flex groups (left/center/right) + single Precision dropdown | `5a4e314` |
| #8 (ratchet) | `HexColorRatchet.test.ts` + per-file baseline + DimensionsTab migration | `7e1a031` |

## 2. Why v9 exists

V8 delivered the *journey*. V9 delivers the *finish*:
- discoverability (Draft tabs, keyboard shortcuts, help catalogue)
- visual consistency (theme tokens, BrickNode unity, mass hex
  migration ratcheted down from 592 to ≤ 150)
- canvas chrome (floating toolbar instead of buried sidebar splicing
  section; responsive TopBar)
- empty-state polish (so first-time operators see prompts instead of
  white space)

## 3. Goal

Operator opens the GUI cold. Without reading docs they can:
1. Recognise every clickable region from visual weight alone.
2. Find every action through ≤ 2 clicks (TopBar + Sidebar grouped).
3. Use `?` overlay to discover all keyboard shortcuts.
4. Switch between concurrent specs via Draft tabs.
5. Splice graphs (parallel/insert/transplant) from a fixed-position
   floating toolbar instead of scrolling a tab.

## 4. Stages — 10 U-epics × 1-4 sub-tasks ≈ 26 sub-tickets

### U01 — Draft tabs strip in TopBar (P2, 3 subs)

**Why**: operator currently sees `[Draft 1] [+ New Draft]` with no
visual taxonomy. Should be browser-tab-style strip wedged into
TopBar left group, right of `project-name`.

**Subs**:
- U01.1 `vbgui/src/components/DraftTabsStrip.tsx`: render N tabs
  `[Untitled] [Draft 1] [Draft 2 ×] [+]`. Active tab highlighted.
  Close button (×) only on inactive tabs.
- U01.2 `vbgui/src/state/drafts.ts`: `useDrafts()` hook persists
  drafts to `localStorage` under `vbgui_drafts_v1` (array of full
  spec snapshots + name + timestamp).
- U01.3 Vitest: switching draft swaps `nodes` + `edges` + `spec`
  on canvas; closing draft does NOT lose data of OTHER drafts.

### U02 — Floating canvas toolbar (P2, 4 subs)

**Why**: Parallel / Insert / Transplant currently live in
`TrainOpsTab` (a hidden tab). They should be a fixed-position
vertical strip 32px wide, left edge of canvas, between Palette
and FlowCanvas.

**Subs**:
- U02.1 `vbgui/src/components/CanvasToolbar.tsx`: vertical icon
  strip with three icons. Each icon opens a popover (positioned to
  the right of the icon) that contains the existing
  `ParallelComposeBar` / `InsertIntoEdgeBar` / `TransplantBar`.
- U02.2 Remove the splicing section from `TrainOpsTab` (revert
  linter's interim placement). Update test
  `tests/TrainOpsTab.test.tsx` to drop the splicing assertions.
- U02.3 Popovers close on click-outside and on Esc; icons show
  active-state ring while popover is open.
- U02.4 Vitest: 3 icons render; opening one closes any other
  already-open popover; ParallelComposeBar still fires `onCompose`.

### U03 — Mass hex-literal migration (P3, 4 subs)

**Why**: `tests/HexColorRatchet.test.ts` locks in baseline 592. The
target is ≤ 150 by end of V9 (75% reduction). High-frequency
literals: `#6b7280` (55), `#374151` (36), `#94a3b8` (30),
`#e5e7eb` (29).

**Subs**:
- U03.1 Map every hex literal to the closest semantic token in
  `src/theme.ts` — produce `vbgui/MIGRATION_TABLE.md` with rows
  `hex → T.token  category  notes`.
- U03.2 Migrate `Sidebar.tsx` + every file under `src/components/sidebar/`
  (target: sidebar drops to ≤ 10 hex total).
- U03.3 Migrate `BrickContextPanel.tsx` (currently 32 hex —
  highest single offender after TrainLiveControls).
- U03.4 Migrate `TrainLiveControls.tsx` (46 hex — relocate its
  ad-hoc colour table into `T.live*` extension if any unique values
  remain after migration).

### U04 — Theme tokens v2 — layered surfaces + elevation (P3, 2 subs)

**Why**: existing tokens (`T.surface`, `T.surface2`, `T.surface3`)
are flat — there's no notion of "card on card" or "modal vs
popover" depth. Adding `T.surfaceLayered{1,2,3}` and `T.elevation*`
gives the migration target enough semantic vocabulary.

**Subs**:
- U04.1 Extend `vbgui/src/theme.css` + `theme.ts` with:
  `surfaceLayered1/2/3` (each a darker variant of `surface` with a
  hairline border); `elevationLow/Mid/High` (box-shadow tokens);
  `liveAccent`, `liveAccentSoft` (semantic for live-train UI).
- U04.2 Pin the additions in `vbgui/STYLE.md` (see Spec §2).

### U05 — Sidebar tab strip overflow (P2, 3 subs)

**Why**: Sidebar now has 10 tabs (Loss, Optim, Rewriters, Side
Ch., Sharding, Gotchas, Dimensions, Train Ops, Ablations, Memory).
At width 320px with `flex: 1 0 33%` they wrap to 4 rows of 3 — eats
sidebar headroom.

**Subs**:
- U05.1 Group tabs into 3 sections (Spec / Diagnostics / Train) —
  visible header per section, tabs underneath as smaller chips.
- U05.2 OR: dropdown picker for less-used tabs (Ablations,
  Memory). A/B both, pick one with screenshots.
- U05.3 Vitest: every existing `sidebar-tab-{name}` testid still
  resolves (back-compat).

### U06 — Empty-state contracts (P3, 2 subs)

**Why**: opening LossTab / MemoryMatrixTab / GotchasTab cold gives
blank panels. Operator can't tell whether the tab is broken or
empty.

**Subs**:
- U06.1 Each of {LossTab, MemoryMatrixTab, GotchasTab,
  AblationsTab} renders an empty-state block when its data array
  is `[]`. Block = icon + headline + 1-sentence prompt + CTA.
  Example for GotchasTab: "✓ No issues yet — run Smoke to populate".
- U06.2 Vitest: each tab in empty state has
  `data-testid="<tab>-empty-state"` and renders the CTA text.

### U07 — BrickNode unified styling (P3, 2 subs)

**Why**: Palette items (left dock) and canvas nodes use different
chrome — Palette is white-bg + grey border, canvas is dark + accent
glow. Single brick should *look the same* in both places (different
backgrounds, identical typography + accent treatment).

**Subs**:
- U07.1 Extract shared `BrickChip` primitive used by both
  `Palette` items and `BrickNode` canvas nodes.
- U07.2 Vitest: Palette item and BrickNode for the same kind have
  same `data-brick-kind` + same `--vb-node-accent` CSS variable.

### U08 — TopBar responsive collapse (P3, 2 subs)

**Why**: UX#7 split into 3 flex groups. Below 1024px width they
overflow horizontally — need to stack groups onto 2 rows when
viewport is narrow.

**Subs**:
- U08.1 `@container` or `min-width` media query — when
  `< 1024px` stack the center group below left+right.
- U08.2 Vitest with jsdom-resize: at 800px width
  `top-bar-group-center.style.flexDirection === "column"` OR a
  wrap container.

### U09 — Help icon catalogue audit (P2, 3 subs)

**Why**: `HelpIcon topic="..."` references ~80 topics. Many may
have drifted (renames, broken slugs) — operator clicks `?` and
sees "missing" modal. Need a programmatic audit + autofix path.

**Subs**:
- U09.1 vitest scans every `<HelpIcon topic="X">` use, every
  `Tooltip name="X">`, and every `ExplainModal name="X">` in
  `vbgui/src/components/**/*.tsx`. For each, asserts the topic
  exists in `HELP_TOPICS` map OR a `catalog.explain` RPC response.
  Failing topics listed.
- U09.2 Add missing topics to `HelpIcon.tsx` HELP_TOPICS map (one
  paragraph each) OR remove the reference if the surface is dead.
- U09.3 Add `topic` autocomplete TS type so future references type-check.

### U10 — Keyboard shortcuts overlay (P3, 3 subs)

**Why**: power users want `?` to bring up a list of all shortcuts.
Currently only Cmd/Ctrl+Z (undo) + Shift+Cmd/Ctrl+Z (redo) are
hinted in button titles.

**Subs**:
- U10.1 `vbgui/src/components/KeyboardShortcutsOverlay.tsx`:
  modal triggered by `?` (and dismissed by `?` or Esc) listing
  every registered shortcut grouped by section (Canvas / Sidebar
  / Train / Help).
- U10.2 `vbgui/src/state/shortcuts.ts`: central registry — every
  bound shortcut registers `{key, label, section, handler}` so the
  overlay is generated, not hand-edited.
- U10.3 Vitest: pressing `?` opens overlay; overlay lists ≥ 1
  shortcut per section.

## 5. Workflow per epic (same as v5/v6/v7/v8)

1. Read `VisualBuilderSpec-v9.md` for the relevant U-section.
2. Create files per spec, with the listed data-testids.
3. Write vitest tests first (TDD); they should fail.
4. Wire the component; tests pass.
5. Run full `npx vitest run` from `vbgui/` to confirm zero regression.
6. Commit with `UX-U0#: <title>` message; close the bd ticket.
7. Push to main.

## 6. Acceptance counters (mirrors §4)

- 10 U-epics → 26 sub-tickets created in bd as
  `cppmega-mlx-v9/U01..U10` (one parent + 10 child epics).
- Each sub-ticket has acceptance criteria copied from §4.
- New vitest count: ≥ 25 (across all U-stages).
- `HexColorRatchet.test.ts` baseline drops from 592 to ≤ 150 by end
  of U03 (≥ 75% reduction).

## 6.5 V9 scope decision (2026-08-01)

After reviewing capacity against the August release blockers, the V9 epic is
scoped down to the two highest-value P2 items:

- **U01 — Draft tabs strip** (operator can switch between concurrent specs).
- **U02 — Floating canvas toolbar** (Parallel / Insert / Transplant are
  discoverable without opening a sidebar tab).

The remaining U03–U10 work is valuable but not release-critical.  It is
formally deferred to **V10** with the following rationale:

| Deferred | Reason |
|---|---|
| U03 mass hex migration | Large surface-area refactor; blocked behind U04 token design anyway. |
| U04 theme tokens v2 | Needs dedicated design slot after V9 surface stabilises. |
| U05 sidebar tab strip | Usability win but current wrapping is functional. |
| U06 empty states | Nice-to-have; can be added incrementally per tab. |
| U07 BrickNode unity | Depends on U04 tokens. |
| U08 responsive TopBar | Desktop viewport ≥ 1024 px is the current supported target. |
| U09 help catalogue audit | Important, but help topics are not blocking training workflows. |
| U10 keyboard shortcuts overlay | Only two shortcuts exist today; low ROI until more are bound. |

The bd epic `cppmega-mlx-zozw` and the U03–U10 child tickets are updated to
reflect this deferral.  U01/U02 tickets remain open and are the active V9 work.

## 7. Out of scope (V10+)

- True theming (user-switchable light/dark/contrast).
- Drag-to-reorder canvas nodes (current ELK auto-align only).
- Multi-user collaboration (CRDT spec sync).
- Mobile/tablet viewports (focused on ≥ 1024px desktop).
- Internationalisation (English-only UI text).

## 8. Done definition

V9 (scoped to U01 + U02) is done when:
- U01 and U02 epics closed in bd (sub-tickets closed).
- U03–U10 tickets formally deferred to V10 with explicit reason.
- `npx vitest run` from `vbgui/`: 0 failures in U01/U02 test files
  (counting only V9-relevant test files; pre-existing HelpModal/Palette/Backward
  failures are tracked separately).
- `VisualBuilderPlan-v9.md` + `VisualBuilderSpec-v9.md` + `vbgui/STYLE.md`
  shipped on `main` with the scope decision recorded.
- UAT footage captures Draft tabs (U01) + Canvas toolbar (U02).
