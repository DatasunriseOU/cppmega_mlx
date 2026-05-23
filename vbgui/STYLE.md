# cppmega.mlx Visual Builder — Style Guide

Single source of truth for visual decisions in
`vbgui/src/components/*`. Companion to `VisualBuilderSpec-v9.md`.

If you're writing inline styles in a `.tsx` file, the answer is
**always** somewhere on this page. If it isn't, propose an addition
via a PR — don't introduce a one-off hex literal.

## 1. Colour

### 1.1 The rule

> No hex literals in `vbgui/src/components/*`. Use tokens from
> `@/theme`. The `HexColorRatchet` test enforces this — adding a hex
> literal fails CI unless you bump `tests/fixtures/hex_color_baseline.json`.

### 1.2 Token catalogue

| Token | CSS var | Use for |
|---|---|---|
| `T.bg` | `--vb-bg` | Page background (outermost) |
| `T.bgCanvas` | `--vb-bg-canvas` | FlowCanvas background |
| `T.surface` | `--vb-surface` | TopBar, Sidebar, modal body |
| `T.surface2` | `--vb-surface-2` | Popovers, dropdown menus |
| `T.surface3` | `--vb-surface-3` | Form inputs, chips |
| `T.surfaceLayered1` (V9) | `--vb-surface-layered-1` | Card on card (1st level) |
| `T.surfaceLayered2` (V9) | `--vb-surface-layered-2` | Canvas BrickNode |
| `T.surfaceLayered3` (V9) | `--vb-surface-layered-3` | Selected node |
| `T.glass` | `--vb-surface-glass` | Tooltip glass-pane |
| `T.border` | `--vb-border` | Default borders |
| `T.borderSoft` | `--vb-border-soft` | Hairline dividers |
| `T.borderStrong` | `--vb-border-strong` | Emphasised borders (active) |
| `T.text` | `--vb-text` | Body text |
| `T.textSecondary` | `--vb-text-secondary` | Labels, hints |
| `T.textMuted` | `--vb-text-muted` | Disabled / placeholder |
| `T.accent` | `--vb-accent` | Primary action (run, save) |
| `T.accentStrong` | `--vb-accent-strong` | Pressed / hover state |
| `T.accentSoft` | `--vb-accent-soft` | Accent backgrounds |
| `T.accentContrast` | `--vb-accent-contrast` | Text on accent surfaces |
| `T.success` | `--vb-success` | Pass, OK, finished |
| `T.warning` | `--vb-warning` | Caution, slow, untested |
| `T.danger` | `--vb-danger` | Error, blocked, destructive |
| `T.info` | `--vb-info` | Neutral informational |
| `T.liveAccent` (V9) | `--vb-live-accent` | Live train banners + pills |
| `T.liveAccentSoft` (V9) | `--vb-live-accent-soft` | Live train backgrounds |

### 1.3 Semantic mapping

| Intent | Use | Don't use |
|---|---|---|
| "Run is failing" | `T.danger` | `T.warning` (= caution, not failure) |
| "Run is in flight" | `T.liveAccent` | `T.warning` (= caution) |
| "Result needs review" | `T.warning` | `T.danger` (= failure) |
| "Tap to perform action" | `T.accent` | `T.success` (= confirmation, not invitation) |
| "Hint / label" | `T.textSecondary` | `T.textMuted` (= disabled) |
| "Placeholder / disabled" | `T.textMuted` | `T.textSecondary` |

### 1.4 Per-category brick accents

`CATEGORY_ACCENT` map in `theme.ts` covers all 8 brick categories.
Don't override unless you have a UX-level reason; let
`accentForCategory(kind)` decide.

## 2. Typography

```css
--vb-font:      ui-sans-serif, system-ui, ...
--vb-font-mono: ui-monospace, "SF Mono", ...
```

| Use | Family | Size | Weight |
|---|---|---|---|
| TopBar primary | `T.font` | 12 | 400 |
| Project name input | `T.font` | 12 | 600 |
| Sidebar tab label | `T.font` | 12 | 500 (active 600) |
| Modal headline | `T.font` | 16 | 600 |
| Modal body | `T.font` | 13 | 400 |
| Form label | `T.font` | 11 | 500 |
| Form input value | `T.font` | 12 | 400 |
| Code / monospace value | `T.fontMono` | 11 | 400 |
| Live stats (loss=…) | `T.fontMono` | 11 | 600 |
| Empty-state headline | `T.font` | 14 | 600 |
| Empty-state hint | `T.font` | 12 | 400 |

## 3. Spacing scale (4-px grid)

| Token | Pixels | Use |
|---|---|---|
| `--vb-space-0` | 0 | reset |
| `--vb-space-1` | 4 | icon ↔ label gap |
| `--vb-space-2` | 8 | within-row gap |
| `--vb-space-3` | 12 | between rows |
| `--vb-space-4` | 16 | between sections |
| `--vb-space-5` | 20 | modal padding |
| `--vb-space-6` | 24 | section headers |
| `--vb-space-8` | 32 | between major regions |

Inline-styles still use raw numbers (`gap: 8`) because the system
isn't fully migrated to CSS Modules. Pick the nearest grid step;
**never use 5, 7, 9, 11**.

## 4. Radius

| Token | Use |
|---|---|
| `T.radiusSm` (4) | inputs, small chips |
| `T.radiusMd` (6) | buttons, cards |
| `T.radiusLg` (8) | modals, popovers |
| `T.radiusXl` (12) | hero cards |
| `T.radiusPill` (999) | live status pills, tabs |

## 5. Elevation (V9)

| Token | Shadow | Use |
|---|---|---|
| `T.elevationLow` | `0 1px 2px rgba(0,0,0,.18)` | Resting cards, sidebar |
| `T.elevationMid` | `0 4px 12px rgba(0,0,0,.22)` | Popovers, dropdowns |
| `T.elevationHigh` | `0 12px 28px rgba(0,0,0,.32)` | Modals, full-screen overlays |

Legacy `T.shadowPanel` (= low) and `T.shadowPop` (= mid) still work.

## 6. Layout primitives

### 6.1 Grid

Top-level grid (see Spec §1 ASCII). All new top-level components
must fit a row in this grid — no floating panels outside the
defined regions, except modals (centered, portal'd to body).

### 6.2 Z-index scale

| Range | Use |
|---|---|
| 0-9 | Canvas + sidebar default |
| 10-19 | TopBar dropdown menus |
| 20-29 | Sticky inputs, BottomStrip |
| 30-39 | LiveTrainPanel slide-up |
| 40-49 | Popovers (CanvasToolbar) |
| 100+ | Modals (HelpModal, LossSurfaceModal, KeyboardShortcutsOverlay) |
| 1000+ | Directory pickers, file dialogs |

## 7. Data-testid conventions

- Kebab-case, prefixed with component name.
- Stable per identity (use entity id, not array index).
- Empty-state: `<tab>-empty-state`.
- Popover: `<owner>-popover-<key>`.
- Group containers (UX#7): `<component>-group-{left,center,right}`.

## 8. Component primitives roadmap

| Primitive | Owner spec | Status |
|---|---|---|
| `MemoryBar` (compact/cluster) | UX#6 | Shipped `23bb694` |
| `HelpModal` (portal) | UX#0 | Shipped `S3615` |
| `FeatureInjectionBar` (chips) | UX#1 | Shipped `6fe5518` |
| `DimEnvEditor` (sidebar host) | UX#2 | Shipped `7bc0b43` |
| `TrainOpsTab` | UX#3 | Shipped `7bc0b43` |
| `TopBar` (3 groups + Precision) | UX#7 | Shipped `5a4e314` |
| `DraftTabsStrip` | U01 | V9 planned |
| `CanvasToolbar` | U02 | V9 planned |
| `EmptyState` | U06 | V9 planned |
| `BrickChip` | U07 | V9 planned |
| `KeyboardShortcutsOverlay` | U10 | V9 planned |

## 9. Forbidden patterns

- No hex literals (see §1.1). Theme tokens only.
- No `backdropFilter: blur(...)` over the canvas — causes Chrome
  per-frame recomposite flicker (root cause of UX#0).
- No `position: fixed` modals without React Portal — transformed
  ancestors break containing-block (root cause of UX#0 part 2).
- No tab-bar with 10+ tabs in a 320px sidebar without grouping (U05).
- No new full-width canvas-top strips (canvas chrome lives in
  CanvasToolbar / TopBar / Sidebar, period).
- No `useEffect(() => setLocal(parentProp), [parentProp])` with
  default-array parent prop — infinite render loop (UX#1 fix).
- No `cursor: pointer` on visual chip body if there's no click handler.
- No emoji in code or test output unless the user explicitly asked.
- No comments that explain WHAT the code does — code is self-evident
  from identifiers. Comments only when explaining WHY a non-obvious
  decision was made.

## 10. Migration policy

- New components: must follow this guide from day 1.
- Existing components: migrate file-by-file under V9-U03. Each
  migration commit must drop the file's count in
  `tests/fixtures/hex_color_baseline.json`.
- When migrating, also align typography + spacing to §2/§3 (do
  not just substitute colours).
- If you find a value not in the catalogue and you can't justify a
  one-off, raise it as a token addition in `theme.css` + `theme.ts`.
