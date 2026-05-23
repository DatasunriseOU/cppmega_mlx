# VisualBuilderSpec-v9 — UX & Visual System Spec

Companion to `VisualBuilderPlan-v9.md`. Defines theme-token contracts,
data-testid conventions, component primitives, layout grid, and the
Playwright story for every U-epic.

## 1. Layout grid

Desktop ≥ 1024px (canonical):

```
┌──────────────────────────────────────────────────────────────────┐
│ TopBar (height 56px)                                             │
│ ┌──── group-left ────┐  ┌── group-center ──┐ ┌── group-right ──┐ │
│ │ project | drafts… │  │ topology compile │ │ memory undo etc │ │
│ │ preset             │  │ precision        │ │ save run-split  │ │
│ └────────────────────┘  └──────────────────┘ └─────────────────┘ │
├─────┬───────────────────────────────────────────────┬────────────┤
│  P  │  Canvas (FlowCanvas, ELK auto-align)          │  Sidebar  │
│  a  │                                               │  (320px)  │
│  l  │  Floating canvas toolbar (32px) ◄── U02       │  tabs:    │
│  e  │  ┌─┐                                          │  Spec /   │
│  t  │  │∥│ Parallel compose                         │  Diagnostics │
│  t  │  ├─┤                                          │  / Train  │
│  e  │  │+│ Insert into edge                         │           │
│     │  ├─┤                                          │           │
│     │  │⇄│ Transplant                               │           │
│  (160) ─┘                                           │           │
├─────┴───────────────────────────────────────────────┴────────────┤
│ BottomStrip (24px) ws status + build id + heartbeat               │
└──────────────────────────────────────────────────────────────────┘
```

Below 1024px (U08): center group wraps under left+right.

## 2. Theme tokens v2 (U04)

### 2.1 New CSS custom properties (`vbgui/src/theme.css`)

Add to `:root`:

```css
/* Layered surfaces — for nested cards inside cards */
--vb-surface-layered-1: rgba(255, 255, 255, 0.02);
--vb-surface-layered-2: rgba(255, 255, 255, 0.04);
--vb-surface-layered-3: rgba(255, 255, 255, 0.06);

/* Elevation shadows */
--vb-elevation-low:  0 1px 2px rgba(0, 0, 0, 0.18);
--vb-elevation-mid:  0 4px 12px rgba(0, 0, 0, 0.22);
--vb-elevation-high: 0 12px 28px rgba(0, 0, 0, 0.32);

/* Live-train accents */
--vb-live-accent:      #f59e0b;  /* amber-500 */
--vb-live-accent-soft: rgba(245, 158, 11, 0.16);
```

### 2.2 TS mirror (`vbgui/src/theme.ts`)

Append to `T`:

```ts
surfaceLayered1: "var(--vb-surface-layered-1)",
surfaceLayered2: "var(--vb-surface-layered-2)",
surfaceLayered3: "var(--vb-surface-layered-3)",
elevationLow:    "var(--vb-elevation-low)",
elevationMid:    "var(--vb-elevation-mid)",
elevationHigh:   "var(--vb-elevation-high)",
liveAccent:      "var(--vb-live-accent)",
liveAccentSoft:  "var(--vb-live-accent-soft)",
```

### 2.3 Hex → token migration table contract (U03.1)

`vbgui/MIGRATION_TABLE.md` columns:

| hex | T.token | category | notes |
|---|---|---|---|
| `#6b7280` | `T.textSecondary` | text | gray-500 |
| `#374151` | `T.text` | text | gray-700 (body) |
| `#94a3b8` | `T.textMuted` | text | slate-400 |
| `#e5e7eb` | `T.border` | border | gray-200 |
| `#dc2626` | `T.danger` | semantic | red-600 |
| `#b91c1c` | `T.danger` | semantic | red-700 (collapse) |
| `#10b981` | `T.success` | semantic | emerald-500 |
| `#16a34a` | `T.success` | semantic | green-600 |
| `#d97706` | `T.warning` | semantic | amber-600 |
| `#f59e0b` | `T.liveAccent` | semantic | amber-500 (new) |
| `#1e40af` | `T.accent` | accent | blue-800 |
| `#2563eb` | `T.accent` | accent | blue-600 |
| `#22d3ee` | `T.accent` | accent | cyan-400 (collapse) |
| `#0f172a` | `T.bg` | surface | slate-900 |
| `#1e293b` | `T.surface` | surface | slate-800 |
| `#334155` | `T.surface3` | surface | slate-700 |
| `#f3f4f6` | `T.surface2` | surface | gray-100 (light theme) |
| `#f9fafb` | `T.surface` | surface | gray-50 |

Files with > 20 hex literals get individual triage paragraphs in
`MIGRATION_TABLE.md`.

## 3. Component primitives

### 3.1 `DraftTabsStrip` (U01)

```tsx
interface Draft {
  id: string;
  name: string;          // user-editable
  spec: SpecState;       // full snapshot
  nodes: Node[];
  edges: Edge[];
  createdAt: number;     // ms epoch
  modifiedAt: number;
}

interface DraftTabsStripProps {
  drafts: readonly Draft[];
  activeDraftId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;       // new draft, becomes active
  onClose:  (id: string) => void;
  onRename: (id: string, name: string) => void;
}
```

testids:
- `draft-tabs-strip`
- `draft-tab-{id}` (clickable)
- `draft-tab-{id}-close` (× button)
- `draft-tab-new` (+ button)

Visual: browser-tab strip, active tab `T.surface` background +
top border `T.accent`, inactive `T.surfaceLayered1`. Close button
only on inactive tabs. Double-click name → inline rename.

### 3.2 `CanvasToolbar` (U02)

```tsx
interface CanvasToolbarProps {
  rpc: RpcClient | null;
  edges: Array<{ source: string; target: string }>;
  presets: readonly string[];
  onParallelCompose: (nodes: any[], edges: any[]) => void;
  onInsertIntoEdge:  (kind: string, edge: any) => void;
  onTransplant:      (kind: string, params: Record<string, unknown>) => void;
}
```

testids:
- `canvas-toolbar`
- `canvas-toolbar-icon-{parallel|insert|transplant}`
- `canvas-toolbar-popover-{parallel|insert|transplant}`

Layout: `position: absolute; left: 0; top: 50%; transform:
translateY(-50%); width: 32px;`. Inside the canvas viewport flex
parent — survives ELK relayout.

Popover anchored to icon's right edge, `T.elevationMid` shadow.
Click outside / Esc closes.

### 3.3 `EmptyState` primitive (U06)

```tsx
interface EmptyStateProps {
  icon: string;          // single glyph
  headline: string;
  hint?: string;
  cta?: { label: string; onClick: () => void };
  testid: string;
}
```

Used by LossTab, GotchasTab, MemoryMatrixTab, AblationsTab when
their data array is empty.

testid pattern: `{tab}-empty-state` (e.g. `gotchas-empty-state`).

### 3.4 `BrickChip` primitive (U07)

Shared by `Palette` items and `BrickNode` canvas nodes:

```tsx
interface BrickChipProps {
  kind: BrickKind;
  category: BrickCategory;
  variant: "palette" | "canvas";
  accent?: string;       // override CATEGORY_ACCENT lookup
  selected?: boolean;
  onSelect?: () => void;
}
```

Sets `--vb-node-accent` CSS variable via `accentVar(...)` regardless
of variant. Background differs (`T.surface` for palette,
`T.surfaceLayered2` for canvas), typography + accent treatment is
identical.

### 3.5 `KeyboardShortcutsOverlay` (U10)

```tsx
interface ShortcutEntry {
  key: string;           // human-printable, e.g. "Cmd+Z" or "?"
  label: string;
  section: "canvas" | "sidebar" | "train" | "help";
  handler: (e: KeyboardEvent) => void;
}
```

`useRegisterShortcut(entry)` hook (and `useShortcuts()` reader) live
in `src/state/shortcuts.ts`. Overlay reads the registry, groups by
section, renders modal triggered by `?`.

testid: `keyboard-shortcuts-overlay`, `shortcut-row-{key}`.

## 4. Data-testid conventions

- Kebab-case throughout.
- Prefix matches the component name (`draft-tab-...`,
  `canvas-toolbar-...`).
- Per-row identifiers use the row's stable id (not array index).
- Empty-state pattern: `<tab-name>-empty-state` + child
  `-empty-state-cta`.
- Popover pattern: `<owner>-popover-<key>`.

## 5. State persistence

- Drafts: `localStorage["vbgui_drafts_v1"]` — JSON array of full
  `Draft` objects. Migration key bumps the suffix when schema changes.
- Active draft: `localStorage["vbgui_active_draft_v1"]` — string id.
- Sidebar tab grouping preference (U05): `localStorage["vbgui_sidebar_group_v1"]`.
- Keyboard shortcuts overlay last-seen-at: `localStorage["vbgui_kb_overlay_seen_v1"]`
  — drives a one-time pulse hint on `?` glyph for first 3 sessions.

## 6. Playwright story (U-epic UAT)

### 6.1 Canonical V9 walk-through

```
1. Cold open → page shows empty canvas, empty LossTab CTA visible.
2. Press `?` → KeyboardShortcutsOverlay appears, lists ≥ 1 row per
   section. Press `?` again → closes.
3. Click `[+]` in DraftTabsStrip → new draft "Draft 2" appears, becomes
   active. Drag a brick onto canvas. Switch back to "Draft 1" via tab
   click → original canvas is restored. Switch to "Draft 2" → new
   canvas restored.
4. Click CanvasToolbar `∥` icon → popover with ParallelComposeBar
   opens. Compose 2 bricks → canvas updates. Press Esc → popover
   closes.
5. Resize window to 800px wide → TopBar groups re-flow (center
   below left+right). All testids still resolve.
6. Click `?` next to dim_env H input → HelpModal renders centered
   (regression for #0/#9 catalogue).
```

### 6.2 Hex ratchet check

`tests/HexColorRatchet.test.ts` re-runs CI-side. After U03 the
baseline JSON must show:
- Total ≤ 150 (down from 592).
- No file > 20 hex literals (target: most are 0).

## 7. Compatibility & rollout

- All new testids: ADD ONLY. Existing testids preserved.
- Theme token additions: ADDITIVE. No existing token renamed.
- Migration files: created, not replaced (`MIGRATION_TABLE.md`,
  `STYLE.md`).
- Drafts feature: gracefully degrades when `localStorage` unavailable
  (single "Untitled" tab, no persistence).
- Keyboard shortcuts: gracefully no-op when `document` undefined (SSR
  guard in `useRegisterShortcut`).

## 8. Acceptance counters (mirrors plan §6)

- 10 U-epics, 26 sub-tickets, all in bd under `cppmega-mlx-v9`.
- ≥ 25 new vitest tests across U01-U10.
- Hex baseline: 592 → ≤ 150.
- `STYLE.md` published at `vbgui/STYLE.md`.
- Spec referenced from every U-epic ticket description.
