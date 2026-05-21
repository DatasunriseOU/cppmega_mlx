# cppmega Visual Builder (vbgui)

React 18 + @xyflow/react v12 + TypeScript shell for the Visual Builder
GUI. Talks to `cppmega_v4.jsonrpc` over POST `/rpc` and WS `/ws`.

See `VisualBuilderPlan.md` (repo root) for the full design.

## Quickstart

```bash
# install (NODE_ENV must allow devDeps)
NODE_ENV=development npm install

# typecheck + test
npm run typecheck
npm test

# dev server
npm run dev
# → http://localhost:5173

# production build (static bundle)
npm run build
```

## Layout

- `src/lib/` — JSON-RPC client, types mirror of Pydantic schema, brick
  + adapter metadata, ELK layout helper.
- `src/components/` — `BrickNode`, `AdapterNode`, `Palette`, `FlowCanvas`.
- `src/App.tsx` — wires the palette to the canvas; drag a brick from
  the left panel onto the canvas to add a node, drag handle-to-handle
  to connect.
- `tests/` — vitest + jsdom + Testing Library.

## Stages shipped

- F-B (this commit): canvas, 25 brick nodes (color-coded by category),
  6 adapter nodes (dashed), palette with drag-source, edge severity
  styling, ELK auto-layout helper, JSON-RPC client.

## Pending

- F-C: 5-tab sidebar (Loss / Optim / Rewriters / Sharding / Gotchas),
  top bar, bottom strip.
- F-D: anywidget shim + cppmega-builder-widget PyPI package.
- F-E: JupyterLite/Pyodide static bundle + GitHub Pages deploy.
- F-G: Universal Tokenizer Playground.
- F-H: Training Data Inspector.
