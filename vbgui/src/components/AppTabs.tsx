// Top-level view switcher used by App.tsx. Canvas vs Tokenizer vs
// Data vs Gallery (V7-F58 per-preset sortable report).

export type AppTab = "canvas" | "tokenizer" | "data"
  | "gallery" | "sweep" | "tokmatrix" | "inference";

export interface AppTabsProps {
  active: AppTab;
  onChange: (t: AppTab) => void;
}

const TABS: { key: AppTab; label: string }[] = [
  { key: "canvas",     label: "Canvas" },
  { key: "tokenizer",  label: "Tokenizer Playground" },
  { key: "tokmatrix",  label: "Tokenizer Matrix" },
  { key: "data",       label: "Data Inspector" },
  { key: "gallery",    label: "Gallery" },
  { key: "sweep",      label: "Scaling Sweep" },
  { key: "inference",  label: "Inference" },
];

export function AppTabs({ active, onChange }: AppTabsProps): JSX.Element {
  return (
    <nav role="tablist" data-testid="app-tabs"
         style={{ display: "flex", gap: 4, padding: "4px 8px",
                  background: "var(--vb-surface)",
                  borderBottom: "1px solid var(--vb-border)",
                  fontFamily: "var(--vb-font)", fontSize: 12 }}>
      {TABS.map((t) => (
        <button key={t.key}
                role="tab"
                aria-selected={active === t.key}
                data-testid={`app-tab-${t.key}`}
                onClick={() => onChange(t.key)}
                style={{
                  padding: "4px 10px",
                  border: "none",
                  background: active === t.key
                    ? "var(--vb-surface-3)"
                    : "transparent",
                  color: active === t.key
                    ? "var(--vb-text)"
                    : "var(--vb-text-secondary)",
                  borderRadius: 3,
                  borderBottom: active === t.key
                    ? "2px solid var(--vb-accent)"
                    : "2px solid transparent",
                  cursor: "pointer", fontSize: 12,
                  fontWeight: active === t.key ? 600 : 400,
                }}>
          {t.label}
        </button>
      ))}
    </nav>
  );
}
