import { useState } from "react";
import { ADAPTERS, BRICKS, type BrickCategory } from "@/lib/bricks";
import { T, accentForCategory, accentVar, CATEGORY_ICON } from "@/theme";
import { HelpIcon } from "@/components/HelpIcon";

export interface PaletteProps {
  onDragStart?: (kind: string, kindClass: "brick" | "adapter") => void;
}

function groupBy<T, K extends string>(
  items: readonly T[],
  fn: (x: T) => K,
): Record<K, T[]> {
  const out = {} as Record<K, T[]>;
  for (const it of items) {
    const k = fn(it);
    (out[k] ??= []).push(it);
  }
  return out;
}

function humanize(cat: string): string {
  return cat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function Palette({ onDragStart }: PaletteProps): JSX.Element {
  const byCat = groupBy(BRICKS, (b) => b.category);
  const cats = (Object.keys(byCat) as BrickCategory[]).sort();
  // All sections expanded by default; clicking a header collapses it.
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggle = (key: string) =>
    setCollapsed((c) => ({ ...c, [key]: !c[key] }));

  return (
    <aside
      data-testid="palette"
      style={{
        width: 232, background: T.surface, padding: "14px 12px",
        fontFamily: T.font, fontSize: 12, color: T.text,
        overflowY: "auto", borderRight: `1px solid ${T.border}`,
        display: "flex", flexDirection: "column", gap: 4,
      }}
    >
      <div style={{ padding: "0 4px 6px" }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>
          Node Library
        </h3>
        <div style={{ color: T.textMuted, fontSize: 11, marginTop: 2 }}>
          Drag a node onto the canvas
        </div>
      </div>

      {cats.map((cat) => {
        const accent = accentForCategory(cat);
        const isOpen = !collapsed[cat];
        return (
          <section key={cat}>
            <button type="button"
                    className="vb-section-head"
                    aria-expanded={isOpen}
                    onClick={() => toggle(cat)}>
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: accent, fontSize: 13 }} aria-hidden="true">
                  {CATEGORY_ICON[cat]}
                </span>
                {humanize(cat)}
              </span>
              <span className={`vb-section-chevron${isOpen ? " vb-open" : ""}`}
                    aria-hidden="true">›</span>
            </button>
            {isOpen && (
              <div style={{ display: "flex", flexDirection: "column",
                            gap: 5, padding: "2px 0 8px" }}>
                {byCat[cat].map((b) => (
                  <div
                    key={b.kind}
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData(
                        "application/x-cppmega-brick", b.kind);
                      e.dataTransfer.effectAllowed = "copy";
                      onDragStart?.(b.kind, "brick");
                    }}
                    data-testid={`palette-brick-${b.kind}`}
                    className="vb-palette-item"
                    style={accentVar(accent)}
                    title={`${b.label} [${b.kind}]`}
                  >
                    <span className="vb-chip" aria-hidden="true"
                          style={{ width: 24, height: 24, fontSize: 13 }}>
                      {CATEGORY_ICON[cat]}
                    </span>
                    <span style={{ minWidth: 0, overflow: "hidden",
                                   textOverflow: "ellipsis",
                                   whiteSpace: "nowrap", flex: 1 }}>
                      {b.label}
                    </span>
                    {/* V7-Q09: per-brick "?" with what/why/I/O/norm
                        popup. Stop drag propagation so opening the
                        modal doesn't also start a brick drag. */}
                    <span
                      draggable={false}
                      onDragStart={(e) => { e.preventDefault();
                                            e.stopPropagation(); }}
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                    >
                      <HelpIcon topic={`brick_${b.kind}`} />
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>
        );
      })}

      <div className="vb-section-head" style={{ cursor: "default",
            marginTop: 4, color: T.textSecondary, fontSize: 12 }}>
        Adapters
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {ADAPTERS.map((a) => (
          <div
            key={a.kind}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(
                "application/x-cppmega-adapter", a.kind);
              e.dataTransfer.effectAllowed = "copy";
              onDragStart?.(a.kind, "adapter");
            }}
            data-testid={`palette-adapter-${a.kind}`}
            className="vb-palette-item"
            style={{ ...accentVar(T.accent), borderStyle: "dashed",
                     fontStyle: "italic" }}
            title={a.label}
          >
            <span aria-hidden="true" style={{ color: T.accent, fontSize: 13,
                  width: 24, textAlign: "center" }}>⇄</span>
            <span style={{ minWidth: 0, overflow: "hidden",
                           textOverflow: "ellipsis", whiteSpace: "nowrap",
                           flex: 1 }}>
              {a.label}
            </span>
            {/* V7-Q09: adapter help icon. */}
            <span
              draggable={false}
              onDragStart={(e) => { e.preventDefault();
                                    e.stopPropagation(); }}
              onMouseDown={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
            >
              <HelpIcon topic={`adapter_${a.kind}`} />
            </span>
          </div>
        ))}
      </div>
    </aside>
  );
}
