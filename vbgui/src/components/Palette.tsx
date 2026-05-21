import { ADAPTERS, BRICKS, colorFor, type BrickCategory } from "@/lib/bricks";

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

export function Palette({ onDragStart }: PaletteProps): JSX.Element {
  const byCat = groupBy(BRICKS, (b) => b.category);
  return (
    <aside
      data-testid="palette"
      style={{
        width: 220, background: "#f3f4f6", padding: 12,
        fontFamily: "system-ui, sans-serif", fontSize: 12,
        overflowY: "auto", borderRight: "1px solid #e5e7eb",
      }}
    >
      <h3 style={{ marginTop: 0, fontSize: 13 }}>Bricks</h3>
      {(Object.keys(byCat) as BrickCategory[]).sort().map((cat) => (
        <div key={cat} style={{ marginBottom: 12 }}>
          <div style={{ color: "#6b7280", fontSize: 11,
                        textTransform: "uppercase", marginBottom: 4 }}>
            {cat.replace(/_/g, " ")}
          </div>
          {byCat[cat].map((b) => (
            <div
              key={b.kind}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData("application/x-cppmega-brick", b.kind);
                e.dataTransfer.effectAllowed = "copy";
                onDragStart?.(b.kind, "brick");
              }}
              data-testid={`palette-brick-${b.kind}`}
              style={{
                padding: "4px 8px",
                background: "#fff",
                borderLeft: `4px solid ${colorFor(cat)}`,
                borderRadius: 3,
                marginBottom: 3,
                cursor: "grab",
              }}
            >
              {b.label}
            </div>
          ))}
        </div>
      ))}

      <h3 style={{ marginTop: 16, fontSize: 13 }}>Adapters</h3>
      {ADAPTERS.map((a) => (
        <div
          key={a.kind}
          draggable
          onDragStart={(e) => {
            e.dataTransfer.setData("application/x-cppmega-adapter", a.kind);
            e.dataTransfer.effectAllowed = "copy";
            onDragStart?.(a.kind, "adapter");
          }}
          data-testid={`palette-adapter-${a.kind}`}
          style={{
            padding: "4px 8px", background: "#fff",
            border: "1px dashed #9ca3af", borderRadius: 3,
            marginBottom: 3, cursor: "grab",
            fontStyle: "italic", color: "#374151",
          }}
        >
          {a.label}
        </div>
      ))}
    </aside>
  );
}
