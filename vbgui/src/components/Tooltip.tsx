// Tooltip — lightweight hover overlay with lazy catalog fetch.
//
// Wraps any inline element. On hover (250 ms delay), fetches the
// ExplainEntry via useCatalog and renders summary + first sentence
// of when_to_use; click ⓘ opens the full ExplainModal.

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useCatalog } from "@/hooks/useCatalog";
import type { RpcClient } from "@/lib/rpc";

export interface TooltipProps {
  rpc: RpcClient | null;
  category: string;
  name: string;
  children: ReactNode;
  /** Delay before tooltip shows on hover (ms). Default 250. */
  delayMs?: number;
  /** Show info icon (ⓘ) that opens a modal. */
  onInfoClick?: () => void;
  testId?: string;
}

export function Tooltip({
  rpc, category, name, children, delayMs = 250, onInfoClick, testId,
}: TooltipProps): JSX.Element {
  const [shown, setShown] = useState(false);
  const [pending, setPending] = useState<ReturnType<typeof setTimeout>
                                          | null>(null);
  const { entry, loading } = useCatalog(rpc, category, name, shown);

  function onEnter() {
    if (pending) clearTimeout(pending);
    const t = setTimeout(() => setShown(true), delayMs);
    setPending(t);
  }
  function onLeave() {
    if (pending) { clearTimeout(pending); setPending(null); }
    setShown(false);
  }
  useEffect(() => () => { if (pending) clearTimeout(pending); },
           [pending]);

  return (
    <span data-testid={testId ?? `tooltip-${category}-${name}`}
          onMouseEnter={onEnter}
          onMouseLeave={onLeave}
          style={{ position: "relative", display: "inline-flex",
                   alignItems: "center", gap: 4 }}>
      {children}
      {onInfoClick && (
        <button data-testid={`tooltip-info-${category}-${name}`}
                onClick={(e) => { e.stopPropagation(); onInfoClick(); }}
                title="Show full explanation"
                style={{ background: "transparent", border: "none",
                         color: "#6b7280", padding: 0, fontSize: 11,
                         cursor: "pointer" }}>
          ⓘ
        </button>
      )}
      {shown && (
        <div data-testid={`tooltip-popup-${category}-${name}`}
             role="tooltip"
             style={{
               position: "absolute", bottom: "calc(100% + 6px)", left: 0,
               background: "#1f2937", color: "white", padding: "6px 8px",
               borderRadius: 4, fontSize: 11, lineHeight: 1.35,
               width: 280, zIndex: 200, pointerEvents: "none",
               boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
             }}>
          {loading && <span>Loading…</span>}
          {!loading && !entry && <span>No info available.</span>}
          {entry && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                {entry.summary}
              </div>
              <div style={{ color: "#d1d5db" }}>
                {entry.when_to_use.split(".")[0]}.
              </div>
              {onInfoClick && (
                <div style={{ marginTop: 4, fontStyle: "italic",
                              color: "#9ca3af" }}>
                  Click ⓘ for full details
                </div>
              )}
            </>
          )}
        </div>
      )}
    </span>
  );
}
