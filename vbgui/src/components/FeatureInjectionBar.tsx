/**
 * V8-R08 FeatureInjectionBar — mid-canvas chooser for inserting
 * MTP / IFIM / MHC rewriters or Engram bricks without leaving the
 * canvas surface.
 *
 * Source of truth: catalog.list_options('feature_injectors') — each
 * option's `paper_ref` is either "rewriter:<Name>" (dispatches a
 * rewriters.add action with sensible defaults) or "brick:<kind>"
 * (signals the parent to insert a brick node on the canvas).
 *
 * The bar keeps a local applied list (so the user sees what's been
 * injected this session) and emits onApply for each click.
 */

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { T } from "@/theme";

interface CatalogOption {
  name: string;
  summary: string;
  paper_ref: string;
}

export interface AppliedInjection {
  name: string;       // option name, e.g. "mtp_weighted"
  paper_ref: string;  // "rewriter:MTPRewriter" or "brick:engram"
}

export interface FeatureInjectionBarProps {
  rpc: RpcClient;
  onApply: (injection: AppliedInjection) => void;
  /** Optional list of already-applied injections; the bar uses this
   *  to populate the applied-list display when the parent persists. */
  applied?: AppliedInjection[];
  /** Optional click callback. Invoked when user clicks the chip body,
   *  useful to route sidebar tab changes. */
  onChipClick?: (name: string) => void;
  /** Optional remove callback. Called with the last-applied injection
   *  matching the chip the user clicked × on. Parent is responsible
   *  for popping the matching rewriter / brick. */
  onRemove?: (injection: AppliedInjection) => void;
}

const EMPTY_APPLIED: AppliedInjection[] = [];

export function FeatureInjectionBar({
  rpc, onApply, applied = EMPTY_APPLIED, onRemove, onChipClick,
}: FeatureInjectionBarProps): JSX.Element {
  const [options, setOptions] = useState<CatalogOption[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);
  const [local, setLocal] = useState<AppliedInjection[]>(applied);

  useEffect(() => {
    setLocal(applied);
  }, [applied]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await rpc.call<{ options?: CatalogOption[] }>(
          "catalog.list_options",
          { category: "feature_injectors" },
        );
        if (!cancelled) {
          const opts = r?.options ?? [];
          setOptions(opts);
          if (opts.length > 0 && !selected) {
            setSelected(opts[0].name);
          }
        }
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => { cancelled = true; };
  }, [rpc, selected]);

  function apply() {
    const opt = options.find((o) => o.name === selected);
    if (!opt) return;
    const injection = { name: opt.name, paper_ref: opt.paper_ref };
    setLocal((prev) => [...prev, injection]);
    onApply(injection);
  }

  function removeOne(name: string) {
    let popped: AppliedInjection | null = null;
    for (let i = local.length - 1; i >= 0; i--) {
      if (local[i].name === name) { popped = local[i]; break; }
    }
    if (!popped) return;
    setLocal((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].name === name) { next.splice(i, 1); break; }
      }
      return next;
    });
    if (onRemove) onRemove(popped);
  }

  const counts = new Map<string, { count: number; paper_ref: string }>();
  for (const a of local) {
    const prev = counts.get(a.name);
    counts.set(a.name, {
      count: (prev?.count ?? 0) + 1,
      paper_ref: a.paper_ref,
    });
  }
  const chipEntries = Array.from(counts.entries());

  return (
    <div data-testid="feature-injection-bar"
         style={{ display: "flex", alignItems: "center", gap: 6,
                  padding: "4px 8px", background: T.surface,
                  borderBottom: `1px solid ${T.border}`,
                  fontSize: 12, fontFamily: T.font, color: T.text }}>
      <strong style={{ color: T.accent, marginRight: 2 }}>Inject</strong>
      <label style={{ display: "flex", alignItems: "center", gap: 4, color: T.textSecondary }}>
        <select
          data-testid="feature-injection-dropdown"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={options.length === 0}
          style={{
            color: T.text,
            background: T.surface3,
            border: `1px solid ${T.border}`,
          }}
        >
          {options.length === 0 && <option value="">…</option>}
          {options.map((o) => (
            <option key={o.name} value={o.name}>{o.name}</option>
          ))}
        </select>
      </label>
      {selected && (
        <span data-testid="feature-injection-summary"
              style={{ color: T.textSecondary, fontWeight: 500 }}>
          {options.find((o) => o.name === selected)?.summary ?? ""}
        </span>
      )}
      <button
        data-testid="feature-injection-apply"
        onClick={apply}
        disabled={!selected}
        style={{
          padding: "2px 8px",
          background: T.surface3,
          color: T.text,
          border: `1px solid ${T.border}`,
        }}>
        Apply
      </button>
      {err && <span data-testid="feature-injection-error"
                    style={{ color: T.danger }}>{err}</span>}
      <span data-testid="feature-injection-applied-list"
            style={{ display: "inline-flex", flexWrap: "wrap", gap: 4,
                     marginLeft: "auto", alignItems: "center" }}>
        {chipEntries.length === 0 ? (
          <span style={{ color: T.textMuted }}>—</span>
        ) : chipEntries.map(([name, { count }]) => (
          <span key={name}
                data-testid={`feature-injection-chip-${name}`}
                onClick={() => onChipClick?.(name)}
                style={{ display: "inline-flex", alignItems: "center",
                         gap: 4, padding: "1px 4px 1px 6px",
                         background: T.surface3,
                         border: `1px solid ${T.border}`,
                         borderRadius: 10, fontSize: 11, color: T.text,
                         cursor: onChipClick ? "pointer" : "default" }}>
            <span>{name}</span>
            {count > 1 && (
              <span data-testid={`feature-injection-chip-${name}-count`}
                    style={{ color: T.textSecondary, fontWeight: 600 }}>
                ×{count}
              </span>
            )}
            <button data-testid={`feature-injection-chip-${name}-remove`}
                    onClick={(e) => { e.stopPropagation(); removeOne(name); }}
                    title={count > 1 ? `remove one ${name}` : `remove ${name}`}
                    style={{ background: "transparent", border: "none",
                             color: T.textSecondary, cursor: "pointer",
                             fontSize: 12, lineHeight: 1, padding: "0 2px" }}>
              ×
            </button>
          </span>
        ))}
      </span>
    </div>
  );
}
