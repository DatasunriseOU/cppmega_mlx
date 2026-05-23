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
}

export function FeatureInjectionBar({
  rpc, onApply, applied = [],
}: FeatureInjectionBarProps): JSX.Element {
  const [options, setOptions] = useState<CatalogOption[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);
  const [local, setLocal] = useState<AppliedInjection[]>(applied);

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

  return (
    <div data-testid="feature-injection-bar"
         style={{ display: "flex", alignItems: "center", gap: 6,
                  padding: 6, background: "#fefce8",
                  border: "1px solid #fde047", borderRadius: 4,
                  fontSize: 12 }}>
      <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span>Inject</span>
        <select
          data-testid="feature-injection-dropdown"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={options.length === 0}
        >
          {options.length === 0 && <option value="">…</option>}
          {options.map((o) => (
            <option key={o.name} value={o.name}>{o.name}</option>
          ))}
        </select>
      </label>
      {selected && (
        <span data-testid="feature-injection-summary"
              style={{ color: "#6b7280" }}>
          {options.find((o) => o.name === selected)?.summary ?? ""}
        </span>
      )}
      <button
        data-testid="feature-injection-apply"
        onClick={apply}
        disabled={!selected}
        style={{ padding: "2px 8px" }}>
        Apply
      </button>
      {err && <span data-testid="feature-injection-error"
                    style={{ color: "#b91c1c" }}>{err}</span>}
      <span data-testid="feature-injection-applied-list"
            style={{ color: "#374151", marginLeft: "auto" }}>
        {local.length === 0 ? "—" :
          local.map((a) => a.name).join(", ")}
      </span>
    </div>
  );
}
