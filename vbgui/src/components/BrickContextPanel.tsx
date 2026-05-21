// BrickContextPanel — per-brick parameter editor overlay (E7-5 / E7-6).
//
// Opens when the user clicks a node in FlowCanvas. Shows an Activation
// dropdown (for mlp/gated_mlp/moe), pre_norm/post_norm dropdowns
// (E7-6), and a free-form params JSON pane.

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { Tooltip } from "@/components/Tooltip";
import { ExplainModal } from "@/components/ExplainModal";

const ACTIVATION_OPTIONS = [
  "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "swiglu",
];
const NORM_OPTIONS = ["rmsnorm", "layernorm", "none"];

const SUPPORTS_ACTIVATION = new Set([
  "mlp", "gated_mlp", "moe", "bailing_moe",
]);
const SUPPORTS_NORM = new Set([
  "attention", "gated_attention", "mla", "mla_absorb", "mistral4_mla",
  "dsv4_attention", "gqa_sliding", "cca_attention", "gdn", "kda",
  "mlp", "gated_mlp", "moe",
]);

export interface BrickContextPanelProps {
  rpc: RpcClient | null;
  brickId: string;
  brickKind: string;
  params: Record<string, unknown>;
  onApply: (newParams: Record<string, unknown>) => void;
  onClose: () => void;
}

const FIELD: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 4, fontSize: 12,
  marginBottom: 10,
};

export function BrickContextPanel({
  rpc, brickId, brickKind, params, onApply, onClose,
}: BrickContextPanelProps): JSX.Element {
  const [draft, setDraft] = useState<Record<string, unknown>>(params);
  const [explain, setExplain] = useState<{ cat: string; name: string }
                                          | null>(null);

  useEffect(() => { setDraft(params); }, [params, brickId]);

  function setField(field: string, value: unknown) {
    setDraft({ ...draft, [field]: value });
  }

  const supportsAct = SUPPORTS_ACTIVATION.has(brickKind);
  const supportsNorm = SUPPORTS_NORM.has(brickKind);
  const activation = (draft.activation as string | undefined) ?? "glu";
  const preNorm = (draft.pre_norm as string | undefined) ?? "rmsnorm";
  const postNorm = (draft.post_norm as string | undefined) ?? "none";

  return (
    <div data-testid={`brick-context-${brickId}`}
         style={{
           position: "absolute", top: 60, right: 8,
           width: 320, maxHeight: "calc(100vh - 200px)",
           background: "white", border: "1px solid #e5e7eb",
           borderRadius: 6, padding: 12, overflowY: "auto",
           boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
           fontFamily: "system-ui, sans-serif", zIndex: 50,
         }}>
      <header style={{ display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 10 }}>
        <h4 style={{ margin: 0, fontSize: 13 }}>
          {brickId}
          <span style={{ marginLeft: 6, color: "#6b7280",
                          fontSize: 10 }}>
            [{brickKind}]
          </span>
        </h4>
        <button data-testid="brick-context-close" onClick={onClose}>×</button>
      </header>

      {supportsAct && (
        <label style={FIELD}>
          <Tooltip rpc={rpc} category="activation" name={activation}
                   onInfoClick={() =>
                     setExplain({ cat: "activation", name: activation })}>
            <span style={{ color: "#6b7280" }}>Activation</span>
          </Tooltip>
          <select data-testid={`brick-context-${brickId}-activation`}
                  value={activation}
                  onChange={(e) => setField("activation", e.target.value)}>
            {ACTIVATION_OPTIONS.map((a) =>
              <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
      )}

      {supportsNorm && (
        <>
          <label style={FIELD}>
            <Tooltip rpc={rpc} category="norm" name={preNorm}
                     onInfoClick={() =>
                       setExplain({ cat: "norm", name: preNorm })}>
              <span style={{ color: "#6b7280" }}>pre_norm</span>
            </Tooltip>
            <select data-testid={`brick-context-${brickId}-pre-norm`}
                    value={preNorm}
                    onChange={(e) => setField("pre_norm", e.target.value)}>
              {NORM_OPTIONS.map((n) =>
                <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label style={FIELD}>
            <Tooltip rpc={rpc} category="norm" name={postNorm}
                     onInfoClick={() =>
                       setExplain({ cat: "norm", name: postNorm })}>
              <span style={{ color: "#6b7280" }}>post_norm</span>
            </Tooltip>
            <select data-testid={`brick-context-${brickId}-post-norm`}
                    value={postNorm}
                    onChange={(e) => setField("post_norm", e.target.value)}>
              {NORM_OPTIONS.map((n) =>
                <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </>
      )}

      <button data-testid={`brick-context-${brickId}-apply`}
              onClick={() => { onApply(draft); onClose(); }}
              style={{ background: "#2563eb", color: "white",
                        border: "none", padding: "5px 12px",
                        borderRadius: 4, cursor: "pointer", fontSize: 12,
                        marginTop: 8 }}>
        Apply
      </button>

      {explain && (
        <ExplainModal rpc={rpc} category={explain.cat} name={explain.name}
                      onClose={() => setExplain(null)} />
      )}
    </div>
  );
}
