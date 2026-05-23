import { useEffect, useState } from "react";
import type { LossKind, LossState } from "@/state/spec";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface LossTabProps {
  loss: LossState;
  onApply: (next: LossState) => void;
}

const KINDS: { key: LossKind; label: string }[] = [
  { key: "cross_entropy", label: "Cross-Entropy" },
  { key: "mtp_weighted",  label: "MTP Weighted" },
  { key: "ifim_shaped",   label: "IFIM Shaped" },
  { key: "mhc_attn_bias", label: "MHC Attn-Bias" },
  { key: "custom",        label: "Custom" },
];

export function LossTab({ loss, onApply }: LossTabProps): JSX.Element {
  const [draft, setDraft] = useState<LossState>(loss);

  useEffect(() => {
    setDraft((prev) => ({
      ...prev,
      head_outputs: loss.head_outputs,
    }));
  }, [loss.head_outputs.join(",")]);

  function setKind(k: LossKind) {
    setDraft({ ...draft, kind: k, params: defaultParamsFor(k) });
  }
  function setParam(k: string, v: number | string) {
    setDraft({ ...draft, params: { ...draft.params, [k]: v } });
  }

  return (
    <div data-testid="loss-tab" style={panel}>
      <label style={labelStyle}>
        <span style={labelTitle}>
          Kind
          <HelpIcon topic="loss_kind" />
        </span>
        <select
          data-testid="loss-kind"
          value={draft.kind}
          onChange={(e) => setKind(e.target.value as LossKind)}
          style={inputStyle}
        >
          {KINDS.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
        </select>
      </label>

      {draft.kind === "mtp_weighted" && (
        <>
          <label style={labelStyle}>
            <span style={labelTitle}>
              K
              <HelpIcon topic="loss_mtp_k" />
            </span>
            <input data-testid="loss-mtp-k" type="number" min={1} max={8}
                   value={Number(draft.params.k ?? 2)}
                   onChange={(e) => setParam("k", Number(e.target.value))}
                   style={inputStyle} />
          </label>
          <label style={labelStyle}>
            <span style={labelTitle}>
              beta
              <HelpIcon topic="loss_mtp_beta" />
            </span>
            <input data-testid="loss-mtp-beta" type="number" min={0} max={1}
                   step={0.05}
                   value={Number(draft.params.beta ?? 0.6)}
                   onChange={(e) => setParam("beta", Number(e.target.value))}
                   style={inputStyle} />
          </label>
        </>
      )}

      {draft.kind === "ifim_shaped" && (
        <label style={labelStyle}>
          <span style={labelTitle}>
            lambda_fim
            <HelpIcon topic="loss_ifim_lambda" />
          </span>
          <input data-testid="loss-ifim-lambda" type="number" min={0} max={1}
                 step={0.05}
                 value={Number(draft.params.lambda_fim ?? 0.1)}
                 onChange={(e) => setParam("lambda_fim", Number(e.target.value))}
                 style={inputStyle} />
        </label>
      )}

      {draft.kind === "mhc_attn_bias" && (
        <label style={labelStyle}>
          <span style={labelTitle}>
            lambda_mhc
            <HelpIcon topic="loss_mhc_lambda" />
          </span>
          <input data-testid="loss-mhc-lambda" type="number" min={0} max={0.5}
                 step={0.01}
                 value={Number(draft.params.lambda_mhc ?? 0.05)}
                 onChange={(e) => setParam("lambda_mhc", Number(e.target.value))}
                 style={inputStyle} />
        </label>
      )}

      {draft.kind === "custom" && (
        <label style={labelStyle}>
          <span style={labelTitle}>
            function-name
            <HelpIcon topic="loss_custom_fn" />
          </span>
          <input data-testid="loss-custom-fn" type="text"
                 value={String(draft.params.function_name ?? "")}
                 onChange={(e) => setParam("function_name", e.target.value)}
                 style={inputStyle} />
        </label>
      )}

      <button data-testid="loss-apply" onClick={() => onApply(draft)} style={buttonStyle}>
        Apply
      </button>
    </div>
  );
}

function defaultParamsFor(k: LossKind): Record<string, number | string> {
  switch (k) {
    case "mtp_weighted":  return { k: 2, beta: 0.6 };
    case "ifim_shaped":   return { lambda_fim: 0.1 };
    case "mhc_attn_bias": return { lambda_mhc: 0.05 };
    case "custom":        return { function_name: "" };
    default:              return {};
  }
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 12, padding: 16,
  fontFamily: T.font, fontSize: 12,
  background: T.surface, color: T.text,
};

const labelStyle: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 6,
  color: T.textSecondary,
};

const labelTitle: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 4,
  fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.05em",
};

const inputStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "8px 10px",
  fontSize: 12,
  fontFamily: T.font,
  outline: "none",
};

const buttonStyle: React.CSSProperties = {
  background: T.accent,
  color: T.accentContrast,
  border: "none",
  borderRadius: "var(--vb-radius-sm)",
  padding: "10px 14px",
  fontSize: 12,
  fontWeight: "bold",
  cursor: "pointer",
  marginTop: 10,
};
