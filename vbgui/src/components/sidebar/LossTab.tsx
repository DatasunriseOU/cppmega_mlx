import { useEffect, useState } from "react";
import type { LossKind, LossState } from "@/state/spec";

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

  // V4-7: keep draft in sync when the parent rebinds loss (e.g. preset
  // auto-binds head_outputs to the last brick). Otherwise the draft
  // captured at first mount sends head_outputs=['logits'] to verify and
  // the whole pipeline fails before train.
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
      <label>Kind
        <select
          data-testid="loss-kind"
          value={draft.kind}
          onChange={(e) => setKind(e.target.value as LossKind)}
        >
          {KINDS.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
        </select>
      </label>

      {draft.kind === "mtp_weighted" && (
        <>
          <label>K
            <input data-testid="loss-mtp-k" type="number" min={1} max={8}
                   value={Number(draft.params.k ?? 2)}
                   onChange={(e) => setParam("k", Number(e.target.value))} />
          </label>
          <label>beta
            <input data-testid="loss-mtp-beta" type="number" min={0} max={1}
                   step={0.05}
                   value={Number(draft.params.beta ?? 0.6)}
                   onChange={(e) => setParam("beta", Number(e.target.value))} />
          </label>
        </>
      )}

      {draft.kind === "ifim_shaped" && (
        <label>lambda_fim
          <input data-testid="loss-ifim-lambda" type="number" min={0} max={1}
                 step={0.05}
                 value={Number(draft.params.lambda_fim ?? 0.1)}
                 onChange={(e) => setParam("lambda_fim", Number(e.target.value))} />
        </label>
      )}

      {draft.kind === "mhc_attn_bias" && (
        <label>lambda_mhc
          <input data-testid="loss-mhc-lambda" type="number" min={0} max={0.5}
                 step={0.01}
                 value={Number(draft.params.lambda_mhc ?? 0.05)}
                 onChange={(e) => setParam("lambda_mhc", Number(e.target.value))} />
        </label>
      )}

      {draft.kind === "custom" && (
        <label>function-name
          <input data-testid="loss-custom-fn" type="text"
                 value={String(draft.params.function_name ?? "")}
                 onChange={(e) => setParam("function_name", e.target.value)} />
        </label>
      )}

      <button data-testid="loss-apply" onClick={() => onApply(draft)}>
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
  display: "flex", flexDirection: "column", gap: 8, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
