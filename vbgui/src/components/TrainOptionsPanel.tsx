// V7-K-block — TrainOptionsPanel exposes the train-time options that
// the backend already reads but the UI had hardcoded. Each input
// drives a single stage_train opt; an Apply isn't necessary because
// the panel pushes back through the controlled `value` prop and the
// parent owns the source of truth.

import { useState } from "react";
import { HelpIcon } from "@/components/HelpIcon";

export interface TrainOptions {
  // K3 — V7-A04 validation cadence.
  val_every?: number;
  // K4 — gradient clipping norm cap.
  grad_clip_max_norm?: number;
  // K5 — fp16 loss-scaler config.
  loss_scaler_init_scale?: number;
  loss_scaler_growth_interval?: number;
  // K6 — synthetic multi-rank simulation (opts.fake_ranks).
  fake_ranks?: number;
  // K8 — explicit abort token (defaults to run_id when empty).
  abort_token?: string;
}

export interface TrainOptionsPanelProps {
  value: TrainOptions;
  onChange: (next: TrainOptions) => void;
}

export function TrainOptionsPanel({
  value, onChange,
}: TrainOptionsPanelProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const v = value;

  function set<K extends keyof TrainOptions>(k: K, val: TrainOptions[K]) {
    onChange({ ...v, [k]: val });
  }

  return (
    <div data-testid="train-options-panel"
         style={{ borderTop: "1px solid #e5e7eb",
                  padding: "4px 8px", fontSize: 12,
                  fontFamily: "system-ui, sans-serif",
                  background: "#fafaf9" }}>
      <button data-testid="train-options-toggle"
              onClick={() => setOpen(!open)}
              style={{ border: "none", background: "transparent",
                       cursor: "pointer", fontWeight: 600,
                       padding: 0, color: "#374151" }}>
        {open ? "▼" : "▶"} Train options (K3–K8){" "}
      </button>
      {open && (
        <div data-testid="train-options-body"
             style={{ display: "flex", flexDirection: "column",
                      gap: 4, marginTop: 4, paddingLeft: 16 }}>
          <Row label="val_every (V7-A04 cadence)"
               help="train_val_every">
            <input data-testid="train-opt-val_every" type="number"
                   min={0} max={10000}
                   value={v.val_every ?? ""}
                   placeholder="off"
                   onChange={(e) => set(
                     "val_every",
                     e.target.value === ""
                       ? undefined : parseInt(e.target.value, 10))}
                   style={INPUT} />
          </Row>
          <Row label="grad_clip_max_norm"
               help="train_grad_clip">
            <input data-testid="train-opt-grad_clip_max_norm"
                   type="number" step="0.1" min={0}
                   value={v.grad_clip_max_norm ?? ""}
                   placeholder="spec default"
                   onChange={(e) => set(
                     "grad_clip_max_norm",
                     e.target.value === ""
                       ? undefined : parseFloat(e.target.value))}
                   style={INPUT} />
          </Row>
          <Row label="loss_scaler init_scale (fp16)"
               help="train_loss_scaler">
            <input data-testid="train-opt-loss_scaler_init_scale"
                   type="number" min={1}
                   value={v.loss_scaler_init_scale ?? ""}
                   placeholder="65536"
                   onChange={(e) => set(
                     "loss_scaler_init_scale",
                     e.target.value === ""
                       ? undefined : parseInt(e.target.value, 10))}
                   style={INPUT} />
          </Row>
          <Row label="loss_scaler growth_interval"
               help="train_loss_scaler">
            <input data-testid="train-opt-loss_scaler_growth_interval"
                   type="number" min={1}
                   value={v.loss_scaler_growth_interval ?? ""}
                   placeholder="2000"
                   onChange={(e) => set(
                     "loss_scaler_growth_interval",
                     e.target.value === ""
                       ? undefined : parseInt(e.target.value, 10))}
                   style={INPUT} />
          </Row>
          <Row label="fake_ranks (multi-rank sim)"
               help="train_fake_ranks">
            <input data-testid="train-opt-fake_ranks"
                   type="range" min={1} max={16}
                   value={v.fake_ranks ?? 1}
                   onChange={(e) => set(
                     "fake_ranks", parseInt(e.target.value, 10))}
                   style={{ width: 120 }} />
            <span data-testid="train-opt-fake_ranks-value"
                  style={{ marginLeft: 6, color: "#374151" }}>
              {v.fake_ranks ?? 1}
            </span>
          </Row>
          <Row label="abort_token (override = run_id)"
               help="train_abort_token">
            <input data-testid="train-opt-abort_token"
                   type="text"
                   value={v.abort_token ?? ""}
                   placeholder="(= run_id)"
                   onChange={(e) => set(
                     "abort_token",
                     e.target.value === "" ? undefined : e.target.value)}
                   style={{ ...INPUT, width: 200 }} />
          </Row>
        </div>
      )}
    </div>
  );
}

function Row({
  label, help, children,
}: { label: string; help: string;
     children: React.ReactNode }): JSX.Element {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ width: 230, color: "#6b7280", fontSize: 11 }}>
        {label}
      </span>
      <HelpIcon topic={help} />
      {children}
    </label>
  );
}

const INPUT: React.CSSProperties = { width: 90, fontSize: 12 };
