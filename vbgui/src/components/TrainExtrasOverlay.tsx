// V7-M-block — orchestrator panel for the long tail of stage_train
// extras that previously dumped into the generic JSON dl. Builds:
//   * LossChart with smoothed (M21) + val (M23) overlays
//   * LRChart (M22)
//   * Scalar metric badges — perplexity / bpb (M24), dtype (M25),
//     fp8 (M26), fim (M28), optimizer (M34), gradient_reduce_ms (M35)
//   * MoE dashboard (M32) — 9 routing keys at-a-glance
//   * Grad-clip activity panel (M33)
//   * Brick-kinds pill row (M31)
//   * Sharding-applied panel (M27)
//   * Side-channels-observed panel (M29)
//   * Per-brick grad-norms bar (M30)

import { LossChart, type LossSeries } from "@/components/LossChart";
import { HelpIcon } from "@/components/HelpIcon";

export interface TrainExtras {
  losses?: number[];
  losses_smoothed?: number[];
  val_losses?: number[];
  lr_trajectory?: number[];
  perplexity?: number;
  bits_per_byte?: number;
  // V7-M25: master_dtype landed in two shapes across backend revs —
  // a flat string ("fp16") and a structured object with requested/
  // actual/fallback flags (V7-D01). dtypeBadges normalises both.
  master_dtype?: string | Record<string, unknown>;
  dtype_actual?: string;
  fp8_active?: boolean;
  sharding_applied?: boolean;
  per_rank_param_bytes?: number;
  fim_active?: boolean;
  fim_ratio?: number;
  side_channels_observed?: string[];
  per_brick_grad_norms?: Record<string, number>;
  routing_entropy?: number;
  load_balance_loss?: number;
  per_expert_load?: number[];
  dropped_token_ratio?: number;
  rerouted_token_ratio?: number;
  overflow_ratio?: number;
  capacity_per_expert?: number;
  capacity_factor?: number;
  num_experts?: number;
  max_grad_norm_seen?: number;
  num_clips?: number;
  optimizer_kind?: string;
  gradient_reduce_ms?: number;
  num_steps?: number;
  loss_scaler_overflows?: number[];
  model_summary?: {
    brick_kinds?: string[] | string;
    num_brick_kinds?: number;
  };
  [k: string]: unknown;
}

export interface TrainExtrasOverlayProps {
  extras: TrainExtras;
}

// V7-M25: backend's master_dtype can be either a string (legacy
// shape) or an object describing requested/actual/fallback flags
// (new dtype_state shape from V7-D01). Coerce defensively so React
// never receives an object as a Badge child.
function dtypeBadges(e: TrainExtras): {
  testid: string; label: string; value: string;
}[] {
  const out: { testid: string; label: string; value: string }[] = [];
  function add(testid: string, label: string, raw: unknown) {
    if (typeof raw === "string" && raw.length > 0) {
      out.push({ testid, label, value: raw });
    }
  }
  // Direct string form.
  add("extras-badge-master_dtype", "master", e.master_dtype);
  add("extras-badge-dtype_actual", "actual", e.dtype_actual);
  // Object form — unpack the standard keys.
  const m = e.master_dtype;
  if (m !== null && typeof m === "object") {
    const o = m as Record<string, unknown>;
    add("extras-badge-master_dtype", "master",
        o.master_dtype_requested ?? o.master_dtype_actual);
    add("extras-badge-dtype_actual", "actual",
        o.master_dtype_actual ?? o.train_dtype_actual);
  }
  return out;
}

export function TrainExtrasOverlay({
  extras,
}: TrainExtrasOverlayProps): JSX.Element {
  const losses = extras.losses ?? [];
  const smoothed = extras.losses_smoothed ?? [];
  const vals = extras.val_losses ?? [];
  const overlays: LossSeries[] = [];
  if (smoothed.length > 0) {
    overlays.push({ label: "smoothed", values: smoothed,
                    color: "#10b981" });
  }
  if (vals.length > 0) {
    overlays.push({ label: "val", values: vals, color: "#a855f7" });
  }

  return (
    <div data-testid="train-extras-overlay"
         style={{ display: "flex", flexDirection: "column",
                  gap: 8, fontFamily: "system-ui, sans-serif",
                  fontSize: 12, marginBottom: 8 }}>
      {/* M21 + M22 + M23 — charts row */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {(losses.length > 0 || overlays.length > 0) && (
          <LossChart
            losses={losses}
            series={overlays}
            width={360} height={140}
            testidPrefix="extras-loss-chart"
            overflowSteps={extras.loss_scaler_overflows ?? []}
          />
        )}
        {Array.isArray(extras.lr_trajectory) && extras.lr_trajectory.length > 0
         && (
          <LossChart
            losses={[]}
            series={[{ label: "lr", values: extras.lr_trajectory,
                       color: "#f59e0b" }]}
            width={300} height={120}
            testidPrefix="extras-lr-chart"
          />
        )}
      </div>

      {/* Scalar badges row — M24..M28 + M34 + M35 */}
      <div data-testid="extras-badges-row"
           style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {typeof extras.perplexity === "number" && (
          <Badge testid="extras-badge-perplexity" label="ppl"
                 value={extras.perplexity.toFixed(3)}
                 help="metric_perplexity" />
        )}
        {typeof extras.bits_per_byte === "number" && (
          <Badge testid="extras-badge-bpb" label="bpb"
                 value={extras.bits_per_byte.toFixed(3)}
                 help="metric_bpb" />
        )}
        {dtypeBadges(extras).map((b) => (
          <Badge key={b.testid} testid={b.testid} label={b.label}
                 value={b.value} help="metric_dtype" />
        ))}
        {extras.fp8_active && (
          <Badge testid="extras-badge-fp8_active" label="fp8" value="ON"
                 tone="ok" help="metric_fp8" />
        )}
        {extras.fim_active && (
          <Badge testid="extras-badge-fim_active" label="FIM"
                 value={typeof extras.fim_ratio === "number"
                   ? `${(extras.fim_ratio * 100).toFixed(1)}%`
                   : "ON"}
                 tone="info" help="metric_fim" />
        )}
        {extras.optimizer_kind && (
          <Badge testid="extras-badge-optimizer_kind" label="opt"
                 value={extras.optimizer_kind}
                 help="metric_optimizer" />
        )}
        {typeof extras.gradient_reduce_ms === "number" && (
          <Badge testid="extras-badge-gradient_reduce_ms"
                 label="reduce_ms"
                 value={extras.gradient_reduce_ms.toFixed(1)}
                 help="metric_gradient_reduce" />
        )}
      </div>

      {/* M31 — brick-kinds pill row */}
      {extras.model_summary && extras.model_summary.brick_kinds && (
        <BrickKindsRow brick_kinds={extras.model_summary.brick_kinds} />
      )}

      {/* M33 — grad-clip activity */}
      {(typeof extras.max_grad_norm_seen === "number"
        || typeof extras.num_clips === "number") && (
        <GradClipPanel
          max_grad_norm_seen={extras.max_grad_norm_seen}
          num_clips={extras.num_clips}
        />
      )}

      {/* M27 — sharding applied */}
      {(extras.sharding_applied
        || typeof extras.per_rank_param_bytes === "number") && (
        <ShardingAppliedPanel
          sharding_applied={extras.sharding_applied}
          per_rank_param_bytes={extras.per_rank_param_bytes}
        />
      )}

      {/* M29 — side channels observed */}
      {Array.isArray(extras.side_channels_observed)
       && extras.side_channels_observed.length > 0 && (
        <SideChannelsObservedPanel
          observed={extras.side_channels_observed}
        />
      )}

      {/* M30 — per-brick grad norms */}
      {extras.per_brick_grad_norms
       && Object.keys(extras.per_brick_grad_norms).length > 0 && (
        <PerBrickGradNormsBar
          per_brick_grad_norms={extras.per_brick_grad_norms}
        />
      )}

      {/* M32 — MoE dashboard */}
      {(typeof extras.routing_entropy === "number"
        || typeof extras.load_balance_loss === "number"
        || Array.isArray(extras.per_expert_load)) && (
        <MoEDashboard extras={extras} />
      )}
    </div>
  );
}

function Badge({
  testid, label, value, tone = "neutral", help,
}: { testid: string; label: string; value: string;
     tone?: "neutral" | "ok" | "info" | "warn";
     help?: string }): JSX.Element {
  const palette = {
    neutral: { bg: "#f3f4f6", fg: "#374151", border: "#d1d5db" },
    ok:      { bg: "#dcfce7", fg: "#166534", border: "#86efac" },
    info:    { bg: "#dbeafe", fg: "#1e40af", border: "#93c5fd" },
    warn:    { bg: "#fef3c7", fg: "#92400e", border: "#fcd34d" },
  }[tone];
  return (
    <span data-testid={testid}
          style={{ display: "inline-flex", alignItems: "center",
                   gap: 4, padding: "2px 8px",
                   background: palette.bg, color: palette.fg,
                   border: `1px solid ${palette.border}`,
                   borderRadius: 9999, fontSize: 11 }}>
      <span style={{ fontWeight: 600 }}>{label}:</span>
      <span data-testid={`${testid}-value`}>{value}</span>
      {help && <HelpIcon topic={help} size={11} />}
    </span>
  );
}

function BrickKindsRow({
  brick_kinds,
}: { brick_kinds: string[] | string }): JSX.Element {
  const arr = Array.isArray(brick_kinds)
    ? brick_kinds
    : String(brick_kinds).split(",").map((s) => s.trim());
  return (
    <div data-testid="extras-brick-kinds-row"
         style={{ display: "flex", gap: 4, flexWrap: "wrap",
                  alignItems: "center" }}>
      <span style={{ color: "#6b7280", fontSize: 11 }}>brick kinds:</span>
      {arr.map((k) => (
        <span key={k}
              data-testid={`extras-brick-kind-${k}`}
              style={{ padding: "1px 6px", background: "#eef2ff",
                       color: "#3730a3",
                       border: "1px solid #c7d2fe",
                       borderRadius: 4, fontSize: 11,
                       fontFamily: "monospace" }}>
          {k}
        </span>
      ))}
    </div>
  );
}

function GradClipPanel({
  max_grad_norm_seen, num_clips,
}: { max_grad_norm_seen?: number;
     num_clips?: number }): JSX.Element {
  return (
    <div data-testid="extras-grad-clip-panel"
         style={{ display: "flex", gap: 12,
                  padding: 6, background: "#fef9c3",
                  border: "1px solid #facc15", borderRadius: 4,
                  alignItems: "center" }}>
      <strong>grad-clip</strong>
      <HelpIcon topic="metric_grad_clip" />
      {typeof max_grad_norm_seen === "number" && (
        <span data-testid="extras-grad-clip-max">
          max ‖g‖ seen: <code>{max_grad_norm_seen.toFixed(4)}</code>
        </span>
      )}
      {typeof num_clips === "number" && (
        <span data-testid="extras-grad-clip-count">
          clips: <code>{num_clips}</code>
        </span>
      )}
    </div>
  );
}

function ShardingAppliedPanel({
  sharding_applied, per_rank_param_bytes,
}: { sharding_applied?: boolean;
     per_rank_param_bytes?: number }): JSX.Element {
  return (
    <div data-testid="extras-sharding-panel"
         style={{ padding: 6, background: "#ecfeff",
                  border: "1px solid #67e8f9", borderRadius: 4 }}>
      <strong>sharding</strong>{" "}
      <HelpIcon topic="metric_sharding" />
      <span data-testid="extras-sharding-applied"
            style={{ marginLeft: 6,
                     color: sharding_applied ? "#0e7490" : "#6b7280" }}>
        applied: <code>{sharding_applied ? "yes" : "no"}</code>
      </span>
      {typeof per_rank_param_bytes === "number" && (
        <span data-testid="extras-sharding-per-rank"
              style={{ marginLeft: 12 }}>
          per-rank param bytes:{" "}
          <code>{per_rank_param_bytes.toLocaleString()}</code>
        </span>
      )}
    </div>
  );
}

function SideChannelsObservedPanel({
  observed,
}: { observed: string[] }): JSX.Element {
  return (
    <div data-testid="extras-side-channels-panel"
         style={{ padding: 6, background: "#f5f3ff",
                  border: "1px solid #c4b5fd", borderRadius: 4 }}>
      <strong>side-channels observed</strong>{" "}
      <HelpIcon topic="metric_side_channels" />
      <ul data-testid="extras-side-channels-list"
          style={{ margin: "4px 0 0 16px", padding: 0 }}>
        {observed.map((c) => (
          <li key={c}
              data-testid={`extras-side-channel-${c}`}
              style={{ color: "#6d28d9", fontFamily: "monospace" }}>
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}

function PerBrickGradNormsBar({
  per_brick_grad_norms,
}: { per_brick_grad_norms: Record<string, number> }): JSX.Element {
  const entries = Object.entries(per_brick_grad_norms);
  const max = Math.max(1e-9, ...entries.map(([, v]) => Math.abs(v)));
  return (
    <div data-testid="extras-per-brick-grad-norms"
         style={{ padding: 6, background: "#f1f5f9",
                  border: "1px solid #cbd5e1", borderRadius: 4 }}>
      <strong>per-brick grad-norm</strong>{" "}
      <HelpIcon topic="metric_per_brick_grad" />
      <div style={{ display: "flex", flexDirection: "column",
                    gap: 2, marginTop: 4 }}>
        {entries.map(([brick, norm]) => {
          const width = (Math.abs(norm) / max) * 100;
          return (
            <div key={brick}
                 data-testid={`extras-grad-norm-${brick}`}
                 data-grad-norm={norm.toFixed(6)}
                 style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 140, fontFamily: "monospace",
                             fontSize: 11, overflow: "hidden",
                             textOverflow: "ellipsis",
                             whiteSpace: "nowrap" }}>
                {brick}
              </span>
              <div style={{ flex: 1, height: 8, background: "#e5e7eb",
                            borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${width}%`, height: "100%",
                              background: "#0ea5e9" }} />
              </div>
              <code style={{ fontSize: 11, minWidth: 60,
                             textAlign: "right" }}>
                {norm.toFixed(4)}
              </code>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MoEDashboard({
  extras,
}: { extras: TrainExtras }): JSX.Element {
  const perExpert = extras.per_expert_load ?? [];
  const maxLoad = perExpert.length > 0
    ? Math.max(...perExpert) : 1;
  return (
    <div data-testid="extras-moe-dashboard"
         style={{ padding: 8, background: "#fdf2f8",
                  border: "1px solid #f9a8d4", borderRadius: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6,
                    marginBottom: 4 }}>
        <strong>MoE routing</strong>
        <HelpIcon topic="metric_moe" />
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap",
                    marginBottom: 6 }}>
        {typeof extras.routing_entropy === "number" && (
          <Badge testid="extras-moe-routing_entropy" label="H(route)"
                 value={extras.routing_entropy.toFixed(3)} />
        )}
        {typeof extras.load_balance_loss === "number" && (
          <Badge testid="extras-moe-load_balance_loss" label="lb_loss"
                 value={extras.load_balance_loss.toFixed(4)} />
        )}
        {typeof extras.dropped_token_ratio === "number" && (
          <Badge testid="extras-moe-dropped_token_ratio" label="dropped"
                 value={`${(extras.dropped_token_ratio * 100).toFixed(2)}%`}
                 tone={extras.dropped_token_ratio > 0.05 ? "warn"
                       : "neutral"} />
        )}
        {typeof extras.rerouted_token_ratio === "number" && (
          <Badge testid="extras-moe-rerouted_token_ratio" label="rerouted"
                 value={`${(extras.rerouted_token_ratio * 100).toFixed(2)}%`} />
        )}
        {typeof extras.overflow_ratio === "number" && (
          <Badge testid="extras-moe-overflow_ratio" label="overflow"
                 value={`${(extras.overflow_ratio * 100).toFixed(2)}%`}
                 tone={extras.overflow_ratio > 0 ? "warn" : "ok"} />
        )}
        {typeof extras.capacity_factor === "number" && (
          <Badge testid="extras-moe-capacity_factor" label="cap_f"
                 value={extras.capacity_factor.toFixed(2)} />
        )}
        {typeof extras.capacity_per_expert === "number" && (
          <Badge testid="extras-moe-capacity_per_expert" label="cap/e"
                 value={String(extras.capacity_per_expert)} />
        )}
        {typeof extras.num_experts === "number" && (
          <Badge testid="extras-moe-num_experts" label="E"
                 value={String(extras.num_experts)} />
        )}
      </div>
      {perExpert.length > 0 && (
        <div data-testid="extras-moe-per-expert-load"
             style={{ display: "flex", gap: 2, alignItems: "flex-end",
                      height: 40 }}>
          {perExpert.map((load, i) => {
            const h = Math.max(2, (load / Math.max(1e-9, maxLoad)) * 38);
            return (
              <div key={i}
                   data-testid={`extras-moe-expert-${i}`}
                   data-load={load.toFixed(6)}
                   title={`expert ${i}: load=${load}`}
                   style={{ width: 14, height: h,
                            background: "#db2777",
                            borderRadius: "2px 2px 0 0" }} />
            );
          })}
        </div>
      )}
    </div>
  );
}
