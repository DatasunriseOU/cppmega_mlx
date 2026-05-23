import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type {
  GotchaState,
  InferenceEnrichmentSource,
  InferenceFailPolicy,
  SideChannelFallback,
  SideChannelMode,
  SideChannelState,
} from "@/state/spec";
import type { RpcClient } from "@/lib/rpc";

export interface SideChannelsTabProps {
  sideChannels: SideChannelState;
  availableChannels: string[];
  selectedTrainChannels: string[];
  gotchas: GotchaState[];
  rpc?: RpcClient | null;
  tokenizerSource?: string | null;
  onApply: (next: SideChannelState) => void;
  onTrainChannelsChange: (next: string[]) => void;
}

interface TensorPreviewPayload {
  shape: number[];
  dtype: string;
  sample: (number | boolean)[];
}

interface SideChannelPreviewPayload {
  token_count: number;
  prompt_ids: TensorPreviewPayload;
  model_kwargs: Record<string, TensorPreviewPayload>;
  side_channels: Record<string, Record<string, TensorPreviewPayload>>;
  provenance: Record<string, string>;
  rendered_platform_context: string;
  cache_key: string;
  elapsed_ms: number;
}

/** V7-H11: side_channels.apply backend verdict. */
interface SideChannelApplyPayload {
  ok: boolean;
  active_count: number;
  inactive_count: number;
  families: Array<{
    family: string;
    mode: string;
    active: boolean;
    reason: string;
    columns_requested: string[];
    columns_present: string[];
    columns_missing: string[];
  }>;
  gotchas: Array<{ id: string; severity: string; message: string;
                    reference?: string }>;
  elapsed_ms: number;
}

const MODES: SideChannelMode[] = ["off", "auto", "require", "if_available"];
const FALLBACKS: SideChannelFallback[] = [
  "zeros", "unknown_id", "drop_family", "error",
];
const SOURCES: InferenceEnrichmentSource[] = [
  "none", "prompt_only", "parse_if_possible", "project_index", "auto",
];
const FAIL_POLICIES: InferenceFailPolicy[] = [
  "drop_family", "text_only", "error",
];
const ADAPTERS = ["none", "cpp", "rust", "go", "python"] as const;
type AdapterName = typeof ADAPTERS[number];

export function SideChannelsTab({
  sideChannels, availableChannels, selectedTrainChannels, gotchas, onApply,
  onTrainChannelsChange, rpc = null, tokenizerSource = null,
}: SideChannelsTabProps): JSX.Element {
  const [draft, setDraft] = useState<SideChannelState>(sideChannels);
  const [platform, setPlatform] = useState({
    os: "macos",
    arch: "arm64",
    compiler: "clang",
    accelerator: "metal",
    standard: "c++20",
  });
  const [prompt, setPrompt] = useState("int add(int a, int b) { return a + b; }");
  const [adapter, setAdapter] = useState<AdapterName>("cpp");
  const [tensorPreview, setTensorPreview] = useState<string[]>([]);
  const [previewError, setPreviewError] = useState<string | null>(null);
  // V7-H11: backend-confirmed Apply verdict (per-family resolution +
  // gotchas + ok/inactive counts). null = not yet applied this draft.
  const [applyResult, setApplyResult] =
    useState<SideChannelApplyPayload | null>(null);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [applying, setApplying] = useState<boolean>(false);

  useEffect(() => setDraft(sideChannels), [sideChannels]);

  const available = useMemo(() => new Set(availableChannels), [availableChannels]);
  const selectedTrain = useMemo(
    () => new Set(selectedTrainChannels),
    [selectedTrainChannels],
  );
  const requiredErrors = gotchas.filter((g) =>
    g.id.startsWith("side_channel_required_"));
  const platformPreview = renderPlatform(platform);
  const enabledFamilies = Object.entries(draft.families)
    .filter(([, f]) => f.mode !== "off")
    .map(([name]) => name);

  return (
    <div data-testid="side-channels-tab" style={panel}>
      <section style={section}>
        <label style={label}>Mode
          <select data-testid="side-channels-mode"
                  value={draft.mode}
                  onChange={(e) =>
                    setDraft({ ...draft,
                      mode: e.target.value as SideChannelMode })}>
            {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <div data-testid="side-channels-available" style={muted}>
          available: {availableChannels.length === 0
            ? "none" : availableChannels.join(", ")}
        </div>
      </section>

      <section data-testid="side-channel-train-selection" style={section}>
        <h4 style={heading}>Train Inputs</h4>
        {availableChannels.length === 0 ? (
          <div data-testid="side-channel-train-empty" style={muted}>
            no available token channels
          </div>
        ) : (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {availableChannels.map((name) => (
              <label key={name}
                     style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <input data-testid={`side-channel-train-${name}`}
                       type="checkbox"
                       checked={selectedTrain.has(name)}
                       onChange={(e) =>
                         setTrainChannel(name, e.target.checked)} />
                {name}
              </label>
            ))}
          </div>
        )}
      </section>

      <section style={section}>
        {Object.entries(draft.families).map(([name, family]) => {
          const present = family.columns.filter((c) => available.has(c));
          const missing = family.columns.filter((c) => !available.has(c));
          return (
            <div key={name} data-testid={`side-channel-family-${name}`}
                 style={familyRow}>
              <div style={{ display: "flex", justifyContent: "space-between",
                            gap: 6 }}>
                <strong>{name}</strong>
                <span data-testid={`side-channel-family-${name}-coverage`}
                      style={{ color: missing.length === 0 ? "#166534" : "#92400e" }}>
                  {present.length}/{family.columns.length}
                </span>
              </div>
              <label style={label}>Mode
                <select data-testid={`side-channel-family-${name}-mode`}
                        value={family.mode}
                        onChange={(e) => setFamily(name, {
                          mode: e.target.value as SideChannelMode,
                        })}>
                  {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              <label style={label}>Dropout
                <input data-testid={`side-channel-family-${name}-dropout`}
                       type="number" min={0} max={1} step={0.05}
                       value={family.dropout}
                       onChange={(e) => setFamily(name, {
                         dropout: Number(e.target.value),
                       })} />
              </label>
              <label style={label}>Fallback
                <select data-testid={`side-channel-family-${name}-fallback`}
                        value={family.fallback}
                        onChange={(e) => setFamily(name, {
                          fallback: e.target.value as SideChannelFallback,
                        })}>
                  {FALLBACKS.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
              <label style={label}>Residual
                <input data-testid={`side-channel-family-${name}-residual`}
                       type="number" min={0} step={0.1}
                       value={family.residual_scale}
                       onChange={(e) => setFamily(name, {
                         residual_scale: Number(e.target.value),
                       })} />
              </label>
              {missing.length > 0 && (
                <div data-testid={`side-channel-family-${name}-missing`}
                     style={muted}>
                  missing: {missing.join(", ")}
                </div>
              )}
            </div>
          );
        })}
      </section>

      <section style={section}>
        <h4 style={heading}>Inference</h4>
        <label style={label}>Source
          <select data-testid="side-channel-inference-source"
                  value={draft.inference.source}
                  onChange={(e) => setDraft({
                    ...draft,
                    inference: {
                      ...draft.inference,
                      source: e.target.value as InferenceEnrichmentSource,
                    },
                  })}>
            {SOURCES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label style={label}>Fail policy
          <select data-testid="side-channel-inference-fail-policy"
                  value={draft.inference.fail_policy}
                  onChange={(e) => setDraft({
                    ...draft,
                    inference: {
                      ...draft.inference,
                      fail_policy: e.target.value as InferenceFailPolicy,
                    },
                  })}>
            {FAIL_POLICIES.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label style={label}>Timeout ms
          <input data-testid="side-channel-inference-timeout"
                 type="number" min={0}
                 value={draft.inference.timeout_ms}
                 onChange={(e) => setDraft({
                   ...draft,
                   inference: {
                     ...draft.inference,
                     timeout_ms: Number(e.target.value),
                   },
                 })} />
        </label>
        <label style={{ ...label, flexDirection: "row", alignItems: "center" }}>
          <input data-testid="side-channel-inference-cache"
                 type="checkbox"
                 checked={draft.inference.cache_enabled}
                 onChange={(e) => setDraft({
                   ...draft,
                   inference: {
                     ...draft.inference,
                     cache_enabled: e.target.checked,
                   },
                 })} />
          Cache
        </label>
        <label style={label}>Adapter
          <select data-testid="side-channel-adapter"
                  value={adapter}
                  onChange={(e) =>
                    setAdapter(e.target.value as typeof ADAPTERS[number])}>
            {ADAPTERS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
      </section>

      <section style={section}>
        <h4 style={heading}>Platform</h4>
        {(["os", "arch", "compiler", "accelerator", "standard"] as const)
          .map((key) => (
            <label key={key} style={label}>{key}
              <input data-testid={`side-channel-platform-${key}`}
                     value={platform[key]}
                     onChange={(e) =>
                       setPlatform({ ...platform, [key]: e.target.value })} />
            </label>
          ))}
        <pre data-testid="side-channel-platform-preview" style={preview}>
          {platformPreview || "unspecified"}
        </pre>
      </section>

      <section style={section}>
        <h4 style={heading}>Preview</h4>
        <textarea data-testid="side-channel-preview-prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  style={{ width: "100%", minHeight: 54,
                           fontFamily: "monospace", fontSize: 11 }} />
        <button data-testid="side-channel-preview-run"
                onClick={() => { void runPreview(); }}>
          Build preview
        </button>
        <pre data-testid="side-channel-preview" style={preview}>
{`tokens=${prompt.length}
source=${draft.inference.source}
fail_policy=${draft.inference.fail_policy}
adapter=${adapter}
platform=${platformPreview || "unspecified"}
families=${enabledFamilies.join(",") || "none"}`}
        </pre>
        <pre data-testid="side-channel-preview-tensors" style={preview}>
          {previewError ?? (
            tensorPreview.length === 0 ? "not built" : tensorPreview.join("\n")
          )}
        </pre>
      </section>

      <section data-testid="side-channel-probe" style={section}>
        <h4 style={heading}>Contract Probe</h4>
        {requiredErrors.length === 0 ? (
          <div data-testid="side-channel-probe-clean" style={muted}>
            0 required-family errors
          </div>
        ) : requiredErrors.map((g) => (
          <div key={g.id} data-testid={`side-channel-probe-error-${g.id}`}
               style={{ color: "#b91c1c", fontSize: 11 }}>
            {g.message}
          </div>
        ))}
      </section>

      <button data-testid="side-channels-apply"
              disabled={applying}
              onClick={() => void handleApply()}>
        {applying ? "Applying…" : "Apply"}
      </button>
      {applyError && (
        <div data-testid="side-channels-apply-error"
             style={{ color: "#b91c1c", fontSize: 11, marginTop: 4 }}>
          {applyError}
        </div>
      )}
      {applyResult && (
        <div data-testid="side-channels-apply-result"
             style={{ marginTop: 6, fontSize: 11,
                      fontFamily: "monospace" }}>
          <div data-testid="side-channels-apply-summary"
               style={{ color: applyResult.ok ? "#047857" : "#b91c1c" }}>
            applied: {applyResult.active_count} active /
            {" "}{applyResult.inactive_count} inactive
            {applyResult.ok ? " · ok" : " · errors"}
          </div>
          {applyResult.families.map((f) => (
            <div key={f.family}
                 data-testid={`side-channels-apply-family-${f.family}`}
                 style={{ color: f.active ? "var(--vb-text-secondary)" : "var(--vb-text-muted)" }}>
              · {f.family} [{f.mode}]: {f.active ? "active" : "inactive"} — {f.reason}
            </div>
          ))}
          {applyResult.gotchas.map((g) => (
            <div key={g.id}
                 data-testid={`side-channels-apply-gotcha-${g.id}`}
                 style={{ color: g.severity === "error"
                                ? "#b91c1c" : "#b45309" }}>
              ! {g.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  async function handleApply() {
    // V7-H11: ship the local draft through side_channels.apply BEFORE
    // committing to spec so the user sees backend's per-family
    // resolution + gotchas inline. Local commit happens regardless so
    // the rest of the UI (verify wiring, gotchas tab) still sees the
    // new config even if the backend flags warnings.
    setApplyError(null);
    setApplyResult(null);
    onApply(draft);
    if (!rpc) {
      setApplyError("no backend connection — local apply only");
      return;
    }
    setApplying(true);
    try {
      const r = await rpc.call<SideChannelApplyPayload>(
        "side_channels.apply",
        {
          side_channels: draft,
          available_side_channels: availableChannels,
        },
      );
      setApplyResult(r);
    } catch (err) {
      setApplyError(
        err instanceof Error ? `error: ${err.message}` : "error: apply failed",
      );
    } finally {
      setApplying(false);
    }
  }

  function setFamily(
    name: string,
    patch: Partial<SideChannelState["families"][string]>,
  ) {
    setDraft({
      ...draft,
      families: {
        ...draft.families,
        [name]: { ...draft.families[name], ...patch },
      },
    });
  }

  function setTrainChannel(name: string, checked: boolean) {
    const selected = new Set(selectedTrainChannels);
    if (checked) {
      selected.add(name);
    } else {
      selected.delete(name);
    }
    const ordered = [
      ...availableChannels.filter((c) => selected.has(c)),
      ...selectedTrainChannels.filter((c) =>
        selected.has(c) && !available.has(c)),
    ];
    onTrainChannelsChange(ordered);
  }

  async function runPreview() {
    setPreviewError(null);
    if (!rpc || !tokenizerSource) {
      setTensorPreview(buildTensorPreview({
        sideChannels: draft,
        prompt,
        platformPreview,
        adapter,
      }));
      return;
    }
    try {
      const result = await rpc.call<SideChannelPreviewPayload>(
        "side_channels.preview",
        {
          tokenizer_source: tokenizerSource,
          text: prompt,
          side_channels: draft,
          platform_context: platform,
          language: adapter === "none" ? undefined : adapter,
          adapter,
        },
      );
      setTensorPreview(formatBackendPreview(result));
    } catch (err) {
      setTensorPreview([]);
      setPreviewError(
        err instanceof Error ? `error: ${err.message}` : "error: preview failed",
      );
    }
  }
}

function formatBackendPreview(result: SideChannelPreviewPayload): string[] {
  const lines = [
    `prompt_ids shape=${shapeText(result.prompt_ids.shape)} dtype=${result.prompt_ids.dtype}`,
  ];
  for (const [name, tensor] of Object.entries(result.model_kwargs)) {
    lines.push(`${name} shape=${shapeText(tensor.shape)} dtype=${tensor.dtype}`);
  }
  for (const [family, columns] of Object.entries(result.side_channels)) {
    for (const [name, tensor] of Object.entries(columns)) {
      lines.push(
        `${name} shape=${shapeText(tensor.shape)} family=${family} dtype=${tensor.dtype}`,
      );
    }
  }
  for (const [key, value] of Object.entries(result.provenance)) {
    lines.push(`${key}=${value}`);
  }
  lines.push(`cache_key=${result.cache_key.slice(0, 12)}`);
  return lines;
}

function shapeText(shape: number[]): string {
  return `(${shape.join(",")})`;
}

function buildTensorPreview({
  sideChannels,
  prompt,
  platformPreview,
  adapter,
}: {
  sideChannels: SideChannelState;
  prompt: string;
  platformPreview: string;
  adapter: AdapterName;
}): string[] {
  const tokenCount = prompt.length;
  const lines = [`prompt_ids shape=(1,${tokenCount}) dtype=int32`];
  if (sideChannels.inference.source === "none") {
    lines.push("side_channels=none");
    return lines;
  }

  if (sideChannels.families.platform?.mode !== "off" && platformPreview) {
    lines.push("platform_ids shape=(1,5) family=platform dtype=int32");
  }

  const parsesSource = (
    adapter !== "none" &&
    ["parse_if_possible", "project_index", "auto"].includes(
      sideChannels.inference.source,
    )
  );
  if (parsesSource && sideChannels.families.structure?.mode !== "off") {
    lines.push(`structure_ids shape=(1,${tokenCount}) family=structure dtype=int32`);
    lines.push(`dep_levels shape=(1,${tokenCount}) family=structure dtype=int32`);
  }
  if (parsesSource && sideChannels.families.syntax?.mode !== "off") {
    lines.push(`ast_depth_ids shape=(1,${tokenCount}) family=syntax dtype=int32`);
    lines.push(`sibling_index_ids shape=(1,${tokenCount}) family=syntax dtype=int32`);
    lines.push(`node_type_ids shape=(1,${tokenCount}) family=syntax dtype=int32`);
  }

  if (lines.length === 1) lines.push("side_channels=none");
  return lines;
}

function renderPlatform(platform: Record<string, string>): string {
  return Object.entries(platform)
    .filter(([, value]) => value.trim().length > 0)
    .map(([key, value]) => `${key}=${value.trim()}`)
    .join("; ");
}

const panel: CSSProperties = {
  display: "flex", flexDirection: "column", gap: 10, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
const section: CSSProperties = {
  display: "flex", flexDirection: "column", gap: 6,
  borderBottom: "1px solid var(--vb-border)", paddingBottom: 8,
};
const familyRow: CSSProperties = {
  display: "flex", flexDirection: "column", gap: 4,
  padding: 6, border: "1px solid var(--vb-border)", borderRadius: 4,
};
const label: CSSProperties = {
  display: "flex", flexDirection: "column", gap: 2,
  color: "var(--vb-text-secondary)", fontSize: 11,
};
const heading: CSSProperties = { margin: 0, fontSize: 13 };
const muted: CSSProperties = { color: "var(--vb-text-muted)", fontSize: 11 };
const preview: CSSProperties = {
  margin: 0, padding: 6, border: "1px solid var(--vb-border)", borderRadius: 3,
  background: "var(--vb-surface-2)", whiteSpace: "pre-wrap", fontSize: 11,
};
