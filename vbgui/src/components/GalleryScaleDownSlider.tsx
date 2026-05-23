/**
 * V8-R02 GalleryScaleDownSlider — slide a target memory budget, see a
 * live estimate of the next ``architectures.scale_down`` result, click
 * Apply to swap the canvas with the scaled preset.
 *
 * Contract:
 *  - props.preset: which preset to scale down (e.g. "llama3_8b")
 *  - props.rpc: RpcClient — used for the scale_down RPC call
 *  - props.onApply(specs, hidden_size, num_layers): invoked when the
 *    user clicks the Apply button. The parent App is expected to
 *    convert specs to canvas nodes/edges + update dim_env.
 *
 * Debounce: 250 ms after the slider stops moving; cancels in-flight
 * requests via an abort flag so the user sees only the last result.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface GalleryScaleDownSliderProps {
  presets: readonly string[];
  rpc: RpcClient;
  onApply: (
    preset: string,
    specs: Array<Record<string, unknown>>,
    hidden_size: number,
    num_layers: number,
  ) => void;
  /** Initial slider value in bytes; defaults to 1 GiB. */
  initialBytes?: number;
  /** Preset preselected in the picker; defaults to presets[0]. */
  initialPreset?: string;
}

interface ScaleDownPreview {
  hidden_size: number;
  num_layers: number;
  estimated_bytes: number;
  target_bytes: number;
  fits: boolean;
  scaled_down_from: { hidden_size: number; num_layers: number };
  specs: Array<Record<string, unknown>>;
}

const ONE_GB = 1_073_741_824;
const MIN_BYTES = 64 * 1024 * 1024;        // 64 MB floor
const MAX_BYTES = 64 * ONE_GB;             // 64 GB ceiling

function fmtBytes(n: number): string {
  if (n >= ONE_GB) return `${(n / ONE_GB).toFixed(2)} GB`;
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(0)} MB`;
  return `${n} B`;
}

interface AutoFitResult {
  scaled: ScaleDownPreview;
  fits: boolean;
  reason: string;
  topology: string;
  headroom: number;
}

export function GalleryScaleDownSlider({
  presets, rpc, onApply, initialBytes = ONE_GB, initialPreset,
}: GalleryScaleDownSliderProps): JSX.Element {
  const [preset, setPreset] = useState<string>(
    initialPreset ?? presets[0] ?? "");
  const [bytes, setBytes] = useState<number>(initialBytes);
  const [preview, setPreview] = useState<ScaleDownPreview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoFitResult, setAutoFitResult] =
    useState<AutoFitResult | null>(null);
  const [autoFitBusy, setAutoFitBusy] = useState(false);
  // Per-request token so stale responses are dropped.
  const reqIdRef = useRef(0);

  const fetchPreview = useCallback(async (p: string, b: number) => {
    if (!p) return;
    const myId = ++reqIdRef.current;
    setLoading(true);
    setErr(null);
    try {
      const r = await rpc.call<ScaleDownPreview>(
        "architectures.scale_down",
        { preset: p, target_bytes: b },
      );
      if (myId === reqIdRef.current) setPreview(r);
    } catch (e) {
      if (myId === reqIdRef.current) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (myId === reqIdRef.current) setLoading(false);
    }
  }, [rpc]);

  // Debounced fetch — 250 ms after the last slider/preset change.
  useEffect(() => {
    const t = setTimeout(() => { void fetchPreview(preset, bytes); }, 250);
    return () => clearTimeout(t);
  }, [preset, bytes, fetchPreview]);

  const runAutoFit = useCallback(async () => {
    if (!preset) return;
    setAutoFitBusy(true);
    setErr(null);
    try {
      const r = await rpc.call<AutoFitResult>(
        "architectures.auto_fit", { preset });
      setAutoFitResult(r);
      // Mirror as the preview block so the user sees identical state.
      setPreview(r.scaled);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setAutoFitBusy(false);
    }
  }, [rpc, preset]);

  return (
    <div data-testid="gallery-scaledown" style={{ padding: 8,
         border: "1px solid #e5e7eb", borderRadius: 6, marginTop: 8,
         display: "flex", flexDirection: "column", gap: 6 }}>
      <label style={{ fontSize: 12, color: "#374151",
                       display: "flex", alignItems: "center", gap: 6 }}>
        <span>Scale-down preset</span>
        <select
          data-testid="gallery-scaledown-preset"
          value={preset}
          onChange={(e) => setPreset(e.target.value)}
        >
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </label>
      <input
        type="range"
        data-testid="gallery-scaledown-slider"
        min={MIN_BYTES} max={MAX_BYTES}
        step={MIN_BYTES} value={bytes}
        onChange={(e) => setBytes(Number(e.target.value))}
      />
      <div style={{ fontSize: 11, color: "#6b7280" }}>
        target: <span data-testid="gallery-scaledown-target">
          {fmtBytes(bytes)}
        </span>
      </div>
      {loading && <span data-testid="gallery-scaledown-loading">…</span>}
      {err && <span data-testid="gallery-scaledown-error"
                    style={{ color: "#b91c1c" }}>{err}</span>}
      <button
        data-testid="gallery-auto-fit"
        onClick={() => { void runAutoFit(); }}
        disabled={autoFitBusy || !preset}
        style={{ alignSelf: "flex-start", padding: "2px 8px" }}>
        {autoFitBusy ? "auto-fitting…" : "Auto-fit to my devbox"}
      </button>
      {autoFitResult && (
        <div data-testid="gallery-auto-fit-result"
             style={{ fontSize: 11, color: "#374151",
                      background: "#eff6ff", padding: 6,
                      borderRadius: 4 }}>
          <strong>{autoFitResult.topology}</strong> · {autoFitResult.reason}
        </div>
      )}
      {preview && (
        <div style={{ fontSize: 11, color: "#374151",
                      display: "flex", flexDirection: "column", gap: 2 }}>
          <span data-testid="gallery-scaledown-est-bytes">
            est: {fmtBytes(preview.estimated_bytes)}
            {preview.estimated_bytes < preview.target_bytes
              ? ` (< ${fmtBytes(preview.target_bytes)})`
              : ` (over ${fmtBytes(preview.target_bytes)})`}
          </span>
          <span data-testid="gallery-scaledown-shape">
            H={preview.hidden_size} L={preview.num_layers}
            {" "}(from H={preview.scaled_down_from.hidden_size}
            {" "}L={preview.scaled_down_from.num_layers})
          </span>
          <span data-testid="gallery-scaledown-fits"
                data-fits={preview.fits}
                style={{ color: preview.fits ? "#15803d" : "#b91c1c" }}>
            {preview.fits ? "fits budget" : "exceeds budget"}
          </span>
          <button
            data-testid="gallery-scaledown-apply"
            disabled={!preview.fits}
            onClick={() => onApply(
              preset, preview.specs,
              preview.hidden_size, preview.num_layers)}
            style={{ marginTop: 4, padding: "2px 8px" }}>
            Apply scaled preset
          </button>
        </div>
      )}
    </div>
  );
}
