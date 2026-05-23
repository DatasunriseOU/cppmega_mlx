// V7-L46 — surface JSON-RPC error.data + Pydantic field-level errors
// alongside the top-line error.message. When the backend rejects a
// pipeline.run or verify call with INVALID_PARAMS, the RpcError.data
// payload typically looks like { errors: [{ loc, msg, type, input }],
// trace?: string, stage?: string }; the UI previously discarded all
// of it.

import { HelpIcon } from "@/components/HelpIcon";

export interface FieldError {
  loc: (string | number)[];
  msg: string;
  type?: string;
  input?: unknown;
}

export interface ErrorDetails {
  code?: number;
  message: string;
  // free-form blob from the backend error.data — we walk known shapes
  // (Pydantic errors[], RuntimeError trace, stage info).
  data?: Record<string, unknown> | null;
}

export interface ErrorDetailsPanelProps {
  error: ErrorDetails;
}

function pickArray(data: Record<string, unknown> | null | undefined,
                   key: string): FieldError[] | null {
  if (!data) return null;
  const v = data[key];
  if (!Array.isArray(v)) return null;
  return v
    .filter((e): e is Record<string, unknown> =>
      e !== null && typeof e === "object")
    .map((e) => ({
      loc: Array.isArray(e.loc)
        ? (e.loc as (string | number)[])
        : [String(e.loc ?? "")],
      msg: String(e.msg ?? e.message ?? "(no message)"),
      type: e.type === undefined ? undefined : String(e.type),
      input: e.input,
    }));
}

export function ErrorDetailsPanel({
  error,
}: ErrorDetailsPanelProps): JSX.Element {
  const fieldErrors = pickArray(error.data, "errors")
                   ?? pickArray(error.data, "detail")
                   ?? [];
  const trace = error.data && typeof error.data.trace === "string"
    ? error.data.trace as string : null;
  const stage = error.data && typeof error.data.stage === "string"
    ? error.data.stage as string : null;
  const errorType = error.data && typeof error.data.type === "string"
    ? error.data.type as string : null;

  return (
    <div data-testid="error-details-panel"
         style={{ background: "#fef2f2", color: "#991b1b",
                  borderLeft: "4px solid #dc2626",
                  borderRadius: 4, padding: 12, fontSize: 12,
                  fontFamily: "system-ui, sans-serif",
                  marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6,
                    marginBottom: 6 }}>
        <strong data-testid="error-details-headline">
          {error.code !== undefined
            ? `Error ${error.code}: ${error.message}`
            : error.message}
        </strong>
        <HelpIcon topic="rpc_error_data" />
      </div>
      {stage && (
        <div data-testid="error-details-stage"
             style={{ color: "#7f1d1d", fontSize: 11 }}>
          stage: <code>{stage}</code>
        </div>
      )}
      {errorType && (
        <div data-testid="error-details-type"
             style={{ color: "#7f1d1d", fontSize: 11 }}>
          type: <code>{errorType}</code>
        </div>
      )}
      {fieldErrors.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ color: "#7f1d1d", fontSize: 11,
                        textTransform: "uppercase", letterSpacing: 0.5,
                        marginBottom: 2 }}>
            Field errors ({fieldErrors.length})
          </div>
          <ul data-testid="error-details-field-errors"
              style={{ margin: 0, padding: "0 0 0 18px",
                       listStyle: "disc" }}>
            {fieldErrors.map((fe, i) => (
              <li key={i}
                  data-testid={`error-details-field-${i}`}>
                <code data-testid={`error-details-field-${i}-loc`}
                      style={{ color: "#7f1d1d", fontWeight: 600 }}>
                  {fe.loc.join(".")}
                </code>{" "}
                — <span data-testid={`error-details-field-${i}-msg`}>
                  {fe.msg}
                </span>
                {fe.type && (
                  <span data-testid={`error-details-field-${i}-type`}
                        style={{ color: "#9ca3af", fontSize: 10,
                                 marginLeft: 6 }}>
                    [{fe.type}]
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {trace && (
        <details style={{ marginTop: 8 }}
                 data-testid="error-details-trace-wrap">
          <summary style={{ cursor: "pointer", fontSize: 11,
                            color: "#7f1d1d" }}>
            traceback
          </summary>
          <pre data-testid="error-details-trace"
               style={{ background: "#fee2e2", padding: 6,
                        borderRadius: 3, fontSize: 11,
                        overflow: "auto", maxHeight: 200,
                        margin: "6px 0 0 0" }}>
            {trace}
          </pre>
        </details>
      )}
    </div>
  );
}
