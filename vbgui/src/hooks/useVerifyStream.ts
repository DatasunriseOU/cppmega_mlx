// V7-H37: WebSocket subscriber for /ws/verify/{spec_hash}.
//
// Mirrors useLiveTrainStream — opens the WS when active=true + specHash
// is known, collects phase events ({phase, ...}) into a buffer, surfaces
// the latest one + a count for a progress indicator. Auto-disconnects
// on backend {finish:'ok'} frame.

import { useEffect, useRef, useState } from "react";

export interface VerifyPhaseEvent {
  phase: string;
  [k: string]: unknown;
}

export interface UseVerifyStreamState {
  events: VerifyPhaseEvent[];
  finished: boolean;
  reset: () => void;
}

export function useVerifyStream(
  baseUrl: string,
  specHash: string | null,
  active: boolean,
): UseVerifyStreamState {
  const [events, setEvents] = useState<VerifyPhaseEvent[]>([]);
  const [finished, setFinished] = useState<boolean>(false);
  const socketRef = useRef<WebSocket | null>(null);

  const reset = () => { setEvents([]); setFinished(false); };

  useEffect(() => {
    if (!active || !specHash) {
      socketRef.current?.close();
      socketRef.current = null;
      return;
    }

    let cancelled = false;
    setEvents([]); setFinished(false);

    const url = baseUrl.replace(/^http/, "ws")
              + `/ws/verify/${specHash}`;
    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch {
      return;
    }
    socketRef.current = socket;

    socket.onmessage = (msg) => {
      if (cancelled) return;
      try {
        const frame = JSON.parse(msg.data) as
          { event?: VerifyPhaseEvent; finish?: string };
        if (frame.event) {
          setEvents((prev) => [...prev, frame.event!]);
        } else if (frame.finish) {
          setFinished(true);
          socket.close();
        }
      } catch { /* ignore malformed */ }
    };

    socket.onerror = () => { /* swallow; reconnect via effect dep */ };

    return () => {
      cancelled = true;
      try { socket.close(); } catch { /* noop */ }
      socketRef.current = null;
    };
  }, [baseUrl, specHash, active]);

  return { events, finished, reset };
}

/**
 * Canonical-JSON serializer matching backend cppmega_v4.runtime.
 * verify_event_bus.spec_hash (json.dumps sort_keys=True default=str).
 */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalJson).join(",") + "]";
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  const parts = keys.map(
    (k) => JSON.stringify(k) + ":" + canonicalJson(obj[k]));
  return "{" + parts.join(",") + "}";
}

/**
 * Compute SHA-256 of canonical JSON, hex-encoded. Matches the
 * backend's `spec_hash`. Uses WebCrypto so requires HTTPS or
 * localhost (jsdom-test environments need a polyfill).
 */
export async function computeSpecHash(value: unknown): Promise<string> {
  const buf = new TextEncoder().encode(canonicalJson(value));
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
