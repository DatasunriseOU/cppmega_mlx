// V7-H42: WebSocket subscriber for /ws/gen/{job_id}.
//
// Mirror of useLiveTrainStream for the gen.run token stream. Collects
// per-token events into an in-memory buffer (consumer decides how to
// render — LiveGenPanel just shows the rolling token tail). Backs off
// on unexpected drops; treats {finish:'ok'} as a one-shot toast.
//
// Backend: cppmega_v4/runtime/gen_event_bus.publish per token and
//          /ws/gen/{job_id} endpoint in cppmega_v4/jsonrpc/server.py.

import { useEffect, useRef, useState } from "react";

export interface GenTokenEvent {
  step: number;
  token_id: number;
  finish_reason?: string | null;
  [k: string]: unknown;
}

export interface UseGenerateStreamState {
  events: GenTokenEvent[];
  reconnectAttempts: number;
  finishToast: boolean;
  reset: () => void;
  dismissToast: () => void;
}

export function useGenerateStream(
  baseUrl: string,
  jobId: string | null,
  active: boolean,
): UseGenerateStreamState {
  const [events, setEvents] = useState<GenTokenEvent[]>([]);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [finishToast, setFinishToast] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  // Tracks normal completion so reconnect only fires on unexpected drops.
  const completedRef = useRef<boolean>(false);

  const reset = () => {
    setEvents([]);
    setReconnectAttempts(0);
    setFinishToast(false);
    completedRef.current = false;
  };
  const dismissToast = () => setFinishToast(false);

  useEffect(() => {
    if (!jobId || !active) return;
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    completedRef.current = false;

    const wsUrl = `${baseUrl.replace(/^http/, "ws")}/ws/gen/${jobId}`;

    const connect = () => {
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl);
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;
      socket.onmessage = (msg) => {
        try {
          const frame = JSON.parse(msg.data) as
            { event?: GenTokenEvent; finish?: string };
          if (frame.event) {
            setEvents((prev) => [...prev, frame.event!]);
          } else if (frame.finish === "ok") {
            completedRef.current = true;
            setFinishToast(true);
            try { socket.close(); } catch { /* noop */ }
          }
        } catch { /* ignore malformed */ }
      };
      socket.onclose = () => {
        if (cancelled) return;
        if (!completedRef.current && active) {
          scheduleReconnect();
        }
      };
      socket.onerror = () => { /* close handles reconnect */ };
    };

    const scheduleReconnect = () => {
      attempt += 1;
      setReconnectAttempts(attempt);
      const delay = Math.min(5_000, 500 * 2 ** Math.min(attempt - 1, 4));
      reconnectTimer = setTimeout(connect, delay);
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try { socketRef.current?.close(); } catch { /* noop */ }
      socketRef.current = null;
    };
  }, [baseUrl, jobId, active]);

  return { events, reconnectAttempts, finishToast, reset, dismissToast };
}
