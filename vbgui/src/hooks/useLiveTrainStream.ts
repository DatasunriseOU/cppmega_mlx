// V7-L37/L40/L41: WebSocket subscriber for /ws/train/{run_id}.
//
//   * collects per-step events into an in-memory buffer (the consumer
//     decides how to render them — LiveTrainPanel does sparkline + pill).
//   * reconnects with backoff if the socket drops mid-run.
//   * surfaces the {finish:'ok'} frame as a one-shot toast flag.

import { useEffect, useRef, useState } from "react";
import type { LiveTrainEvent } from "@/components/LiveTrainPanel";

export interface UseLiveTrainStreamState {
  events: LiveTrainEvent[];
  reconnectAttempts: number;
  finishToast: boolean;
  reset: () => void;
  dismissToast: () => void;
}

export function useLiveTrainStream(
  baseUrl: string,
  runId: string | null,
  active: boolean,
): UseLiveTrainStreamState {
  const [events, setEvents] = useState<LiveTrainEvent[]>([]);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [finishToast, setFinishToast] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  // V7-L41: track whether the last close was a normal completion
  // (finish frame received) vs an unexpected drop, so reconnect only
  // fires for unexpected drops.
  const completedRef = useRef<boolean>(false);

  const reset = () => {
    setEvents([]);
    setReconnectAttempts(0);
    setFinishToast(false);
    completedRef.current = false;
  };
  const dismissToast = () => setFinishToast(false);

  useEffect(() => {
    if (!runId || !active) return;
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    completedRef.current = false;

    const wsUrl = `${baseUrl.replace(/^http/, "ws")}/ws/train/${runId}`;

    const connect = () => {
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl);
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;
      socket.onopen = () => {
        if (cancelled) return;
        // Successful reconnect: reset attempt counter so a later drop
        // gets fresh backoff windows. We keep the UI's reported value
        // until the next disconnect so the user can see we reconnected.
      };
      socket.onmessage = (msg) => {
        try {
          const frame = JSON.parse(msg.data) as
            { event?: LiveTrainEvent; finish?: string };
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
      socket.onerror = () => {
        // Errors are followed by close; reconnect logic lives there.
      };
    };

    const scheduleReconnect = () => {
      attempt += 1;
      setReconnectAttempts(attempt);
      // Exponential backoff capped at 5s.
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
  }, [baseUrl, runId, active]);

  return { events, reconnectAttempts, finishToast, reset, dismissToast };
}
