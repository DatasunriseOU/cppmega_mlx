// Singleton RpcClient bound to the FastAPI server URL plus an
// optional WebSocket subscription for backend.status heartbeats.

import { useEffect, useMemo, useRef } from "react";
import { RpcClient } from "@/lib/rpc";

export interface UseRpcOptions {
  baseUrl?: string;             // defaults to localhost:8765 in dev
  onBackendStatus?: (s: "connected" | "reconnecting" | "disconnected") => void;
  enableWs?: boolean;
  timeoutMs?: number;
}

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";

/** Returns a stable RpcClient; manages a WS connection if requested. */
export function useRpc(opts: UseRpcOptions = {}): RpcClient {
  const baseUrl = opts.baseUrl ?? DEFAULT_BASE_URL;
  const enableWs = opts.enableWs ?? false;
  const client = useMemo(
    () => new RpcClient({ baseUrl, timeoutMs: opts.timeoutMs ?? 30_000 }),
    [baseUrl, opts.timeoutMs],
  );
  const wsRef = useRef<WebSocket | null>(null);

  // Stash the status callback in a ref so its identity drift across
  // renders doesn't re-run the WS effect (which would reconnect on
  // every render and leak sockets).
  const onStatusRef = useRef<UseRpcOptions["onBackendStatus"]>(undefined);
  useEffect(() => { onStatusRef.current = opts.onBackendStatus; },
           [opts.onBackendStatus]);

  useEffect(() => {
    if (!enableWs) return;
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    const fire = (s: "connected" | "reconnecting" | "disconnected") =>
      onStatusRef.current?.(s);

    const open = () => {
      const url = baseUrl.replace(/^http/, "ws") + "/ws";
      let socket: WebSocket;
      try {
        socket = new WebSocket(url);
      } catch {
        fire("disconnected");
        return;
      }
      wsRef.current = socket;
      fire("reconnecting");
      socket.onopen = () => {
        if (!cancelled) fire("connected");
      };
      socket.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data) as { method?: string };
          if (payload.method === "backend.status") fire("connected");
        } catch { /* ignore */ }
      };
      socket.onclose = () => {
        if (cancelled) return;
        fire("disconnected");
        reconnectTimer = setTimeout(open, 2000);
      };
      socket.onerror = () => fire("disconnected");
    };

    open();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [baseUrl, enableWs]);

  return client;
}
