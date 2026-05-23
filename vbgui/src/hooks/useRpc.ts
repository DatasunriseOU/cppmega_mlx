// Singleton RpcClient bound to the FastAPI server URL plus an
// optional WebSocket subscription for backend.status heartbeats.

import { useEffect, useMemo, useRef } from "react";
import { RpcClient } from "@/lib/rpc";

export interface UseRpcOptions {
  baseUrl?: string;             // defaults to localhost:8765 in dev
  onBackendStatus?: (s: "connected" | "reconnecting" | "disconnected") => void;
  /** V7-H48: invoked whenever the heartbeat reports a build_id we
   *  haven't seen before in this session — used by usePresets (V7-H47)
   *  to invalidate cached lists on backend restart. */
  onBackendBuildId?: (buildId: string) => void;
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
  const onBuildIdRef = useRef<UseRpcOptions["onBackendBuildId"]>(undefined);
  useEffect(() => { onBuildIdRef.current = opts.onBackendBuildId; },
           [opts.onBackendBuildId]);
  const lastBuildIdRef = useRef<string | null>(null);

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
          const payload = JSON.parse(ev.data) as
            { method?: string;
              params?: { status?: string; build_id?: string } };
          if (payload.method === "backend.status") {
            fire("connected");
            // V7-H48: fire onBackendBuildId only when the id changes,
            // so subscribers aren't spammed on every 1Hz heartbeat.
            const bid = payload.params?.build_id;
            if (bid && bid !== lastBuildIdRef.current) {
              lastBuildIdRef.current = bid;
              onBuildIdRef.current?.(bid);
            }
          }
        } catch { /* ignore */ }
      };
      socket.onclose = () => {
        if (cancelled) return;
        fire("disconnected");
        reconnectTimer = setTimeout(open, 2000);
      };
      socket.onerror = () => fire("disconnected");
    };

    reconnectTimer = setTimeout(open, 100);
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        const socket = wsRef.current;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        if (socket.readyState === WebSocket.CONNECTING) {
          socket.onopen = () => {
            try { socket.close(); } catch {}
          };
        } else {
          try { socket.close(); } catch {}
        }
        wsRef.current = null;
      }
    };
  }, [baseUrl, enableWs]);

  return client;
}
