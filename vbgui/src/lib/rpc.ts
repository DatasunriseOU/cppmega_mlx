// Thin JSON-RPC 2.0 client over HTTP POST.
// WebSocket client is added in F-C when the live-update loop wires in.

import type { JsonRpcRequest, JsonRpcResponse } from "./types";

export interface RpcClientOptions {
  baseUrl: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export class RpcError extends Error {
  readonly code: number;
  readonly data?: Record<string, unknown>;
  constructor(code: number, message: string, data?: Record<string, unknown>) {
    super(message);
    this.code = code;
    this.data = data;
  }
}

export class RpcClient {
  private nextId = 1;
  constructor(private readonly opts: RpcClientOptions) {}

  async call<R = unknown, P = Record<string, unknown>>(
    method: string,
    params?: P,
  ): Promise<R> {
    const fetcher = this.opts.fetchImpl ?? fetch;
    const envelope: JsonRpcRequest<P> = {
      jsonrpc: "2.0",
      id: this.nextId++,
      method,
      params,
    };

    const controller = new AbortController();
    const timer = this.opts.timeoutMs
      ? setTimeout(() => controller.abort(), this.opts.timeoutMs)
      : undefined;

    try {
      const res = await fetcher(`${this.opts.baseUrl}/rpc`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(envelope),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new RpcError(res.status, `HTTP ${res.status}`);
      }
      const body = (await res.json()) as JsonRpcResponse<R>;
      if (body.error) {
        throw new RpcError(body.error.code, body.error.message, body.error.data);
      }
      return body.result as R;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}
