import { describe, it, expect, vi } from "vitest";
import { RpcClient, RpcError } from "@/lib/rpc";

function mockFetch(response: object, init: ResponseInit = { status: 200 }) {
  return vi.fn(async () =>
    new Response(JSON.stringify(response), {
      ...init,
      headers: { "content-type": "application/json", ...(init.headers ?? {}) },
    }),
  );
}

describe("RpcClient", () => {
  it("sends a JSON-RPC envelope to /rpc", async () => {
    const fetchImpl = mockFetch({ jsonrpc: "2.0", id: 1, result: { ok: true } });
    const client = new RpcClient({ baseUrl: "http://test", fetchImpl: fetchImpl as never });
    const result = await client.call<{ ok: boolean }>("backend.status");
    expect(result).toEqual({ ok: true });
    const call = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://test/rpc");
    const body = JSON.parse(call[1].body as string);
    expect(body.jsonrpc).toBe("2.0");
    expect(body.method).toBe("backend.status");
    expect(typeof body.id).toBe("number");
  });

  it("increments the id between calls", async () => {
    const fetchImpl = mockFetch({ jsonrpc: "2.0", id: 0, result: {} });
    const client = new RpcClient({ baseUrl: "http://x", fetchImpl: fetchImpl as never });
    await client.call("a");
    await client.call("b");
    const c0 = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    const c1 = fetchImpl.mock.calls[1] as unknown as [string, RequestInit];
    const id1 = JSON.parse(c0[1].body as string).id;
    const id2 = JSON.parse(c1[1].body as string).id;
    expect(id2).toBe(id1 + 1);
  });

  it("throws RpcError on JSON-RPC error envelope", async () => {
    const fetchImpl = mockFetch({
      jsonrpc: "2.0", id: 1,
      error: { code: -32601, message: "Method not found",
               data: { available: ["verify"] } },
    });
    const client = new RpcClient({ baseUrl: "http://x", fetchImpl: fetchImpl as never });
    await expect(client.call("missing")).rejects.toMatchObject({
      code: -32601,
      message: "Method not found",
      data: { available: ["verify"] },
    });
    await expect(client.call("missing")).rejects.toBeInstanceOf(RpcError);
  });

  it("throws RpcError on HTTP failure", async () => {
    const fetchImpl = mockFetch({}, { status: 500 });
    const client = new RpcClient({ baseUrl: "http://x", fetchImpl: fetchImpl as never });
    await expect(client.call("anything")).rejects.toMatchObject({ code: 500 });
  });

  it("aborts on timeout", async () => {
    const fetchImpl = vi.fn((_input: unknown, init: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(
          new DOMException("aborted", "AbortError"),
        ));
      }),
    );
    const client = new RpcClient({
      baseUrl: "http://x", fetchImpl: fetchImpl as never, timeoutMs: 10,
    });
    await expect(client.call("slow")).rejects.toThrow();
  });
});
