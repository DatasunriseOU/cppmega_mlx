// E-AUDIT-01: DataInspector file upload picker.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DataInspector } from "@/components/DataInspector";
import type { RpcClient } from "@/lib/rpc";

function makeRpc(): RpcClient {
  return { call: vi.fn(async () => ({} as never)) } as unknown as RpcClient;
}

describe("E-AUDIT-01 DataInspector file upload", () => {
  let origFetch: typeof fetch | undefined;
  beforeEach(() => {
    origFetch = globalThis.fetch;
  });
  afterEach(() => {
    if (origFetch) globalThis.fetch = origFetch;
  });

  it("renders file input with the spec'd testid + accept=.parquet", () => {
    render(<DataInspector rpc={makeRpc()} />);
    const input = screen.getByTestId(
      "data-inspector-file-upload") as HTMLInputElement;
    expect(input).toBeTruthy();
    expect(input.type).toBe("file");
    expect(input.accept).toBe(".parquet");
  });

  it("uploads selected file via POST and auto-populates path field",
     async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ path: "/tmp/vbgui_uploads/abc123.parquet",
                            bytes: 42, filename: "shard.parquet" }),
    } as Response));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<DataInspector rpc={makeRpc()} />);
    const fileInput = screen.getByTestId(
      "data-inspector-file-upload") as HTMLInputElement;
    const blob = new File([new Uint8Array(42)], "shard.parquet",
                          { type: "application/octet-stream" });
    fireEvent.change(fileInput, { target: { files: [blob] } });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    const call = (fetchMock.mock.calls[0] ?? []) as unknown as
      [string, RequestInit];
    expect(call[0]).toMatch(/\/upload\/parquet$/);
    expect(call[1].method).toBe("POST");
    expect(call[1].body).toBeInstanceOf(FormData);

    await waitFor(() => {
      const pathInput = screen.getByTestId("data-path") as HTMLInputElement;
      expect(pathInput.value).toBe("/tmp/vbgui_uploads/abc123.parquet");
    });
  });

  it("surfaces HTTP error into the upload-error span", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false, status: 400,
      json: async () => ({}),
    } as Response));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<DataInspector rpc={makeRpc()} />);
    const fileInput = screen.getByTestId(
      "data-inspector-file-upload") as HTMLInputElement;
    const blob = new File([new Uint8Array(8)], "x.parquet",
                          { type: "application/octet-stream" });
    fireEvent.change(fileInput, { target: { files: [blob] } });

    await waitFor(() => {
      const err = screen.getByTestId(
        "data-inspector-file-upload-error");
      expect(err.textContent).toContain("400");
    });
  });
});
