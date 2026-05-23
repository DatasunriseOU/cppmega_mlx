import { describe, it, expect } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TokenizerMatrixTab } from "@/components/TokenizerMatrixTab";

function fakeRpc(handler: (m: string, p: unknown) => Promise<unknown>): {
  call: <T>(m: string, p: unknown) => Promise<T>;
} {
  return {
    call: async <T,>(m: string, p: unknown) => (await handler(m, p)) as T,
  };
}

describe("V7-F55 TokenizerMatrixTab", () => {
  it("renders the preset × tokenizer grid", () => {
    render(<TokenizerMatrixTab
      rpc={null}
      presets={["alpha", "beta"]}
      tokenizers={["/x/T1.json", "/x/T2.json"]}
    />);
    expect(screen.getByTestId("tokenizer-matrix-tab")).toBeDefined();
    expect(screen.getByTestId("tokmatrix-row-alpha")).toBeDefined();
    expect(screen.getByTestId("tokmatrix-row-beta")).toBeDefined();
    expect(screen.getByTestId("tokmatrix-alpha-T1")).toBeDefined();
    expect(screen.getByTestId("tokmatrix-beta-T2")).toBeDefined();
  });

  it("probe-all fires sequential RPCs and pills turn ok", async () => {
    const calls: string[] = [];
    const rpc = fakeRpc(async (_m, p) => {
      const params = p as { tokenizer_source: string };
      calls.push(params.tokenizer_source);
      return { tokens: [{ id: 100 }, { id: 200 }],
               token_count: 2, bytes_per_token_avg: 1.5 };
    });
    render(<TokenizerMatrixTab
      rpc={rpc as never}
      presets={["alpha"]}
      tokenizers={["/x/T1.json", "/x/T2.json"]}
    />);
    fireEvent.click(screen.getByTestId("tokmatrix-probe-all"));
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T1")
        .getAttribute("data-status")).toBe("ok");
    });
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T2")
        .getAttribute("data-status")).toBe("ok");
    });
    expect(calls).toEqual(["/x/T1.json", "/x/T2.json"]);
  });

  it("clicking an idle cell probes only that cell", async () => {
    const rpc = fakeRpc(async () => ({
      tokens: [{ id: 9 }], token_count: 1, bytes_per_token_avg: 2.0,
    }));
    render(<TokenizerMatrixTab
      rpc={rpc as never}
      presets={["alpha"]}
      tokenizers={["/x/T1.json", "/x/T2.json"]}
    />);
    fireEvent.click(screen.getByTestId("tokmatrix-alpha-T1-pill"));
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T1")
        .getAttribute("data-status")).toBe("ok");
    });
    // T2 untouched.
    expect(screen.getByTestId("tokmatrix-alpha-T2")
      .getAttribute("data-status")).toBe("idle");
  });

  it("incompat status fires when token_count is 0", async () => {
    const rpc = fakeRpc(async () => ({
      tokens: [], token_count: 0, bytes_per_token_avg: 0,
    }));
    render(<TokenizerMatrixTab
      rpc={rpc as never}
      presets={["alpha"]} tokenizers={["/x/T1.json"]}
    />);
    fireEvent.click(screen.getByTestId("tokmatrix-alpha-T1-pill"));
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T1")
        .getAttribute("data-status")).toBe("incompat");
    });
  });

  it("error status fires when rpc rejects", async () => {
    const rpc = fakeRpc(async () => {
      throw new Error("file not found");
    });
    render(<TokenizerMatrixTab
      rpc={rpc as never}
      presets={["alpha"]} tokenizers={["/x/T1.json"]}
    />);
    fireEvent.click(screen.getByTestId("tokmatrix-alpha-T1-pill"));
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T1")
        .getAttribute("data-status")).toBe("error");
    });
  });

  it("clicking a populated cell expands the inline ids panel", async () => {
    const rpc = fakeRpc(async () => ({
      tokens: [{ id: 1 }, { id: 2 }, { id: 3 }],
      token_count: 3, bytes_per_token_avg: 1.7,
    }));
    render(<TokenizerMatrixTab
      rpc={rpc as never}
      presets={["alpha"]} tokenizers={["/x/T1.json"]}
    />);
    fireEvent.click(screen.getByTestId("tokmatrix-alpha-T1-pill"));
    await waitFor(() => {
      expect(screen.getByTestId("tokmatrix-alpha-T1")
        .getAttribute("data-status")).toBe("ok");
    });
    // Second click on the same pill toggles the expand panel.
    fireEvent.click(screen.getByTestId("tokmatrix-alpha-T1-pill"));
    const expand = screen.getByTestId("tokmatrix-alpha-T1-expand");
    expect(expand.textContent).toContain("1, 2, 3");
    expect(expand.textContent).toContain("count: 3");
  });
});
