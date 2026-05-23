import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TransplantBar } from "@/components/TransplantBar";

function fakeRpc(handler: (m: string, p: unknown) => Promise<unknown>) {
  return {
    call: async <T,>(m: string, p: unknown) => (await handler(m, p)) as T,
  } as never;
}

describe("V7-F54 TransplantBar", () => {
  it("renders preset dropdown + Load button", () => {
    render(<TransplantBar rpc={null}
                           presets={["alpha", "beta"]}
                           onTransplant={() => {}} />);
    const sel = screen.getByTestId(
      "transplant-source-preset") as HTMLSelectElement;
    expect(sel.value).toBe("alpha");
    expect(screen.getByTestId("transplant-load-source")).toBeDefined();
    expect(screen.queryByTestId("transplant-source-brick")).toBeNull();
  });

  it("Load populates brick dropdown from RPC response", async () => {
    const rpc = fakeRpc(async () => ({
      specs: [
        { kind: "moe", name: "src_moe", params: { num_experts: 8 } },
        { kind: "attention", name: "src_attn", params: {} },
      ],
    }));
    render(<TransplantBar rpc={rpc}
                           presets={["alpha"]}
                           onTransplant={() => {}} />);
    fireEvent.click(screen.getByTestId("transplant-load-source"));
    await waitFor(() => {
      expect(screen.getByTestId("transplant-source-brick"))
        .toBeDefined();
    });
    const sel = screen.getByTestId(
      "transplant-source-brick") as HTMLSelectElement;
    expect(sel.value).toBe("src_moe");
    expect(sel.options.length).toBe(2);
  });

  it("Import fires onTransplant with kind + params of selected brick", async () => {
    const onT = vi.fn();
    const rpc = fakeRpc(async () => ({
      specs: [{ kind: "moe", name: "src_moe",
                 params: { num_experts: 8, top_k: 2 } }],
    }));
    render(<TransplantBar rpc={rpc}
                           presets={["alpha"]}
                           onTransplant={onT} />);
    fireEvent.click(screen.getByTestId("transplant-load-source"));
    await waitFor(() => {
      expect(screen.getByTestId("transplant-source-brick")).toBeDefined();
    });
    fireEvent.click(screen.getByTestId("transplant-import"));
    expect(onT).toHaveBeenCalledWith("moe",
      { num_experts: 8, top_k: 2 });
  });

  it("Import button is disabled before Load", () => {
    render(<TransplantBar rpc={null}
                           presets={["alpha"]}
                           onTransplant={() => {}} />);
    const btn = screen.getByTestId(
      "transplant-import") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("RPC rejection surfaces transplant-error span", async () => {
    const rpc = fakeRpc(async () => {
      throw new Error("preset not found");
    });
    render(<TransplantBar rpc={rpc}
                           presets={["alpha"]}
                           onTransplant={() => {}} />);
    fireEvent.click(screen.getByTestId("transplant-load-source"));
    await waitFor(() => {
      expect(screen.getByTestId("transplant-error").textContent)
        .toContain("preset not found");
    });
  });
});
