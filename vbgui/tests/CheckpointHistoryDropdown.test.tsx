import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import {
  CheckpointHistoryDropdown,
} from "@/components/CheckpointHistoryDropdown";

function fakeRpc(handler: (m: string, p: unknown) => Promise<unknown>) {
  return { call: async <T,>(m: string, p: unknown) =>
                  (await handler(m, p)) as T } as never;
}

describe("V7-Q03.2 CheckpointHistoryDropdown", () => {
  it("renders disabled toggle when rpc is null", () => {
    render(<CheckpointHistoryDropdown rpc={null} directory="."
                                      onSelect={() => {}} />);
    const btn = screen.getByTestId("ckpt-history-toggle") as
      HTMLButtonElement;
    expect(btn).toBeDefined();
    expect(btn.disabled).toBe(true);
  });

  it("fetches and renders entries on first open", async () => {
    const rpc = fakeRpc(async (method: string, params: unknown) => {
      expect(method).toBe("ckpt.list_history");
      expect((params as { directory: string }).directory).toBe(".");
      return {
        directory: ".",
        scanned: 2,
        entries: [
          {
            path: "/tmp/a.safetensors",
            mtime: 1700000200,
            size_bytes: 1024,
            arch_hash: "abc12345xyz",
            opt_kind: "adamw",
            global_step: 9,
            has_opt_sidecar: true,
          },
          {
            path: "/tmp/b.safetensors",
            mtime: 1700000100,
            size_bytes: 2048,
            arch_hash: "def67890",
            opt_kind: "muon",
            global_step: 4,
            has_opt_sidecar: false,
          },
        ],
      };
    });
    render(<CheckpointHistoryDropdown rpc={rpc} directory="."
                                      onSelect={() => {}} />);
    fireEvent.click(screen.getByTestId("ckpt-history-toggle"));
    await waitFor(() => {
      const rows = screen.getAllByTestId("ckpt-history-row");
      expect(rows.length).toBe(2);
    });
  });

  it("calls onSelect when a row is clicked", async () => {
    const onSelect = vi.fn();
    const rpc = fakeRpc(async () => ({
      directory: ".", scanned: 1,
      entries: [{
        path: "/tmp/pick.safetensors", mtime: 1700000300,
        size_bytes: 512, arch_hash: "hhh", opt_kind: "adamw",
        global_step: 1, has_opt_sidecar: false,
      }],
    }));
    render(<CheckpointHistoryDropdown rpc={rpc} directory="."
                                      onSelect={onSelect} />);
    fireEvent.click(screen.getByTestId("ckpt-history-toggle"));
    await waitFor(() => {
      expect(screen.getAllByTestId("ckpt-history-row").length).toBe(1);
    });
    fireEvent.click(screen.getAllByTestId("ckpt-history-row")[0]!);
    expect(onSelect).toHaveBeenCalledWith("/tmp/pick.safetensors");
  });

  it("shows empty message when scan returns no entries", async () => {
    const rpc = fakeRpc(async () => ({
      directory: "/empty", scanned: 0, entries: [],
    }));
    render(<CheckpointHistoryDropdown rpc={rpc} directory="/empty"
                                      onSelect={() => {}} />);
    fireEvent.click(screen.getByTestId("ckpt-history-toggle"));
    await waitFor(() => {
      expect(screen.getByTestId("ckpt-history-empty")).toBeDefined();
    });
  });

  it("surfaces backend error", async () => {
    const rpc = fakeRpc(async () => ({
      directory: ".", scanned: 0, entries: [],
      error: "directory does not exist",
    }));
    render(<CheckpointHistoryDropdown rpc={rpc} directory="/missing"
                                      onSelect={() => {}} />);
    fireEvent.click(screen.getByTestId("ckpt-history-toggle"));
    await waitFor(() => {
      const err = screen.getByTestId("ckpt-history-error");
      expect(err.textContent).toContain("does not exist");
    });
  });
});
