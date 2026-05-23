/**
 * V8-R09 vitest: HFQuickStartModal.
 *
 * Asserts:
 *  - modal renders when open=true, hidden when open=false
 *  - Run button fires data.hf_quickstart with the chosen dataset + n_tokens
 *  - result block renders the parquet_path after the RPC resolves
 *  - error banner appears when RPC throws
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { HFQuickStartModal } from "@/components/HFQuickStartModal";

function makeFakeRpc(responses: Record<string, unknown>) {
  const calls: { method: string; params: unknown }[] = [];
  return {
    calls,
    rpc: {
      call: vi.fn(async (method: string, params: unknown) => {
        calls.push({ method, params });
        const r = responses[method];
        if (r instanceof Error) throw r;
        return r;
      }),
    } as never,
  };
}

const RPC_OK = {
  parquet_path: "/tmp/vbgui/hf-1.parquet",
  n_tokens_written: 8200,
  n_docs_seen: 17,
  elapsed_ms: 432,
};

describe("V8-R09 HFQuickStartModal", () => {
  it("renders nothing when closed", () => {
    const { rpc } = makeFakeRpc({});
    const { container } = render(
      <HFQuickStartModal rpc={rpc} open={false} onClose={() => {}} />);
    expect(container.querySelector('[data-testid="hf-quickstart-modal"]'))
      .toBeNull();
  });

  it("Run fires data.hf_quickstart with chosen dataset + n_tokens",
    async () => {
      const { rpc, calls } = makeFakeRpc({
        "data.hf_quickstart": RPC_OK });
      const onResult = vi.fn();
      render(
        <HFQuickStartModal rpc={rpc} open={true} onClose={() => {}}
                            onResult={onResult} />);
      fireEvent.change(screen.getByTestId("hf-quickstart-dataset-id"),
        { target: { value: "HuggingFaceFW/fineweb-edu" } });
      fireEvent.change(screen.getByTestId("hf-quickstart-n-tokens"),
        { target: { value: "8192" } });
      fireEvent.click(screen.getByTestId("hf-quickstart-run"));
      await waitFor(() => {
        expect(calls.some((c) => c.method === "data.hf_quickstart"))
          .toBe(true);
      });
      const c = calls.find((x) => x.method === "data.hf_quickstart")!;
      expect(c.params).toMatchObject({
        dataset_id: "HuggingFaceFW/fineweb-edu", n_tokens: 8192,
      });
      // Result block renders
      await waitFor(() => {
        expect(screen.getByTestId("hf-quickstart-result-path").textContent)
          .toContain("/tmp/vbgui/hf-1.parquet");
      });
      expect(onResult).toHaveBeenCalledWith("/tmp/vbgui/hf-1.parquet", 8200);
    });

  it("GitHub tab fires data.github_corpus instead of data.hf_quickstart",
    async () => {
      const { rpc, calls } = makeFakeRpc({
        "data.github_corpus": {
          ...RPC_OK,
          parquet_path: "/tmp/vbgui/gh-1.parquet",
        },
      });
      const onResult = vi.fn();
      render(
        <HFQuickStartModal rpc={rpc} open={true} onClose={() => {}}
                            onResult={onResult} />);
      fireEvent.click(screen.getByTestId("github-corpus-tab"));
      fireEvent.change(screen.getByTestId("github-corpus-repo-url"),
        { target: { value: "https://github.com/karpathy/nanochat" } });
      fireEvent.change(screen.getByTestId("github-corpus-max-commits"),
        { target: { value: "10" } });
      fireEvent.click(screen.getByTestId("github-corpus-run"));
      await waitFor(() => {
        expect(calls.some((c) => c.method === "data.github_corpus"))
          .toBe(true);
      });
      const c = calls.find((x) => x.method === "data.github_corpus")!;
      expect(c.params).toMatchObject({
        repo_url: "https://github.com/karpathy/nanochat",
        max_commits: 10,
        use_treesitter: true,
      });
      await waitFor(() => {
        expect(screen.getByTestId("hf-quickstart-result-path").textContent)
          .toContain("gh-1.parquet");
      });
    });

  it("error banner appears when the RPC throws", async () => {
    const { rpc } = makeFakeRpc({
      "data.hf_quickstart": new Error("dataset not found") });
    render(
      <HFQuickStartModal rpc={rpc} open={true} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("hf-quickstart-run"));
    await waitFor(() => {
      expect(screen.getByTestId("hf-quickstart-error").textContent ?? "")
        .toContain("dataset not found");
    });
  });
});
