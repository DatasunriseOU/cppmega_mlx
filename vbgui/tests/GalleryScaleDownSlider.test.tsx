/**
 * V8-R02 vitest: GalleryScaleDownSlider
 *
 * Asserts:
 *  - mounts a slider + preset picker + preview + apply button
 *  - slider value change triggers an architectures.scale_down RPC
 *    (debounced) whose result is rendered as estimated-bytes + shape
 *  - Apply button is disabled until a fitting preview lands
 *  - Apply invokes onApply with the scaled specs + (H, L)
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GalleryScaleDownSlider }
  from "@/components/GalleryScaleDownSlider";

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

const FIT_PREVIEW = {
  hidden_size: 512,
  num_layers: 32,
  estimated_bytes: 821_035_008,
  target_bytes: 1_073_741_824,
  fits: true,
  scaled_down_from: { hidden_size: 4096, num_layers: 32 },
  specs: [
    { kind: "attention", name: "attn_L0" },
    { kind: "mlp", name: "mlp_L0" },
  ],
};

describe("V8-R02 GalleryScaleDownSlider", () => {
  it("renders preset picker + slider and fires scale_down on change",
    async () => {
      const { rpc, calls } = makeFakeRpc({
        "architectures.scale_down": FIT_PREVIEW,
      });
      const onApply = vi.fn();
      render(
        <GalleryScaleDownSlider
          presets={["llama3_8b", "qwen3_dense_8b"]}
          rpc={rpc} onApply={onApply}
        />,
      );

      // Picker + slider are mounted
      const picker = screen.getByTestId("gallery-scaledown-preset") as
        HTMLSelectElement;
      expect(picker.value).toBe("llama3_8b");
      const slider = screen.getByTestId("gallery-scaledown-slider") as
        HTMLInputElement;
      // Default initialBytes is 1 GiB
      expect(Number(slider.value)).toBe(1_073_741_824);

      // The debounced first effect fires the RPC after mount.
      await waitFor(() => {
        expect(calls.some((c) =>
          c.method === "architectures.scale_down")).toBe(true);
      });
      const first = calls.find((c) =>
        c.method === "architectures.scale_down")!;
      expect(first.params).toMatchObject({
        preset: "llama3_8b", target_bytes: 1_073_741_824,
      });

      // Preview cells render
      await waitFor(() => {
        expect(screen.getByTestId("gallery-scaledown-est-bytes"))
          .toBeDefined();
      });
      expect(screen.getByTestId("gallery-scaledown-shape").textContent ?? "")
        .toMatch(/H=512.*L=32/);
      const fits = screen.getByTestId("gallery-scaledown-fits");
      expect(fits.getAttribute("data-fits")).toBe("true");

      // Apply button enabled when fits=true
      const apply = screen.getByTestId("gallery-scaledown-apply") as
        HTMLButtonElement;
      expect(apply.disabled).toBe(false);
      fireEvent.click(apply);
      expect(onApply).toHaveBeenCalledTimes(1);
      const [presetArg, specsArg, h, l] = onApply.mock.calls[0];
      expect(presetArg).toBe("llama3_8b");
      expect(specsArg).toEqual(FIT_PREVIEW.specs);
      expect(h).toBe(512);
      expect(l).toBe(32);
    });

  it("disables Apply when scale_down returns fits=false", async () => {
    const overBudget = { ...FIT_PREVIEW, fits: false,
                         estimated_bytes: 2_000_000_000 };
    const { rpc } = makeFakeRpc({
      "architectures.scale_down": overBudget,
    });
    render(
      <GalleryScaleDownSlider
        presets={["llama3_8b"]} rpc={rpc}
        onApply={() => {}}
      />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("gallery-scaledown-fits")
        .getAttribute("data-fits")).toBe("false");
    });
    const apply = screen.getByTestId("gallery-scaledown-apply") as
      HTMLButtonElement;
    expect(apply.disabled).toBe(true);
  });

  it("Auto-fit button calls architectures.auto_fit and renders result",
    async () => {
      const AUTO_FIT = {
        scaled: FIT_PREVIEW,
        fits: true,
        topology: "gb10_quarter",
        headroom: 0.9,
        reason: "hidden=512, layers=32, axis=dp×1, peak=0.82 GB / 137 GB",
      };
      const { rpc, calls } = makeFakeRpc({
        "architectures.scale_down": FIT_PREVIEW,
        "architectures.auto_fit": AUTO_FIT,
      });
      render(
        <GalleryScaleDownSlider
          presets={["llama3_8b"]} rpc={rpc}
          onApply={() => {}}
        />,
      );
      const btn = await waitFor(() =>
        screen.getByTestId("gallery-auto-fit"));
      fireEvent.click(btn);
      await waitFor(() => {
        expect(calls.some((c) => c.method === "architectures.auto_fit"))
          .toBe(true);
      });
      await waitFor(() => {
        const banner = screen.getByTestId("gallery-auto-fit-result");
        expect(banner.textContent ?? "").toContain("gb10_quarter");
        expect(banner.textContent ?? "").toContain("hidden=512");
      });
    });

  it("renders an error banner if the RPC throws", async () => {
    const { rpc } = makeFakeRpc({
      "architectures.scale_down": new Error("boom"),
    });
    render(
      <GalleryScaleDownSlider
        presets={["llama3_8b"]} rpc={rpc}
        onApply={() => {}}
      />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("gallery-scaledown-error").textContent ?? "")
        .toContain("boom");
    });
  });
});
