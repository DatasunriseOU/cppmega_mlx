// H12: simulate a WS drop mid-training (via Playwright's routeWebSocket
// to close the open ws://…/ws connection after a short delay) and
// assert the BottomStrip transitions through reconnecting →
// connected once the network/route recovers, while the Train
// pipeline.run HTTP request still completes (modal shows train ok).
// No spurious error modal should surface during the offline window —
// the WS lives on its own reconnect timer (useRpc).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H12: WS drop+reconnect during Train preserves modal completion",
  async ({ page }) => {
    test.setTimeout(180_000);

    // Wrap WebSocket so the test can close all open instances on demand.
    // useRpc's onclose handler then fires → disconnected → 2s reconnect.
    await page.addInitScript(() => {
      const NativeWS = window.WebSocket;
      const all: WebSocket[] = [];
      (window as unknown as { __vbgui_sockets: WebSocket[] }).__vbgui_sockets
        = all;
      class TrackedWS extends NativeWS {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          all.push(this as unknown as WebSocket);
        }
      }
      (window as unknown as { WebSocket: typeof WebSocket })
        .WebSocket = TrackedWS as unknown as typeof WebSocket;
    });

    await gotoApp(page);

    // Wait for initial backend.status=connected.
    await expect.poll(async () =>
      await page.getByTestId("backend-status").textContent(),
      { timeout: 8_000 },
    ).toContain("Backend connected");

    await selectPreset(page, "llama3_8b");

    // Kick off a Train.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("32");
    await page.getByTestId("run-pipeline-train").click();
    await page.waitForTimeout(200);

    // Force-close every open WebSocket on the page → useRpc.onclose
    // fires → fire("disconnected") → 2s reconnect timer.
    await page.evaluate(() => {
      const w = window as unknown as { __vbgui_sockets: WebSocket[] };
      for (const s of w.__vbgui_sockets) {
        try { s.close(); } catch { /* ignore */ }
      }
    });

    // UI acknowledges the drop.
    await expect.poll(async () =>
      await page.getByTestId("backend-status").textContent(),
      { timeout: 8_000 },
    ).toMatch(/Reconnecting|Disconnected/);

    // useRpc retries on a 2s timer → reconnects to the real backend.
    await expect.poll(async () =>
      await page.getByTestId("backend-status").textContent(),
      { timeout: 15_000 },
    ).toContain("Backend connected");

    // The pipeline.run HTTP call should still complete; modal shows up.
    await page.getByTestId("run-result-modal").waitFor({ timeout: 90_000 });
    const trainRow = page.getByTestId("run-result-stage-train");
    await expect(trainRow).toBeVisible();
    await expect(trainRow).toContainText("ok");
    await closeModal(page);
  });
