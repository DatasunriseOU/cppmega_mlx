// V7-I03: tightens H22 with a synchronous lock. Two rapid Train
// clicks dispatched in the same microtask (no artificial delay)
// must result in only ONE pipeline.run on the backend.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-I03: zero-delay double Train click → single pipeline.run",
  async ({ page }) => {
    test.setTimeout(60_000);

    // Count pipeline.run HTTP requests across the test.
    let pipelineRunCount = 0;
    page.on("request", (req) => {
      if (!req.url().endsWith("/rpc") || req.method() !== "POST") return;
      try {
        const body = JSON.parse(req.postData() ?? "{}");
        if (body.method === "pipeline.run") pipelineRunCount += 1;
      } catch { /* ignore */ }
    });

    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    // Dispatch two clicks in the SAME microtask via JS so neither
    // sees a React render before the second fires.
    await page.evaluate(() => {
      const btn = document.querySelector<HTMLButtonElement>(
        "[data-testid='run-pipeline-train']")!;
      btn.click();
      btn.click();
    });

    // Wait for the modal of the (one and only) pipeline run.
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await closeModal(page);

    // Allow a moment for any straggler request to land.
    await page.waitForTimeout(200);
    expect(pipelineRunCount).toBe(1);
  });
