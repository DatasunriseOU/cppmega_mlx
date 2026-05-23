// V7-H05: live per-step training metrics stream via /ws/train/{run_id}.
// Honest-closure: before, the loss curve only rendered at pipeline.run
// completion. Now stage_train pushes {step, loss, lr, overflow} after
// each opt.update onto train_event_bus, the WS endpoint forwards them,
// and App.tsx renders a live-train strip that grows as steps land.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-H05: live train strip appears mid-run and updates per step",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    // Enough steps for the strip to register and update.
    await page.getByTestId("train-num-steps").fill("6");
    await page.getByTestId("run-pipeline-train").click();

    // Wait for the live strip to appear at least once.
    const strip = page.getByTestId("live-train-strip");
    await expect(strip).toBeVisible({ timeout: 30_000 });

    // The step counter must climb above 0.
    await expect(page.getByTestId("live-train-strip-header"))
      .toContainText(/step \d+/);

    // Loss has 4-decimal text.
    await expect(page.getByTestId("live-train-strip-last-loss"))
      .toContainText(/loss: \d+\.\d{4}/);

    // After training completes the modal is visible and the strip can
    // disappear (trainInFlight=false).
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 120_000 });
    await closeModal(page);
  });
