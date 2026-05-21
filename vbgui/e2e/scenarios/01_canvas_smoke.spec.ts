// Proof-of-life for the Playwright scaffolding. Five short scenarios:
// each loads a preset via the top-bar launcher, asserts canvas nodes,
// then runs the Smoke pipeline through the backend and screenshots
// the final modal.

import { expect, test } from "@playwright/test";
import {
  assertOverallStatus, clickRunPipeline,
  closeModal, gotoApp, selectPreset,
} from "../fixtures";
import { snapshot } from "../utils/screenshot";

const PROOF_PRESETS = [
  "qwen3_next", "kimi_linear", "deepseek_v3", "mistral4", "gemma4",
];

test.describe("canvas smoke (proof of life)", () => {
  for (const preset of PROOF_PRESETS) {
    test(`preset ${preset} runs smoke pipeline through GUI`,
      async ({ page }) => {
        await gotoApp(page);
        await snapshot(page, "01_canvas_smoke", `${preset}__01_empty`);

        await selectPreset(page, preset);
        await snapshot(page, "01_canvas_smoke", `${preset}__02_loaded`);

        const modal = await clickRunPipeline(page, "smoke");
        await snapshot(page, "01_canvas_smoke", `${preset}__03_modal`);
        await assertOverallStatus(modal, "ok");

        // At least the first stage row should be rendered.
        await expect(modal.getByTestId("run-result-stage-parse")).toBeVisible();
        await closeModal(page);
      });
  }

  test("smoke on empty canvas surfaces error in modal", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("run-pipeline").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor();
    await expect(modal.getByTestId("run-result-error"))
      .toContainText("empty");
    await snapshot(page, "01_canvas_smoke", "empty_canvas_error");
  });
});
