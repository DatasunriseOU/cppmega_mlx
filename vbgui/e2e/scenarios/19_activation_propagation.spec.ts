// V4-5: Every ActivationName must propagate UI → BrickContextPanel →
// stage_train → extras.model_summary.mlp_activation. V3-5 proved only
// swiglu; this proves all 11.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickRunPipeline, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// Order matches ACTIVATION_OPTIONS in BrickContextPanel.tsx.
const ACTIVATIONS = [
  "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
  "swiglu", "geglu", "reglu", "xielu",
];

for (const act of ACTIVATIONS) {
  test(`V4-5: UI activation '${act}' reaches extras.model_summary`,
    async ({ page }) => {
      test.setTimeout(60_000);
      await gotoApp(page);
      await selectPreset(page, "llama3_8b");

      await page.locator("[data-testid='brick-node-llama3_8b_mlp']").click();
      const panel = page.getByTestId("brick-context-llama3_8b_mlp");
      await expect(panel).toBeVisible({ timeout: 4_000 });
      await page.locator(
        "[data-testid='brick-context-llama3_8b_mlp-activation']")
        .selectOption(act);
      await page.locator(
        "[data-testid='brick-context-llama3_8b_mlp-apply']").click();

      await clickRunPipeline(page, "train");
      const extras = await readTrainExtras(page);

      // Propagation assertion
      expect(extras.model_summary.mlp_activation).toBe(act);
      // Math sanity
      expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);
      expect(extras.weight_delta_norm).toBeGreaterThan(0);

      await closeModal(page);
    });
}
