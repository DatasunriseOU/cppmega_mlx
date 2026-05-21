// V3-11: deep UI→train assertions across multiple presets. Same
// strict-content pattern as 11_ui_to_train.spec.ts but parametrised
// across 4 family-rep presets × 3 mutation axes.
//
// Lighter than the original 6×3=18 target — we picked 4 reps that
// each contain attention+mlp bricks (necessary for activation / norm
// mutations) and run the optimizer mutation across all four.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickRunPipeline, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// Family reps known to expose preset-id-suffixed mlp/attn bricks.
const FAMILY_REPS = [
  "llama3_8b", "mistral_small_3_1", "gemma3_270m", "tiny_aya",
];

// ---------------------------------------------------------------------------
// Optimizer mutation across all 4 reps (Lion). Most robust mutation —
// every preset has at least one ParamGroup that V3-1 swaps.
// ---------------------------------------------------------------------------

for (const preset of FAMILY_REPS) {
  test(`cross-arch optimizer Lion: ${preset} extras.optimizer_kind=lion`,
    async ({ page }) => {
      test.setTimeout(90_000);
      await gotoApp(page);
      await selectPreset(page, preset);

      await page.getByTestId("sidebar-tab-optim").click();
      await page.getByTestId("optim-kind").selectOption("lion");
      await page.getByTestId("optim-apply").click();

      await clickRunPipeline(page, "train");
      const extras = await readTrainExtras(page);

      expect(extras.optimizer_kind).toBe("lion");
      expect(extras.model_summary.optimizer_kind).toBe("lion");
      expect(extras.weight_delta_norm).toBeGreaterThan(0);
      expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

      await closeModal(page);
    });
}

// ---------------------------------------------------------------------------
// Schedule mutation across the same reps (linear_warmup w=2)
// ---------------------------------------------------------------------------

for (const preset of FAMILY_REPS) {
  test(`cross-arch schedule linear_warmup: ${preset} extras.schedule_kind`,
    async ({ page }) => {
      test.setTimeout(90_000);
      await gotoApp(page);
      await selectPreset(page, preset);

      await page.getByTestId("sidebar-tab-optim").click();
      await page.getByTestId("optim-group-0-schedule-toggle").click();
      await page.getByTestId("schedule-kind-0").selectOption("linear_warmup");
      await page.getByTestId("schedule-warmup-0").fill("2");
      await page.getByTestId("optim-apply").click();

      await clickRunPipeline(page, "train");
      const extras = await readTrainExtras(page);

      expect(extras.schedule_kind).toBe("linear_warmup");
      expect(extras.model_summary.schedule_kind).toBe("linear_warmup");
      expect(extras.lr_trajectory[0]).toBeCloseTo(0, 6);

      await closeModal(page);
    });
}
