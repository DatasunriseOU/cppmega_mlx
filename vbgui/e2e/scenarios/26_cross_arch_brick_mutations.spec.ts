// V4-12: cross-architecture activation + pre_norm mutations propagate
// to extras.model_summary for 12 family-rep presets. V3-11 hardcoded
// 4 presets × 2 mutations; this expands to 12 × 2 = 24 cells using
// runtime brick-context-{nodeId} testid introspection.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickRunPipeline, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// 12 family reps that build with both mlp + attention bricks under the
// canonical `{preset}_mlp` / `{preset}_attn` node-naming convention.
// Selected so that the canonical brick-node naming convention
// (`{preset}_mlp` / `{preset}_attn`) matches the actual node ids
// produced by build_preset_specs. Presets whose factories use other
// prefixes (e.g. glm_45 → glm_45_shared, qwen3_dense_* → qwen3_*) are
// excluded — they need runtime testid introspection deferred to v5.
const PRESETS = [
  "llama3_8b", "mistral_small_3_1", "granite_4_1",
  "llama3_2_1b", "llama3_2_3b", "phi4",
  "olmo2_7b", "nanbeige_4_1", "qwen3_6_27b",
  "smollm3",
] as const;

for (const preset of PRESETS) {
  test(`V4-12: ${preset} activation mutation propagates`, async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, preset);

    const mlpNode = page.locator(`[data-testid='brick-node-${preset}_mlp']`);
    await mlpNode.click();
    const panel = page.getByTestId(`brick-context-${preset}_mlp`);
    await expect(panel).toBeVisible({ timeout: 4_000 });
    await page.locator(`[data-testid='brick-context-${preset}_mlp-activation']`)
      .selectOption("swiglu");
    await page.locator(`[data-testid='brick-context-${preset}_mlp-apply']`)
      .click();

    await clickRunPipeline(page, "train");
    const extras = await readTrainExtras(page);
    expect(extras.model_summary.mlp_activation).toBe("swiglu");
    expect(extras.weight_delta_norm).toBeGreaterThan(0);
    await closeModal(page);
  });

  test(`V4-12: ${preset} pre_norm mutation propagates`, async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, preset);

    const attnNode = page.locator(`[data-testid='brick-node-${preset}_attn']`);
    await attnNode.click();
    const panel = page.getByTestId(`brick-context-${preset}_attn`);
    await expect(panel).toBeVisible({ timeout: 4_000 });
    await page.locator(`[data-testid='brick-context-${preset}_attn-pre-norm']`)
      .selectOption("layernorm");
    await page.locator(`[data-testid='brick-context-${preset}_attn-apply']`)
      .click();

    await clickRunPipeline(page, "train");
    const extras = await readTrainExtras(page);
    expect(extras.model_summary.attention_pre_norm).toBe("layernorm");
    expect(extras.weight_delta_norm).toBeGreaterThan(0);
    await closeModal(page);
  });
}
