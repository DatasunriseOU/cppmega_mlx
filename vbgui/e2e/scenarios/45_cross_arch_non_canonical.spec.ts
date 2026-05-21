// G16: cross-arch coverage for presets whose brick names don't follow
// `{preset}_mlp` / `{preset}_attn` convention. V4-12 hardcoded the
// canonical pattern and skipped these 9. v5 supplies explicit
// mlp_id / attn_id per preset.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickRunPipeline, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

interface Entry { preset: string; mlpId: string; attnId: string; }

const ENTRIES: Entry[] = [
  { preset: "glm_45",            mlpId: "glm_45_shared",     attnId: "glm_45_attn" },
  { preset: "glm_45_air",        mlpId: "glm_45_air_shared", attnId: "glm_45_air_attn" },
  { preset: "glm_47",            mlpId: "glm_47_shared",     attnId: "glm_47_attn" },
  { preset: "intellect_3",       mlpId: "intellect_3_shared",attnId: "intellect_3_attn" },
  { preset: "qwen3_dense_0_6b",  mlpId: "qwen3_0_6b_mlp",    attnId: "qwen3_0_6b_attn" },
  { preset: "qwen3_dense_32b",   mlpId: "qwen3_32b_mlp",     attnId: "qwen3_32b_attn" },
  { preset: "qwen3_dense_4b",    mlpId: "qwen3_4b_mlp",      attnId: "qwen3_4b_attn" },
  { preset: "qwen3_dense_8b",    mlpId: "qwen3_8b_mlp",      attnId: "qwen3_8b_attn" },
  { preset: "sarvam_30b",        mlpId: "sarvam_30b_shared", attnId: "sarvam_30b_attn" },
];

for (const { preset, mlpId, attnId } of ENTRIES) {
  test(`G16: ${preset} activation mutation propagates`, async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, preset);
    await page.locator(`[data-testid='brick-node-${mlpId}']`).click();
    const panel = page.getByTestId(`brick-context-${mlpId}`);
    await expect(panel).toBeVisible({ timeout: 4_000 });
    await page.locator(`[data-testid='brick-context-${mlpId}-activation']`)
      .selectOption("swiglu");
    await page.locator(`[data-testid='brick-context-${mlpId}-apply']`).click();
    await clickRunPipeline(page, "train");
    const extras = await readTrainExtras(page);
    expect(extras.model_summary.mlp_activation).toBe("swiglu");
    expect(extras.weight_delta_norm).toBeGreaterThan(0);
    await closeModal(page);
  });

  test(`G16: ${preset} pre_norm mutation propagates`, async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, preset);
    await page.locator(`[data-testid='brick-node-${attnId}']`).click();
    const panel = page.getByTestId(`brick-context-${attnId}`);
    await expect(panel).toBeVisible({ timeout: 4_000 });
    await page.locator(`[data-testid='brick-context-${attnId}-pre-norm']`)
      .selectOption("layernorm");
    await page.locator(`[data-testid='brick-context-${attnId}-apply']`).click();
    await clickRunPipeline(page, "train");
    const extras = await readTrainExtras(page);
    expect(extras.model_summary.attention_pre_norm).toBe("layernorm");
    expect(extras.weight_delta_norm).toBeGreaterThan(0);
    await closeModal(page);
  });
}
