// V7-E01/E02: capacity_factor < 1 in V4MoE causes real drop/reroute,
// stage_train surfaces dropped_token_ratio + rerouted_token_ratio +
// overflow_ratio + capacity_per_expert + capacity_factor.
//
// The honest-closure gap: extras.moe.dropped_token_ratio was hardcoded
// 0.0 even when the user wired capacity_factor in the MoE brick. After
// V7-E01/E02 plumbing it reflects compute_drop_reroute_stats.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-E01/E02: capacity_factor=0.25 surfaces non-zero overflow_ratio",
  async ({ page }) => {
    test.setTimeout(90_000);
    await gotoApp(page);
    // deepseek_v3 has an MoE block — use it as the platform.
    await selectPreset(page, "deepseek_v3_pro");

    // Open the MoE brick and set capacity_factor=0.25 via the canvas
    // edit affordance. The vbgui param editor is JSON-text driven so
    // we click into the BrickContextPanel.
    const moeBrick = page.locator("[data-testid^='brick-node-']")
      .filter({ hasText: /moe/i }).first();
    if (await moeBrick.count() > 0) {
      await moeBrick.click();
      const editor = page.locator("[data-testid='brick-params-editor']");
      if (await editor.count() > 0) {
        const current = (await editor.inputValue()) || "{}";
        const parsed = JSON.parse(current);
        parsed.capacity_factor = 0.25;
        await editor.fill(JSON.stringify(parsed));
        await page.locator("[data-testid='brick-params-apply']").click();
      }
    }

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    // extras.moe is a nested object; recursive StageExtras render emits
    // -moe-<field> testids.
    const moeBlock = page.getByTestId("run-result-extras-train-moe");
    await expect(moeBlock).toBeVisible();

    const drop = await page.getByTestId(
      "run-result-extras-train-moe-dropped_token_ratio").textContent();
    const reroute = await page.getByTestId(
      "run-result-extras-train-moe-rerouted_token_ratio").textContent();
    const dropN = parseFloat(drop ?? "0");
    const rerouteN = parseFloat(reroute ?? "0");
    expect(dropN).toBeGreaterThanOrEqual(0);
    expect(rerouteN).toBeGreaterThanOrEqual(0);
    // With C=0.25 some overflow must fire.
    expect(dropN + rerouteN).toBeGreaterThan(0);

    await closeModal(page);
  });
