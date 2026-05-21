// G05: sharding axis_assignments propagate to extras.sharding_applied.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G05: INITIAL_SPEC fsdp2 axis surfaces in extras.sharding_applied",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    // Topology must match axis degree
    await page.getByTestId("topology-selector").selectOption("h100_8x");
    await page.waitForTimeout(500);
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const shardDim = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-sharding_applied-shard_dim")
        .textContent()) ?? "0", 10);
    const compile = await page.getByTestId(
      "run-result-extras-train-sharding_applied-compile_mode")
      .textContent();
    // INITIAL_SPEC has fsdp2 degree=8
    expect(shardDim).toBe(8);
    expect(compile?.trim()).toMatch(/regional|off|whole_model/);
    await closeModal(page);
  });
