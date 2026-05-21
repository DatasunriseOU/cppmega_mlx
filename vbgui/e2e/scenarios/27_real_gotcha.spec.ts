// V4-13: real config triggers a backend gotcha=error with NO
// page.route() injection. INITIAL_SPEC sharding has fsdp2 axis;
// flipping compile-mode to whole_model triggers the fsdp2_whole_compile
// gotcha (PyTorch #144376) which is severity=error. UI Train button
// must then become disabled with the real reason text.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("V4-13: fsdp2 + whole_model compile triggers real gotcha, gates Train",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Set topology with an 8-way device mesh so the fsdp2 axis degree
    // matches, THEN flip compile-mode → whole_model. With the INITIAL_SPEC
    // fsdp2 dp=8 axis already present this is the real-world
    // fsdp2_whole_compile footgun (PyTorch #144376).
    await page.getByTestId("topology-selector").selectOption("h100_8x");
    await page.getByTestId("compile-mode").selectOption("whole_model");
    // Give the 200ms debounced verify + RPC roundtrip time to complete.
    await page.waitForTimeout(1_500);

    // Open the run dropdown so the conditionally-rendered Train button
    // and its disabled-reason span attach to the DOM.
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("run-pipeline-train")).toBeDisabled();

    await expect(page.getByTestId("top-bar-train-disabled-reason"))
      .toBeVisible({ timeout: 4_000 });
    const reason = await page.getByTestId("top-bar-train-disabled-reason")
      .textContent();
    expect(reason).toMatch(/FSDP|compile|gradients|flat loss/i);
  });

test("V4-13: switching compile-mode back to off clears the gotcha gate",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Trigger then clear
    await page.getByTestId("topology-selector").selectOption("h100_8x");
    await page.getByTestId("compile-mode").selectOption("whole_model");
    await page.waitForTimeout(1_500);
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("top-bar-train-disabled-reason"))
      .toBeVisible({ timeout: 4_000 });
    // Close the dropdown so the second toggle re-opens cleanly
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("compile-mode").selectOption("off");
    await page.waitForTimeout(1_500);
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("run-pipeline-train")).toBeEnabled();
    await expect(page.getByTestId("top-bar-train-disabled-reason"))
      .toHaveCount(0);
  });
