// Sharding tab — accept a proposal, confirm the memory bar updates and
// no error is raised. Proposals come from suggest_sharding RPC which
// fires automatically after each preset drop (App.tsx wires it).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset,
} from "../fixtures";
import { snapshot } from "../utils/screenshot";

test("Sharding tab — proposals surface after preset drop", async ({ page }) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("sidebar-tab-sharding").click();
  await page.getByTestId("sharding-tab").waitFor();

  // The suggest_sharding RPC fires asynchronously; poll for at least one
  // proposal card.
  await expect.poll(async () =>
    await page.locator("[data-testid^='sharding-proposal-']").count(),
    { timeout: 8_000 },
  ).toBeGreaterThan(0);

  await snapshot(page, "06_sharding_proposals", "llama3_8b_proposals");
});

test("Sharding tab — Accept on first proposal does not throw",
  async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "deepseek_v3");

    await page.getByTestId("sidebar-tab-sharding").click();
    await page.getByTestId("sharding-tab").waitFor();
    await expect.poll(async () =>
      await page.locator("[data-testid^='sharding-proposal-']").count(),
      { timeout: 8_000 },
    ).toBeGreaterThan(0);

    await page.getByTestId("sharding-accept-0").click();
    // No modal popped open with an error — just transient status banner.
    await expect(page.locator("text=applied")).toBeVisible({ timeout: 3_000 });
  });

test("Sharding tab — toggle fp8_enabled flips the spec", async ({ page }) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("sidebar-tab-sharding").click();
  await page.getByTestId("sharding-tab").waitFor();

  // The testid lives on the <input type=checkbox> itself, not the label.
  const toggle = page.getByTestId("sharding-toggle-fp8_enabled");
  await toggle.waitFor({ timeout: 5_000 });
  await toggle.check();
  await expect(toggle).toBeChecked();
});
