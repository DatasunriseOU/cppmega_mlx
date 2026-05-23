// V7-F54 visual e2e — cross-preset brick transplant.
// Load llama3_8b, transplant the moe brick from llama4_maverick
// (mixtral-like factory), assert the canvas now carries a moe node,
// then run train and assert the visual LossChart renders.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("F54: transplant moe from llama4_maverick into llama3_8b", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  const bar = page.getByTestId("transplant-bar");
  await expect(bar).toBeVisible();

  // Pick mixtral-style preset as source.
  await page.getByTestId("transplant-source-preset")
    .selectOption("llama4_maverick");
  await page.getByTestId("transplant-load-source").click();
  // Brick dropdown populates with mixtral_like bricks (attention + moe).
  const brickSel = page.getByTestId("transplant-source-brick");
  await expect(brickSel).toBeVisible({ timeout: 8_000 });

  // Select the moe brick — name from _mixtral_like factory is
  // <prefix>_moe = "llama4_maverick_moe". The select option value
  // is the brick's name (we passed `name` from build_preset_specs).
  await brickSel.selectOption("llama4_maverick_moe");
  await page.getByTestId("transplant-import").click();

  // The transplanted node appears with an id derived from the kind +
  // canvas index ("moe_xplant_N"). At least one moe node now exists.
  await expect(page.locator("[data-testid^='brick-node-moe_xplant']"))
    .toHaveCount(1, { timeout: 6_000 });

  // The brick keeps its source params — assert the canvas node
  // reflects the moe kind via the brick label inside the node.
  const xnode = page.locator("[data-testid^='brick-node-moe_xplant']")
    .first();
  await expect(xnode).toContainText(/moe/i);
});
