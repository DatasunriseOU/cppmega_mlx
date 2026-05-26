import { expect, test } from "@playwright/test";

test("Capture console errors during gpt2_xl preset load", async ({ page }) => {
  const consoleMsgs: string[] = [];
  page.on("console", (msg) => {
    consoleMsgs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    consoleMsgs.push(`[PAGE_ERROR] ${err.message}\nStack: ${err.stack}`);
  });

  await page.goto("/");
  await page.waitForTimeout(1500);

  console.log("Selecting preset: gpt2_xl");
  await page.getByTestId("preset-launcher").selectOption("gpt2_xl");

  const wizard = page.getByTestId("llm-wizard-generate");
  await expect(wizard).toBeVisible();

  // Click 1x Full scale
  await page.locator("button:has-text('1x Full')").click();
  // Set 4 layers
  await page.locator("input[type='number']").fill("4");
  
  console.log("Generating architecture...");
  await wizard.click();

  await page.waitForTimeout(3000);

  console.log("--- BROWSER CONSOLE LOGS ---");
  consoleMsgs.forEach(msg => console.log(msg));
  console.log("----------------------------");

  const nodesCount = await page.locator("[data-testid^='brick-node-']").count();
  console.log(`Found ${nodesCount} brick nodes on canvas`);
});
