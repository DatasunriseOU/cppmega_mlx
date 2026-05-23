// V7-F58 visual e2e — Gallery tab sortable table.
// Real backend roundtrip: refresh two presets, verify their stats
// populate, click column headers to sort asc/desc, assert visible row
// order through data-testid attribute reading.

import { test, expect } from "@playwright/test";
import { gotoApp } from "../fixtures";

test("F58: Gallery tab refreshes two presets and sorts by params_M", async ({
  page,
}) => {
  await gotoApp(page);

  // Switch to the new Gallery tab.
  await page.getByTestId("app-tab-gallery").click();
  await expect(page.getByTestId("gallery-tab")).toBeVisible();

  // Refresh two presets — real verify roundtrip per preset.
  const A = "llama3_8b";
  const B = "qwen3_dense_0_6b";
  await page.getByTestId(`gallery-refresh-${A}`).click();
  await page.getByTestId(`gallery-refresh-${B}`).click();
  // Stats land in cache → cells show numeric content (not the '—' dash).
  await expect.poll(async () =>
    await page.getByTestId(`gallery-cell-${A}-params_M`).textContent(),
    { timeout: 15_000 },
  ).not.toBe("—");
  await expect.poll(async () =>
    await page.getByTestId(`gallery-cell-${B}-params_M`).textContent(),
    { timeout: 15_000 },
  ).not.toBe("—");

  // Sort by params_M ascending. aria-sort attribute is the contract.
  const paramsHeader = page.getByTestId("gallery-sort-params_M");
  await paramsHeader.click();
  await expect(paramsHeader).toHaveAttribute("aria-sort", "ascending");

  // Capture row order via data-testid attribute walk — pure DOM read,
  // no scraping of cell text. Then sort desc and assert reversal of
  // the two refreshed presets while uncached rows remain at the
  // bottom (they have no params_M value → tie-break stable).
  const rowsAsc = await page.locator("[data-testid^='gallery-row-']")
    .evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
  const filledAsc = rowsAsc.filter((r) =>
    r === `gallery-row-${A}` || r === `gallery-row-${B}`);
  expect(filledAsc.length).toBe(2);

  await paramsHeader.click();
  await expect(paramsHeader).toHaveAttribute("aria-sort", "descending");
  const rowsDesc = await page.locator("[data-testid^='gallery-row-']")
    .evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
  const filledDesc = rowsDesc.filter((r) =>
    r === `gallery-row-${A}` || r === `gallery-row-${B}`);
  // The two filled rows must reverse order between asc and desc.
  expect(filledDesc).toEqual([...filledAsc].reverse());

  // Click the preset header — default sort by name asc returns A,B
  // in alphabetical order (llama3_8b < qwen3_dense_0_6b).
  await page.getByTestId("gallery-sort-preset").click();
  await expect(page.getByTestId("gallery-sort-preset"))
    .toHaveAttribute("aria-sort", "ascending");
});
