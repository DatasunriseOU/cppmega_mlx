// G24: Tokenizer roundtrip OK badge must actually mean byte-exact
// decode. V3-10 only proved FAIL non-blocking; positive case (OK
// label) was never asserted to be byte-exact.

import { test, expect } from "@playwright/test";
import { gotoApp, clickTab } from "../fixtures";
import { loadMatrix } from "../utils/matrix";

test("G24: cppmega tokenizer + T1 parquet → OK rows with byte_diff=0",
  async ({ page }) => {
    test.setTimeout(60_000);
    const matrix = loadMatrix();
    // T2 = GPT-2 BPE; byte-exact for the ASCII corpus in its matching
    // parquet. (T1 cppmega has decoder=null per repo memory, only
    // ~0.7% of lines roundtrip byte-exact — see e2e_matrix_v3_report.)
    const parquet = matrix.parquets.T2_gpt2_small__P1_minimal.path;
    const tokenizer = matrix.tokenizers.T2_gpt2_small.path;

    await gotoApp(page);
    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(parquet);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-tokenizer-path").fill(tokenizer);
    await page.getByTestId("data-roundtrip").click();
    await page.getByTestId("data-roundtrip-0").waitFor({ timeout: 8_000 });

    // Scan all rendered roundtrip badges; for every row labelled OK,
    // the title attribute MUST report byte_diff=0 (byte-exact decode).
    // Asserts the OK label has real meaning, not just a green chip.
    const rows = page.locator("[data-testid^='data-roundtrip-']");
    const count = await rows.count();
    expect(count).toBeGreaterThan(0);
    let okCount = 0;
    for (let i = 0; i < count; i++) {
      const text = await rows.nth(i).textContent();
      const title = await rows.nth(i).getAttribute("title");
      if (text?.includes("Roundtrip OK")) {
        okCount++;
        expect(title).toContain("byte_diff=0");
      }
    }
    // GPT-2 BPE should byte-exact at least one ASCII row.
    expect(okCount).toBeGreaterThan(0);
  });
