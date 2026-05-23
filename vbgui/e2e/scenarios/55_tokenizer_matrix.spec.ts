// V7-F55 visual e2e — tokenizer × preset compatibility matrix.
// Click "Probe all cells" → each cell makes a real
// tokenizer.encode_visualize RPC against the fixtures shipped at
// tests/fixtures/tokenizers/T{1..4}_*.json. Asserts the pills land
// on visible 'ok' / 'incompat' / 'error' statuses (via data-status
// attribute) and the inline expand panel shows real token ids.

import { test, expect } from "@playwright/test";
import { gotoApp } from "../fixtures";

test("F55: tokenizer matrix probes 8 cells with real RPC", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await page.getByTestId("app-tab-tokmatrix").click();
  await expect(page.getByTestId("tokenizer-matrix-tab")).toBeVisible();

  // 2 presets × 4 tokenizers = 8 cells. All start in 'idle'.
  const presets = ["llama3_8b", "mistral_small_3_1"];
  const toks = ["T1_cppmega_v3", "T2_gpt2_small",
                "T3_minimal_no_fim", "T4_fim_only"];

  // Initial pills are all 'idle'.
  for (const p of presets) {
    for (const t of toks) {
      await expect(page.getByTestId(`tokmatrix-${p}-${t}`))
        .toHaveAttribute("data-status", "idle");
    }
  }

  // Probe every cell.
  await page.getByTestId("tokmatrix-probe-all").click();

  // Wait until at least one cell on every row settles into 'ok'.
  for (const p of presets) {
    await expect.poll(async () => {
      let any_ok = false;
      for (const t of toks) {
        const s = await page.getByTestId(`tokmatrix-${p}-${t}`)
          .getAttribute("data-status");
        if (s === "ok") { any_ok = true; break; }
      }
      return any_ok;
    }, { timeout: 60_000 }).toBe(true);
  }

  // All 8 cells must have left 'idle' (either ok / incompat / error).
  for (const p of presets) {
    for (const t of toks) {
      const status = await page.getByTestId(`tokmatrix-${p}-${t}`)
        .getAttribute("data-status");
      expect(status, `${p}/${t} status`).not.toBe("idle");
    }
  }

  // Click a known-ok cell pill twice to expand the inline ids panel.
  // First click (after probe) toggles expand.
  const pill = page.getByTestId(`tokmatrix-${presets[0]}-${toks[1]}-pill`);
  // ensure ok then expand.
  await expect(page.getByTestId(`tokmatrix-${presets[0]}-${toks[1]}`))
    .toHaveAttribute("data-status", "ok");
  await pill.click();
  await expect(page.getByTestId(
    `tokmatrix-${presets[0]}-${toks[1]}-expand`)).toBeVisible();
  await expect(page.getByTestId(
    `tokmatrix-${presets[0]}-${toks[1]}-expand`)).toContainText("ids:");
});
