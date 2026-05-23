// V7-F57 visual e2e — manually-composed parallel block (tiny-aya).
// Click "Compose tiny-aya" → assert 5 brick nodes appear on the
// canvas (input, attn, mlp, join, norm) with the parallel topology
// (input fans to both attn and mlp, both feed the join, join → norm).
// The verify roundtrip runs automatically; we assert that the
// expected brick nodes are visible.

import { test, expect } from "@playwright/test";
import { gotoApp } from "../fixtures";

test("F57: compose tiny-aya parallel block on canvas", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await page.getByTestId("parallel-compose-tiny-aya").click();

  // Five nodes appear with the F57 contract ids.
  for (const id of ["aya_input", "aya_attn", "aya_mlp",
                    "aya_join", "aya_norm"]) {
    await expect(page.getByTestId(`brick-node-${id}`)).toBeVisible({
      timeout: 6_000,
    });
  }

  // Brick-count surfaced in the bottom strip reflects 5 nodes after
  // the verify debouncer fires (BottomStrip reads spec.brick_count,
  // which is written by the verify.complete dispatch).
  await expect(page.getByTestId("brick-count")).toContainText("5 bricks", {
    timeout: 15_000,
  });

  // The norm node ends up labelled with RMSNorm (the adapter label),
  // proving the kind landed correctly.
  await expect(page.getByTestId("brick-node-aya_norm"))
    .toContainText(/rmsnorm|RMSNorm/i);
});
