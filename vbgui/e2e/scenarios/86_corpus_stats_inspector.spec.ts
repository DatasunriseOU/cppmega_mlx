// V7-G04: DataInspector renders the corpus_stats sidecar emitted by
// clang_enriched_to_parquet. This test creates a tiny parquet shard,
// writes a synthetic sidecar JSON, opens DataInspector via the UI,
// and asserts the corpus-stats block surfaces token coverage,
// doc-length percentiles, and long-tail count.

import { test, expect } from "@playwright/test";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";
import { execSync } from "node:child_process";
import { gotoApp, clickTab } from "../fixtures";

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "v7g04-"));
}

test("V7-G04: corpus_stats sidecar renders in DataInspector",
  async ({ page }) => {
    test.setTimeout(60_000);
    const dir = tmpDir();
    const parquet = path.join(dir, "shard.parquet");
    const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../../..");
    execSync(
      `${repoRoot}/.venv/bin/python -c "import pyarrow as pa, pyarrow.parquet as pq, json; ` +
        `pq.write_table(pa.table({'token_ids':[[1,2,3]]}), '${parquet}'); ` +
        `open('${parquet}.corpus_stats.json','w').write(` +
        `json.dumps({'token_coverage_pct': 42.5, 'doc_length_p50': 3, ` +
        `'doc_length_p90': 4, 'doc_length_p99': 5, 'n_docs': 1, ` +
        `'long_tail_count': 7}))"`,
      { encoding: "utf-8" },
    );

    await gotoApp(page);
    await clickTab(page, "data");
    await page.getByTestId("data-path").fill(parquet);
    await page.getByTestId("data-load").click();

    const block = page.getByTestId("data-corpus-stats");
    await expect(block).toBeVisible({ timeout: 10_000 });

    await expect(page.getByTestId("data-corpus-stats-token-coverage"))
      .toContainText("42.50%");
    await expect(page.getByTestId("data-corpus-stats-doc-length"))
      .toContainText("3/4/5");
    await expect(page.getByTestId("data-corpus-stats-long-tail"))
      .toContainText("7");
    await expect(page.getByTestId("data-corpus-stats-n-docs"))
      .toContainText("1");
  });
