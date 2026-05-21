import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Page } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "screenshots");

/** Save a PNG into vbgui/e2e/screenshots/<subdir>/<slug>.png */
export async function snapshot(page: Page, subdir: string,
                               name: string): Promise<string> {
  const dir = resolve(ROOT, subdir);
  mkdirSync(dir, { recursive: true });
  const slug = name.replace(/[^a-z0-9_.-]/gi, "_");
  const path = resolve(dir, `${slug}.png`);
  await page.screenshot({ path, fullPage: false });
  return path;
}
