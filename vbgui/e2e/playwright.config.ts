import { defineConfig, devices } from "@playwright/test";

// Dev ports may be overridden by globalSetup if it picks dynamic ones.
export const FRONTEND_PORT = Number(process.env.VBGUI_E2E_FRONTEND_PORT ?? 5176);
export const BACKEND_PORT  = Number(process.env.VBGUI_E2E_BACKEND_PORT  ?? 8767);
export const FRONTEND_URL  = `http://127.0.0.1:${FRONTEND_PORT}`;
export const BACKEND_URL   = `http://127.0.0.1:${BACKEND_PORT}`;

export default defineConfig({
  testDir: "./scenarios",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  // One retry everywhere — the 1100+ cell matrix occasionally drops a
  // request under workers=4 contention; a retry catches genuine
  // flakiness without masking real failures (Playwright reports both).
  retries: 1,
  workers: process.env.CI ? 2 : 4,
  reporter: process.env.CI ? "github" : "list",
  globalSetup: "./globalSetup.ts",
  globalTeardown: "./globalTeardown.ts",
  outputDir: "./test-results",
  use: {
    baseURL: FRONTEND_URL,
    headless: true,
    viewport: { width: 1280, height: 800 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    actionTimeout: 8_000,
    navigationTimeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  expect: { timeout: 5_000 },
});
