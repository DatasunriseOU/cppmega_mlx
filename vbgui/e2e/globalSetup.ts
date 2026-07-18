// Spawn the FastAPI backend (uvicorn) + the Vite dev server before the
// Playwright workers attach. Both are torn down by globalTeardown.ts.

import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync } from "node:fs";
import { createWriteStream } from "node:fs";
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  BACKEND_PORT, BACKEND_URL,
  FRONTEND_PORT, FRONTEND_URL,
} from "./playwright.config";

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOG_DIR = resolve(__dirname, "logs");

interface Handles {
  backend?: ChildProcess;
  frontend?: ChildProcess;
}

const handles: Handles = {};

function spawnLogged(cmd: string, args: string[],
                     cwd: string, label: string,
                     env: NodeJS.ProcessEnv = process.env): ChildProcess {
  mkdirSync(LOG_DIR, { recursive: true });
  const out = createWriteStream(`${LOG_DIR}/${label}.stdout.log`,
                                { flags: "w" });
  const err = createWriteStream(`${LOG_DIR}/${label}.stderr.log`,
                                { flags: "w" });
  const child = spawn(cmd, args, {
    cwd,
    env: { ...env, FORCE_COLOR: "0" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout?.pipe(out);
  child.stderr?.pipe(err);
  child.on("exit", (code) =>
    err.write(`[${label}] exited code=${code}\n`));
  return child;
}

function assertRunning(child: ChildProcess, label: string): void {
  if (child.exitCode !== null || child.signalCode !== null || child.pid == null) {
    throw new Error(
      `${label} exited before readiness ` +
      `(code=${child.exitCode}, signal=${child.signalCode})`,
    );
  }
  try {
    process.kill(child.pid, 0);
  } catch {
    throw new Error(`${label} process ${child.pid} is not running`);
  }
}

async function waitFor(url: string, label: string, timeoutMs: number,
                       child: ChildProcess) {
  const deadline = Date.now() + timeoutMs;
  let lastErr: string = "no attempts";
  while (Date.now() < deadline) {
    assertRunning(child, label);
    try {
      const res = await fetch(url);
      if (res.ok) {
        assertRunning(child, label);
        return;
      }
      lastErr = `HTTP ${res.status}`;
    } catch (e) {
      lastErr = String(e);
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`${label} not ready at ${url} after ${timeoutMs}ms; ` +
                  `last error: ${lastErr}`);
}

export default async function globalSetup(): Promise<void> {
  const repoRoot = resolve(__dirname, "../..");
  const vbguiRoot = resolve(__dirname, "..");

  // Backend (uvicorn). Use the same Python venv the tests use.
  handles.backend = spawnLogged(
    process.env.VBGUI_E2E_PYTHON
      ?? "/Users/dave/sources/nanochat/.venv/bin/python",
    [
      "-m", "uvicorn",
      "cppmega_v4.jsonrpc.server:create_app",
      "--factory",
      "--port", String(BACKEND_PORT),
      "--host", "127.0.0.1",
    ],
    repoRoot, "backend",
  );

  // Frontend (vite dev). Force NODE_ENV=development so devDeps are honoured.
  // Point the bundled RpcClient at the e2e backend port via Vite env var.
  handles.frontend = spawnLogged(
    "npx",
    ["vite", "--port", String(FRONTEND_PORT), "--strictPort",
     "--host", "127.0.0.1"],
    vbguiRoot, "frontend",
    {
      ...process.env,
      NODE_ENV: "development",
      VITE_BACKEND_URL: BACKEND_URL,
    },
  );

  // Record PIDs so globalTeardown can find them even after re-import.
  writeFileSync(`${LOG_DIR}/handles.json`, JSON.stringify({
    backend_pid: handles.backend?.pid ?? null,
    frontend_pid: handles.frontend?.pid ?? null,
  }, null, 2));

  await Promise.all([
    waitFor(`${BACKEND_URL}/health`, "backend", 30_000, handles.backend),
    waitFor(FRONTEND_URL, "frontend", 30_000, handles.frontend),
  ]);
}
