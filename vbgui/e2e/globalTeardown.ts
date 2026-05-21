// Kill the dev servers spawned in globalSetup via the PID file.

import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOG_DIR = resolve(__dirname, "logs");
const HANDLES = `${LOG_DIR}/handles.json`;

function kill(pid: number | null): void {
  if (!pid) return;
  try { process.kill(pid, "SIGTERM"); } catch { /* already gone */ }
  // SIGTERM should suffice for uvicorn + vite; if not, follow up later.
}

export default async function globalTeardown(): Promise<void> {
  if (!existsSync(HANDLES)) return;
  const raw = JSON.parse(readFileSync(HANDLES, "utf-8")) as {
    backend_pid?: number | null;
    frontend_pid?: number | null;
  };
  kill(raw.frontend_pid ?? null);
  kill(raw.backend_pid ?? null);
  // Give them a moment to exit cleanly so port:5176/8767 is free for the
  // next run; ignore any post-exit chatter.
  await new Promise((r) => setTimeout(r, 500));
  try { unlinkSync(HANDLES); } catch { /* ok */ }
}
