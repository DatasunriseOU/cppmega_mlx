#!/bin/bash
# ---------------------------------------------------------------------------
# cppmega Real E2E System Integration Test Runner
#
# Starts the background FastAPI RPC server, executes the real zero-mock
# Playwright integration suite, captures console/API/training logs, 
# and shuts down safely on exit.
# ---------------------------------------------------------------------------

set -euo pipefail

CWD="/Users/dave/sources/cppmega.mlx"
export VBGUI_E2E_PYTHON="${CWD}/.venv/bin/python"
API_PORT=8767
API_HOST="127.0.0.1"
API_URL="http://${API_HOST}:${API_PORT}"

# 1. Cleanup old logs
echo "[Real E2E] Cleaning up old test logs..."
rm -f "${CWD}/backend.log" "${CWD}/frontend.log"

# 2. Start cppmega JSON-RPC backend server in background
echo "[Real E2E] Launching background FastAPI server on port ${API_PORT}..."
cd "${CWD}"
.venv/bin/python -m uvicorn cppmega_v4.jsonrpc.server:create_app --factory --port ${API_PORT} --host ${API_HOST} > backend.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "[Real E2E] Stopping backend server PID ${SERVER_PID}..."
    kill -9 "${SERVER_PID}" || true
}
trap cleanup EXIT

# 3. Wait for backend to be healthy
echo "[Real E2E] Waiting for server health check..."
for i in {1..30}; do
    if curl -s "${API_URL}/health" > /dev/null; then
        echo "[Real E2E] Server is healthy and running!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[Real E2E] ERROR: Timeout waiting for FastAPI server."
        cat backend.log
        exit 1
    fi
    sleep 0.5
done

# 4. Run Playwright Real E2E Spec
echo "[Real E2E] Starting Playwright Real E2E integration test..."
cd "${CWD}/vbgui"

set +e
npx playwright test e2e/scenarios/real_e2e_integration.spec.ts --config=e2e/playwright.config.ts
TEST_EXIT_CODE=$?
set -e

if [ "${TEST_EXIT_CODE}" -ne 0 ]; then
    echo "[Real E2E] ERROR: Real E2E integration test failed! Printing backend logs..."
    cat "${CWD}/backend.log"
else
    echo "[Real E2E] SUCCESS: Real E2E system integration test passed successfully!"
fi

exit "${TEST_EXIT_CODE}"
