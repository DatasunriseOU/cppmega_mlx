#!/bin/bash
# ---------------------------------------------------------------------------
# cppmega E2E Mega-Matrix Test Runner & Diagnostician
#
# Starts the background FastAPI RPC server, executes the 30-scenario E2E
# Playwright suite, captures console/API/training logs, collects screenshot
# assets, and shuts down safely on exit.
# ---------------------------------------------------------------------------

set -euo pipefail

CWD="/Users/dave/sources/cppmega.mlx"
API_PORT=8767
API_HOST="127.0.0.1"
API_URL="http://${API_HOST}:${API_PORT}"

# 1. Cleanup old logs & screenshots
echo "[E2E Mega-Matrix] Cleaning up old test logs and assets..."
rm -f "${CWD}/backend.log" "${CWD}/frontend.log"
rm -rf "${CWD}/vbgui/e2e/screenshots/*"

# 2. Start cppmega JSON-RPC backend server in background
echo "[E2E Mega-Matrix] Launching background FastAPI server on port ${API_PORT}..."
cd "${CWD}"
.venv/bin/python -m uvicorn cppmega_v4.jsonrpc.server:create_app --factory --port ${API_PORT} --host ${API_HOST} > backend.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "[E2E Mega-Matrix] Stopping backend server PID ${SERVER_PID}..."
    kill -9 "${SERVER_PID}" || true
}
trap cleanup EXIT

# 3. Wait for backend to be healthy
echo "[E2E Mega-Matrix] Waiting for server health check..."
for i in {1..30}; do
    if curl -s "${API_URL}/health" > /dev/null; then
        echo "[E2E Mega-Matrix] Server is healthy and running!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[E2E Mega-Matrix] ERROR: Timeout waiting for FastAPI server."
        cat backend.log
        exit 1
    fi
    sleep 0.5
done

# 4. Run Playwright E2E Mega-Matrix Suite
echo "[E2E Mega-Matrix] Starting Playwright E2E Matrix tests..."
cd "${CWD}/vbgui"

# Run tests
set +e
npx playwright test e2e/scenarios/mega_matrix.spec.ts --config=e2e/playwright.config.ts
TEST_EXIT_CODE=$?
set -e

# 5. Extract screenshots and console logs on error
if [ "${TEST_EXIT_CODE}" -ne 0 ]; then
    echo "[E2E Mega-Matrix] ERROR: Some E2E scenarios failed! Extracting logs and screenshots..."
    # Archive failure logs for debugging
    mkdir -p "${CWD}/reports/e2e_failures"
    cp "${CWD}/backend.log" "${CWD}/reports/e2e_failures/api_backend.log"
    echo "[E2E Mega-Matrix] Logs copied to reports/e2e_failures/."
else
    echo "[E2E Mega-Matrix] SUCCESS: All 30 E2E scenarios passed successfully!"
fi

exit "${TEST_EXIT_CODE}"
