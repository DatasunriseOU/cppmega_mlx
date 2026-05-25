#!/bin/bash
# ---------------------------------------------------------------------------
# cppmega Unified Single-Port Startup Script
#
# Builds the React visual builder frontend statically (if dist/ is missing
# or if clean build is requested) and launches the FastAPI backend server,
# serving both the visual builder UI and the JSON-RPC API on port 8765!
# ---------------------------------------------------------------------------

set -euo pipefail

CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VBGUI_CACHE_DIR="${CWD}/data/cache/datasets"

CLEAN_BUILD=false

# Help menu
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --clean    Force a clean static build of the visual builder React UI before launch"
    echo "  -h, --help     Show this help menu"
    exit 0
}

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "=========================================================================="
echo "⚡ cppmega Visual Builder Single-Port Launcher"
echo "=========================================================================="

# 1. Verify virtual environment
if [ ! -d "${CWD}/.venv" ]; then
    echo "⚠️  WARNING: Virtual environment (.venv) not found at ${CWD}/.venv!"
    echo "Please configure your cppmega python virtual environment first."
    exit 1
fi

# 2. Check and compile static UI if dist is missing or clean build requested
cd "${CWD}/vbgui"

if [ ! -d "dist" ] || [ "${CLEAN_BUILD}" = true ]; then
    echo "📦 Compiling frontend React visual builder UI..."
    if [ ! -d "node_modules" ]; then
        echo "📥 Installing frontend node dependencies..."
        npm install
    fi
    echo "🚀 Building static assets with Vite..."
    npm run build
    echo "✅ Frontend build completed successfully! Static files saved to vbgui/dist/."
else
    echo "✨ Static visual builder build folder found. Skipping frontend compilation."
    echo "💡 Run '$0 --clean' if you want to rebuild frontend modifications."
fi

# 3. Launch single-port unified server
echo "⚡ Launching unified FastAPI server on port 8765..."
echo "👉 Open your browser to: http://127.0.0.1:8765"
echo "=========================================================================="

cd "${CWD}"
exec .venv/bin/python -m uvicorn cppmega_v4.jsonrpc.server:create_app --factory --port 8765 --host 127.0.0.1
