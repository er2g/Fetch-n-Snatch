#!/bin/bash
set -euo pipefail

# Resolve script directory and switch there so uvicorn sees project modules
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-8002}"
HOST="0.0.0.0"
APP_MODULE="web_app.backend.main:app"

# Informational banner for convenience
cat <<BANNER
Starting FastAPI server from WSL
  App   : ${APP_MODULE}
  Host  : ${HOST}
  Port  : ${PORT}
  Dir   : ${SCRIPT_DIR}
Press Ctrl+C to stop.
BANNER

exec python3 -m uvicorn "${APP_MODULE}" --host "$HOST" --port "$PORT"
