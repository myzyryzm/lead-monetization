#!/usr/bin/env bash
# Start the app locally.
#
#   ./start_local.sh            # build the SPA if needed, serve it all on :5000
#   ./start_local.sh --dev      # hot reload: Flask on :5000 + Vite on :5173
#   ./start_local.sh --rebuild  # force a fresh frontend build, then serve on :5000
#
# First run bootstraps everything (venv, pip deps, npm deps, SPA build, model
# training). Needs ANTHROPIC_API_KEY in backend/.env for chat to work.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"

# --- backend: venv + deps -----------------------------------------------------
if [ ! -d backend/.venv ]; then
  echo "[start_local] creating backend venv..."
  python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate
echo "[start_local] syncing python deps..."
pip install -q -r backend/requirements.txt

if ! grep -q "^ANTHROPIC_API_KEY=" backend/.env 2>/dev/null; then
  echo "[start_local] WARNING: no ANTHROPIC_API_KEY= line in backend/.env — chat will return 503."
fi

# --- frontend -----------------------------------------------------------------
if [ "$MODE" = "--dev" ]; then
  # Flask (API) in the background, Vite (hot-reload UI, proxies /api) in front.
  ( cd backend && python app.py ) &
  FLASK_PID=$!
  trap 'kill "$FLASK_PID" 2>/dev/null' EXIT INT TERM
  cd frontend
  [ -d node_modules ] || { echo "[start_local] npm install..."; npm install; }
  echo "[start_local] dev mode: UI at http://localhost:5173 (API on :5000)"
  npm run dev
else
  if [ "$MODE" = "--rebuild" ] || [ ! -d frontend/dist ]; then
    cd frontend
    [ -d node_modules ] || { echo "[start_local] npm install..."; npm install; }
    echo "[start_local] building the SPA..."
    npm run build
    cd ..
  fi
  echo "[start_local] serving at http://localhost:5000"
  cd backend && exec python app.py
fi
