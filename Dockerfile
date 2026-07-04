# --- Stage 1: build the React SPA -------------------------------------------
# --platform=$BUILDPLATFORM: run natively on the build host (esbuild crashes
# under QEMU emulation); the dist/ output is static files, so the target
# architecture doesn't matter here.
FROM --platform=$BUILDPLATFORM node:20-alpine AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

# --- Stage 2: Python runtime -------------------------------------------------
# python:3.12-slim matches the dev venv so backend/model.pkl unpickles as-is.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# backend/ and frontend/dist must stay siblings: app.py resolves the SPA at
# ../frontend/dist relative to itself.
COPY backend/ /app/backend/
COPY --from=frontend /frontend/dist /app/frontend/dist

WORKDIR /app/backend
EXPOSE 5000

# /api/chat chains up to 8 sequential LLM calls, so the default 30s worker
# timeout would kill it; gthread keeps one slow chat from blocking the worker.
CMD ["sh", "-c", "exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-5000} \
    --worker-class gthread --workers 2 --threads 8 \
    --timeout ${GUNICORN_TIMEOUT:-600} --graceful-timeout 30 \
    --preload --access-logfile - --error-logfile -"]
