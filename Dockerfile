# ---- Stage 1: build the React/Vite frontend ----
FROM node:20-alpine AS frontend-build

WORKDIR /build
COPY package.json package-lock.json vite.config.js ./
COPY frontend ./frontend
RUN npm ci && npm run build
# Build output: /build/backend/static

# ---- Stage 2: Python runtime with the packaged app ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md MANIFEST.in ./
COPY backend ./backend
COPY --from=frontend-build /build/backend/static ./backend/static

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install CPU-only torch first so the CUDA wheels pulled in by the default PyPI
# index are skipped; local embedding inference does not need them.
RUN pip install --no-cache-dir .

# All mutable state lives under /app; mount volumes to persist it.
ENV AI_STUDY_ASSISTANT_HOME=/app \
    EMBEDDING_MODEL_LOCAL_ONLY=false
RUN mkdir -p /app/docs /app/data /app/rag_index /app/logs

VOLUME ["/app/docs", "/app/data", "/app/rag_index", "/app/logs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["rulebook", "serve", "--host", "0.0.0.0", "--port", "8000"]
