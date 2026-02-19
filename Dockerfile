# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
# Install system deps: libpq5 for asyncpg, libgomp1 for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
CMD exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 120 \
    --graceful-timeout 120 \
    --keep-alive 75 \
    --access-logfile - \
    --error-logfile -
