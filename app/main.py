"""
Harmonia V3 — FastAPI Application Entry Point

Production-ready application with:
- Async lifespan management (DB pool, Redis, GCS model-weight sync)
- CORS, timeout, and structured-logging middleware
- Health-check endpoints (liveness + deep readiness)
- Active-request tracking for graceful shutdown
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings
from app.database import async_session_factory, engine

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger("harmonia")

# ---------------------------------------------------------------------------
# Active request counter for graceful shutdown
# ---------------------------------------------------------------------------

_active_requests: int = 0
_active_requests_lock = asyncio.Lock()
_shutdown_event = asyncio.Event()

DRAIN_TIMEOUT_SECONDS = 15


async def _increment_active() -> None:
    global _active_requests
    async with _active_requests_lock:
        _active_requests += 1


async def _decrement_active() -> None:
    global _active_requests
    async with _active_requests_lock:
        _active_requests -= 1


async def _drain_active_requests() -> None:
    """Wait until all in-flight requests complete or timeout expires."""
    deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
    while True:
        async with _active_requests_lock:
            if _active_requests <= 0:
                break
        if time.monotonic() >= deadline:
            logger.warning(
                "drain_timeout_exceeded",
                remaining_requests=_active_requests,
            )
            break
        await asyncio.sleep(0.25)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

_redis_client = None


async def _connect_redis() -> None:
    global _redis_client
    import redis.asyncio as aioredis

    settings = get_settings()
    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    # Verify connectivity
    await _redis_client.ping()
    logger.info("redis_connected", url=settings.REDIS_URL)


async def _close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("redis_closed")


def get_redis():
    """Return the shared Redis client (for use in health checks, etc.)."""
    return _redis_client


# ---------------------------------------------------------------------------
# GCS model-weight download with SHA-256 integrity
# ---------------------------------------------------------------------------

async def _ensure_metafbp_weights() -> None:
    """Download MetaFBP weights from GCS if they are not present locally.

    The function checks for both the extractor and generator weight files.
    Each downloaded blob is verified against a companion ``.sha256`` sidecar
    stored in the same GCS prefix.  If the sidecar does not exist the
    download proceeds without integrity verification (the weights may not
    have been published yet).
    """
    from google.cloud import storage as gcs

    settings = get_settings()

    if not settings.GCS_BUCKET_NAME:
        logger.info("gcs_skip", reason="GCS_BUCKET_NAME not configured")
        return

    client = gcs.Client(project=settings.GCP_PROJECT_ID or None)
    bucket = client.bucket(settings.GCS_BUCKET_NAME)

    weight_files = [
        settings.METAFBP_EXTRACTOR_PATH,
        settings.METAFBP_GENERATOR_PATH,
    ]

    for weight_path_str in weight_files:
        local_path = Path(weight_path_str)

        if local_path.exists():
            logger.info("weights_present", path=str(local_path))
            continue

        blob_name = f"{settings.GCS_MODEL_WEIGHTS_PREFIX}{local_path.name}"
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.warning("weights_blob_missing", blob=blob_name)
            continue

        # Ensure parent directories exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download to a temporary file first, then rename for atomicity
        tmp_path = local_path.with_suffix(".tmp")
        blob.download_to_filename(str(tmp_path))

        # SHA-256 integrity check (sidecar file: <blob>.sha256)
        sha_blob_name = f"{blob_name}.sha256"
        sha_blob = bucket.blob(sha_blob_name)

        if sha_blob.exists():
            expected_hash = sha_blob.download_as_text().strip().split()[0]
            sha256 = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()

            if actual_hash != expected_hash:
                tmp_path.unlink(missing_ok=True)
                logger.error(
                    "weights_integrity_failure",
                    blob=blob_name,
                    expected=expected_hash,
                    actual=actual_hash,
                )
                continue

            logger.info("weights_integrity_ok", blob=blob_name)
        else:
            logger.info(
                "weights_sha256_sidecar_missing",
                blob=sha_blob_name,
                note="skipping integrity check",
            )

        tmp_path.rename(local_path)
        logger.info("weights_downloaded", path=str(local_path))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of long-lived resources."""
    settings = get_settings()

    # -- Startup --------------------------------------------------------- #
    logger.info(
        "startup_begin",
        environment=settings.ENVIRONMENT,
        log_level=settings.LOG_LEVEL,
    )

    # 1. Database connection pool — engine is already created at module level
    #    in app.database; issuing a simple query warms the pool.
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("database_pool_initialised")

    # 2. Redis
    await _connect_redis()

    # 3. MetaFBP weights (best-effort)
    try:
        await asyncio.to_thread(_sync_ensure_weights)
    except Exception:
        logger.exception("metafbp_weight_download_failed")

    logger.info("startup_complete")

    yield

    # -- Shutdown -------------------------------------------------------- #
    logger.info("shutdown_begin")

    # 1. Drain in-flight requests
    _shutdown_event.set()
    await _drain_active_requests()

    # 2. Close Redis
    await _close_redis()

    # 3. Dispose DB engine (closes the connection pool)
    await engine.dispose()
    logger.info("database_pool_closed")

    logger.info("shutdown_complete")


def _sync_ensure_weights() -> None:
    """Synchronous wrapper so the blocking GCS SDK can run in a thread."""
    import asyncio as _asyncio

    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(_ensure_metafbp_weights())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Middleware classes
# ---------------------------------------------------------------------------

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests that exceed a configurable wall-clock timeout."""

    def __init__(self, app, timeout_seconds: float = 70.0) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "request_timeout",
                method=request.method,
                path=request.url.path,
                timeout=self.timeout_seconds,
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"},
            )


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()

        await _increment_active()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_error",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )
            raise
        finally:
            await _decrement_active()

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="Harmonia V3",
    description="Whole-the-Match compatibility platform",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# -- Middleware (applied in reverse order — last added runs first) ---------- #

app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(TimeoutMiddleware, timeout_seconds=70.0)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Health-check endpoints ------------------------------------------------ #


@app.get("/health", tags=["health"])
async def health_liveness() -> dict:
    """Lightweight liveness probe — always returns healthy if the process is
    running."""
    return {"status": "healthy"}


@app.get("/health/deep", tags=["health"])
async def health_deep() -> dict:
    """Deep readiness probe — verifies database, Redis, and GCS connectivity."""
    result: dict = {
        "status": "healthy",
        "database": "connected",
        "redis": "connected",
        "gcs": "accessible",
    }

    # Database
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("health_db_failure", error=str(exc))
        result["database"] = f"error: {exc}"
        result["status"] = "degraded"

    # Redis
    try:
        redis = get_redis()
        if redis is None:
            raise RuntimeError("Redis client not initialised")
        await redis.ping()
    except Exception as exc:
        logger.error("health_redis_failure", error=str(exc))
        result["redis"] = f"error: {exc}"
        result["status"] = "degraded"

    # GCS
    try:
        from google.cloud import storage as gcs

        gcs_settings = get_settings()
        if not gcs_settings.GCS_BUCKET_NAME:
            result["gcs"] = "not_configured"
        else:
            client = gcs.Client(project=gcs_settings.GCP_PROJECT_ID or None)
            bucket = client.bucket(gcs_settings.GCS_BUCKET_NAME)
            bucket.exists()
    except Exception as exc:
        logger.error("health_gcs_failure", error=str(exc))
        result["gcs"] = f"error: {exc}"
        result["status"] = "degraded"

    return result


# -- API router ------------------------------------------------------------ #

from app.api.router import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")
