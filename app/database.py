"""
Harmonia V3 — Async Database Engine & Session Factory

Provides two connection strategies:

1. **Cloud Run (production)** – Uses ``cloud-sql-python-connector`` with
   automatic IAM authentication over a Unix domain socket.  Activated when
   ``CLOUD_SQL_USE_UNIX_SOCKET`` is *True* **and** a valid
   ``CLOUD_SQL_INSTANCE_CONNECTION`` is provided.

2. **Local development** – Falls back to a standard ``asyncpg`` connection
   string read from ``DATABASE_URL``.

Both paths share the same pool tuning parameters and expose the same
``get_db`` async generator for FastAPI dependency injection.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Declarative base for all ORM models
# ------------------------------------------------------------------ #

class Base(DeclarativeBase):
    """Shared declarative base.

    Every SQLAlchemy model in the project should inherit from this class::

        from app.database import Base

        class User(Base):
            __tablename__ = "users"
            ...
    """
    pass


# ------------------------------------------------------------------ #
# Pool configuration (shared across both connection strategies)
# ------------------------------------------------------------------ #

_POOL_KWARGS = {
    "pool_size": 10,
    "max_overflow": 5,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "pool_pre_ping": True,
}


# ------------------------------------------------------------------ #
# Engine construction helpers
# ------------------------------------------------------------------ #

def _build_cloud_sql_engine():
    """Create an async engine that connects through the Cloud SQL Python
    Connector with automatic IAM authentication.

    The connector manages the SSL tunnel / Unix socket transparently so
    the application only needs the *instance connection name*
    (``project:region:instance``).
    """
    from google.cloud.sql.connector import Connector

    settings = get_settings()

    connector = Connector()

    async def _get_connection():
        return await connector.connect_async(
            settings.CLOUD_SQL_INSTANCE_CONNECTION,
            "asyncpg",
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            db=settings.DB_NAME,
            enable_iam_auth=True,
        )

    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=_get_connection,
        echo=(settings.LOG_LEVEL == "DEBUG"),
        **_POOL_KWARGS,
    )

    logger.info(
        "Database engine created via Cloud SQL Connector (%s)",
        settings.CLOUD_SQL_INSTANCE_CONNECTION,
    )
    return engine


def _build_local_engine():
    """Create an async engine from the plain ``DATABASE_URL`` string.

    Expects a URL of the form::

        postgresql+asyncpg://user:pass@host:5432/harmonia
    """
    settings = get_settings()
    url = settings.DATABASE_URL

    # Transparently upgrade a plain ``postgresql://`` scheme so that
    # developers do not need to remember the asyncpg dialect prefix.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(
        url,
        echo=(settings.LOG_LEVEL == "DEBUG"),
        **_POOL_KWARGS,
    )

    logger.info("Database engine created from DATABASE_URL (local / dev)")
    return engine


def _create_engine():
    """Select the appropriate engine builder based on configuration."""
    settings = get_settings()

    use_cloud_sql = (
        settings.CLOUD_SQL_USE_UNIX_SOCKET
        and settings.CLOUD_SQL_INSTANCE_CONNECTION
    )

    if use_cloud_sql:
        return _build_cloud_sql_engine()

    return _build_local_engine()


# ------------------------------------------------------------------ #
# Module-level engine & session factory (lazy-initialised)
# ------------------------------------------------------------------ #

engine = _create_engine()

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ------------------------------------------------------------------ #
# FastAPI dependency
# ------------------------------------------------------------------ #

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` and ensure it is closed afterwards.

    Usage in a FastAPI route::

        from fastapi import Depends
        from app.database import get_db

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
