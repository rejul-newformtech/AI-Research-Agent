"""Database session and engine management using asynchronous SQLAlchemy."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("app.db.session")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def create_async_db_engine() -> AsyncEngine:
    """Create and configure an asynchronous SQLAlchemy engine adapted to the database dialect."""
    url = settings.effective_async_database_url

    kwargs: dict[str, Any] = {
        "echo": settings.debug,
    }

    if "sqlite" in url:
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
        else:
            # Extract filesystem path for SQLite to ensure directory exists
            cleaned_path = url.split("sqlite+aiosqlite:///")[-1].split("sqlite:///")[-1]
            if cleaned_path and not cleaned_path.startswith(":"):
                db_path = Path(cleaned_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # PostgreSQL / asyncpg connection pooling
        kwargs.update(
            {
                "pool_size": settings.db_pool_size,
                "max_overflow": settings.db_max_overflow,
                "pool_timeout": settings.db_pool_timeout,
                "pool_recycle": settings.db_pool_recycle,
                "pool_pre_ping": True,
            }
        )

    logger.info(f"Connecting to database backend: {url.split('://')[0]}")
    return create_async_engine(url, **kwargs)


async_engine = create_async_db_engine()
engine = async_engine  # Alias for backward compatibility

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
SessionLocal = AsyncSessionLocal  # Alias


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an independent asynchronous database session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Initialize database tables asynchronously for registered SQLAlchemy models."""
    logger.info("Initializing database tables asynchronously...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")
