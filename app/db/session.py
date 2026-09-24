"""Database session and engine management using asynchronous SQLAlchemy and SQLite."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.logger import get_logger
from app.db.base import Base

logger = get_logger("app.db.session")


def create_async_db_engine() -> AsyncEngine:
    """Create and configure an asynchronous SQLite engine using aiosqlite."""
    url = settings.effective_async_database_url

    kwargs: dict[str, Any] = {
        "echo": settings.debug,
        "connect_args": {"check_same_thread": False},
    }

    if ":memory:" in url:
        kwargs["poolclass"] = StaticPool
    else:
        # Extract filesystem path for SQLite to ensure directory exists
        cleaned_path = (
            url.split("sqlite+aiosqlite:///")[-1]
            .split("sqlite+aiosqlite://")[-1]
            .split("sqlite:///")[-1]
            .split("sqlite://")[-1]
        )
        if cleaned_path and not cleaned_path.startswith(":"):
            db_path = Path(cleaned_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to SQLite database backend")
    return create_async_engine(url, **kwargs)


engine = create_async_db_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an independent asynchronous database session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Initialize database tables asynchronously for registered SQLAlchemy models."""
    logger.info("Initializing database tables asynchronously...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")
