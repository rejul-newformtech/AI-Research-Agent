"""Database session and engine management using SQLAlchemy."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from research_agent.core.config import settings
from research_agent.core.logger import get_logger

logger = get_logger("research_agent.db.session")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def create_db_engine():
    """Create and configure a SQLAlchemy engine adapted to the database dialect."""
    url = settings.effective_database_url

    # Normalize standard postgres URI prefixes if needed
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and not any(
        driver in url for driver in ("+psycopg", "+asyncpg")
    ):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    kwargs: dict[str, Any] = {
        "echo": settings.debug,
    }

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        db_path_str = url.replace("sqlite:///", "")
        if db_path_str and not db_path_str.startswith(":memory:"):
            db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # PostgreSQL / MySQL enterprise connection pooling
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
    return create_engine(url, **kwargs)


engine = create_db_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields an independent database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables for registered SQLAlchemy models."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
