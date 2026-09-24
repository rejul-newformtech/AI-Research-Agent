"""Pytest fixtures for database, API clients, and authentication."""

import os
from collections.abc import AsyncGenerator, Generator

# 1. Configure test environment variables before importing app
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password
from app.db.base import Base
from app.db.session import AsyncSessionLocal, engine
from app.main import app
from app.models.user import User, UserRole
from app.service.chunking import DocumentChunk

# ============================================================================
# Database Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def setup_test_database():
    """Create in-memory SQLite tables once for the test session."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated database session that rolls back and closes after each test."""
    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


# ============================================================================
# Client Fixtures (Sync & Async)
# ============================================================================


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a synchronous TestClient."""
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an unauthenticated AsyncClient."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================================
# User & Auth Fixtures
# ============================================================================


async def _get_or_create_user(db: AsyncSession, username: str, email: str, role: UserRole) -> User:
    stmt = select(User).where(User.username == username)
    user = await db.scalar(stmt)
    if not user:
        user = User(
            email=email,
            username=username,
            hashed_password=hash_password("Password123!"),
            role=role,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def researcher_user(db: AsyncSession) -> User:
    """Create or return a standard researcher user."""
    return await _get_or_create_user(
        db,
        username="fixture_researcher",
        email="fixture_researcher@example.com",
        role=UserRole.RESEARCHER,
    )


# Alias for backward compatibility
test_researcher_user = researcher_user


@pytest.fixture
def researcher_token(researcher_user: User) -> str:
    """JWT bearer token for the researcher user."""
    return create_access_token(
        {
            "sub": researcher_user.username,
            "user_id": researcher_user.id,
            "role": researcher_user.role,
        }
    )


@pytest.fixture
def researcher_headers(researcher_token: str) -> dict[str, str]:
    """Authorization header for researcher requests."""
    return {"Authorization": f"Bearer {researcher_token}"}


@pytest_asyncio.fixture
async def async_researcher_client(
    researcher_headers: dict[str, str],
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an authenticated AsyncClient with researcher credentials."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=researcher_headers
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def admin_user(db: AsyncSession) -> User:
    """Create or return a standard admin user."""
    return await _get_or_create_user(
        db,
        username="fixture_admin",
        email="fixture_admin@example.com",
        role=UserRole.ADMIN,
    )


# Alias for backward compatibility
test_admin_user = admin_user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """JWT bearer token for the admin user."""
    return create_access_token(
        {
            "sub": admin_user.username,
            "user_id": admin_user.id,
            "role": admin_user.role,
        }
    )


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    """Authorization header for admin requests."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest_asyncio.fixture
async def async_admin_client(admin_headers: dict[str, str]) -> AsyncGenerator[AsyncClient, None]:
    """Provide an authenticated AsyncClient with admin credentials."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=admin_headers
    ) as ac:
        yield ac


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def sample_document_chunks() -> list[DocumentChunk]:
    """Sample document chunks for testing search and retrieval."""
    return [
        DocumentChunk(
            chunk_index=0,
            text="Photosynthesis converts solar light into chemical glucose.",
            metadata={"source": "botany.pdf", "page_number": 1, "topic": "biology"},
        ),
        DocumentChunk(
            chunk_index=1,
            text="Quantum mechanics governs discrete energy transitions in atoms.",
            metadata={"source": "physics.pdf", "page_number": 10, "topic": "physics"},
        ),
        DocumentChunk(
            chunk_index=2,
            text="Operational amplifiers with negative feedback provide linear amplification.",
            metadata={"source": "circuits.pdf", "page_number": 42, "topic": "electronics"},
        ),
    ]
