"""Shared pytest fixtures, in-memory database setup, and async test client configuration."""

import os
import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

# Ensure project root is in python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set in-memory test database and test JWT secret before application imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import app.models  # noqa: F401, E402
from app.core.security import create_access_token  # noqa: E402
from app.db.session import Base  # noqa: E402
from app.db.session import SessionLocal as TestSessionLocal  # noqa: E402
from app.db.session import engine as test_engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.service.chunking import DocumentChunk  # noqa: E402

__all__ = [
    "Base",
    "TestSessionLocal",
    "async_admin_client",
    "async_client",
    "async_researcher_client",
    "client",
    "db_session",
    "init_test_db",
    "test_client",
    "test_engine",
]


def init_test_db():
    """Ensure all SQLAlchemy tables are created in the shared in-memory database."""
    Base.metadata.create_all(bind=test_engine)


# Initialize schema immediately upon test module load
init_test_db()

# Pre-configured global synchronous test clients (for backward-compatible imports)
test_client = TestClient(app)


# =========================================================================
# Pytest & Pytest-Asyncio Fixtures
# =========================================================================


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Yield a synchronous TestClient backed by the FastAPI application."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db() -> Generator[Session, None, None]:
    """Yield an isolated SQLAlchemy session that closes after the test."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Alias for db fixture."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Yield an async HTTP test client backed by the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_researcher_user(db: Session) -> User:
    """Fixture ensuring a standard researcher user exists in the test database."""
    user = db.query(User).filter(User.username == "fixture_researcher").first()
    if not user:
        user = User(
            email="fixture_researcher@example.com",
            username="fixture_researcher",
            hashed_password="hashed_fixture_password",
            role=UserRole.RESEARCHER,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@pytest.fixture
def researcher_token(test_researcher_user: User) -> str:
    """Fixture returning a valid JWT bearer token for a researcher."""
    return create_access_token(
        {
            "sub": test_researcher_user.username,
            "user_id": test_researcher_user.id,
            "role": test_researcher_user.role,
        }
    )


@pytest.fixture
def researcher_headers(researcher_token: str) -> dict[str, str]:
    """Fixture returning authorization headers for a researcher."""
    return {"Authorization": f"Bearer {researcher_token}"}


@pytest.fixture
def test_admin_user(db: Session) -> User:
    """Fixture ensuring an admin user exists in the test database."""
    user = db.query(User).filter(User.username == "fixture_admin").first()
    if not user:
        user = User(
            email="fixture_admin@example.com",
            username="fixture_admin",
            hashed_password="hashed_fixture_password",
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@pytest.fixture
def admin_token(test_admin_user: User) -> str:
    """Fixture returning a valid JWT bearer token for an admin."""
    return create_access_token(
        {
            "sub": test_admin_user.username,
            "user_id": test_admin_user.id,
            "role": test_admin_user.role,
        }
    )


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    """Fixture returning authorization headers for an admin."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest_asyncio.fixture
async def async_admin_client(admin_headers: dict[str, str]) -> AsyncGenerator[AsyncClient, None]:
    """Yield an authenticated AsyncClient with admin credentials."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=admin_headers
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def async_researcher_client(
    researcher_headers: dict[str, str],
) -> AsyncGenerator[AsyncClient, None]:
    """Yield an authenticated AsyncClient with researcher credentials."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=researcher_headers
    ) as ac:
        yield ac


@pytest.fixture
def sample_document_chunks() -> list[DocumentChunk]:
    """Fixture providing sample document chunks for retrieval tests."""
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
