"""Shared in-memory test database engine, session, and FastAPI test client."""

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import research_agent.models  # noqa: F401 - Register all models with Base.metadata
from research_agent.api.dependencies.auth import get_db as auth_get_db
from research_agent.db.session import Base, get_db
from research_agent.main import app

# Shared in-memory SQLite engine with StaticPool
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_test_db():
    """Ensure all SQLAlchemy tables are created in the shared in-memory database."""
    Base.metadata.create_all(bind=test_engine)


# Initialize schema immediately upon test module load
init_test_db()

# Configure FastAPI dependency overrides once globally
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[auth_get_db] = override_get_db
test_client = TestClient(app)
