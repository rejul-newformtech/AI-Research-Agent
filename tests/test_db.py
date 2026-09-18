"""Shared in-memory test database engine, session, and FastAPI test client."""

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from fastapi.testclient import TestClient

import app.models  # noqa: F401 - Register all models with Base.metadata
from app.db.session import Base
from app.db.session import SessionLocal as TestSessionLocal
from app.db.session import engine as test_engine
from app.main import app

__all__ = ["Base", "TestSessionLocal", "init_test_db", "test_client", "test_engine"]


def init_test_db():
    """Ensure all SQLAlchemy tables are created in the shared in-memory database."""
    Base.metadata.create_all(bind=test_engine)


# Initialize schema immediately upon test module load
init_test_db()
test_client = TestClient(app)
