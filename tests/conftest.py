"""Shared test configuration, in-memory database setup, and client fixtures."""

import os
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set in-memory test database and test JWT secret before any application imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from fastapi.testclient import TestClient  # noqa: E402

import app.models  # noqa: F401, E402
from app.db.session import Base  # noqa: E402
from app.db.session import SessionLocal as TestSessionLocal  # noqa: E402
from app.db.session import engine as test_engine  # noqa: E402
from app.main import app  # noqa: E402

__all__ = [
    "Base",
    "TestSessionLocal",
    "client",
    "init_test_db",
    "test_client",
    "test_engine",
]


def init_test_db():
    """Ensure all SQLAlchemy tables are created in the shared in-memory database."""
    Base.metadata.create_all(bind=test_engine)


# Initialize schema immediately upon test module load
init_test_db()

# Pre-configured test clients
test_client = TestClient(app)
client = test_client
