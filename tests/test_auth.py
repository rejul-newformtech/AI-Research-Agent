import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure test in-memory SQLite database before imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from research_agent.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from tests.test_db import init_test_db
from tests.test_db import test_client as client


class TestSecurityUtilities(unittest.TestCase):
    """Test core password hashing and JWT token operations."""

    def test_password_hashing(self):
        raw_password = "SuperSecretPassword123!"
        hashed = hash_password(raw_password)
        self.assertNotEqual(raw_password, hashed)
        self.assertTrue(verify_password(raw_password, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_jwt_token_flow(self):
        data = {"sub": "testuser", "user_id": 42, "role": "researcher"}
        token = create_access_token(data, expires_delta=timedelta(minutes=15))
        decoded = decode_access_token(token)
        self.assertEqual(decoded["sub"], "testuser")
        self.assertEqual(decoded["user_id"], 42)
        self.assertEqual(decoded["role"], "researcher")


class TestAuthAPI(unittest.TestCase):
    """Test registration, login, profile, and role authorization endpoints."""

    @classmethod
    def setUpClass(cls):
        init_test_db()

    def test_01_register_user_success(self):
        payload = {
            "email": "researcher@example.com",
            "username": "dr_watson",
            "password": "Password123!",
            "role": "researcher",
        }
        res = client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["username"], "dr_watson")
        self.assertEqual(data["email"], "researcher@example.com")
        self.assertEqual(data["role"], "researcher")
        self.assertTrue(data["is_active"])
        self.assertIn("id", data)

    def test_02_register_duplicate_username_fails(self):
        payload = {
            "email": "another@example.com",
            "username": "dr_watson",
            "password": "Password123!",
            "role": "researcher",
        }
        res = client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("username already exists", res.json()["detail"])

    def test_03_login_json_success(self):
        payload = {
            "username": "dr_watson",
            "password": "Password123!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["username"], "dr_watson")
        self.assertEqual(data["role"], "researcher")
        self.assertIn("access_token", data)

    def test_04_login_with_email_success(self):
        # Test login using registered email address
        payload = {
            "username": "researcher@example.com",
            "password": "Password123!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("access_token", res.json())

    def test_05_login_invalid_password_fails(self):
        payload = {
            "username": "dr_watson",
            "password": "WrongPassword!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 401)

    def test_06_get_profile_authenticated(self):
        # Login to get token
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "dr_watson", "password": "Password123!"},
        )
        token = login_res.json()["access_token"]

        # Call /me with valid token
        headers = {"Authorization": f"Bearer {token}"}
        res = client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["username"], "dr_watson")

        # Call /me without token -> 401
        res_no_token = client.get("/api/v1/auth/me")
        self.assertEqual(res_no_token.status_code, 401)

    def test_07_rbac_admin_user_listing(self):
        # Register an admin user
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "admin@example.com",
                "username": "superadmin",
                "password": "AdminPassword123!",
                "role": "admin",
            },
        )

        # Login as researcher
        researcher_token = client.post(
            "/api/v1/auth/login",
            json={"username": "dr_watson", "password": "Password123!"},
        ).json()["access_token"]

        # Login as admin
        admin_token = client.post(
            "/api/v1/auth/login",
            json={"username": "superadmin", "password": "AdminPassword123!"},
        ).json()["access_token"]

        # Researcher should be forbidden (403) from accessing admin user listing
        res_forbidden = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {researcher_token}"},
        )
        self.assertEqual(res_forbidden.status_code, 403)

        # Admin should succeed (200)
        res_admin = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(res_admin.status_code, 200)
        users = res_admin.json()
        self.assertGreaterEqual(len(users), 2)

    def test_08_protected_endpoints_reject_unauthenticated(self):
        # Test agent chat rejects requests without auth token
        res = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        self.assertEqual(res.status_code, 401)

        # Test ingestion text rejects requests without auth token
        res = client.post("/api/v1/ingest/text", json={"text": "Research data"})
        self.assertEqual(res.status_code, 401)

        # Test vector search rejects requests without auth token
        res = client.post("/api/v1/ingest/search", json={"query": "Test search"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
