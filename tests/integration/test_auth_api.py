"""Integration tests for Authentication API endpoints and Role-Based Access Control (RBAC)."""

import unittest

from tests.conftest import TestSessionLocal, client, init_test_db


class TestAuthAPI(unittest.TestCase):
    """Test registration, login, profile, and role authorization endpoints."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        from app.models.user import User

        db = TestSessionLocal()
        db.query(User).filter(User.username.in_(["dr_watson_api", "superadmin_api"])).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()

    def test_01_register_user_success(self):
        payload = {
            "email": "researcher_api@example.com",
            "username": "dr_watson_api",
            "password": "Password123!",
            "role": "researcher",
        }
        res = client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["username"], "dr_watson_api")
        self.assertEqual(data["email"], "researcher_api@example.com")
        self.assertEqual(data["role"], "researcher")
        self.assertTrue(data["is_active"])
        self.assertIn("id", data)

    def test_02_register_duplicate_username_fails(self):
        payload = {
            "email": "another_api@example.com",
            "username": "dr_watson_api",
            "password": "Password123!",
            "role": "researcher",
        }
        res = client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("username already exists", res.json()["detail"])

    def test_03_login_json_success(self):
        payload = {
            "username": "dr_watson_api",
            "password": "Password123!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["username"], "dr_watson_api")
        self.assertEqual(data["role"], "researcher")
        self.assertIn("access_token", data)

    def test_04_login_with_email_success(self):
        payload = {
            "username": "researcher_api@example.com",
            "password": "Password123!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("access_token", res.json())

    def test_05_login_invalid_password_fails(self):
        payload = {
            "username": "dr_watson_api",
            "password": "WrongPassword!",
        }
        res = client.post("/api/v1/auth/login", json=payload)
        self.assertEqual(res.status_code, 401)

    def test_06_get_profile_authenticated(self):
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "dr_watson_api", "password": "Password123!"},
        )
        token = login_res.json()["access_token"]

        headers = {"Authorization": f"Bearer {token}"}
        res = client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["username"], "dr_watson_api")

        res_no_token = client.get("/api/v1/auth/me")
        self.assertEqual(res_no_token.status_code, 401)

    def test_07_rbac_admin_user_listing(self):
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "admin_api@example.com",
                "username": "superadmin_api",
                "password": "AdminPassword123!",
                "role": "admin",
            },
        )

        researcher_token = client.post(
            "/api/v1/auth/login",
            json={"username": "dr_watson_api", "password": "Password123!"},
        ).json()["access_token"]

        admin_token = client.post(
            "/api/v1/auth/login",
            json={"username": "superadmin_api", "password": "AdminPassword123!"},
        ).json()["access_token"]

        res_forbidden = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {researcher_token}"},
        )
        self.assertEqual(res_forbidden.status_code, 403)

        res_admin = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(res_admin.status_code, 200)
        users = res_admin.json()
        self.assertGreaterEqual(len(users), 2)

    def test_08_protected_endpoints_reject_unauthenticated(self):
        res = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        self.assertEqual(res.status_code, 401)

        res = client.post("/api/v1/ingest/text", json={"text": "Research data"})
        self.assertEqual(res.status_code, 401)

        res = client.post("/api/v1/ingest/search", json={"query": "Test search"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
