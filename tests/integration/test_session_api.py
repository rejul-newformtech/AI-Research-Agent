"""Integration tests for session history endpoints and multi-user isolation."""

import unittest

from app.service.memory import ConversationMemoryService
from tests.conftest import TestSessionLocal, client, init_test_db


class TestSessionAPI(unittest.TestCase):
    """Integration tests for session history endpoints and user isolation."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        from app.models.user import User

        db = TestSessionLocal()
        db.query(User).filter(
            User.username.in_(["session_user_alpha_int", "session_user_beta_int"])
        ).delete(synchronize_session=False)
        db.commit()
        db.close()

        reg_a = client.post(
            "/api/v1/auth/register",
            json={
                "email": "session_user_alpha_int@example.com",
                "username": "session_user_alpha_int",
                "password": "Password123!",
                "role": "researcher",
            },
        ).json()
        cls.user_a_id = reg_a["id"]
        cls.token_a = client.post(
            "/api/v1/auth/login",
            json={"username": "session_user_alpha_int", "password": "Password123!"},
        ).json()["access_token"]

        reg_b = client.post(
            "/api/v1/auth/register",
            json={
                "email": "session_user_beta_int@example.com",
                "username": "session_user_beta_int",
                "password": "Password123!",
                "role": "researcher",
            },
        ).json()
        cls.user_b_id = reg_b["id"]
        cls.token_b = client.post(
            "/api/v1/auth/login",
            json={"username": "session_user_beta_int", "password": "Password123!"},
        ).json()["access_token"]

    def test_session_endpoints_and_isolation(self):
        db = TestSessionLocal()
        mem = ConversationMemoryService()

        # Seed a session for User A
        mem.get_or_create_session(
            db,
            session_id="user_a_session_int",
            user_id=self.user_a_id,
            initial_prompt="Hello from A",
        )
        mem.save_message(db, session_id="user_a_session_int", role="user", content="Question A")
        mem.save_message(db, session_id="user_a_session_int", role="assistant", content="Answer A")
        db.close()

        # User A can list their sessions
        res_a = client.get(
            "/api/v1/agent/sessions",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(res_a.status_code, 200)
        sessions_a = [s for s in res_a.json() if s["id"] == "user_a_session_int"]
        self.assertEqual(len(sessions_a), 1)
        self.assertEqual(sessions_a[0]["id"], "user_a_session_int")
        self.assertEqual(sessions_a[0]["message_count"], 2)

        # User B lists sessions -> should not contain User A's session
        res_b = client.get(
            "/api/v1/agent/sessions",
            headers={"Authorization": f"Bearer {self.token_b}"},
        )
        self.assertEqual(res_b.status_code, 200)
        sessions_b = [s for s in res_b.json() if s["id"] == "user_a_session_int"]
        self.assertEqual(len(sessions_b), 0)

        # User A gets history
        history_res = client.get(
            "/api/v1/agent/sessions/user_a_session_int",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(history_res.status_code, 200)
        self.assertEqual(len(history_res.json()["messages"]), 2)

        # User B tries to get User A's session -> 404
        forbidden_get = client.get(
            "/api/v1/agent/sessions/user_a_session_int",
            headers={"Authorization": f"Bearer {self.token_b}"},
        )
        self.assertEqual(forbidden_get.status_code, 404)

        # User A deletes their session
        del_res = client.delete(
            "/api/v1/agent/sessions/user_a_session_int",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(del_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
