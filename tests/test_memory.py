import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure test in-memory SQLite database
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from research_agent.models.chat import ChatMessage
from research_agent.models.user import User
from research_agent.service.memory import ConversationMemoryService
from tests.test_db import TestSessionLocal, init_test_db
from tests.test_db import test_client as client


class TestMemoryService(unittest.TestCase):
    """Unit tests for ConversationMemoryService logic and context windowing."""

    @classmethod
    def setUpClass(cls):
        init_test_db()

    def setUp(self):
        self.db = TestSessionLocal()
        self.memory = ConversationMemoryService(default_window_size=4)

        # Get or create test user
        self.user = self.db.query(User).filter(User.username == "memuser").first()
        if not self.user:
            self.user = User(
                email="memory_tester@example.com",
                username="memuser",
                hashed_password="hashed_test_password",
                role="researcher",
            )
            self.db.add(self.user)
            self.db.commit()
            self.db.refresh(self.user)

    def tearDown(self):
        self.db.close()

    def test_01_session_creation_and_auto_titling(self):
        prompt = "Explain quantum electrodynamics and Feynman diagrams in detail."
        session = self.memory.get_or_create_session(
            db=self.db,
            session_id="session_qed",
            user_id=self.user.id,
            initial_prompt=prompt,
        )
        self.assertEqual(session.id, "session_qed")
        self.assertTrue(session.title.startswith("Explain quantum electrodynamics"))
        self.assertTrue(session.title.endswith("..."))

    def test_02_message_persistence_and_sliding_window(self):
        session_id = "window_test_session"
        self.memory.get_or_create_session(
            db=self.db,
            session_id=session_id,
            user_id=self.user.id,
            initial_prompt="Turn 1",
        )

        # Insert 6 message turns (Turn 1 to 6)
        for i in range(1, 7):
            role = "user" if i % 2 != 0 else "assistant"
            self.memory.save_message(
                db=self.db,
                session_id=session_id,
                role=role,
                content=f"Message {i}",
            )

        # Query with sliding window size of 4
        windowed = self.memory.get_windowed_history(
            db=self.db,
            session_id=session_id,
            max_messages=4,
        )

        # Should only return the 4 most recent messages (Message 3, 4, 5, 6) in chronological order
        self.assertEqual(len(windowed), 4)
        self.assertEqual(
            [m.content for m in windowed], ["Message 3", "Message 4", "Message 5", "Message 6"]
        )

    def test_03_list_and_delete_session(self):
        sessions = self.memory.list_user_sessions(db=self.db, user_id=self.user.id)
        self.assertGreaterEqual(len(sessions), 2)

        # Delete session
        deleted = self.memory.delete_session(
            db=self.db,
            session_id="window_test_session",
            user_id=self.user.id,
        )
        self.assertTrue(deleted)

        # Verify cascade deletion of messages
        remaining = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == "window_test_session")
            .count()
        )
        self.assertEqual(remaining, 0)


class TestSessionAPI(unittest.TestCase):
    """Integration tests for session history endpoints and user isolation."""

    @classmethod
    def setUpClass(cls):
        init_test_db()

        # Register User A
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "user_a@example.com",
                "username": "user_a",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token_a = client.post(
            "/api/v1/auth/login",
            json={"username": "user_a", "password": "Password123!"},
        ).json()["access_token"]

        # Register User B
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "user_b@example.com",
                "username": "user_b",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token_b = client.post(
            "/api/v1/auth/login",
            json={"username": "user_b", "password": "Password123!"},
        ).json()["access_token"]

    def test_session_endpoints_and_isolation(self):
        db = TestSessionLocal()
        user_a = db.query(User).filter(User.username == "user_a").first()
        mem = ConversationMemoryService()

        # Seed a session for User A
        mem.get_or_create_session(
            db, session_id="user_a_session", user_id=user_a.id, initial_prompt="Hello from A"
        )
        mem.save_message(db, session_id="user_a_session", role="user", content="Question A")
        mem.save_message(db, session_id="user_a_session", role="assistant", content="Answer A")
        db.close()

        # User A can list their sessions
        res_a = client.get(
            "/api/v1/agent/sessions",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(res_a.status_code, 200)
        sessions_a = res_a.json()
        self.assertEqual(len(sessions_a), 1)
        self.assertEqual(sessions_a[0]["id"], "user_a_session")
        self.assertEqual(sessions_a[0]["message_count"], 2)

        # User B lists sessions -> should be empty (User isolation)
        res_b = client.get(
            "/api/v1/agent/sessions",
            headers={"Authorization": f"Bearer {self.token_b}"},
        )
        self.assertEqual(res_b.status_code, 200)
        self.assertEqual(len(res_b.json()), 0)

        # User A gets history
        history_res = client.get(
            "/api/v1/agent/sessions/user_a_session",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(history_res.status_code, 200)
        self.assertEqual(len(history_res.json()["messages"]), 2)

        # User B tries to get User A's session -> 404
        forbidden_get = client.get(
            "/api/v1/agent/sessions/user_a_session",
            headers={"Authorization": f"Bearer {self.token_b}"},
        )
        self.assertEqual(forbidden_get.status_code, 404)

        # User A deletes their session
        del_res = client.delete(
            "/api/v1/agent/sessions/user_a_session",
            headers={"Authorization": f"Bearer {self.token_a}"},
        )
        self.assertEqual(del_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
