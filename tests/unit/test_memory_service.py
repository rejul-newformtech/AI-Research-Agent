"""Unit tests for ConversationMemoryService logic, persistence, and context windowing."""

import unittest

from app.models.chat import ChatMessage
from app.models.user import User
from app.service.memory import ConversationMemoryService
from tests.conftest import TestSessionLocal, init_test_db


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


if __name__ == "__main__":
    unittest.main()
