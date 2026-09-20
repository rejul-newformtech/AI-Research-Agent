"""Unit tests for ConversationMemoryService persistence and sliding window logic."""

from sqlalchemy.orm import Session

from app.models.chat import ChatMessage
from app.models.user import User
from app.service.memory import ConversationMemoryService


def test_session_creation_and_auto_titling(db: Session, test_researcher_user: User):
    memory = ConversationMemoryService(default_window_size=4)
    prompt = "Explain quantum electrodynamics and Feynman diagrams in detail."
    session = memory.get_or_create_session(
        db=db,
        session_id="session_qed_test",
        user_id=test_researcher_user.id,
        initial_prompt=prompt,
    )
    assert session.id == "session_qed_test"
    assert session.title.startswith("Explain quantum electrodynamics")
    assert session.title.endswith("...")


def test_message_persistence_and_sliding_window(db: Session, test_researcher_user: User):
    memory = ConversationMemoryService(default_window_size=4)
    session_id = "window_unit_test_session"
    memory.get_or_create_session(
        db=db,
        session_id=session_id,
        user_id=test_researcher_user.id,
        initial_prompt="Turn 1",
    )

    for i in range(1, 7):
        role = "user" if i % 2 != 0 else "assistant"
        memory.save_message(
            db=db,
            session_id=session_id,
            role=role,
            content=f"Message {i}",
        )

    windowed = memory.get_windowed_history(
        db=db,
        session_id=session_id,
        max_messages=4,
    )

    assert len(windowed) == 4
    assert [m.content for m in windowed] == ["Message 3", "Message 4", "Message 5", "Message 6"]


def test_list_and_delete_session(db: Session, test_researcher_user: User):
    memory = ConversationMemoryService()
    session_id = "to_delete_session"
    memory.get_or_create_session(
        db=db,
        session_id=session_id,
        user_id=test_researcher_user.id,
        initial_prompt="Delete me",
    )
    memory.save_message(db, session_id=session_id, role="user", content="Test")

    deleted = memory.delete_session(
        db=db,
        session_id=session_id,
        user_id=test_researcher_user.id,
    )
    assert deleted is True

    remaining = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
    assert remaining == 0
