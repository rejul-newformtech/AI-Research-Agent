"""Integration tests for session history endpoints and multi-user isolation."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.models.user import User, UserRole
from app.service.memory import ConversationMemoryService


def test_session_endpoints_and_isolation(client: TestClient, db: Session):
    # Ensure fresh test users
    db.query(User).filter(
        User.username.in_(["session_user_alpha_int", "session_user_beta_int"])
    ).delete(synchronize_session=False)
    db.commit()

    user_a = User(
        email="session_user_alpha_int@example.com",
        username="session_user_alpha_int",
        hashed_password="hashed_fixture_password",
        role=UserRole.RESEARCHER,
        is_active=True,
    )
    user_b = User(
        email="session_user_beta_int@example.com",
        username="session_user_beta_int",
        hashed_password="hashed_fixture_password",
        role=UserRole.RESEARCHER,
        is_active=True,
    )
    db.add(user_a)
    db.add(user_b)
    db.commit()
    db.refresh(user_a)
    db.refresh(user_b)

    token_a = create_access_token(
        {"sub": user_a.username, "user_id": user_a.id, "role": user_a.role}
    )
    token_b = create_access_token(
        {"sub": user_b.username, "user_id": user_b.id, "role": user_b.role}
    )
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    mem = ConversationMemoryService()

    # Seed a session for User A
    mem.get_or_create_session(
        db,
        session_id="user_a_session_int",
        user_id=user_a.id,
        initial_prompt="Hello from A",
    )
    mem.save_message(db, session_id="user_a_session_int", role="user", content="Question A")
    mem.save_message(db, session_id="user_a_session_int", role="assistant", content="Answer A")

    # User A can list their sessions
    res_a = client.get(
        "/api/v1/agent/sessions",
        headers=headers_a,
    )
    assert res_a.status_code == 200
    sessions_a = [s for s in res_a.json() if s["id"] == "user_a_session_int"]
    assert len(sessions_a) == 1
    assert sessions_a[0]["id"] == "user_a_session_int"
    assert sessions_a[0]["message_count"] == 2

    # User B lists sessions -> should not contain User A's session
    res_b = client.get(
        "/api/v1/agent/sessions",
        headers=headers_b,
    )
    assert res_b.status_code == 200
    sessions_b = [s for s in res_b.json() if s["id"] == "user_a_session_int"]
    assert len(sessions_b) == 0

    # User A gets history
    history_res = client.get(
        "/api/v1/agent/sessions/user_a_session_int",
        headers=headers_a,
    )
    assert history_res.status_code == 200
    assert len(history_res.json()["messages"]) == 2

    # User B tries to get User A's session -> 404
    forbidden_get = client.get(
        "/api/v1/agent/sessions/user_a_session_int",
        headers=headers_b,
    )
    assert forbidden_get.status_code == 404

    # User A deletes their session
    del_res = client.delete(
        "/api/v1/agent/sessions/user_a_session_int",
        headers=headers_a,
    )
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_async_session_listing(async_researcher_client: AsyncClient):
    """Test session listing asynchronously with pytest-asyncio fixture."""
    res = await async_researcher_client.get("/api/v1/agent/sessions")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
