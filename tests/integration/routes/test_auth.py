"""Integration tests for Authentication API endpoints and Role-Based Access Control (RBAC)."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.asyncio
async def test_register_user_success(client: TestClient, db: AsyncSession):
    # Ensure cleanup
    stmt = select(User).where(User.username == "dr_watson_api")
    user = await db.scalar(stmt)
    if user:
        await db.delete(user)
        await db.commit()

    payload = {
        "email": "researcher_api@example.com",
        "username": "dr_watson_api",
        "password": "Password123!",
        "role": "researcher",
    }
    res = client.post("/api/v1/auth/register", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["username"] == "dr_watson_api"
    assert data["email"] == "researcher_api@example.com"
    assert data["role"] == "researcher"
    assert data["is_active"] is True
    assert "id" in data


def test_register_duplicate_username_fails(client: TestClient):
    payload = {
        "email": "another_api@example.com",
        "username": "dr_watson_api",
        "password": "Password123!",
        "role": "researcher",
    }
    res = client.post("/api/v1/auth/register", json=payload)
    assert res.status_code == 400
    assert "username already exists" in res.json()["detail"]


def test_login_json_success(client: TestClient):
    payload = {
        "username": "dr_watson_api",
        "password": "Password123!",
    }
    res = client.post("/api/v1/auth/login", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["token_type"] == "bearer"
    assert data["username"] == "dr_watson_api"
    assert data["role"] == "researcher"
    assert "access_token" in data


def test_login_with_email_success(client: TestClient):
    payload = {
        "username": "researcher_api@example.com",
        "password": "Password123!",
    }
    res = client.post("/api/v1/auth/login", json=payload)
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_login_invalid_password_fails(client: TestClient):
    payload = {
        "username": "dr_watson_api",
        "password": "WrongPassword!",
    }
    res = client.post("/api/v1/auth/login", json=payload)
    assert res.status_code == 401


def test_get_profile_authenticated(client: TestClient, researcher_headers: dict[str, str]):
    res = client.get("/api/v1/auth/me", headers=researcher_headers)
    assert res.status_code == 200
    assert res.json()["username"] == "fixture_researcher"

    res_no_token = client.get("/api/v1/auth/me")
    assert res_no_token.status_code == 401


def test_rbac_admin_user_listing(
    client: TestClient, researcher_headers: dict[str, str], admin_headers: dict[str, str]
):
    # Researcher attempting to list users -> 403 Forbidden
    res_forbidden = client.get("/api/v1/auth/users", headers=researcher_headers)
    assert res_forbidden.status_code == 403

    # Admin listing users -> 200 OK
    res_admin = client.get("/api/v1/auth/users", headers=admin_headers)
    assert res_admin.status_code == 200
    users = res_admin.json()
    assert len(users) >= 2


def test_protected_endpoints_reject_unauthenticated(client: TestClient):
    res = client.post("/api/v1/agent/chat", json={"message": "Hello"})
    assert res.status_code == 401

    res = client.post("/api/v1/ingest/text", json={"text": "Research data"})
    assert res.status_code == 401

    res = client.post("/api/v1/ingest/search", json={"query": "Test search"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_async_auth_profile(async_researcher_client: AsyncClient):
    """Test authenticated profile endpoint using pytest-asyncio fixture async_researcher_client."""
    res = await async_researcher_client.get("/api/v1/auth/me")
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "fixture_researcher"
    assert data["role"] == "researcher"
