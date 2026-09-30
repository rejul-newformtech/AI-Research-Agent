"""Unit tests for security utilities: bcrypt hashing and JWT encoding/decoding."""

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing():
    plain_password = "SuperSecretPassword123!"
    hashed = hash_password(plain_password)
    assert plain_password != hashed
    assert verify_password(plain_password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_token_flow():
    data = {"sub": "testuser", "user_id": 42, "role": "researcher"}
    token = create_access_token(data)
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
    assert payload["user_id"] == 42
    assert payload["role"] == "researcher"
