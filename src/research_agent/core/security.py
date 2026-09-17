"""Security utilities: password hashing with bcrypt and JWT handling with PyJWT."""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from research_agent.core.config import settings


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token containing the provided payload."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update(
        {
            "exp": expire,
            "iat": now,
        }
    )
    encoded_jwt = jwt.encode(
        payload=to_encode,
        key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a signed JWT access token.
    Raises jwt.PyJWTError on expiration or signature errors.
    """
    return jwt.decode(
        jwt=token,
        key=settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
