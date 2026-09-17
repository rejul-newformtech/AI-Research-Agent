"""FastAPI dependencies for OAuth2 authentication and role-based access control (RBAC)."""

from collections.abc import Callable
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from research_agent.api.schemas.auth import TokenPayload
from research_agent.core.security import decode_access_token
from research_agent.db.session import get_db
from research_agent.models import User

# Standard HTTP Bearer scheme for Swagger UI and API clients
http_bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Validate bearer JWT token and return the corresponding database user."""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        username: str | None = payload.get("sub")
        user_id: int | None = payload.get("user_id")
        if username is None and user_id is None:
            raise credentials_exception
        token_data = TokenPayload(sub=username, user_id=user_id, role=payload.get("role"))
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.PyJWTError as err:
        raise credentials_exception from err

    # Query user by username or id
    stmt = select(User).where((User.username == token_data.sub) | (User.id == token_data.user_id))
    user = db.scalar(stmt)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the authenticated user account is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    return current_user


def require_roles(*allowed_roles: str) -> Callable[..., Any]:
    """Dependency factory that restricts route access to specific user roles (RBAC).

    Usage:
        @router.post("/protected", dependencies=[Depends(require_roles("admin", "researcher"))])
    """

    def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: role '{current_user.role}' lacks required permissions ({list(allowed_roles)}).",
            )
        return current_user

    return role_checker
