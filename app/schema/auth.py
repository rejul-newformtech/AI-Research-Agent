"""Pydantic schemas for authentication, user registration, and token management."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserRole


class UserRegisterRequest(BaseModel):
    """Payload for registering a new user."""

    email: EmailStr = Field(..., description="Valid email address")
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=6, description="Password with minimum 6 characters")
    role: UserRole = Field(
        default=UserRole.RESEARCHER,
        description="Assigned authorization role: 'admin', 'researcher', or 'user'",
    )


class UserLoginRequest(BaseModel):
    """Payload for JSON-based login."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Account password")


class TokenResponse(BaseModel):
    """OAuth2 compatible token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    username: str


class TokenPayload(BaseModel):
    """Decoded JWT payload data."""

    sub: str | None = None
    user_id: int | None = None
    role: str | None = None


class UserResponse(BaseModel):
    """Public user profile response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    role: str
    is_active: bool
    created_at: datetime
