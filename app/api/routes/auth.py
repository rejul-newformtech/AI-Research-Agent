"""Authentication and authorization API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import (
    get_current_active_user,
    require_roles,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.models import User, UserRole
from app.schema.auth import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

logger = get_logger("app.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication & Authorization"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register_user(
    payload: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Register a new user account with unique email and username."""
    # Check for existing email or username
    existing_user = await db.scalar(
        select(User).where((User.email == payload.email) | (User.username == payload.username))
    )
    if existing_user:
        if existing_user.email == payload.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email address already exists.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists.",
        )

    # Hash password and persist user
    new_user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
        role=payload.role.value if isinstance(payload.role, UserRole) else str(payload.role),
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    logger.info(f"New user registered: '{new_user.username}' with role '{new_user.role}'.")
    return new_user


async def _authenticate_and_create_token(
    username: str, password: str, db: AsyncSession
) -> TokenResponse:
    """Internal helper to authenticate credentials and issue a JWT token."""
    user = await db.scalar(
        select(User).where((User.username == username) | (User.email == username))
    )
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role,
        }
    )
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        role=user.role,
        username=user.username,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with JSON credentials",
)
async def login_json(
    payload: UserLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate with username/email and password via JSON payload."""
    return await _authenticate_and_create_token(payload.username, payload.password, db)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_my_profile(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Return the profile and role details of the currently authenticated user."""
    return current_user


@router.get(
    "/users",
    response_model=list[UserResponse],
    dependencies=[Depends(require_roles("admin"))],
    summary="List all users (Admin only)",
)
async def list_all_users(
    db: AsyncSession = Depends(get_db),
) -> list[User]:
    """Return a list of all registered users. Restricted to users with the 'admin' role."""
    result = await db.scalars(select(User).order_by(User.id))
    return list(result.all())
