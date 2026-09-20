from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Settings
    app_name: str = "Research Agent API"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Google Gemini API Key (accepts GEMINI_API_KEY or GOOGLE_API_KEY)
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key"),
        description="Google Gemini API key for embeddings and generation",
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model", "MODEL_NAME"),
        description="Google Gemini LLM generation model",
    )

    # Embedding & Chunking Settings
    embedding_model: str = "gemini-embedding-001"
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_breakpoint_percentile: float = 85.0

    # Storage Paths
    document_storage_dir: Path = Field(
        default=Path("data/documents"),
        validation_alias=AliasChoices(
            "DOCUMENT_STORAGE_DIR", "PDF_STORAGE_DIR", "document_storage_dir", "pdf_storage_dir"
        ),
        description="Path to store uploaded research documents",
    )
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "research_documents"
    log_dir: Path = Path("logs")

    # Database & Authentication Settings (Hot-swappable via URL or individual fields)
    db_type: str = Field(
        default="sqlite",
        validation_alias=AliasChoices("DB_TYPE", "db_type"),
        description="Database type: 'sqlite', 'postgres', or 'mysql'",
    )
    db_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DB_USER", "db_user", "POSTGRES_USER"),
        description="Database user",
    )
    db_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DB_PASSWORD", "db_password", "POSTGRES_PASSWORD"),
        description="Database password",
    )
    db_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("DB_HOST", "db_host", "POSTGRES_HOST"),
        description="Database server host",
    )
    db_port: int = Field(
        default=5432,
        validation_alias=AliasChoices("DB_PORT", "db_port", "POSTGRES_PORT"),
        description="Database server port",
    )
    db_name: str = Field(
        default="research_agent",
        validation_alias=AliasChoices("DB_NAME", "db_name", "POSTGRES_DB"),
        description="Database name",
    )
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
        description="Explicit connection URL (takes precedence over individual DB fields if provided)",
    )

    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    @property
    def effective_database_url(self) -> str:
        """Construct the database URL from explicit URL or individual credentials."""
        if self.database_url:
            return self.database_url
        if self.db_type.lower() in ("sqlite", "sqlite3"):
            filename = self.db_name if self.db_name.endswith(".db") else f"{self.db_name}.db"
            return f"sqlite:///data/{filename}"
        user_part = (
            f"{self.db_user}:{self.db_password}@"
            if self.db_user and self.db_password
            else (f"{self.db_user}@" if self.db_user else "")
        )
        if self.db_type.lower() in ("postgres", "postgresql"):
            return f"postgresql+psycopg2://{user_part}{self.db_host}:{self.db_port}/{self.db_name}"
        if self.db_type.lower() == "mysql":
            return f"mysql+pymysql://{user_part}{self.db_host}:{self.db_port}/{self.db_name}"
        return f"sqlite:///data/{self.db_name}.db"

    @property
    def effective_async_database_url(self) -> str:
        """Construct an async-compatible database URL (aiosqlite / asyncpg)."""
        url = self.effective_database_url
        if url.startswith("sqlite+aiosqlite://"):
            return url
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        if url.startswith("postgresql+asyncpg://"):
            return url
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    jwt_secret_key: str = Field(
        default="research-agent-secret-key-please-change-in-production",
        validation_alias=AliasChoices("JWT_SECRET_KEY", "jwt_secret_key"),
        description="Secret key used for signing JWT access tokens",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @property
    def pdf_storage_dir(self) -> Path:
        """Backward-compatible alias for document_storage_dir."""
        return self.document_storage_dir

    def model_post_init(self, __context: Any) -> None:
        """Propagate configured settings for third-party SDKs that inspect os.environ."""
        import os

        if self.gemini_api_key and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings singleton instance."""
    return Settings()


settings = get_settings()
