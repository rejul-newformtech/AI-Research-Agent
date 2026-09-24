import logging
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("app.core.config")


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
        default="gemini-3-flash-preview",
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

    # Database Settings (SQLite)
    database_url: str = Field(description="SQLite database connection URL")

    @property
    def effective_database_url(self) -> str:
        """Return the database URL."""
        return self.database_url

    @property
    def effective_async_database_url(self) -> str:
        """Return the configured async database URL directly."""
        logger.info(f"Effective database URL: {self.database_url}")
        return self.database_url

    jwt_secret_key: str = Field(
        validation_alias=AliasChoices("JWT_SECRET_KEY", "jwt_secret_key"),
        description="Secret key used for signing JWT access tokens",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @property
    def pdf_storage_dir(self) -> Path:
        """Backward-compatible alias for document_storage_dir."""
        return self.document_storage_dir


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings singleton instance."""
    return Settings()


settings = get_settings()
