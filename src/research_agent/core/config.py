from functools import lru_cache
from pathlib import Path

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
    pdf_storage_dir: Path = Path("data/pdfs")
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "research_documents"
    log_dir: Path = Path("logs")


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings singleton instance."""
    return Settings()


settings = get_settings()
