"""Pydantic schemas for document ingestion, text parsing, and hybrid search."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    """Payload for ingesting raw text or markdown."""

    text: str = Field(..., min_length=1, description="Raw text content to ingest")
    source_name: str = Field(
        default="raw_text", description="Name or identifier for the text source"
    )
    strategy: Literal["fixed", "semantic", "both"] = Field(
        default="both",
        description="Chunking strategy: 'fixed', 'semantic', or 'both' (runs both chunkers)",
    )
    generate_embeddings: bool = Field(
        default=False, description="Whether to generate vector embeddings"
    )
    store_in_chroma: bool = Field(
        default=False, description="Whether to store chunks into ChromaDB"
    )
    collection_name: str = Field(
        default="research_documents", description="ChromaDB collection name"
    )
    chunk_size: int = Field(
        default=1000, ge=50, le=8000, description="Target chunk size for fixed chunking"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, description="Overlap characters for fixed chunking"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata to attach"
    )


class SearchRequest(BaseModel):
    """Query payload for vector, keyword, or hybrid search in ChromaDB."""

    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    mode: Literal["hybrid", "dense", "sparse"] = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid' (BM25 + Dense RRF), 'dense' (ChromaDB vector), or 'sparse' (BM25)",
    )
    dense_weight: float = Field(
        default=1.0, ge=0.0, description="Weight for dense vector ranking in RRF"
    )
    sparse_weight: float = Field(
        default=1.0, ge=0.0, description="Weight for sparse BM25 ranking in RRF"
    )
    strategy_filter: Literal["fixed", "semantic"] | None = Field(
        default=None,
        description="Filter results by chunking strategy: 'fixed', 'semantic', or None (both)",
    )
    use_hyde: bool = Field(
        default=False,
        description="Whether to use HyDE (Hypothetical Document Embeddings) before searching",
    )


class ChunkResponseSchema(BaseModel):
    """Individual retrieved document chunk schema."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None


class SearchResponse(BaseModel):
    """Search response returning ranked passages."""

    query: str
    mode: str
    total_results: int
    results: list[ChunkResponseSchema]
