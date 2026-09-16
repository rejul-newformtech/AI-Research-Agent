from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
import pypdf.errors

from research_agent.db.chroma import ChromaService
from research_agent.service.ingestion import (
    ChunkingStrategy,
    IngestionResult,
    IngestionService,
)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class TextIngestRequest(BaseModel):
    """Payload for ingesting raw text or markdown."""

    text: str = Field(..., min_length=1, description="Raw text content to ingest")
    source_name: str = Field(default="raw_text", description="Name or identifier for the text source")
    strategy: Literal["fixed", "semantic", "both"] = Field(
        default="both",
        description="Chunking strategy: 'fixed', 'semantic', or 'both' (runs both chunkers)",
    )
    generate_embeddings: bool = Field(default=False, description="Whether to generate vector embeddings")
    store_in_chroma: bool = Field(default=False, description="Whether to store chunks into ChromaDB")
    collection_name: str = Field(default="research_documents", description="ChromaDB collection name")
    chunk_size: int = Field(default=1000, ge=50, le=8000, description="Target chunk size for fixed chunking")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap characters for fixed chunking")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata to attach")


class SearchRequest(BaseModel):
    """Query payload for vector search in ChromaDB."""

    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    strategy_filter: Literal["fixed", "semantic"] | None = Field(
        default=None,
        description="Filter results by chunking strategy: 'fixed', 'semantic', or None (both)",
    )
    collection_name: str = Field(default="research_documents", description="ChromaDB collection to query")


class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    collection_name: str
    results: list[SearchResultItem]


@router.post(
    "/text",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest raw text or markdown content",
)
async def ingest_text_endpoint(payload: TextIngestRequest) -> IngestionResult:
    """Split using fixed, semantic, or both chunkers, and optionally embed/store."""
    need_embeddings = payload.generate_embeddings or payload.store_in_chroma

    service = IngestionService(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    try:
        result = service.ingest_text(
            text=payload.text,
            source_name=payload.source_name,
            strategy=payload.strategy,
            generate_embeddings=need_embeddings,
            metadata=payload.metadata,
        )

        if payload.store_in_chroma and result.chunks:
            vs = ChromaService()
            vs.add_chunks(result.chunks, collection_name=payload.collection_name)

        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {e}",
        ) from e


@router.post(
    "/pdf",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest a PDF file",
)
async def ingest_pdf_endpoint(
    file: UploadFile = File(..., description="PDF document to upload and parse"),
    strategy: Literal["fixed", "semantic", "both"] = Query(
        default="both",
        description="Chunking strategy: 'fixed', 'semantic', or 'both'",
    ),
    generate_embeddings: bool = Query(
        default=False,
        description="Whether to generate Gemini vector embeddings for chunks",
    ),
    store_in_chroma: bool = Query(
        default=False,
        description="Whether to store embedded chunks into ChromaDB",
    ),
    collection_name: str = Query(
        default="research_documents",
        description="ChromaDB collection name",
    ),
    chunk_size: int = Query(
        default=1000,
        ge=50,
        le=8000,
        description="Target chunk size for fixed chunking",
    ),
    chunk_overlap: int = Query(
        default=200,
        ge=0,
        description="Target chunk overlap for fixed chunking",
    ),
) -> IngestionResult:
    """Upload PDF, extract text, chunk with fixed and/or semantic chunking, and optionally index in ChromaDB."""
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format for '{filename}'. Only PDF files are supported.",
        )

    need_embeddings = generate_embeddings or store_in_chroma

    service = IngestionService(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        result = service.ingest_pdf(
            file_input=file_bytes,
            filename=filename,
            strategy=strategy,
            generate_embeddings=need_embeddings,
            save_to_storage=True,
        )

        if store_in_chroma and result.chunks:
            vs = ChromaService()
            vs.add_chunks(result.chunks, collection_name=collection_name)

        return result
    except pypdf.errors.PdfReadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid PDF file: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF: {e}",
        ) from e


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search using ChromaDB",
)
async def search_endpoint(payload: SearchRequest) -> SearchResponse:
    """Embed the search query and perform vector similarity search against ChromaDB."""
    service = IngestionService()
    vs = ChromaService()

    try:
        query_embedding = service.embed_query(payload.query)

        where_filter = None
        if payload.strategy_filter:
            where_filter = {"strategy": payload.strategy_filter}

        matches = vs.search(
            query_embedding=query_embedding,
            top_k=payload.top_k,
            collection_name=payload.collection_name,
            where=where_filter,
        )
        return SearchResponse(
            query=payload.query,
            total_results=len(matches),
            collection_name=payload.collection_name,
            results=[SearchResultItem(**m) for m in matches],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        ) from e
