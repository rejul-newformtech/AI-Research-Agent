import io
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Literal

import pypdf
from google import genai
from pydantic import BaseModel, Field

from research_agent.core.config import settings
from research_agent.service.chunking import (
    DocumentChunk,
    FixedChunker,
    SemanticChunker,
)


class DocumentMetadata(BaseModel):
    """Metadata describing the ingested document."""

    source: str
    title: str | None = None
    author: str | None = None
    total_pages: int = 1
    file_type: str = "pdf"
    chunking_strategy: str = "fixed"
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Output summary of document ingestion supporting fixed, semantic, or both chunkings."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    total_pages: int
    total_chunks: int
    chunking_strategy: str = "fixed"
    metadata: DocumentMetadata
    chunks: list[DocumentChunk]
    fixed_chunks: list[DocumentChunk] = Field(default_factory=list)
    semantic_chunks: list[DocumentChunk] = Field(default_factory=list)


class IngestionService:
    """Service to handle document loading (PDF/text), chunking (fixed and semantic), and embedding."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        document_storage_dir: str | Path | None = None,
        pdf_storage_dir: str | Path | None = None,
    ):
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap
        )
        self.embedding_model = embedding_model or settings.embedding_model
        self.api_key = api_key or settings.gemini_api_key
        storage_path = document_storage_dir or pdf_storage_dir or settings.document_storage_dir
        self.document_storage_dir = Path(storage_path)
        self.document_storage_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_storage_dir = self.document_storage_dir

        self._client: genai.Client | None = None

        # Fixed Chunker
        self.fixed_chunker = FixedChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Semantic Chunker
        self.semantic_chunker = SemanticChunker(
            embed_fn=self._safe_embed_texts,
            fallback_chunk_size=self.chunk_size,
            fallback_chunk_overlap=self.chunk_overlap,
        )

    def _safe_embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            return []
        try:
            return self.embed_texts(texts)
        except Exception:
            return []

    @property
    def client(self) -> genai.Client:
        """Lazily initialize and return the Google GenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Google GenAI API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                    "environment variable, or pass api_key to IngestionService."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def extract_pdf(
        self, file_input: str | Path | bytes | BinaryIO
    ) -> tuple[list[tuple[int, str]], int, dict[str, Any]]:
        """Extract text per page and metadata from a PDF file."""
        stream: BinaryIO
        should_close = False

        if isinstance(file_input, str | Path):
            stream = open(file_input, "rb")
            should_close = True
        elif isinstance(file_input, bytes):
            stream = io.BytesIO(file_input)
        else:
            stream = file_input

        try:
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)
            extracted_pages: list[tuple[int, str]] = []

            pdf_info: dict[str, Any] = {}
            if reader.metadata:
                for key in ["/Title", "/Author", "/Subject", "/Creator", "/Producer"]:
                    val = reader.metadata.get(key)
                    if val:
                        clean_key = key.lstrip("/").lower()
                        pdf_info[clean_key] = str(val)

            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned_text = page_text.replace("\x00", "").strip()
                if cleaned_text:
                    extracted_pages.append((page_idx, cleaned_text))

            return extracted_pages, total_pages, pdf_info
        finally:
            if should_close:
                stream.close()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate vector embeddings for a list of strings using Google GenAI."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
            )
            if response.embeddings:
                for item in response.embeddings:
                    if item.values:
                        all_embeddings.append(list(item.values))
                    else:
                        all_embeddings.append([])
            else:
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate vector embedding for a search query string."""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

    def embed_chunks(
        self, chunks: list[DocumentChunk], batch_size: int = 32
    ) -> list[DocumentChunk]:
        """Generate and attach vector embeddings to DocumentChunks in-place."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk.embedding = embedding

        return chunks

    def chunk_pages(
        self,
        pages: list[tuple[int, str]],
        strategy: Literal["fixed", "semantic", "both"] = "both",
        base_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[DocumentChunk], list[DocumentChunk], list[DocumentChunk]]:
        """Chunk pages using fixed, semantic, or both strategies.

        Returns:
            Tuple of (all_chunks, fixed_chunks, semantic_chunks).
        """
        fixed_chunks: list[DocumentChunk] = []
        semantic_chunks: list[DocumentChunk] = []

        if strategy in ("fixed", "both"):
            fixed_chunks = self.fixed_chunker.chunk_pages(pages, base_metadata=base_metadata)

        if strategy in ("semantic", "both"):
            semantic_chunks = self.semantic_chunker.chunk_pages(pages, base_metadata=base_metadata)

        all_chunks: list[DocumentChunk] = []
        if strategy == "fixed":
            all_chunks = fixed_chunks
        elif strategy == "semantic":
            all_chunks = semantic_chunks
        else:  # both
            all_chunks = fixed_chunks + semantic_chunks

        return all_chunks, fixed_chunks, semantic_chunks

    def save_pdf_to_storage(self, file_bytes: bytes, filename: str) -> Path:
        """Persist uploaded PDF file into the pdfs folder."""
        safe_filename = Path(filename).name
        target_path = self.pdf_storage_dir / safe_filename
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return target_path

    def ingest_pdf(
        self,
        file_input: str | Path | bytes | BinaryIO,
        filename: str | None = None,
        strategy: Literal["fixed", "semantic", "both"] = "both",
        generate_embeddings: bool = False,
        save_to_storage: bool = True,
    ) -> IngestionResult:
        """Extract, chunk (fixed, semantic, or both), and optionally embed a PDF document."""
        source_name = filename
        if not source_name and isinstance(file_input, str | Path):
            source_name = Path(file_input).name
        source_name = source_name or "uploaded_document.pdf"

        if save_to_storage and isinstance(file_input, bytes):
            self.save_pdf_to_storage(file_input, source_name)

        pages, total_pages, pdf_info = self.extract_pdf(file_input)

        metadata = DocumentMetadata(
            source=source_name,
            title=pdf_info.get("title"),
            author=pdf_info.get("author"),
            total_pages=total_pages,
            file_type="pdf",
            chunking_strategy=strategy,
            extra=pdf_info,
        )

        all_chunks, fixed_chunks, semantic_chunks = self.chunk_pages(
            pages,
            strategy=strategy,
            base_metadata={"source": source_name},
        )

        if generate_embeddings and all_chunks:
            self.embed_chunks(all_chunks)

        return IngestionResult(
            source=source_name,
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            chunking_strategy=strategy,
            metadata=metadata,
            chunks=all_chunks,
            fixed_chunks=fixed_chunks,
            semantic_chunks=semantic_chunks,
        )

    def ingest_text(
        self,
        text: str,
        source_name: str = "raw_text",
        strategy: Literal["fixed", "semantic", "both"] = "both",
        generate_embeddings: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Chunk (fixed, semantic, or both) and optionally embed raw text or markdown content."""
        pages = [(1, text)]
        doc_metadata = DocumentMetadata(
            source=source_name,
            total_pages=1,
            file_type="text",
            chunking_strategy=strategy,
            extra=metadata or {},
        )

        all_chunks, fixed_chunks, semantic_chunks = self.chunk_pages(
            pages,
            strategy=strategy,
            base_metadata={"source": source_name},
        )

        if generate_embeddings and all_chunks:
            self.embed_chunks(all_chunks)

        return IngestionResult(
            source=source_name,
            total_pages=1,
            total_chunks=len(all_chunks),
            chunking_strategy=strategy,
            metadata=doc_metadata,
            chunks=all_chunks,
            fixed_chunks=fixed_chunks,
            semantic_chunks=semantic_chunks,
        )
