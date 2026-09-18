import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    BOTH = "both"


class DocumentChunk(BaseModel):
    """A single text chunk with provenance, strategy tag, and optional embedding."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: int
    page_number: int | None = None
    strategy: str = "fixed"  # "fixed" or "semantic"
    text: str
    char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.char_count and self.text:
            self.char_count = len(self.text)


def cosine_similarity(vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors using NumPy."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into a list of chunk strings."""
        pass

    def chunk_pages(
        self,
        pages: list[tuple[int, str]],
        base_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk a list of (page_number, text) tuples into DocumentChunk objects."""
        chunks: list[DocumentChunk] = []
        global_index = 0
        base_meta = base_metadata or {}
        strategy_name = getattr(self, "strategy_name", "custom")

        for page_num, page_text in pages:
            splits = self.split_text(page_text)
            for split_text in splits:
                cleaned = split_text.strip()
                if not cleaned:
                    continue

                chunk_meta = {
                    **base_meta,
                    "page_number": page_num,
                    "chunk_index": global_index,
                    "strategy": strategy_name,
                }

                chunk = DocumentChunk(
                    chunk_index=global_index,
                    page_number=page_num,
                    strategy=strategy_name,
                    text=cleaned,
                    char_count=len(cleaned),
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                global_index += 1

        return chunks


class FixedChunker(BaseChunker):
    """Fixed-window chunking using character or token length with overlap."""

    strategy_name = "fixed"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, int(chunk_size * 0.2))
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_text(self, text: str) -> list[str]:
        """Split text strictly by character length window with overlap."""
        return self.splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """Semantic chunking based on cosine similarity transitions between consecutive sentences.

    Algorithm:
    1. Splits raw text into sentences using punctuation boundaries.
    2. Embeds each sentence using the provided embedding function.
    3. Calculates cosine distance (1 - cosine_similarity) between adjacent sentences.
    4. Identifies semantic transition points where distance exceeds the percentile threshold.
    5. Groups sentences into cohesive semantic chunks.
    """

    strategy_name = "semantic"

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        breakpoint_percentile: float = 85.0,
        max_chunk_size: int = 1500,
        min_sentences_per_chunk: int = 1,
        fallback_chunk_size: int = 1000,
        fallback_chunk_overlap: int = 200,
    ):
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_size = max_chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.fallback_splitter = FixedChunker(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        """Split text into semantic chunks based on sentence similarity."""
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        if not self.embed_fn:
            # Fall back to fixed chunker if no embedding function is provided
            return self.fallback_splitter.split_text(text)

        try:
            embeddings = self.embed_fn(sentences)
        except Exception:
            return self.fallback_splitter.split_text(text)

        if not embeddings or len(embeddings) < 2:
            return self.fallback_splitter.split_text(text)

        # Calculate cosine distances between adjacent sentences
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            dist = 1.0 - cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(dist)

        # Determine breakpoint distance threshold using percentile
        threshold = float(np.percentile(distances, self.breakpoint_percentile))

        # Group sentences into semantic chunks
        chunks: list[str] = []
        current_chunk_sentences: list[str] = [sentences[0]]

        for i, dist in enumerate(distances):
            next_sentence = sentences[i + 1]
            current_len = sum(len(s) for s in current_chunk_sentences)

            should_split = (
                dist > threshold and len(current_chunk_sentences) >= self.min_sentences_per_chunk
            ) or (current_len + len(next_sentence) > self.max_chunk_size)

            if should_split:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [next_sentence]
            else:
                current_chunk_sentences.append(next_sentence)

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks
