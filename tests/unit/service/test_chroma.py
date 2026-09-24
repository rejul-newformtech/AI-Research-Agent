"""Unit tests for ChromaService and AsyncChromaService."""

from pathlib import Path

import pytest

from app.db.chroma import AsyncChromaService, ChromaService
from app.service.chunking import DocumentChunk


def test_chroma_service_sync_flow(tmp_path: Path):
    """Test synchronous indexing, counting, and vector search."""
    service = ChromaService(persist_directory=tmp_path / "chroma_sync")
    col_name = "test_sync_collection"

    assert service.count(col_name) == 0

    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Artificial intelligence transforms research workflows and automation.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "paper1.pdf"},
        ),
        DocumentChunk(
            chunk_index=1,
            text="Quantum computing utilizes qubits to achieve superposition and entanglement.",
            embedding=[0.9, 0.8, 0.7, 0.6],
            metadata={"source": "paper2.pdf"},
        ),
    ]

    added = service.add_chunks(chunks, collection_name=col_name)
    assert added == 2
    assert service.count(col_name) == 2

    # Vector search with exact embedding of first chunk
    results = service.search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=1,
        collection_name=col_name,
    )
    assert len(results) == 1
    assert "artificial intelligence" in results[0]["text"].lower()
    assert results[0]["similarity"] > 0.99


@pytest.mark.asyncio
async def test_chroma_service_async_methods(tmp_path: Path):
    """Test asynchronous non-blocking methods on ChromaService."""
    service = ChromaService(persist_directory=tmp_path / "chroma_async")
    col_name = "test_async_collection"

    initial_count = await service.acount(col_name)
    assert initial_count == 0

    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Deep learning architectures rely on backpropagation and gradient descent.",
            embedding=[0.05, 0.15, 0.25, 0.35],
            metadata={"topic": "deep_learning"},
        ),
    ]

    added = await service.aadd_chunks(chunks, collection_name=col_name)
    assert added == 1
    assert (await service.acount(col_name)) == 1

    results = await service.asearch(
        query_embedding=[0.05, 0.15, 0.25, 0.35],
        top_k=1,
        collection_name=col_name,
    )
    assert len(results) == 1
    assert "backpropagation" in results[0]["text"]


@pytest.mark.asyncio
async def test_async_chroma_service_wrapper(tmp_path: Path):
    """Test AsyncChromaService wrapper class for pure async/await pipelines."""
    async_service = AsyncChromaService(persist_directory=tmp_path / "chroma_wrapper")
    col_name = "test_wrapper_collection"

    assert (await async_service.count(col_name)) == 0

    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Autonomous agents reason through cyclic thought-action-observation loops.",
            embedding=[0.5, 0.5, 0.5, 0.5],
            metadata={"topic": "agents"},
        ),
    ]

    added = await async_service.add_chunks(chunks, collection_name=col_name)
    assert added == 1
    assert (await async_service.count(col_name)) == 1

    results = await async_service.search(
        query_embedding=[0.5, 0.5, 0.5, 0.5],
        top_k=1,
        collection_name=col_name,
    )
    assert len(results) == 1
    assert "Autonomous agents" in results[0]["text"]
