"""Integration tests for document ingestion search APIs and HybridSearchService."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.service.chunking import DocumentChunk
from app.service.retrieval import BM25Index, HybridSearchService


def test_service_sparse_and_fallback(tmp_path: Path):
    index_path = tmp_path / "test_bm25_integration.pkl"
    bm25 = BM25Index(index_path=index_path)
    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Semiconductor physics governs modern microprocessors and silicon wafers.",
            metadata={"source": "physics_paper"},
        )
    ]
    bm25.add_documents(chunks, persist=False)

    service = HybridSearchService(bm25_index=bm25)
    sparse_matches = service.sparse_search(query="semiconductor silicon", top_k=2)
    assert len(sparse_matches) == 1
    assert "semiconductor" in sparse_matches[0]["text"].lower()


def test_search_api_modes(client: TestClient, researcher_headers: dict[str, str]):
    # 1. Test search with mode='sparse'
    res_sparse = client.post(
        "/api/v1/ingest/search",
        headers=researcher_headers,
        json={
            "query": "research and testing",
            "top_k": 3,
            "mode": "sparse",
        },
    )
    assert res_sparse.status_code == 200
    data_sparse = res_sparse.json()
    assert data_sparse["mode"] == "sparse"
    assert "results" in data_sparse

    # 2. Test search with mode='hybrid'
    res_hybrid = client.post(
        "/api/v1/ingest/search",
        headers=researcher_headers,
        json={
            "query": "research and testing",
            "top_k": 3,
            "mode": "hybrid",
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
    )
    assert res_hybrid.status_code == 200
    data_hybrid = res_hybrid.json()
    assert data_hybrid["mode"] == "hybrid"
    assert "results" in data_hybrid


@pytest.mark.asyncio
async def test_async_search_api(async_researcher_client: AsyncClient):
    """Test search endpoint asynchronously with pytest-asyncio fixture."""
    res = await async_researcher_client.post(
        "/api/v1/ingest/search",
        json={
            "query": "quantum computing",
            "top_k": 2,
            "mode": "sparse",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "sparse"
    assert "results" in data
