"""Unit tests for BM25 lexical sparse retrieval and Reciprocal Rank Fusion (RRF)."""

from pathlib import Path

from app.service.chunking import DocumentChunk
from app.service.retrieval import (
    BM25Index,
    ReciprocalRankFusion,
    tokenize_text,
)


def test_tokenization():
    text = "Hello, World! AI-Powered Research Agent: 2026."
    tokens = tokenize_text(text)
    assert "hello" in tokens
    assert "world" in tokens
    assert "ai" in tokens
    assert "powered" in tokens
    assert "2026" in tokens


def test_indexing_and_sparse_search(tmp_path: Path):
    index_path = tmp_path / "test_bm25.pkl"
    index = BM25Index(index_path=index_path)

    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Photosynthesis is the biochemical process by which plants convert sunlight into chemical energy.",
            metadata={"topic": "biology"},
        ),
        DocumentChunk(
            chunk_index=1,
            text="Quantum electrodynamics describes how light and matter interact at relativistic subatomic scales.",
            metadata={"topic": "physics"},
        ),
        DocumentChunk(
            chunk_index=2,
            text="Chlorophyll pigments in chloroplasts absorb blue and red light for plant cellular respiration.",
            metadata={"topic": "biology"},
        ),
    ]

    added = index.add_documents(chunks, persist=False)
    assert added == 3
    assert index.is_empty() is False

    # Exact keyword search for physics
    results = index.search(query="quantum electrodynamics", top_k=2)
    assert len(results) > 0
    assert results[0]["metadata"]["topic"] == "physics"
    assert "quantum" in results[0]["text"].lower()
    assert results[0]["sparse_score"] > 0.0

    # Keyword search with metadata filter
    bio_results = index.search(query="light", top_k=5, where={"topic": "biology"})
    assert all(r["metadata"]["topic"] == "biology" for r in bio_results)


def test_persistence_save_and_reload(tmp_path: Path):
    index_path = tmp_path / "persistent_bm25.pkl"
    index = BM25Index(index_path=index_path)

    chunks = [
        DocumentChunk(
            chunk_index=0,
            text="Persistence test chunk for BM25 serialization.",
            metadata={"source": "test"},
        )
    ]
    index.add_documents(chunks, persist=True)
    assert index_path.exists()

    # Load into new instance
    reloaded_index = BM25Index(index_path=index_path)
    assert len(reloaded_index.doc_ids) == 1
    res = reloaded_index.search(query="serialization", top_k=1)
    assert len(res) == 1
    assert "serialization" in res[0]["text"]


def test_rrf_fuses_and_prioritizes_overlapping_documents():
    dense_results = [
        {"id": "doc_a", "text": "Text A", "similarity": 0.95},
        {"id": "doc_b", "text": "Text B", "similarity": 0.88},
    ]
    sparse_results = [
        {"id": "doc_a", "text": "Text A", "sparse_score": 3.4},
        {"id": "doc_c", "text": "Text C", "sparse_score": 2.8},
    ]

    fused = ReciprocalRankFusion.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=3,
        k=60,
    )

    assert len(fused) == 3
    assert fused[0]["id"] == "doc_a"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["sparse_rank"] == 1


def test_rrf_weighting():
    dense_results = [{"id": "doc_dense", "text": "Dense item", "similarity": 0.9}]
    sparse_results = [{"id": "doc_sparse", "text": "Sparse item", "sparse_score": 4.0}]

    fused = ReciprocalRankFusion.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        dense_weight=0.1,
        sparse_weight=2.0,
        top_k=2,
        k=60,
    )

    assert len(fused) == 2
    assert fused[0]["id"] == "doc_sparse"
