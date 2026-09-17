"""Unit and integration tests for Hybrid Search, BM25 sparse retrieval, and Reciprocal Rank Fusion."""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure test in-memory SQLite database before other imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from research_agent.service.chunking import DocumentChunk
from research_agent.service.retrieval import (
    BM25Index,
    HybridSearchService,
    ReciprocalRankFusion,
    tokenize_text,
)
from tests.test_db import init_test_db
from tests.test_db import test_client as client


class TestBM25Index(unittest.TestCase):
    """Unit tests for BM25 lexical tokenization, indexing, and keyword scoring."""

    def setUp(self):
        self.temp_index_path = Path("data/test_bm25_index.pkl")
        if self.temp_index_path.exists():
            self.temp_index_path.unlink()
        self.index = BM25Index(index_path=self.temp_index_path)

    def tearDown(self):
        if self.temp_index_path.exists():
            self.temp_index_path.unlink()

    def test_tokenization(self):
        text = "Hello, World! AI-Powered Research Agent: 2026."
        tokens = tokenize_text(text)
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("ai", tokens)
        self.assertIn("powered", tokens)
        self.assertIn("2026", tokens)

    def test_indexing_and_sparse_search(self):
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

        added = self.index.add_documents(chunks, persist=False)
        self.assertEqual(added, 3)
        self.assertFalse(self.index.is_empty())

        # Exact keyword search for physics
        results = self.index.search(query="quantum electrodynamics", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["metadata"]["topic"], "physics")
        self.assertIn("quantum", results[0]["text"].lower())
        self.assertGreater(results[0]["sparse_score"], 0.0)

        # Keyword search with metadata filter
        bio_results = self.index.search(query="light", top_k=5, where={"topic": "biology"})
        self.assertTrue(all(r["metadata"]["topic"] == "biology" for r in bio_results))

    def test_persistence_save_and_reload(self):
        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="Persistence test chunk for BM25 serialization.",
                metadata={"source": "test"},
            )
        ]
        self.index.add_documents(chunks, persist=True)
        self.assertTrue(self.temp_index_path.exists())

        # Load into new instance
        reloaded_index = BM25Index(index_path=self.temp_index_path)
        self.assertEqual(len(reloaded_index.doc_ids), 1)
        res = reloaded_index.search(query="serialization", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("serialization", res[0]["text"])


class TestReciprocalRankFusion(unittest.TestCase):
    """Unit tests for Reciprocal Rank Fusion (RRF) formula and re-ranking."""

    def test_rrf_fuses_and_prioritizes_overlapping_documents(self):
        # Doc A is #1 in dense and #1 in sparse
        # Doc B is #2 in dense only
        # Doc C is #2 in sparse only
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

        self.assertEqual(len(fused), 3)
        # Doc A should be rank #1 because it scored high in both modalities
        self.assertEqual(fused[0]["id"], "doc_a")
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertEqual(fused[0]["sparse_rank"], 1)

    def test_rrf_weighting(self):
        dense_results = [{"id": "doc_dense", "text": "Dense item", "similarity": 0.9}]
        sparse_results = [{"id": "doc_sparse", "text": "Sparse item", "sparse_score": 4.0}]

        # Heavily weight sparse search
        fused_sparse_heavy = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=2,
            dense_weight=0.1,
            sparse_weight=2.0,
        )
        self.assertEqual(fused_sparse_heavy[0]["id"], "doc_sparse")

        # Heavily weight dense search
        fused_dense_heavy = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=2,
            dense_weight=2.0,
            sparse_weight=0.1,
        )
        self.assertEqual(fused_dense_heavy[0]["id"], "doc_dense")


class TestHybridSearchIntegration(unittest.TestCase):
    """Integration tests for HybridSearchService and the /api/v1/ingest/search endpoint."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        # Register and log in a test researcher
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "hybrid_tester@example.com",
                "username": "hybrid_tester",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token = client.post(
            "/api/v1/auth/login",
            json={"username": "hybrid_tester", "password": "Password123!"},
        ).json()["access_token"]

    def test_service_sparse_and_fallback(self):
        bm25 = BM25Index(index_path="data/test_hybrid_bm25.pkl")
        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="Semiconductor physics governs modern microprocessors and silicon wafers.",
                metadata={"source": "physics_paper"},
            )
        ]
        bm25.add_documents(chunks, persist=False)

        service = HybridSearchService(bm25_index=bm25)
        # Search sparse
        sparse_matches = service.sparse_search(query="semiconductor silicon", top_k=2)
        self.assertEqual(len(sparse_matches), 1)
        self.assertIn("semiconductor", sparse_matches[0]["text"].lower())

    def test_search_api_modes(self):
        # 1. Test search with mode='sparse'
        res_sparse = client.post(
            "/api/v1/ingest/search",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "query": "research and testing",
                "top_k": 3,
                "mode": "sparse",
            },
        )
        self.assertEqual(res_sparse.status_code, 200)
        data_sparse = res_sparse.json()
        self.assertEqual(data_sparse["mode"], "sparse")
        self.assertIn("results", data_sparse)

        # 2. Test search with mode='hybrid'
        res_hybrid = client.post(
            "/api/v1/ingest/search",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "query": "research and testing",
                "top_k": 3,
                "mode": "hybrid",
                "dense_weight": 1.0,
                "sparse_weight": 1.0,
            },
        )
        self.assertEqual(res_hybrid.status_code, 200)
        data_hybrid = res_hybrid.json()
        self.assertEqual(data_hybrid["mode"], "hybrid")
        self.assertIn("results", data_hybrid)


if __name__ == "__main__":
    unittest.main()
