"""Integration tests for document ingestion search APIs and HybridSearchService."""

import unittest

from app.service.chunking import DocumentChunk
from app.service.retrieval import BM25Index, HybridSearchService
from tests.conftest import client, init_test_db


class TestHybridSearchIntegration(unittest.TestCase):
    """Integration tests for HybridSearchService and the /api/v1/ingest/search endpoint."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "hybrid_integration@example.com",
                "username": "hybrid_integration_user",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token = client.post(
            "/api/v1/auth/login",
            json={"username": "hybrid_integration_user", "password": "Password123!"},
        ).json()["access_token"]

    def test_service_sparse_and_fallback(self):
        bm25 = BM25Index(index_path="data/test_hybrid_bm25_integration.pkl")
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
