"""Unit and integration tests for Advanced Retrieval (HyDE, Multi-Query) and Chained RAG Pipeline."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure test in-memory SQLite database before other imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-for-hmac-sha256"

from research_agent.service.advanced_retrieval import (
    ChainedRAGPipeline,
    HyDEService,
    MultiQueryService,
)
from research_agent.service.retrieval import HybridSearchService
from tests.test_db import init_test_db
from tests.test_db import test_client as client


class TestHyDEService(unittest.TestCase):
    """Unit tests for HyDE generation and document-to-document retrieval."""

    @patch("google.genai.Client")
    def test_generate_hypothetical_document(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "Photosynthesis is the biochemical synthesis of organic compounds from light."
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = HyDEService()
        hypo_doc = service.generate_hypothetical_document("How do plants produce energy?")

        self.assertIn("Photosynthesis", hypo_doc)
        mock_client.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_hyde_search_flow(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hypothetical academic passage on quantum entanglement."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        mock_search_svc = MagicMock(spec=HybridSearchService)
        mock_search_svc.dense_search.return_value = [
            {"id": "chunk_1", "text": "Real textbook chunk on entanglement", "similarity": 0.92}
        ]

        hyde = HyDEService(hybrid_search_service=mock_search_svc)
        hypo_text, matches = hyde.search("What is quantum entanglement?", top_k=3)

        self.assertEqual(hypo_text, "Hypothetical academic passage on quantum entanglement.")
        self.assertEqual(len(matches), 1)
        mock_search_svc.dense_search.assert_called_once_with(
            query="Hypothetical academic passage on quantum entanglement.",
            top_k=3,
            where=None,
        )


class TestMultiQueryService(unittest.TestCase):
    """Unit tests for Multi-Query expansion and multi-list RRF fusion."""

    @patch("google.genai.Client")
    def test_generate_query_variations(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "bipolar junction transistor operation\n"
            "transistor current gain alpha beta\n"
            "semiconductor amplifier circuits"
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = MultiQueryService()
        queries = service.generate_query_variations("How do transistors work?", num_queries=3)

        self.assertEqual(len(queries), 3)
        self.assertIn("bipolar junction transistor operation", queries)

    def test_multi_list_rrf_fusion(self):
        list_1 = [
            {"id": "doc_a", "text": "A", "similarity": 0.9},
            {"id": "doc_b", "text": "B", "similarity": 0.8},
        ]
        list_2 = [
            {"id": "doc_c", "text": "C", "similarity": 0.95},
            {"id": "doc_a", "text": "A", "similarity": 0.85},
        ]

        fused = MultiQueryService._fuse_multiple_rankings([list_1, list_2], top_k=3)
        self.assertEqual(len(fused), 3)
        # doc_a appeared in both rankings, so it should rank first
        self.assertEqual(fused[0]["id"], "doc_a")


class TestChainedRAGPipeline(unittest.TestCase):
    """Unit tests for the 2-call LLM chained pipeline."""

    @patch("google.genai.Client")
    def test_chained_pipeline_executes_two_calls(self, mock_client_cls):
        mock_client = MagicMock()

        # Call 1: HyDE / Multi-Query generation
        # Call 2: Grounded answer synthesis
        resp_call1 = MagicMock()
        resp_call1.text = (
            "Hypothetical document on Maxwell's equations and electromagnetic induction."
        )

        resp_call2 = MagicMock()
        resp_call2.text = "According to [Source: physics.pdf, Page 12], Maxwell's equations unify electricity and magnetism."

        mock_client.models.generate_content.side_effect = [
            resp_call1,  # HyDE
            MagicMock(
                text="maxwell equations\nelectromagnetism laws\ngauss law faraday"
            ),  # MultiQuery
            resp_call2,  # Synthesis
        ]
        mock_client_cls.return_value = mock_client

        mock_search_svc = MagicMock(spec=HybridSearchService)
        mock_search_svc.hybrid_search.return_value = [
            {
                "id": "c1",
                "text": "Maxwell's equations govern electromagnetic fields.",
                "metadata": {"source": "physics.pdf", "page_number": 12},
                "rrf_score": 0.032,
            }
        ]
        mock_search_svc.dense_search.return_value = [
            {
                "id": "c1",
                "text": "Maxwell's equations govern electromagnetic fields.",
                "metadata": {"source": "physics.pdf", "page_number": 12},
                "similarity": 0.91,
            }
        ]

        pipeline = ChainedRAGPipeline(hybrid_search_service=mock_search_svc)
        result = pipeline.run(
            query="Explain Maxwell's equations",
            top_k=2,
            use_hyde=True,
            use_multiquery=True,
        )

        self.assertEqual(result.total_llm_calls, 2)
        self.assertIn("physics.pdf", result.synthesized_answer)
        self.assertEqual(len(result.retrieved_chunks), 1)
        self.assertIsNotNone(result.hypothetical_document)


class TestAdvancedRetrievalAPI(unittest.TestCase):
    """Integration tests for the /api/v1/agent/research and search endpoints."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        # Register and login test researcher
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "adv_tester@example.com",
                "username": "adv_tester",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token = client.post(
            "/api/v1/auth/login",
            json={"username": "adv_tester", "password": "Password123!"},
        ).json()["access_token"]

    @patch("research_agent.service.advanced_retrieval.ChainedRAGPipeline.run")
    def test_chained_research_endpoint(self, mock_pipeline_run):
        from research_agent.service.advanced_retrieval import ChainedRAGResult

        mock_pipeline_run.return_value = ChainedRAGResult(
            query="What is Moore's Law?",
            hypothetical_document="Moore's Law is an empirical observation.",
            expanded_queries=["semiconductor scaling", "transistor count density"],
            retrieved_chunks=[
                {
                    "id": "chunk_moore",
                    "text": "Moore's Law states that transistor density doubles every 2 years.",
                    "metadata": {"source": "electronics.pdf", "page_number": 4},
                    "rrf_score": 0.033,
                }
            ],
            synthesized_answer="Moore's Law predicts transistor doubling [Source: electronics.pdf, Page 4].",
            total_llm_calls=2,
        )

        res = client.post(
            "/api/v1/agent/research",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "query": "What is Moore's Law?",
                "top_k": 3,
                "use_hyde": True,
                "use_multiquery": True,
            },
        )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_llm_calls"], 2)
        self.assertIn("electronics.pdf", data["answer"])
        self.assertEqual(len(data["retrieved_chunks"]), 1)
        self.assertIsNotNone(data["session_id"])

        # Verify conversational memory recorded this turn
        session_id = data["session_id"]
        sess_res = client.get(
            f"/api/v1/agent/sessions/{session_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(sess_res.status_code, 200)
        sess_data = sess_res.json()
        self.assertEqual(len(sess_data["messages"]), 2)
        self.assertEqual(sess_data["messages"][0]["role"], "user")
        self.assertEqual(sess_data["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
