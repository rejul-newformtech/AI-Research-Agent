"""Unit tests for HyDE (Hypothetical Document Embeddings) and Multi-Query services."""

import unittest
from unittest.mock import MagicMock, patch

from app.service.advanced_retrieval import (
    ChainedRAGPipeline,
    HyDEService,
    MultiQueryService,
)
from app.service.retrieval import HybridSearchService


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
        self.assertEqual(fused[0]["id"], "doc_a")


class TestChainedRAGPipelineUnit(unittest.TestCase):
    """Unit test verifying the 2-call LLM sequence in ChainedRAGPipeline."""

    @patch("google.genai.Client")
    def test_chained_pipeline_executes_two_calls(self, mock_client_cls):
        mock_client = MagicMock()

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


if __name__ == "__main__":
    unittest.main()
