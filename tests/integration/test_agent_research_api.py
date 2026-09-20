"""Integration tests for the /api/v1/agent/research endpoint and structured response."""

import unittest
from unittest.mock import patch

from app.schema.structured_output import CitationModel, ResearchSynthesisModel
from app.service.advanced_retrieval import ChainedRAGResult
from tests.conftest import client, init_test_db


class TestAgentResearchAPI(unittest.TestCase):
    """Integration tests for the 2-call chained research endpoint and conversational recording."""

    @classmethod
    def setUpClass(cls):
        init_test_db()
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "adv_research_user@example.com",
                "username": "adv_research_user",
                "password": "Password123!",
                "role": "researcher",
            },
        )
        cls.token = client.post(
            "/api/v1/auth/login",
            json={"username": "adv_research_user", "password": "Password123!"},
        ).json()["access_token"]

    @patch("app.service.advanced_retrieval.ChainedRAGPipeline.run")
    def test_chained_research_endpoint(self, mock_pipeline_run):
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
            structured_synthesis=ResearchSynthesisModel(
                summary="Moore's Law predicts exponential growth in transistor density.",
                detailed_findings="Transistor count doubles roughly every two years.",
                citations=[
                    CitationModel(
                        source="electronics.pdf",
                        page_number=4,
                        quote_or_fact="Transistor density doubles every 2 years.",
                    )
                ],
                key_takeaways=["Exponential compute growth", "Fabrication cost economics"],
                confidence_score=0.95,
            ),
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
                "expertise_level": "expert",
                "target_tone": "academic",
            },
        )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_llm_calls"], 2)
        self.assertIn("electronics.pdf", data["answer"])
        self.assertEqual(len(data["retrieved_chunks"]), 1)
        self.assertIsNotNone(data["session_id"])
        self.assertIsNotNone(data["structured_synthesis"])
        self.assertEqual(data["structured_synthesis"]["confidence_score"], 0.95)
        self.assertEqual(len(data["structured_synthesis"]["citations"]), 1)

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
