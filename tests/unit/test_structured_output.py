"""Unit and integration tests for Structured Output (typed Pydantic models)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from app.schema.structured_output import (
    CitationModel,
    HyDEPassageModel,
    MultiQueryExpansionModel,
    ResearchSynthesisModel,
)
from app.service.advanced_retrieval import ChainedRAGPipeline


class TestPydanticStructuredSchemas(unittest.TestCase):
    """Test schema validation, JSON dumping, and schema generation."""

    def test_citation_model_validation(self):
        cite = CitationModel(
            source="electronics_handbook.pdf",
            page_number=42,
            section="BJT Amplifiers",
            quote_or_fact="Small-signal voltage gain is proportional to transconductance gm.",
        )
        self.assertEqual(cite.source, "electronics_handbook.pdf")
        self.assertEqual(cite.page_number, 42)
        self.assertIn("voltage gain", cite.quote_or_fact)

        d = cite.model_dump()
        self.assertEqual(d["page_number"], 42)
        j = cite.model_dump_json()
        self.assertIn("electronics_handbook.pdf", j)

    def test_research_synthesis_model_validation(self):
        synthesis = ResearchSynthesisModel(
            summary="Transistors function as electronically controlled switches or amplifiers.",
            detailed_findings="A bipolar junction transistor (BJT) utilizes both electron and hole carriers.",
            citations=[
                CitationModel(
                    source="physics.pdf",
                    page_number=10,
                    quote_or_fact="BJT operates in active, saturation, and cutoff regions.",
                )
            ],
            key_takeaways=[
                "Active mode is used for linear amplification.",
                "Saturation and cutoff are used in digital logic.",
            ],
            confidence_score=0.95,
            missing_evidence=None,
        )
        self.assertEqual(len(synthesis.citations), 1)
        self.assertEqual(len(synthesis.key_takeaways), 2)
        self.assertEqual(synthesis.confidence_score, 0.95)

        json_str = synthesis.model_dump_json()
        restored = ResearchSynthesisModel.model_validate_json(json_str)
        self.assertEqual(restored.summary, synthesis.summary)
        self.assertEqual(restored.citations[0].page_number, 10)

    def test_multi_query_and_hyde_models(self):
        mq = MultiQueryExpansionModel(
            queries=[
                "transistor amplification mechanism",
                "bjt small signal equivalent circuit",
                "semiconductor current gain calculations",
            ]
        )
        self.assertEqual(len(mq.queries), 3)

        hyde = HyDEPassageModel(
            hypothetical_passage="A field-effect transistor controls conductance via an electric field.",
            domain="electronics",
        )
        self.assertEqual(hyde.domain, "electronics")


class TestStructuredSynthesisExecution(unittest.TestCase):
    """Test ChainedRAGPipeline synthesis with structured outputs and fallbacks."""

    @patch("google.genai.Client")
    def test_synthesis_parses_structured_json_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()

        json_payload = {
            "summary": "Superconductivity occurs below critical temperature Tc.",
            "detailed_findings": "Cooper pairs form via electron-phonon coupling in BCS theory.",
            "citations": [
                {
                    "source": "condensed_matter.pdf",
                    "page_number": 88,
                    "section": "BCS Theory",
                    "quote_or_fact": "Cooper pairs have zero electrical resistance below Tc.",
                }
            ],
            "key_takeaways": ["Zero electrical resistance", "Meissner effect expulsion of B field"],
            "confidence_score": 0.98,
            "missing_evidence": None,
        }
        mock_response.text = json.dumps(json_payload)
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        chunks = [
            {
                "id": "c1",
                "text": "Superconductivity was discovered in 1911 by Onnes.",
                "metadata": {"source": "condensed_matter.pdf", "page_number": 88},
            }
        ]

        pipeline = ChainedRAGPipeline()
        text_answer, structured = pipeline._synthesize_answer("What is superconductivity?", chunks)

        self.assertIsInstance(structured, ResearchSynthesisModel)
        self.assertEqual(structured.summary, json_payload["summary"])
        self.assertEqual(len(structured.citations), 1)
        self.assertEqual(structured.citations[0].page_number, 88)
        self.assertEqual(structured.confidence_score, 0.98)
        self.assertIn("References & Citations", text_answer)
        self.assertIn("condensed_matter.pdf", text_answer)

    @patch("google.genai.Client")
    def test_synthesis_plain_text_fallback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Plain text answer without JSON markers."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        chunks = [
            {
                "id": "c1",
                "text": "Sample passage content for fallback testing.",
                "metadata": {"source": "fallback.pdf", "page_number": 5},
            }
        ]

        pipeline = ChainedRAGPipeline()
        text_answer, structured = pipeline._synthesize_answer("Query?", chunks)

        self.assertIsInstance(structured, ResearchSynthesisModel)
        self.assertEqual(structured.detailed_findings, "Plain text answer without JSON markers.")
        self.assertEqual(len(structured.citations), 1)
        self.assertEqual(structured.citations[0].source, "fallback.pdf")

    def test_synthesis_empty_chunks(self):
        pipeline = ChainedRAGPipeline()
        text_answer, structured = pipeline._synthesize_answer("Nonexistent topic", [])

        self.assertEqual(structured.confidence_score, 0.0)
        self.assertIn("No relevant research documents found", structured.summary)
        self.assertEqual(len(structured.citations), 0)


if __name__ == "__main__":
    unittest.main()
