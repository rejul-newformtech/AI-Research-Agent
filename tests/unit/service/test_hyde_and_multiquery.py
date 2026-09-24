"""Unit tests for HyDE and Multi-Query retrieval services."""

from unittest.mock import MagicMock, patch

from app.service.advanced_retrieval import (
    ChainedRAGPipeline,
    HyDEService,
    MultiQueryService,
)
from app.service.retrieval import HybridSearchService


@patch("google.genai.Client")
def test_generate_hypothetical_document(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "Photosynthesis is the biochemical synthesis of organic compounds from light."
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    service = HyDEService()
    hypo_doc = service.generate_hypothetical_document("How do plants produce energy?")

    assert "Photosynthesis" in hypo_doc
    mock_client.models.generate_content.assert_called_once()


@patch("google.genai.Client")
def test_hyde_search_flow(mock_client_cls):
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

    assert hypo_text == "Hypothetical academic passage on quantum entanglement."
    assert len(matches) == 1
    mock_search_svc.dense_search.assert_called_once_with(
        query="Hypothetical academic passage on quantum entanglement.",
        top_k=3,
        where=None,
    )


@patch("google.genai.Client")
def test_generate_query_variations(mock_client_cls):
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

    assert len(queries) == 3
    assert "bipolar junction transistor operation" in queries


def test_multi_list_rrf_fusion():
    list_1 = [
        {"id": "doc_a", "text": "A", "similarity": 0.9},
        {"id": "doc_b", "text": "B", "similarity": 0.8},
    ]
    list_2 = [
        {"id": "doc_c", "text": "C", "similarity": 0.95},
        {"id": "doc_a", "text": "A", "similarity": 0.85},
    ]

    fused = MultiQueryService._fuse_multiple_rankings([list_1, list_2], top_k=3)
    assert len(fused) == 3
    assert fused[0]["id"] == "doc_a"


@patch("google.genai.Client")
def test_chained_pipeline_executes_two_calls(mock_client_cls):
    mock_client = MagicMock()

    resp_call1 = MagicMock()
    resp_call1.text = "Hypothetical document on Maxwell's equations and electromagnetic induction."

    resp_call2 = MagicMock()
    resp_call2.text = "According to [Source: physics.pdf, Page 12], Maxwell's equations unify electricity and magnetism."

    mock_client.models.generate_content.side_effect = [
        resp_call1,
        MagicMock(text="maxwell equations\nelectromagnetism laws\ngauss law faraday"),
        resp_call2,
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

    assert result.total_llm_calls == 2
    assert "physics.pdf" in result.synthesized_answer
    assert len(result.retrieved_chunks) == 1
    assert result.hypothetical_document is not None


@patch("google.genai.Client")
def test_synthesis_parses_structured_json_response(mock_client_cls):
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
    import json

    from app.schema.structured_output import ResearchSynthesisModel

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

    assert isinstance(structured, ResearchSynthesisModel)
    assert structured.summary == json_payload["summary"]
    assert len(structured.citations) == 1
    assert structured.citations[0].page_number == 88
    assert structured.confidence_score == 0.98
    assert "References & Citations" in text_answer
    assert "condensed_matter.pdf" in text_answer


@patch("google.genai.Client")
def test_synthesis_plain_text_fallback(mock_client_cls):
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

    from app.schema.structured_output import ResearchSynthesisModel

    pipeline = ChainedRAGPipeline()
    _text_answer, structured = pipeline._synthesize_answer("Query?", chunks)

    assert isinstance(structured, ResearchSynthesisModel)
    assert structured.detailed_findings == "Plain text answer without JSON markers."
    assert len(structured.citations) == 1
    assert structured.citations[0].source == "fallback.pdf"


def test_synthesis_empty_chunks():
    pipeline = ChainedRAGPipeline()
    _text_answer, structured = pipeline._synthesize_answer("Nonexistent topic", [])

    assert structured.confidence_score == 0.0
    assert "No relevant research documents found" in structured.summary
    assert len(structured.citations) == 0
