"""Integration tests for the /api/v1/agent/research endpoint and structured response."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.schema.structured_output import CitationModel, ResearchSynthesisModel
from app.service.advanced_retrieval import ChainedRAGResult


@patch("app.service.advanced_retrieval.ChainedRAGPipeline.run")
def test_chained_research_endpoint(
    mock_pipeline_run, client: TestClient, researcher_headers: dict[str, str]
):
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
        headers=researcher_headers,
        json={
            "query": "What is Moore's Law?",
            "top_k": 3,
            "use_hyde": True,
            "use_multiquery": True,
            "expertise_level": "expert",
            "target_tone": "academic",
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["total_llm_calls"] == 2
    assert "electronics.pdf" in data["answer"]
    assert len(data["retrieved_chunks"]) == 1
    assert data["session_id"] is not None
    assert data["structured_synthesis"] is not None
    assert data["structured_synthesis"]["confidence_score"] == 0.95
    assert len(data["structured_synthesis"]["citations"]) == 1

    # Verify conversational memory recorded this turn
    session_id = data["session_id"]
    sess_res = client.get(
        f"/api/v1/agent/sessions/{session_id}",
        headers=researcher_headers,
    )
    assert sess_res.status_code == 200
    sess_data = sess_res.json()
    assert len(sess_data["messages"]) == 2
    assert sess_data["messages"][0]["role"] == "user"
    assert sess_data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
@patch("app.service.advanced_retrieval.ChainedRAGPipeline.run")
async def test_async_agent_research_endpoint(
    mock_pipeline_run, async_researcher_client: AsyncClient
):
    """Test agent research endpoint asynchronously with pytest-asyncio fixture."""
    mock_pipeline_run.return_value = ChainedRAGResult(
        query="Async query test",
        hypothetical_document="Hypothetical doc",
        expanded_queries=["q1", "q2"],
        retrieved_chunks=[],
        synthesized_answer="Async synthesized answer",
        structured_synthesis=ResearchSynthesisModel(
            summary="Async summary",
            detailed_findings="Async findings",
            citations=[],
            key_takeaways=["Key takeaway"],
            confidence_score=0.9,
        ),
        total_llm_calls=2,
    )

    res = await async_researcher_client.post(
        "/api/v1/agent/research",
        json={
            "query": "Async query test",
            "top_k": 2,
            "use_hyde": False,
            "use_multiquery": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Async synthesized answer"
    assert data["structured_synthesis"]["confidence_score"] == 0.9
