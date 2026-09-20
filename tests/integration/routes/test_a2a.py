"""Integration tests for Agent-to-Agent (A2A) protocol and Agent Card discovery."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.schema.agent import ReActExecutionTrace, ReActStep
from app.schema.structured_output import CitationModel, ResearchSynthesisModel
from app.service.advanced_retrieval import ChainedRAGResult


def test_well_known_agent_card(client: TestClient):
    """Verify GET /.well-known/agent.json returns standard Agent Card manifest."""
    res = client.get("/.well-known/agent.json")
    assert res.status_code == 200
    data = res.json()

    assert data["name"] == "AI Research Assistant Agent"
    assert "FastMCP" in data["protocols"]
    assert "A2A" in data["protocols"]
    assert "react_reasoning" in data["skills"]
    assert len(data["tools"]) >= 3
    tool_names = [t["name"] for t in data["tools"]]
    assert "search_research_documents" in tool_names
    assert "advanced_research_query" in tool_names
    assert "ingest_stored_document" in tool_names
    assert data["authentication"]["type"] == "bearer"


def test_agent_card_namespaced(client: TestClient):
    """Verify GET /api/v1/agent/card alias returns the same Agent Card."""
    res = client.get("/api/v1/agent/card")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "AI Research Assistant Agent"
    assert data["version"] == "0.1.0"


@patch("app.service.react_agent.ReActAgentService.run")
def test_a2a_query_react_mode(
    mock_react_run, client: TestClient, researcher_headers: dict[str, str]
):
    """Verify POST /api/v1/agent/a2a/query executing under ReAct multi-step mode."""
    mock_react_run.return_value = (
        "Silicon photonics integrates optics onto silicon substrates [Source: photonics.pdf, Page 3].",
        ReActExecutionTrace(
            steps=[
                ReActStep(
                    step_number=1,
                    thought="Need to search literature.",
                    action="search_research_documents",
                    action_input={"query": "silicon photonics"},
                    observation="Found photonics.pdf page 3.",
                ),
                ReActStep(
                    step_number=2,
                    thought="Sufficient evidence.",
                    action=None,
                    action_input=None,
                    observation=None,
                ),
            ],
            total_iterations=2,
            is_terminated=True,
            termination_reason="final_answer_reached",
        ),
        ResearchSynthesisModel(
            summary="Silicon photonics enables optical interconnects.",
            detailed_findings="High bandwidth, low latency data communication.",
            citations=[
                CitationModel(
                    source="photonics.pdf",
                    page_number=3,
                    quote_or_fact="Silicon photonics integrates optics onto silicon.",
                )
            ],
            key_takeaways=["Optical interconnects", "CMOS compatibility"],
            confidence_score=0.92,
        ),
    )

    res = client.post(
        "/api/v1/agent/a2a/query",
        headers=researcher_headers,
        json={
            "sender_agent_id": "literature_review_bot_01",
            "query": "What are the latest advances in silicon photonics?",
            "mode": "react",
            "top_k": 3,
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["agent_id"] == "research_agent"
    assert data["mode_used"] == "react"
    assert "photonics.pdf" in data["response"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source"] == "photonics.pdf"
    assert len(data["tool_traces"]) == 2
    assert data["execution_time_ms"] > 0


@patch("app.service.advanced_retrieval.ChainedRAGPipeline.run")
def test_a2a_query_chained_rag_mode(
    mock_rag_run, client: TestClient, researcher_headers: dict[str, str]
):
    """Verify POST /api/v1/agent/a2a/query executing under 2-call Chained RAG mode."""
    mock_rag_run.return_value = ChainedRAGResult(
        query="Explain quantum error correction",
        hypothetical_document="Quantum error correction uses surface codes.",
        expanded_queries=["surface codes", "topological qubits"],
        retrieved_chunks=[],
        synthesized_answer="QEC preserves quantum state coherence [Source: qec.pdf, Page 1].",
        structured_synthesis=ResearchSynthesisModel(
            summary="Quantum error correction protects qubits.",
            detailed_findings="Surface codes provide fault tolerance.",
            citations=[
                CitationModel(
                    source="qec.pdf",
                    page_number=1,
                    quote_or_fact="Preserves quantum coherence.",
                )
            ],
            key_takeaways=["Fault tolerance"],
            confidence_score=0.9,
        ),
        total_llm_calls=2,
    )

    res = client.post(
        "/api/v1/agent/a2a/query",
        headers=researcher_headers,
        json={
            "sender_agent_id": "quantum_agent_02",
            "query": "Explain quantum error correction",
            "mode": "chained_rag",
            "top_k": 5,
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["mode_used"] == "chained_rag"
    assert "qec.pdf" in data["response"]
    assert len(data["citations"]) == 1


@patch("app.service.retrieval.HybridSearchService.hybrid_search")
def test_a2a_query_hybrid_search_mode(
    mock_hybrid, client: TestClient, researcher_headers: dict[str, str]
):
    """Verify POST /api/v1/agent/a2a/query executing under direct hybrid search mode."""
    mock_hybrid.return_value = [
        {
            "id": "c1",
            "text": "GaN semiconductors have wide bandgap properties.",
            "metadata": {"source": "gan_paper.pdf", "page_number": 2},
            "rrf_score": 0.045,
        }
    ]

    res = client.post(
        "/api/v1/agent/a2a/query",
        headers=researcher_headers,
        json={
            "sender_agent_id": "materials_bot",
            "query": "GaN wide bandgap",
            "mode": "hybrid_search",
            "top_k": 1,
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["mode_used"] == "hybrid_search"
    assert "Found 1 relevant passage(s)" in data["response"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source"] == "gan_paper.pdf"


@pytest.mark.asyncio
@patch("app.service.react_agent.ReActAgentService.run")
async def test_async_a2a_query(mock_react_run, async_researcher_client: AsyncClient):
    """Verify A2A query endpoint asynchronously with pytest-asyncio fixture."""
    mock_react_run.return_value = (
        "Async A2A delegated answer.",
        ReActExecutionTrace(
            steps=[],
            total_iterations=1,
            is_terminated=True,
            termination_reason="direct_answer",
        ),
        None,
    )

    res = await async_researcher_client.post(
        "/api/v1/agent/a2a/query",
        json={
            "sender_agent_id": "async_caller",
            "query": "What is graphene?",
            "mode": "react",
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["response"] == "Async A2A delegated answer."
    assert data["mode_used"] == "react"
