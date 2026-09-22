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
        "/api/v1/agent/chat",
        headers=researcher_headers,
        json={
            "message": "What is Moore's Law?",
            "mode": "rag",
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
    assert "electronics.pdf" in data["response"]
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
        "/api/v1/agent/chat",
        json={
            "message": "Async query test",
            "mode": "rag",
            "top_k": 2,
            "use_hyde": False,
            "use_multiquery": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["response"] == "Async synthesized answer"
    assert data["structured_synthesis"]["confidence_score"] == 0.9


@patch("app.service.react_agent.ReActAgentService.run")
def test_react_endpoint_success(
    mock_react_run, client: TestClient, researcher_headers: dict[str, str]
):
    from app.schema.agent import ReActExecutionTrace, ReActStep

    mock_react_run.return_value = (
        "Transistors amplify signals [Source: electronics.pdf, Page 12].",
        ReActExecutionTrace(
            steps=[
                ReActStep(
                    step_number=1,
                    thought="Need to search for transistor amplification.",
                    action="search_research_documents",
                    action_input={"query": "transistor amplification"},
                    observation="Found electronics.pdf page 12.",
                ),
                ReActStep(
                    step_number=2,
                    thought="Sufficient evidence gathered.",
                    action=None,
                    action_input=None,
                    observation=None,
                ),
            ],
            total_iterations=2,
            is_terminated=True,
            termination_reason="final_answer_reached",
        ),
        None,
    )

    res = client.post(
        "/api/v1/agent/chat",
        headers=researcher_headers,
        json={
            "message": "How do transistors amplify signals?",
            "max_iterations": 3,
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert "electronics.pdf" in data["response"]
    assert data["trace"]["total_iterations"] == 2
    assert len(data["trace"]["steps"]) == 2
    assert data["trace"]["steps"][0]["action"] == "search_research_documents"


@pytest.mark.asyncio
@patch("app.service.react_agent.ReActAgentService.run")
async def test_async_react_endpoint(mock_react_run, async_researcher_client: AsyncClient):
    """Test ReAct endpoint asynchronously with pytest-asyncio fixture."""
    from app.schema.agent import ReActExecutionTrace, ReActStep

    mock_react_run.return_value = (
        "Async ReAct answer.",
        ReActExecutionTrace(
            steps=[
                ReActStep(
                    step_number=1,
                    thought="Direct answer.",
                    action=None,
                    action_input=None,
                    observation=None,
                )
            ],
            total_iterations=1,
            is_terminated=True,
            termination_reason="direct_answer",
        ),
        None,
    )

    res = await async_researcher_client.post(
        "/api/v1/agent/chat",
        json={
            "message": "What is quantum computing?",
            "max_iterations": 3,
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["response"] == "Async ReAct answer."
    assert data["trace"]["total_iterations"] == 1


@patch("app.service.react_agent.ReActAgentService.run")
def test_chat_endpoint_uses_react_service(
    mock_react_run, client: TestClient, researcher_headers: dict[str, str]
):
    """Verify that POST /api/v1/agent/chat runs the ReAct thinking engine and exposes thoughts in tool_traces."""
    from app.schema.agent import ReActExecutionTrace, ReActStep

    mock_react_run.return_value = (
        "ReAct reasoned answer [Source: paper.pdf, Page 1].",
        ReActExecutionTrace(
            steps=[
                ReActStep(
                    step_number=1,
                    thought="Need to check literature for photonics.",
                    action="search_research_documents",
                    action_input={"query": "photonics"},
                    observation="Found paper.pdf page 1.",
                ),
                ReActStep(
                    step_number=2,
                    thought="Sufficient evidence found to formulate answer.",
                    action=None,
                    action_input=None,
                    observation=None,
                ),
            ],
            total_iterations=2,
            is_terminated=True,
            termination_reason="final_answer_reached",
        ),
        None,
    )

    res = client.post(
        "/api/v1/agent/chat",
        headers=researcher_headers,
        json={
            "message": "Explain silicon photonics.",
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["agent_name"] == "research_agent"
    assert data["response"] == "ReAct reasoned answer [Source: paper.pdf, Page 1]."
    assert len(data["tool_traces"]) == 3
    # Step 1 thought
    assert data["tool_traces"][0]["type"] == "thought"
    assert data["tool_traces"][0]["response"] == "Need to check literature for photonics."
    # Step 1 action
    assert data["tool_traces"][1]["type"] == "function_call"
    assert data["tool_traces"][1]["name"] == "search_research_documents"
    # Step 2 thought
    assert data["tool_traces"][2]["type"] == "thought"
    assert data["tool_traces"][2]["response"] == "Sufficient evidence found to formulate answer."


def test_agent_info_endpoint(client: TestClient):
    """Verify GET /api/v1/agent/info reports the unified ReAct agent and all 5 research tools."""
    res = client.get("/api/v1/agent/info")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "research_agent"
    assert "ReAct" in data["framework"]
    assert len(data["tools"]) == 5
    assert "search_research_documents" in data["tools"]
    assert "advanced_research_query" in data["tools"]
    assert "list_stored_documents" in data["tools"]
    assert "ingest_stored_document" in data["tools"]
    assert "ingest_research_notes" in data["tools"]
