"""Agent-to-Agent (A2A) protocol endpoints and Agent Card declarations."""

import time
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_active_user
from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_db
from app.models import User
from app.schema.a2a import (
    A2AQueryRequest,
    A2AQueryResponse,
    AgentCard,
    AgentCardToolDeclaration,
)
from app.schema.structured_output import UserProfileContext
from app.service.react_agent import ReActAgentService

logger = get_logger("app.api.a2a")

# Router for root .well-known endpoints
well_known_router = APIRouter(tags=["A2A Protocols"])

# Router for API v1 agent endpoints
a2a_router = APIRouter(prefix="/agent", tags=["A2A Protocols"])


def build_agent_card(request: Request | None = None) -> AgentCard:
    """Construct the standardized machine-readable Agent Card."""
    base_url = str(request.base_url).rstrip("/") if request else "http://localhost:8000"

    react_service = ReActAgentService()
    tools_declarations = [
        AgentCardToolDeclaration(
            name=name,
            description=(getattr(func, "__doc__", "") or "").split("\n\n")[0].strip(),
            parameters=(
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or research question",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 5,
                            "description": "Passages to retrieve",
                        },
                    },
                    "required": ["query"],
                }
                if name != "ingest_stored_document"
                else {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Local filename to ingest",
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["fixed", "semantic", "both"],
                            "default": "fixed",
                        },
                    },
                    "required": ["filename"],
                }
            ),
        )
        for name, func in react_service.tool_registry.items()
    ]

    return AgentCard(
        name="AI Research Assistant Agent",
        version=settings.app_version,
        description=(
            "Autonomous academic and scientific research agent with Hybrid Search (BM25 + ChromaDB RRF), "
            "HyDE expansion, Multi-Query reformulation, and ReAct thinking loop."
        ),
        url=base_url,
        protocols=["REST", "FastMCP", "A2A"],
        skills=[
            "document_ingestion",
            "hybrid_retrieval",
            "hyde_expansion",
            "react_reasoning",
            "academic_synthesis",
        ],
        tools=tools_declarations,
        authentication={
            "type": "bearer",
            "token_url": f"{base_url}/api/v1/auth/login",
            "header": "Authorization",
            "scheme": "Bearer",
        },
    )


@well_known_router.get(
    "/.well-known/agent.json",
    response_model=AgentCard,
    summary="Agent Card Discovery (Well-Known Protocol)",
)
async def get_well_known_agent_card(request: Request) -> AgentCard:
    """Return the standardized Agent Card manifest for autonomous agent discovery."""
    return build_agent_card(request)


@a2a_router.post(
    "/a2a/query",
    response_model=A2AQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Inter-Agent Research Query Execution",
)
async def execute_a2a_query(
    payload: A2AQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> A2AQueryResponse:
    """Execute an incoming research query delegated by a peer agent.

    Supports 'react' multi-step reasoning, 'chained_rag' 2-call HyDE expansion, or direct 'hybrid_search'.
    """
    start_time = time.perf_counter()
    citations: list[dict[str, Any]] = []
    tool_traces: list[dict[str, Any]] = []

    user_role_str = (
        current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    )
    user_profile = UserProfileContext(
        username=current_user.username,
        role=user_role_str,
        expertise_level="expert",
        target_tone="academic",
    )

    if payload.mode == "react":
        react_service = ReActAgentService()
        session_id = f"a2a_{payload.sender_agent_id}_{int(time.time())}"
        final_answer, trace, structured = await react_service.run(
            query=payload.query,
            db=db,
            user_id=current_user.username,
            session_id=session_id,
            max_iterations=5,
            user_profile=user_profile,
        )
        answer = final_answer
        if structured and structured.citations:
            citations = [c.model_dump() for c in structured.citations]
        tool_traces = [step.model_dump() for step in trace.steps]

    elif payload.mode == "chained_rag":
        from app.service.advanced_retrieval import ChainedRAGPipeline

        pipeline = ChainedRAGPipeline()
        rag_res = pipeline.run(
            query=payload.query,
            top_k=payload.top_k,
            use_hyde=True,
            use_multiquery=True,
            user_profile=user_profile,
        )
        answer = rag_res.synthesized_answer
        if rag_res.structured_synthesis and rag_res.structured_synthesis.citations:
            citations = [c.model_dump() for c in rag_res.structured_synthesis.citations]
        tool_traces = [
            {
                "type": "chained_rag",
                "hypothetical_document": rag_res.hypothetical_document,
                "expanded_queries": rag_res.expanded_queries,
                "retrieved_count": len(rag_res.retrieved_chunks),
            }
        ]

    else:  # hybrid_search
        from app.service.retrieval import HybridSearchService

        search_service = HybridSearchService()
        matches = search_service.hybrid_search(query=payload.query, top_k=payload.top_k)
        answer = (
            f"Found {len(matches)} relevant passage(s) via hybrid search for '{payload.query}'."
        )
        for m in matches:
            meta = m.get("metadata", {})
            citations.append(
                {
                    "source": meta.get("source", "unknown"),
                    "page_number": meta.get("page_number"),
                    "rrf_score": m.get("rrf_score"),
                }
            )
        tool_traces = [{"type": "hybrid_search", "results_count": len(matches)}]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return A2AQueryResponse(
        agent_id="research_agent",
        query=payload.query,
        response=answer,
        citations=citations,
        tool_traces=tool_traces,
        mode_used=payload.mode,
        execution_time_ms=round(elapsed_ms, 2),
    )
