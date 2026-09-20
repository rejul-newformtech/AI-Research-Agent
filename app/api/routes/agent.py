"""FastAPI routes for the unified ReAct Research Assistant Agent, conversational memory, and session management."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.research_assistant.agent import ReActAgentService
from app.api.dependencies.auth import get_current_active_user
from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_db
from app.models import ChatSession, User
from app.schema.agent import (
    AgentChatRequest,
    AgentChatResponse,
    ChainedResearchRequest,
    ChainedResearchResponse,
    ReActAgentRequest,
    ReActAgentResponse,
    ToolTrace,
)
from app.schema.chat import (
    ChatSessionDetailResponse,
    ChatSessionSummaryResponse,
)
from app.schema.structured_output import (
    UserProfileContext,
)
from app.service.memory import ConversationMemoryService

logger = get_logger("app.api.agent")

router = APIRouter(prefix="/agent", tags=["Agent"])

# Conversational memory service with a default sliding window of 10 messages (5 turns)
memory_service = ConversationMemoryService(default_window_size=10)


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Research Assistant Agent (ReAct Reasoning Loop)",
)
def chat_with_agent(
    payload: AgentChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> AgentChatResponse:
    """Send a prompt to the unified ReAct Research Assistant Agent with persistent conversational memory and step tracing.

    Executes the ReAct (Reasoning + Action + Observation) loop, captures thoughts and tool executions,
    and returns a clean response with full tool_traces.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    user_id = payload.user_id or current_user.username

    user_role_str = (
        current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    )
    user_profile = UserProfileContext(
        username=current_user.username,
        role=user_role_str,
        expertise_level="expert",
        target_tone="academic",
    )

    react_service = ReActAgentService(memory_service=memory_service)
    final_answer, trace, _ = react_service.run(
        query=payload.message,
        db=db,
        user_id=user_id,
        session_id=session_id,
        max_iterations=5,
        user_profile=user_profile,
    )

    # Convert ReAct steps into ToolTrace objects for UI visibility
    tool_traces: list[ToolTrace] = []
    for step in trace.steps:
        if step.thought:
            tool_traces.append(
                ToolTrace(
                    type="thought",
                    name=f"Step {step.step_number}",
                    response=step.thought,
                )
            )
        if step.action:
            tool_traces.append(
                ToolTrace(
                    type="function_call",
                    name=step.action,
                    args=step.action_input,
                    response=step.observation,
                )
            )

    return AgentChatResponse(
        response=final_answer,
        session_id=session_id,
        user_id=user_id,
        agent_name="research_agent",
        model=settings.gemini_model,
        tool_traces=tool_traces,
    )


@router.post(
    "/research",
    response_model=ChainedResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Deep research using 2-call chained HyDE, Multi-Query, and Grounded Synthesis",
)
def run_chained_research(
    payload: ChainedResearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ChainedResearchResponse:
    """Execute a 2-call chained RAG pipeline:

    Call 1: Query expansion (HyDE hypothetical answer + Multi-Query variations).
    Retrieval: Hybrid RRF search across all query representations.
    Call 2: Grounded answer synthesis citing document names and page numbers.
    The conversation turn is automatically recorded into persistent session memory.
    """
    from app.service.advanced_retrieval import ChainedRAGPipeline

    session_id = payload.session_id or str(uuid.uuid4())

    # Ensure chat session exists
    memory_service.get_or_create_session(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
        initial_prompt=payload.query,
    )

    # Construct user profile context dynamically
    user_profile = UserProfileContext(
        username=current_user.username,
        role=current_user.role,
        expertise_level=payload.expertise_level,
        target_tone=payload.target_tone,
        custom_instructions=payload.custom_instructions,
    )

    # Fetch recent history window for conversation continuity
    recent_messages = memory_service.get_windowed_history(db=db, session_id=session_id)
    history_context = [
        {"role": msg.role, "content": msg.content}
        for msg in recent_messages
        if msg.role in ("user", "assistant")
    ]

    pipeline = ChainedRAGPipeline()
    result = pipeline.run(
        query=payload.query,
        top_k=payload.top_k,
        use_hyde=payload.use_hyde,
        use_multiquery=payload.use_multiquery,
        user_profile=user_profile,
        history=history_context,
    )

    # Persist the conversation turn to conversational memory
    memory_service.save_message(
        db=db,
        session_id=session_id,
        role="user",
        content=payload.query,
    )
    memory_service.save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=result.synthesized_answer,
        tool_traces=[
            {
                "type": "chained_rag",
                "provenance": [
                    {
                        "id": c.get("id"),
                        "source": c.get("metadata", {}).get("source"),
                        "page_number": c.get("metadata", {}).get("page_number"),
                        "rrf_score": c.get("rrf_score"),
                    }
                    for c in result.retrieved_chunks
                ],
            }
        ],
    )

    return ChainedResearchResponse(
        session_id=session_id,
        query=payload.query,
        hypothetical_document=result.hypothetical_document,
        expanded_queries=result.expanded_queries,
        retrieved_chunks=result.retrieved_chunks,
        answer=result.synthesized_answer,
        structured_synthesis=result.structured_synthesis,
        total_llm_calls=result.total_llm_calls,
    )


@router.post(
    "/react",
    response_model=ReActAgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-step ReAct (Reasoning + Action + Observation) Agent Loop",
)
def run_react_agent_loop(
    payload: ReActAgentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ReActAgentResponse:
    """Execute the iterative ReAct (Reasoning + Action + Observation) loop with transparent step tracing."""
    from app.agents.research_assistant.agent import ReActAgentService

    user_role_str = (
        current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    )
    user_profile = UserProfileContext(
        username=current_user.username,
        role=user_role_str,
        expertise_level=payload.expertise_level,
        target_tone=payload.target_tone,
        custom_instructions=payload.custom_instructions,
    )

    react_service = ReActAgentService(memory_service=memory_service)
    final_answer, trace, structured = react_service.run(
        query=payload.query,
        db=db,
        user_id=current_user.username,
        session_id=payload.session_id,
        max_iterations=payload.max_iterations,
        user_profile=user_profile,
    )

    return ReActAgentResponse(
        answer=final_answer,
        session_id=payload.session_id or f"react_sess_{abs(hash(payload.query)) % 1000000}",
        trace=trace,
        structured_synthesis=structured,
    )


@router.get(
    "/sessions",
    response_model=list[ChatSessionSummaryResponse],
    summary="List all chat sessions for the authenticated user",
)
def list_user_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return all conversation sessions created by the currently authenticated user."""
    return memory_service.list_user_sessions(db=db, user_id=current_user.id)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    summary="Get full conversation history for a specific session",
)
def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ChatSession:
    """Retrieve full chronological conversation message thread for a given session."""
    session_obj = memory_service.get_session_details(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
    )
    if not session_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return session_obj


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a chat session and its history",
)
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a conversation session and all its associated messages."""
    deleted = memory_service.delete_session(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )

    return {"status": "deleted", "session_id": session_id}


@router.get(
    "/info",
    summary="Retrieve Agent Details",
)
def get_agent_info() -> dict[str, Any]:
    """Return metadata about the unified ReAct Research Assistant agent and its configured tools."""
    react_service = ReActAgentService()
    tool_names = list(react_service.tool_registry.keys())
    tools_manifest = [
        {
            "name": name,
            "description": (getattr(func, "__doc__", "") or "").split("\n\n")[0].strip(),
        }
        for name, func in react_service.tool_registry.items()
    ]
    return {
        "name": "research_agent",
        "framework": "ReAct (Reasoning + Action + Observation)",
        "model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "tools": tool_names,
        "tools_manifest": tools_manifest,
        "instruction": "ReAct framework with iterative Thought, Action, and Observation reasoning loop.",
    }
