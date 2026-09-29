"""FastAPI routes for the unified ReAct Research Assistant Agent, conversational memory, and session management."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_active_user
from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_db
from app.models.chat import ChatSession
from app.models.user import User
from app.schema.agent import (
    AgentChatRequest,
    AgentChatResponse,
    ToolTrace,
)
from app.schema.chat import (
    ChatSessionDetailResponse,
    ChatSessionSummaryResponse,
)
from app.schema.structured_output import (
    UserProfileContext,
)
from app.service.a2a_client import A2AServerClient, get_a2a_client
from app.service.memory import ConversationMemoryService

logger = get_logger("app.api.agent")

router = APIRouter(prefix="/agent", tags=["Agent"])

# Conversational memory service with a default sliding window of 10 messages (5 turns)
memory_service = ConversationMemoryService(default_window_size=10)


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Research Assistant Agent (ReAct or Chained RAG)",
)
async def chat_with_agent(
    payload: AgentChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    a2a_client: A2AServerClient = Depends(get_a2a_client),
) -> AgentChatResponse:
    """Send a prompt to the unified Research Assistant Agent with persistent conversational memory.

    Dispatches remote network execution to the standalone A2A Intelligence microservice.
    Supports two execution modes:
    - 'react' (default): Iterative ReAct (Reasoning + Action + Observation) loop with tool execution.
    - 'rag' / 'research': 2-call Chained RAG pipeline (HyDE query expansion + Hybrid RRF + Grounded Synthesis).
    """
    session_id = payload.session_id or str(uuid.uuid4())
    user_id = payload.user_id or current_user.username

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

    # 1. Ensure chat session exists in SQLite
    await memory_service.get_or_create_session(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
        initial_prompt=payload.message,
    )

    # 2. Fetch recent history window for conversation continuity
    recent_messages = await memory_service.get_windowed_history(db=db, session_id=session_id)
    history_context = [
        {"role": msg.role, "content": msg.content}
        for msg in recent_messages
        if msg.role in ("user", "assistant")
    ]

    # 3. Persist incoming user turn to conversational memory
    await memory_service.save_message(
        db=db,
        session_id=session_id,
        role="user",
        content=payload.message,
    )

    # 4. Remote invocation of A2A Intelligence Microservice over HTTP
    if payload.mode in ("rag", "research"):
        remote_data = await a2a_client.run_query(
            query=payload.message,
            top_k=payload.top_k,
            use_hyde=payload.use_hyde,
            use_multiquery=payload.use_multiquery,
            user_profile=user_profile,
            history=history_context,
        )

        synthesized_answer = remote_data.get("synthesized_answer", "")
        retrieved_chunks = remote_data.get("retrieved_chunks", [])
        tool_traces_raw = [
            {
                "type": "chained_rag",
                "provenance": [
                    {
                        "id": c.get("id"),
                        "source": c.get("metadata", {}).get("source"),
                        "page_number": c.get("metadata", {}).get("page_number"),
                        "rrf_score": c.get("rrf_score"),
                    }
                    for c in retrieved_chunks
                ],
            }
        ]

        await memory_service.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=synthesized_answer,
            tool_traces=tool_traces_raw,
        )

        tool_traces = [
            ToolTrace(
                type="chained_rag",
                name="hybrid_retrieval_and_synthesis",
                args={
                    "top_k": payload.top_k,
                    "use_hyde": payload.use_hyde,
                    "use_multiquery": payload.use_multiquery,
                },
                response=f"Retrieved {len(retrieved_chunks)} passages.",
            )
        ]

        return AgentChatResponse(
            response=synthesized_answer,
            session_id=session_id,
            user_id=user_id,
            agent_name="research_agent",
            model=settings.gemini_model,
            mode=payload.mode,
            tool_traces=tool_traces,
            hypothetical_document=remote_data.get("hypothetical_document"),
            expanded_queries=remote_data.get("expanded_queries", []),
            retrieved_chunks=retrieved_chunks,
            structured_synthesis=remote_data.get("structured_synthesis"),
            total_llm_calls=remote_data.get("total_llm_calls", 2),
        )

    # Default: Remote ReAct agent loop execution
    remote_data = await a2a_client.run_chat(
        message=payload.message,
        session_id=session_id,
        max_iterations=payload.max_iterations,
        user_profile=user_profile,
        history=history_context,
    )

    final_answer = remote_data.get("response", "")
    tool_traces_raw = remote_data.get("tool_traces", [])

    await memory_service.save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=final_answer,
        tool_traces=tool_traces_raw,
    )

    tool_traces = [ToolTrace(**t) for t in tool_traces_raw]

    return AgentChatResponse(
        response=final_answer,
        session_id=session_id,
        user_id=user_id,
        agent_name="research_agent",
        model=settings.gemini_model,
        mode="react",
        tool_traces=tool_traces,
        trace=remote_data.get("trace"),
        structured_synthesis=remote_data.get("structured_synthesis"),
    )


@router.get(
    "/sessions",
    response_model=list[ChatSessionSummaryResponse],
    summary="List all chat sessions for the authenticated user",
)
async def list_user_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return all conversation sessions created by the currently authenticated user."""
    return await memory_service.list_user_sessions(db=db, user_id=current_user.id)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    summary="Get full conversation history for a specific session",
)
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    """Retrieve full chronological conversation message thread for a given session."""
    session_obj = await memory_service.get_session_details(
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
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete a conversation session and all its associated messages."""
    deleted = await memory_service.delete_session(
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
async def get_agent_info(
    a2a_client: A2AServerClient = Depends(get_a2a_client),
) -> dict[str, Any]:
    """Return metadata about the unified ReAct Research Assistant agent and its configured tools."""
    try:
        return await a2a_client.get_info()
    except Exception as e:
        logger.warning(f"Failed to fetch remote agent info: {e}")
        return {
            "name": "research_agent",
            "framework": "ReAct (Reasoning + Action + Observation)",
            "model": settings.gemini_model,
            "status": "remote_pending",
        }
