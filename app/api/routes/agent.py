"""FastAPI routes for Google ADK Agent interaction, conversational memory, and session management."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.research_assistant.agent import root_agent
from app.api.dependencies.auth import get_current_active_user
from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import get_db
from app.models import ChatSession, User
from app.schema.agent import ReActAgentRequest, ReActAgentResponse
from app.schema.chat import (
    ChatSessionDetailResponse,
    ChatSessionSummaryResponse,
)
from app.schema.structured_output import (
    ResearchSynthesisModel,
    UserProfileContext,
)
from app.service.memory import ConversationMemoryService

logger = get_logger("app.api.agent")

router = APIRouter(prefix="/agent", tags=["Agent"])

# Global ADK runner for the research agent
adk_runner = InMemoryRunner(agent=root_agent)

# Conversational memory service with a default sliding window of 10 messages (5 turns)
memory_service = ConversationMemoryService(default_window_size=10)


class AgentChatRequest(BaseModel):
    """Request schema for interacting with the Google ADK Research Agent."""

    message: str = Field(..., min_length=1, description="User question or research prompt")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")
    user_id: str | None = Field(
        default=None, description="Optional user identifier (defaults to authenticated username)"
    )


class ToolTrace(BaseModel):
    """Execution trace of an agent tool invocation."""

    type: str
    name: str | None = None
    args: dict[str, Any] | None = None
    response: Any | None = None


class AgentChatResponse(BaseModel):
    """Response schema returned by the Google ADK Research Agent."""

    response: str
    session_id: str
    user_id: str
    agent_name: str
    model: str
    tool_traces: list[ToolTrace] = Field(default_factory=list)


class ChainedResearchRequest(BaseModel):
    """Payload for executing a 2-call chained research query."""

    query: str = Field(..., min_length=1, description="Research query or scientific question")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve")
    use_hyde: bool = Field(
        default=True, description="Enable HyDE (Hypothetical Document Embeddings)"
    )
    use_multiquery: bool = Field(default=True, description="Enable Multi-Query expansion")
    expertise_level: str = Field(
        default="expert",
        description="Target audience expertise ('expert', 'intermediate', 'novice')",
    )
    target_tone: str = Field(
        default="academic", description="Response tone ('academic', 'executive', 'didactic')"
    )
    custom_instructions: str | None = Field(
        default=None, description="Optional custom prompt directives"
    )


class ChainedResearchResponse(BaseModel):
    """Response returned by the 2-call chained research pipeline."""

    session_id: str
    query: str
    hypothetical_document: str | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    answer: str
    structured_synthesis: ResearchSynthesisModel | None = None
    total_llm_calls: int = 2


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Google ADK Research Agent",
)
async def chat_with_agent(
    payload: AgentChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> AgentChatResponse:
    """Send a prompt to the Google ADK Agent with persistent conversational memory and context windowing.

    Previous turns are persisted to SQLite and context-windowed to provide conversation continuity.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    user_id = payload.user_id or current_user.username

    # 1. Ensure persistent ChatSession exists in SQLite
    memory_service.get_or_create_session(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
        initial_prompt=payload.message,
    )

    # 2. Ensure ADK session exists in the runner
    try:
        session = await adk_runner.session_service.get_session(
            app_name=adk_runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = None

    if not session:
        session = await adk_runner.session_service.create_session(
            app_name=adk_runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        # Context Window Restoration:
        # If this session already existed from a previous server run, preload the windowed history
        windowed_history = memory_service.get_windowed_history(
            db=db,
            session_id=session_id,
            max_messages=10,
        )
        for hist_msg in windowed_history:
            author_name = root_agent.name if hist_msg.role == "assistant" else user_id
            role_name = "model" if hist_msg.role == "assistant" else "user"
            hist_content = types.Content(
                role=role_name,
                parts=[types.Part.from_text(text=hist_msg.content)],
            )
            event = Event(author=author_name, content=hist_content)
            await adk_runner.session_service.append_event(session=session, event=event)

    # 3. Save incoming user message to SQLite
    memory_service.save_message(
        db=db,
        session_id=session_id,
        role="user",
        content=payload.message,
    )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=payload.message)],
    )

    text_parts: list[str] = []
    tool_traces: list[ToolTrace] = []

    try:
        async for event in adk_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        text_parts.append(part.text)
                    if part.function_call:
                        tool_traces.append(
                            ToolTrace(
                                type="function_call",
                                name=part.function_call.name,
                                args=part.function_call.args,
                            )
                        )
                    if part.function_response:
                        tool_traces.append(
                            ToolTrace(
                                type="function_response",
                                name=part.function_response.name,
                                response=part.function_response.response,
                            )
                        )

        final_response = "".join(text_parts).strip()

        # 4. Persist generated assistant response and tool traces into SQLite
        serialized_traces = [t.model_dump() for t in tool_traces] if tool_traces else None
        memory_service.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=final_response,
            tool_traces=serialized_traces,
        )

        return AgentChatResponse(
            response=final_response,
            session_id=session_id,
            user_id=user_id,
            agent_name=root_agent.name,
            model=settings.gemini_model,
            tool_traces=tool_traces,
        )
    except Exception as exc:
        logger.exception(f"ADK Agent execution failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ADK Agent execution failed: {str(exc)}",
        ) from exc


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

    # Clean up runner memory
    try:
        await adk_runner.session_service.delete_session(
            app_name=adk_runner.app_name,
            user_id=current_user.username,
            session_id=session_id,
        )
    except Exception:
        pass

    return {"status": "deleted", "session_id": session_id}


@router.get(
    "/info",
    summary="Retrieve Agent Details",
)
async def get_agent_info() -> dict[str, Any]:
    """Return metadata about the current Google ADK agent and its configured tools."""
    tool_names = [getattr(t, "__name__", str(t)) for t in (root_agent.tools or [])]
    return {
        "name": root_agent.name,
        "model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "tools": tool_names,
        "instruction": root_agent.instruction,
    }
