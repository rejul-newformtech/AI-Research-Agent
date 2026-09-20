"""Pydantic schemas for Google ADK Agent interaction, tool traces, and chained RAG."""

from typing import Any

from pydantic import BaseModel, Field

from app.schema.structured_output import ResearchSynthesisModel


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
