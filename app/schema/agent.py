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


class ReActStep(BaseModel):
    """A single ReAct reasoning and tool invocation step."""

    step_number: int = Field(..., description="1-indexed step number")
    thought: str = Field(
        ..., description="Agent reasoning or internal thought before taking action"
    )
    action: str | None = Field(
        default=None, description="Selected tool name, or None if final answer"
    )
    action_input: dict[str, Any] | None = Field(
        default=None, description="Parameters passed to the tool"
    )
    observation: str | None = Field(
        default=None, description="Observation or result returned by the tool"
    )


class ReActExecutionTrace(BaseModel):
    """Full execution trace of the ReAct reasoning and action cycle."""

    steps: list[ReActStep] = Field(default_factory=list, description="Ordered ReAct steps")
    total_iterations: int = Field(..., description="Total ReAct iterations executed")
    is_terminated: bool = Field(..., description="Whether the loop terminated cleanly")
    termination_reason: str = Field(..., description="Reason for loop termination")


class ReActAgentRequest(BaseModel):
    """Request payload for executing the explicit ReAct agent loop."""

    query: str = Field(..., min_length=1, description="Research question or multi-step problem")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")
    max_iterations: int = Field(
        default=5, ge=1, le=10, description="Max reasoning and action loops allowed"
    )
    expertise_level: str = Field(
        default="expert", description="Audience expertise ('expert', 'intermediate', 'novice')"
    )
    target_tone: str = Field(
        default="academic", description="Response tone ('academic', 'executive', 'didactic')"
    )
    custom_instructions: str | None = Field(
        default=None, description="Optional custom prompt instructions"
    )


class ReActAgentResponse(BaseModel):
    """Response returned by the explicit ReAct agent loop."""

    answer: str = Field(..., description="Final synthesized evidence-backed answer")
    session_id: str = Field(..., description="Active conversation session ID")
    trace: ReActExecutionTrace = Field(..., description="Full reasoning and action trace")
    structured_synthesis: ResearchSynthesisModel | None = Field(
        default=None, description="Optional structured synthesis object if generated"
    )
