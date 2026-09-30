"""Pydantic schemas for the unified Research Assistant Agent, tool traces, and conversational RAG."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schema.structured_output import ResearchSynthesisModel


class AgentChatRequest(BaseModel):
    """Request schema for interacting with the Research Assistant Agent."""

    message: str = Field(..., min_length=1, description="User question or research prompt")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")
    user_id: str | None = Field(
        default=None, description="Optional user identifier (defaults to authenticated username)"
    )
    mode: Literal["react", "rag", "research"] = Field(
        default="react",
        description="Execution mode: 'react' (autonomous reasoning loop) or 'rag'/'research' (2-call chained pipeline)",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve")
    use_hyde: bool = Field(
        default=True, description="Enable HyDE (Hypothetical Document Embeddings)"
    )
    use_multiquery: bool = Field(default=True, description="Enable Multi-Query expansion")
    max_iterations: int = Field(
        default=5, ge=1, le=10, description="Max reasoning and action loops allowed (ReAct mode)"
    )
    expertise_level: str = Field(
        default="expert", description="Audience expertise ('expert', 'intermediate', 'novice')"
    )
    target_tone: str = Field(
        default="academic", description="Response tone ('academic', 'executive', 'didactic')"
    )
    custom_instructions: str | None = Field(
        default=None, description="Optional custom prompt directives"
    )


class ToolTrace(BaseModel):
    """Execution trace of an agent tool invocation."""

    type: str
    name: str | None = None
    args: dict[str, Any] | None = None
    response: Any | None = None


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


class AgentChatResponse(BaseModel):
    """Response schema returned by the Research Assistant Agent."""

    response: str
    session_id: str
    user_id: str
    agent_name: str
    model: str
    mode: str = "react"
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    trace: ReActExecutionTrace | None = Field(
        default=None, description="Full reasoning and action trace (ReAct mode)"
    )
    hypothetical_document: str | None = Field(
        default=None, description="HyDE passage generated (RAG mode)"
    )
    expanded_queries: list[str] = Field(
        default_factory=list, description="Expanded queries generated (RAG mode)"
    )
    retrieved_chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved context passages (RAG mode)"
    )
    structured_synthesis: ResearchSynthesisModel | None = Field(
        default=None, description="Structured synthesis model with verified citations"
    )
    total_llm_calls: int | None = Field(default=None, description="Total LLM calls executed")
