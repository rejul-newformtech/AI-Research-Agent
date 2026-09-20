"""Pydantic schemas for Agent-to-Agent (A2A) protocol and Agent Card declarations."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentCardToolDeclaration(BaseModel):
    """Schema for a tool or capability exposed in an Agent Card."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Description of the tool purpose and behavior")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema of input parameters"
    )


class AgentCard(BaseModel):
    """Standardized Agent Card metadata for autonomous agent discovery."""

    name: str = Field(default="AI Research Assistant Agent", description="Agent display name")
    version: str = Field(default="0.1.0", description="Agent protocol version")
    description: str = Field(
        default=(
            "Autonomous academic and scientific research agent with Hybrid Search (BM25 + ChromaDB RRF), "
            "HyDE expansion, Multi-Query reformulation, and ReAct thinking loop."
        ),
        description="Detailed agent summary",
    )
    url: str = Field(default="http://localhost:8000", description="Base URL of the agent service")
    protocols: list[str] = Field(
        default_factory=lambda: ["REST", "FastMCP", "A2A"],
        description="Supported communication protocols",
    )
    skills: list[str] = Field(
        default_factory=lambda: [
            "document_ingestion",
            "hybrid_retrieval",
            "hyde_expansion",
            "react_reasoning",
            "academic_synthesis",
        ],
        description="High-level capabilities of the agent",
    )
    tools: list[AgentCardToolDeclaration] = Field(
        default_factory=list, description="List of exposed callable research tools"
    )
    authentication: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "bearer",
            "token_url": "/api/v1/auth/login",
            "header": "Authorization",
            "scheme": "Bearer",
        },
        description="Authentication requirements",
    )


class A2AQueryRequest(BaseModel):
    """Request payload sent by a peer agent initiating an inter-agent task delegation."""

    sender_agent_id: str = Field(
        ..., min_length=1, description="Identifier of the requesting peer agent"
    )
    query: str = Field(..., min_length=1, description="Research query or scientific question")
    mode: Literal["react", "chained_rag", "hybrid_search"] = Field(
        default="react",
        description="Execution mode: 'react' (multi-step loop), 'chained_rag' (HyDE+MultiQuery), or 'hybrid_search'",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max passages to retrieve")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional metadata or context forwarded by the caller"
    )


class A2AQueryResponse(BaseModel):
    """Response payload returned to a peer agent containing grounded evidence and citations."""

    agent_id: str = Field(default="research_agent", description="Responding agent identifier")
    query: str = Field(..., description="The original query executed")
    response: str = Field(..., description="Grounded research answer or synthesized findings")
    citations: list[dict[str, Any]] = Field(
        default_factory=list, description="Document sources and page references"
    )
    tool_traces: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution steps or tool invocations"
    )
    mode_used: str = Field(..., description="The processing mode executed")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
