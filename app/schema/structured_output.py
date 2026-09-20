"""Typed Pydantic models for structured LLM generation and dynamic prompting."""

from pydantic import BaseModel, ConfigDict, Field


class CitationModel(BaseModel):
    """Explicit source citation for an extracted fact or research claim."""

    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., description="Filename or identifier of the source document")
    page_number: int | None = Field(
        default=None, description="Page number where the evidence appears, if available"
    )
    section: str | None = Field(
        default=None, description="Section, chapter, or heading title if identifiable"
    )
    quote_or_fact: str = Field(
        ..., description="Specific excerpt, fact, or metric directly cited from the document"
    )


class ResearchSynthesisModel(BaseModel):
    """Structured research synthesis returned by the LLM in the 2-call Chained RAG pipeline."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ..., description="Executive summary providing a direct answer to the user's inquiry"
    )
    detailed_findings: str = Field(
        ..., description="Detailed academic or technical breakdown grounded in the evidence"
    )
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Structured list of verified citations extracted from context",
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Key conclusions or actionable takeaways",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) based on coverage and factual support",
    )
    missing_evidence: str | None = Field(
        default=None,
        description="Information gaps or ambiguities not addressed by the provided excerpts",
    )


class MultiQueryExpansionModel(BaseModel):
    """Structured query expansion result for multi-query retrieval."""

    model_config = ConfigDict(extra="ignore")

    queries: list[str] = Field(
        ...,
        min_length=1,
        description="List of diverse reformulations of the original query",
    )


class HyDEPassageModel(BaseModel):
    """Structured hypothetical document output for HyDE retrieval."""

    model_config = ConfigDict(extra="ignore")

    hypothetical_passage: str = Field(
        ...,
        description="Plausible academic or technical passage answering the query",
    )
    domain: str = Field(
        default="general_science",
        description="Scientific or technical field identified for the query",
    )


class UserProfileContext(BaseModel):
    """User profile metadata used to dynamically steer prompt generation."""

    model_config = ConfigDict(extra="ignore")

    username: str = "researcher"
    role: str = "researcher"  # e.g., 'admin', 'researcher', 'user'
    expertise_level: str = "expert"  # 'expert', 'intermediate', 'novice'
    target_tone: str = "academic"  # 'academic', 'executive', 'didactic'
    custom_instructions: str | None = None
