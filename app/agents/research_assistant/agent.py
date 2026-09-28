"""ADK research_assistant agent proxy forwarding to a2a_core.agent.agent."""

from a2a_core.agent.agent import (
    ReActAgentService,
    advanced_research_query,
    ingest_research_notes,
    ingest_stored_document,
    list_stored_documents,
    root_agent,
    search_research_documents,
)

__all__ = [
    "ReActAgentService",
    "advanced_research_query",
    "ingest_research_notes",
    "ingest_stored_document",
    "list_stored_documents",
    "root_agent",
    "search_research_documents",
]
