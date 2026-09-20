"""Proxy module redirecting to unified ReAct agent in app.agents.research_assistant.agent."""

from app.agents.research_assistant.agent import (
    REACT_SYSTEM_PROMPT_TEMPLATE,
    ReActAgentService,
)

__all__ = [
    "REACT_SYSTEM_PROMPT_TEMPLATE",
    "ReActAgentService",
]
