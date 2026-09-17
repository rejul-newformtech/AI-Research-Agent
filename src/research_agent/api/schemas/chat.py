"""Pydantic schemas for chat sessions, message history, and conversational memory."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessageResponse(BaseModel):
    """Schema representing an individual message in a chat history."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    tool_traces: list[dict[str, Any]] | None = None
    created_at: datetime


class ChatSessionSummaryResponse(BaseModel):
    """Schema summarizing a chat session in lists."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatSessionDetailResponse(BaseModel):
    """Detailed chat session schema including full message history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse]
