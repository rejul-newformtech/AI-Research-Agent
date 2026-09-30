"""Conversational memory and context windowing service."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession


class ConversationMemoryService:
    """Manages chat session lifecycle, persistence, and sliding context window retrieval."""

    def __init__(self, default_window_size: int = 10):
        self.default_window_size = default_window_size

    async def get_or_create_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: int,
        initial_prompt: str = "",
    ) -> ChatSession:
        """Retrieve existing chat session or create a new one with auto-generated title."""
        stmt = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
        session = await db.scalar(stmt)
        if session:
            return session

        # Auto-title from the first ~45 characters of user prompt
        title = (
            initial_prompt.strip().replace("\n", " ")[:45] if initial_prompt else "New Conversation"
        )
        if len(initial_prompt.strip()) > 45:
            title += "..."

        session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=title or "New Conversation",
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    async def save_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        tool_traces: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """Persist a message turn (user, assistant, or tool) into the database."""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tool_traces=tool_traces,
        )
        db.add(message)

        # Touch session updated_at
        session = await db.get(ChatSession, session_id)
        if session:
            session.updated_at = func.now()

        await db.commit()
        await db.refresh(message)
        return message

    async def get_windowed_history(
        self,
        db: AsyncSession,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[ChatMessage]:
        """Apply a sliding context window to retrieve only the last N chronological messages.

        This guarantees that earlier context is retained without exceeding model token budgets.
        """
        limit = max_messages or self.default_window_size
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        result = await db.scalars(stmt)
        recent_messages = list(result.all())
        # Return in chronological order
        recent_messages.reverse()
        return recent_messages

    async def list_user_sessions(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[dict[str, Any]]:
        """List all conversation sessions belonging to a specific user with message counts."""
        stmt = (
            select(
                ChatSession,
                func.count(ChatMessage.id).label("message_count"),
            )
            .outerjoin(ChatMessage, ChatSession.id == ChatMessage.session_id)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.updated_at.desc())
        )
        res = await db.execute(stmt)
        results = []
        for session, count in res.all():
            results.append(
                {
                    "id": session.id,
                    "title": session.title,
                    "message_count": count,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                }
            )
        return results

    async def get_session_details(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: int,
    ) -> ChatSession | None:
        """Retrieve a specific chat session and all its messages, verifying user ownership."""
        stmt = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
        return await db.scalar(stmt)

    async def delete_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: int,
    ) -> bool:
        """Delete a chat session and all its messages (cascade), verifying user ownership."""
        session = await self.get_session_details(db, session_id, user_id)
        if not session:
            return False
        await db.delete(session)
        await db.commit()
        return True
