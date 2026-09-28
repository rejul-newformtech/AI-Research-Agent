"""A2A Core Client for the FastAPI Gateway.

Provides a unified client interface for the FastAPI gateway to communicate
with the A2A Core Server (port 8082 or in-process core engine).
"""

import asyncio
from typing import Any

import httpx

from app.core.config import settings
from app.core.logger import get_logger
from app.schema.agent import ReActExecutionTrace
from app.schema.structured_output import ResearchSynthesisModel, UserProfileContext

logger = get_logger("app.gateway.a2a_client")


class A2ACoreClient:
    """Client for delegating research reasoning and execution to the A2A Core Engine."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or getattr(settings, "a2a_server_url", "http://localhost:8082")

    async def execute_react(
        self,
        query: str,
        user_id: int,
        session_id: str,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
        db: Any = None,
        react_service: Any = None,
    ) -> tuple[str, ReActExecutionTrace, ResearchSynthesisModel | None]:
        """Delegate ReAct loop execution to the A2A Core Engine."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/.well-known/agent-card.json")
                if res.status_code == 200:
                    logger.debug(f"A2A Core Server active at {self.base_url}")
        except Exception:
            logger.debug(f"A2A Core Server at {self.base_url} executing via local core engine.")

        if react_service is None:
            from a2a_server.agent.agent import ReActAgentService

            react_service = ReActAgentService()

        return await react_service.run(
            query=query,
            db=db,
            user_id=user_id,
            session_id=session_id,
            max_iterations=max_iterations,
            user_profile=user_profile,
        )

    async def execute_rag(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_multiquery: bool = True,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
        pipeline: Any = None,
    ):
        """Delegate Chained RAG pipeline execution to the A2A Core Engine."""
        if pipeline is None:
            from a2a_server.rag.advanced_retrieval import ChainedRAGPipeline

            pipeline = ChainedRAGPipeline()

        return await asyncio.to_thread(
            pipeline.run,
            query=query,
            top_k=top_k,
            use_hyde=use_hyde,
            use_multiquery=use_multiquery,
            user_profile=user_profile,
            history=history,
        )


# Singleton gateway client instance
a2a_client = A2ACoreClient()
