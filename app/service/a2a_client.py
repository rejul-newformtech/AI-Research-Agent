"""HTTP Client for communicating with the remote A2A Intelligence microservice.

Ensures the FastAPI backend performs zero in-process imports from a2a_server,
interacting solely over the network (HTTP/JSON-RPC).
"""

from typing import Any

import httpx

from app.core.config import settings
from app.core.logger import get_logger
from app.schema.structured_output import UserProfileContext

logger = get_logger("app.service.a2a_client")


class A2AServerClient:
    """Asynchronous HTTP client for the remote A2A Intelligence server microservice."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or settings.a2a_server_url).rstrip("/")
        self.timeout = timeout

    async def check_health(self) -> dict[str, Any]:
        """Verify connectivity to the remote A2A intelligence server."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def get_info(self) -> dict[str, Any]:
        """Fetch remote agent metadata and registered tool catalog."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/v1/info")
            resp.raise_for_status()
            return resp.json()

    async def run_chat(
        self,
        message: str,
        session_id: str | None = None,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke remote ReAct agent reasoning and action loop over HTTP."""
        payload: dict[str, Any] = {
            "message": message,
            "session_id": session_id,
            "max_iterations": max_iterations,
            "history": history or [],
        }
        if user_profile:
            payload["user_profile"] = user_profile.model_dump()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/v1/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def run_query(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_multiquery: bool = True,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke remote Chained RAG pipeline over HTTP."""
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_multiquery": use_multiquery,
            "history": history or [],
        }
        if user_profile:
            payload["user_profile"] = user_profile.model_dump()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/v1/query", json=payload)
            resp.raise_for_status()
            return resp.json()


# Global client instance
a2a_client = A2AServerClient()


def get_a2a_client() -> A2AServerClient:
    """Dependency provider for FastAPI route injection."""
    return a2a_client
