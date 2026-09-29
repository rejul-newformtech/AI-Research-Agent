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

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch official A2A protocol Agent Card from /.well-known/agent-card.json using A2ACardResolver."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                from a2a.client import A2ACardResolver
                from google.protobuf.json_format import MessageToDict

                resolver = A2ACardResolver(client, self.base_url)
                card = await resolver.get_agent_card()
                return MessageToDict(card)
            except Exception as e:
                logger.debug(f"A2ACardResolver fallback to direct HTTP: {e}")
                resp = await client.get(f"{self.base_url}/.well-known/agent-card.json")
                resp.raise_for_status()
                return resp.json()

    async def resolve_agent_endpoint(self) -> str:
        """Resolve and verify target communication endpoint from the official Agent Card."""
        try:
            card = await self.get_agent_card()
            interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces", [])
            if interfaces and isinstance(interfaces, list) and len(interfaces) > 0:
                first_url = interfaces[0].get("url")
                if first_url:
                    # In docker networks, preserve hostname override if pointing to localhost
                    if "localhost" in first_url and "localhost" not in self.base_url:
                        return self.base_url
                    return first_url.rstrip("/")
        except Exception as e:
            logger.warning(f"Could not resolve interface URL from agent card: {e}")
        return self.base_url

    async def get_info(self) -> dict[str, Any]:
        """Fetch remote agent metadata and registered tool catalog from the official Agent Card."""
        card = await self.get_agent_card()
        skills = card.get("skills", [])
        tool_skills = [s for s in skills if "tools" in s.get("tags", [])]
        tool_names = [s["name"] for s in tool_skills]
        tools_manifest = [
            {
                "name": s["name"],
                "description": (s.get("description", "") or "").split("\n\n")[0].strip(),
            }
            for s in tool_skills
        ]
        return {
            "name": card.get("name", "research_agent"),
            "description": card.get("description", ""),
            "framework": "ReAct (Reasoning + Action + Observation)",
            "model": settings.gemini_model,
            "version": card.get("version", "1.0.0"),
            "capabilities": card.get("capabilities", {}),
            "tools": tool_names,
            "tools_manifest": tools_manifest,
            "agent_card": card,
        }

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
