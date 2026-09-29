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
    """Asynchronous client for communicating with the remote A2A microservice via its official Agent Card."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self._server_url = (base_url or settings.a2a_server_url).rstrip("/")
        self.timeout = timeout
        self._resolved_endpoint: str | None = None
        self._cached_card: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        """Return the resolved communication endpoint, falling back to initial server URL."""
        return self._resolved_endpoint or self._server_url

    async def get_agent_card(self, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch official A2A protocol Agent Card from /.well-known/agent-card.json using A2ACardResolver."""
        if self._cached_card and not force_refresh:
            return self._cached_card

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                from a2a.client import A2ACardResolver
                from google.protobuf.json_format import MessageToDict

                resolver = A2ACardResolver(client, self._server_url)
                card = await resolver.get_agent_card()
                card_dict = MessageToDict(card)
            except Exception as e:
                logger.debug(f"A2ACardResolver fallback to direct HTTP: {e}")
                resp = await client.get(f"{self._server_url}/.well-known/agent-card.json")
                resp.raise_for_status()
                card_dict = resp.json()

            self._cached_card = card_dict
            self._update_resolved_endpoint(card_dict)
            return card_dict

    def _update_resolved_endpoint(self, card: dict[str, Any]) -> str:
        """Extract and cache the communication endpoint declared in supportedInterfaces of the Agent Card."""
        interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces", [])
        if interfaces and isinstance(interfaces, list) and len(interfaces) > 0:
            url = interfaces[0].get("url")
            if url:
                url = url.rstrip("/")
                # In container networks, preserve container hostname if localhost was advertised
                if "localhost" in url and "localhost" not in self._server_url:
                    self._resolved_endpoint = self._server_url
                else:
                    self._resolved_endpoint = url
                return self._resolved_endpoint
        self._resolved_endpoint = self._server_url
        return self._resolved_endpoint

    async def get_target_endpoint(self) -> str:
        """Resolve target communication endpoint from the official Agent Card before communicating."""
        if not self._resolved_endpoint:
            await self.get_agent_card()
        return self._resolved_endpoint or self._server_url

    async def check_health(self) -> dict[str, Any]:
        """Verify connectivity to the remote A2A microservice via resolved endpoint."""
        endpoint = await self.get_target_endpoint()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{endpoint}/health")
            resp.raise_for_status()
            return resp.json()

    async def get_info(self) -> dict[str, Any]:
        """Fetch remote agent metadata and registered tool catalog derived from the official Agent Card."""
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

    async def get_a2a_protocol_client(self):
        """Create an official A2A protocol Client resolved from the Agent Card."""
        from a2a.client import ClientConfig, ClientFactory

        hc = httpx.AsyncClient(timeout=self.timeout)
        config = ClientConfig(streaming=True, httpx_client=hc)
        factory = ClientFactory(config)
        return await factory.create_from_url(self._server_url)

    async def run_chat(
        self,
        message: str,
        session_id: str | None = None,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke remote agent using official A2A protocol resolved from the Agent Card."""
        import uuid

        from a2a.types import Message, Part, Role, SendMessageRequest
        from google.protobuf.json_format import MessageToDict

        try:
            client = await self.get_a2a_protocol_client()
            parts = []
            if user_profile:
                parts.append(Part(text=f"[User Context: {user_profile.model_dump_json()}]\n"))
            if history:
                history_text = "\n".join(
                    [f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-5:]]
                )
                parts.append(Part(text=f"[Conversation History:\n{history_text}\n]\n"))
            parts.append(Part(text=message))

            user_msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.ROLE_USER,
                parts=parts,
            )
            req = SendMessageRequest(message=user_msg)

            final_text = ""
            tool_traces = []

            async for event in client.send_message(req):
                event_dict = MessageToDict(event)
                status_update = event_dict.get("statusUpdate", {})
                status_msg = status_update.get("status", {}).get("message", {})
                for part in status_msg.get("parts", []):
                    if part.get("text"):
                        final_text += part.get("text")
                metadata = status_update.get("metadata", {})
                if metadata.get("adk_author"):
                    tool_traces.append(
                        {
                            "type": "thought",
                            "name": metadata.get("adk_author"),
                            "response": f"Status: {status_update.get('status', {}).get('state', '')}",
                        }
                    )

            if final_text:
                return {
                    "response": final_text,
                    "session_id": session_id,
                    "tool_traces": tool_traces,
                }
        except Exception as e:
            logger.warning(
                f"Official A2A protocol stream encountered an issue ({e}), using resolved endpoint: {e}"
            )

        # Fallback to endpoint discovered from the Agent Card
        endpoint = await self.get_target_endpoint()
        payload: dict[str, Any] = {
            "message": message,
            "session_id": session_id,
            "max_iterations": max_iterations,
            "history": history or [],
        }
        if user_profile:
            payload["user_profile"] = user_profile.model_dump()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{endpoint}/api/v1/chat", json=payload)
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
        """Invoke remote Chained RAG pipeline through endpoint resolved dynamically from the well-known Agent Card."""
        endpoint = await self.get_target_endpoint()
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
            resp = await client.post(f"{endpoint}/api/v1/query", json=payload)
            resp.raise_for_status()
            return resp.json()


# Global client instance
a2a_client = A2AServerClient()


def get_a2a_client() -> A2AServerClient:
    """Dependency provider for FastAPI route injection."""
    return a2a_client
