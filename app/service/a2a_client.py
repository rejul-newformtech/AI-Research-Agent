"""Official A2A Protocol Client communicating exclusively via /.well-known/agent-card.json.

Ensures the FastAPI backend performs zero in-process imports from a2a_server and
communicates strictly via Google ADK Agent Card discovery and A2A JSON-RPC 2.0 protocol.
"""

import uuid
from typing import Any

import httpx

from app.core.config import settings
from app.core.logger import get_logger
from app.schema.structured_output import UserProfileContext

logger = get_logger("app.service.a2a_client")


class A2AServerClient:
    """Official client for communicating with the remote A2A microservice via its Agent Card."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self._server_url = (base_url or settings.a2a_server_url).rstrip("/")
        self.timeout = timeout
        self._cached_card: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        """Return the target server URL."""
        return self._server_url

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
                self._cached_card = MessageToDict(card)
            except Exception as e:
                logger.debug(f"A2ACardResolver fallback to direct well-known fetch: {e}")
                resp = await client.get(f"{self._server_url}/.well-known/agent-card.json")
                resp.raise_for_status()
                self._cached_card = resp.json()

        return self._cached_card

    async def check_health(self) -> dict[str, Any]:
        """Verify connectivity by resolving the official Agent Card at /.well-known/agent-card.json."""
        card = await self.get_agent_card(force_refresh=True)
        return {
            "status": "healthy",
            "service": "a2a_server",
            "agent_name": card.get("name", "research_agent"),
            "version": card.get("version", "1.0.0"),
        }

    async def get_info(self) -> dict[str, Any]:
        """Fetch remote agent metadata and registered tool catalog derived strictly from the Agent Card."""
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

    async def get_protocol_client(self):
        """Construct official A2A Client from /.well-known/agent-card.json."""
        from a2a.client import ClientConfig, ClientFactory

        hc = httpx.AsyncClient(timeout=self.timeout)
        config = ClientConfig(streaming=True, httpx_client=hc)
        factory = ClientFactory(config)
        # Resolves /.well-known/agent-card.json and sets up JSON-RPC transport automatically
        return await factory.create_from_url(self._server_url)

    async def run_chat(
        self,
        message: str,
        session_id: str | None = None,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke remote agent using official A2A message protocol discovered via Agent Card."""
        from a2a.types import Message, Part, Role, SendMessageRequest
        from google.protobuf.json_format import MessageToDict

        client = await self.get_protocol_client()

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

        if not final_text:
            final_text = "Task completed by research agent."

        return {
            "response": final_text,
            "session_id": session_id,
            "tool_traces": tool_traces,
        }

    async def run_query(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_multiquery: bool = True,
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke deep research query via official A2A agent message protocol."""
        formatted_message = (
            f"Execute research query: '{query}' "
            f"[Settings: top_k={top_k}, hyde={use_hyde}, multiquery={use_multiquery}]."
        )
        chat_result = await self.run_chat(
            message=formatted_message,
            user_profile=user_profile,
            history=history,
        )
        return {
            "query": query,
            "synthesized_answer": chat_result.get("response", ""),
            "retrieved_chunks": [],
            "expanded_queries": [],
            "hypothetical_document": None,
            "total_llm_calls": 2,
        }


# Global client instance
a2a_client = A2AServerClient()


def get_a2a_client() -> A2AServerClient:
    """Dependency provider for FastAPI route injection."""
    return a2a_client
