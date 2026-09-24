"""Official Agent-to-Agent (A2A) Protocol Server.

Powered by `a2a-sdk` and Google ADK to expose the Research Assistant Agent
over standard A2A JSON-RPC 2.0 and Agent Card discovery transports.
"""

import sys

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.applications import Starlette

from app.agents.research_assistant.agent import root_agent
from app.core.logger import get_logger

logger = get_logger("research_agent.a2a")


def create_a2a_app(host: str = "0.0.0.0", port: int = 8082) -> Starlette:
    """Create the standard A2A Starlette application using official a2a-sdk."""
    return to_a2a(
        agent=root_agent,
        host=host,
        port=port,
        protocol="http",
    )


a2a_app = create_a2a_app()

if __name__ == "__main__":
    port = 8082
    host = "0.0.0.0"
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass
    if "--host" in sys.argv:
        try:
            host = sys.argv[sys.argv.index("--host") + 1]
        except IndexError:
            pass

    logger.info(f"Starting official A2A Server on http://{host}:{port}")
    logger.info(f"Agent Card available at: http://{host}:{port}/.well-known/agent-card.json")
    uvicorn.run(a2a_app, host=host, port=port)
