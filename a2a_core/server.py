"""Official Agent-to-Agent (A2A) Protocol Server.

Powered by Google ADK and a2a-sdk to expose the Research Assistant Agent
as the core intelligence service over standard A2A JSON-RPC 2.0.
"""

import sys

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a_core.agent.agent import root_agent
from app.core.logger import get_logger

logger = get_logger("research_agent.a2a_core")

a2a_app = to_a2a(root_agent, port=8082)

if __name__ == "__main__":
    port = 8082
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass

    logger.info(f"Starting official A2A Core Server on port {port}")
    app = a2a_app if port == 8082 else to_a2a(root_agent, port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
