import sys

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from app.agents.research_assistant.agent import root_agent
from app.core.logger import get_logger

logger = get_logger("research_agent.a2a")

#
a2a_app = to_a2a(root_agent, port=8082)

if __name__ == "__main__":
    port = 8082
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass

    logger.info(f"Starting official A2A Server on port {port}")
    app = a2a_app if port == 8082 else to_a2a(root_agent, port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
