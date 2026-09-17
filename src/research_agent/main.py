from contextlib import asynccontextmanager

from fastapi import FastAPI

from research_agent.api.routes.agent import router as agent_router
from research_agent.api.routes.auth import router as auth_router
from research_agent.api.routes.ingestion import router as ingestion_router
from research_agent.core.config import settings
from research_agent.db.session import init_db
from research_agent.middleware.logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Register request logging middleware
app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Agent API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("research_agent.main:app", host="0.0.0.0", port=8000, reload=True)
