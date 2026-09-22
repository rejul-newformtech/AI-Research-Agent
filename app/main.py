from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.a2a import a2a_router, well_known_router
from app.api.routes.agent import router as agent_router
from app.api.routes.auth import router as auth_router
from app.api.routes.ingestion import router as ingestion_router
from app.core.config import settings
from app.db.session import init_db
from app.middleware.logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    await init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Register request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Root-level discovery routers
app.include_router(well_known_router)

# Versioned API routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")
app.include_router(a2a_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Agent API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
