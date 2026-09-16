from fastapi import FastAPI

from research_agent.api.routes.ingestion import router as ingestion_router
from research_agent.core.config import settings
from research_agent.middleware.logging import RequestLoggingMiddleware

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

# Register request logging middleware
app.add_middleware(RequestLoggingMiddleware)

app.include_router(ingestion_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Agent API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("research_agent.main:app", host="0.0.0.0", port=8000, reload=True)
