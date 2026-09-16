from fastapi import FastAPI

app = FastAPI(title="Research Agent API", version="0.1.0")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Agent API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("research_agent.main:app", host="0.0.0.0", port=8000, reload=True)
