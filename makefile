.DEFAULT_GOAL: dev

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  dev - Run the FastAPI server in development mode"
	@echo "  run - Run the FastAPI server"
	@echo "  build - Build the project"
	@echo "  install - Install the project"

.PHONY: dev
dev:
	uv run fastapi dev app/main.py

.PHONY: run
run:
	uv run fastapi run app/main.py

.PHONY: adk
adk:
	uv run adk web app/agents --port 8081

.PHONY: eval
eval:
	uv run python tests/evaluation/run_eval.py

.PHONY: mcp
mcp:
	uv run python -m app.service.mcp_server

.PHONY: mcp-http
mcp-http:
	uv run python -m app.service.mcp_server --http
