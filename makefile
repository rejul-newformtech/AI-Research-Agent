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
	uv run fastapi dev src/research_agent/main.py

.PHONY: run
run:
	uv run fastapi run src/research_agent/main.py

.PHONY: adk
adk:
	uv run adk web src/research_agent/agents --port 8080
