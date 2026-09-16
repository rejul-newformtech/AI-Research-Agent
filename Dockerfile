# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.14
FROM python:${PYTHON_VERSION}-slim AS base

# Copy uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment configurations for Python and uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies using cached mounts (without installing the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the application source code
COPY . /app

# Sync and install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Set permissions for the non-privileged user
RUN chown -R appuser:appuser /app

# Put the uv virtualenv into PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to the non-privileged user
USER appuser

# Expose the application port
EXPOSE 8000

# Run FastAPI
CMD ["fastapi", "run", "src/research_agent/main.py", "--port", "8000", "--host", "0.0.0.0"]
