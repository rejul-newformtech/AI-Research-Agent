"""FastMCP Model Context Protocol (MCP) Server for the AI Research Assistant Agent.

Exposes 3 core research tools and 1 resource catalog over standard MCP transports.
"""

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from app.agents.research_assistant.agent import (
    advanced_research_query as _advanced_research_query,
)
from app.agents.research_assistant.agent import (
    ingest_stored_document as _ingest_stored_document,
)
from app.agents.research_assistant.agent import (
    search_research_documents as _search_research_documents,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.db.chroma import ChromaService

logger = get_logger("research_agent.mcp")

mcp = FastMCP(
    name="research-assistant",
)


@mcp.tool(
    name="search_research_documents",
    description=(
        "Search ingested research documents and academic literature using Hybrid Search "
        "(BM25 lexical sparse + vector dense embeddings + Reciprocal Rank Fusion re-ranking)."
    ),
)
def search_research_documents(query: str, top_k: int = 5) -> str:
    """Execute hybrid search over stored documents."""
    return _search_research_documents(query=query, top_k=top_k)


@mcp.tool(
    name="advanced_research_query",
    description=(
        "Execute deep research using 2-call chained HyDE (Hypothetical Document Embeddings) "
        "and Multi-Query expansion with grounded academic synthesis and citations."
    ),
)
def advanced_research_query(query: str, top_k: int = 5) -> str:
    """Execute deep 2-call chained research."""
    return _advanced_research_query(query=query, top_k=top_k)


@mcp.tool(
    name="ingest_stored_document",
    description=(
        "Ingest, parse, chunk, embed, and index a document from local storage into "
        "ChromaDB and the BM25 index."
    ),
)
def ingest_stored_document(filename: str, strategy: str = "fixed") -> str:
    """Chunk and embed a document from local storage."""
    return _ingest_stored_document(filename=filename, strategy=strategy)


@mcp.resource("research://documents/catalog")
def get_document_catalog() -> str:
    """Return a comprehensive catalog of all available research documents and vector index status."""
    storage_dir = Path(settings.document_storage_dir)
    docs: list[dict[str, Any]] = []

    if storage_dir.exists():
        for file in sorted(storage_dir.iterdir()):
            if file.is_file() and not file.name.startswith("."):
                docs.append(
                    {
                        "filename": file.name,
                        "extension": file.suffix.lower(),
                        "size_bytes": file.stat().st_size,
                        "modified_timestamp": file.stat().st_mtime,
                    }
                )

    # Vector store status
    chroma_info = {"collection": settings.chroma_collection_name, "total_chunks": 0}
    try:
        chroma = ChromaService()
        chroma_info["total_chunks"] = chroma.count()
    except Exception:
        pass

    catalog = {
        "server": "research-assistant-mcp",
        "version": settings.app_version,
        "available_files_count": len(docs),
        "files": docs,
        "vector_store": chroma_info,
    }
    return json.dumps(catalog, indent=2)


if __name__ == "__main__":
    mcp.run()
