"""Unit tests for the FastMCP Research Assistant Server (3 tools + 1 resource catalog)."""

import json
from unittest.mock import patch

import pytest

from app.service.mcp_server import (
    advanced_research_query,
    get_document_catalog,
    ingest_stored_document,
    mcp,
    search_research_documents,
)


@pytest.mark.asyncio
async def test_mcp_server_initialization_and_manifest():
    """Verify FastMCP instance name, registered tools, and resource catalog."""
    assert mcp.name == "research-assistant"

    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]

    assert "search_research_documents" in tool_names
    assert "advanced_research_query" in tool_names
    assert "ingest_stored_document" in tool_names


@patch("app.service.mcp_server._search_research_documents")
def test_mcp_tool_search_documents(mock_search):
    """Verify search_research_documents FastMCP tool delegates properly."""
    mock_search.return_value = "Found 2 passages citing paper.pdf."

    res = search_research_documents(query="quantum dots", top_k=3)
    assert res == "Found 2 passages citing paper.pdf."
    mock_search.assert_called_once_with(query="quantum dots", top_k=3)


@patch("app.service.mcp_server._advanced_research_query")
def test_mcp_tool_advanced_research_query(mock_advanced):
    """Verify advanced_research_query FastMCP tool executes chained research."""
    mock_advanced.return_value = "Deep synthesized answer with citations."

    res = advanced_research_query(query="neuromorphic computing", top_k=4)
    assert res == "Deep synthesized answer with citations."
    mock_advanced.assert_called_once_with(query="neuromorphic computing", top_k=4)


@patch("app.service.mcp_server._ingest_stored_document")
def test_mcp_tool_ingest_stored_document(mock_ingest):
    """Verify ingest_stored_document FastMCP tool calls ingestion pipeline."""
    mock_ingest.return_value = "Successfully ingested 'test.pdf': created 5 chunks."

    res = ingest_stored_document(filename="test.pdf", strategy="semantic")
    assert "Successfully ingested 'test.pdf'" in res
    mock_ingest.assert_called_once_with(filename="test.pdf", strategy="semantic")


def test_mcp_resource_catalog():
    """Verify research://documents/catalog resource returns a valid JSON catalog."""
    catalog_str = get_document_catalog()
    catalog = json.loads(catalog_str)

    assert catalog["server"] == "research-assistant-mcp"
    assert "files" in catalog
    assert isinstance(catalog["files"], list)
    assert "vector_store" in catalog
    assert "collection" in catalog["vector_store"]
