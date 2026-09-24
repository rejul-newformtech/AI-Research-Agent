"""Integration tests for official Agent-to-Agent (A2A) protocol server powered by a2a-sdk."""

from starlette.testclient import TestClient

from app.service.a2a_server import a2a_app


def test_official_a2a_agent_card():
    """Verify /.well-known/agent-card.json returns compliant A2A protocol Agent Card."""
    with TestClient(a2a_app) as client:
        res = client.get("/.well-known/agent-card.json")
        assert res.status_code == 200
        card = res.json()

        assert card["name"] == "research_agent"
        assert "description" in card
        assert "skills" in card
        assert len(card["skills"]) > 0


def test_official_a2a_jsonrpc_endpoint():
    """Verify root JSON-RPC 2.0 endpoint handles compliant A2A RPC messages."""
    with TestClient(a2a_app) as client:
        # Test an RPC call with standard envelope
        res = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "test-req-1",
                "method": "tasks.list",
                "params": {},
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-req-1"
