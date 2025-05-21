"""Tests for the REST and MCP transports.

Exercised through the real ASGI app so routing, validation, serialization, and
error handlers are all covered rather than just the handler bodies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from enrich.server.api import app
from enrich.server.mcp_server import TOOL_DEFINITIONS, EnrichmentMCPServer

SIGNATURE = """Jane Doe
Chief Technology Officer
Acme Robotics
+1 (415) 555-2671
"""


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


class TestOpsEndpoints:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_providers(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert "inference" in response.json()["providers"]

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestEnrichEndpoint:
    def test_enriches_a_corporate_address(self, client: TestClient) -> None:
        response = client.post(
            "/v1/enrich", json={"email": "jane.doe@acmerobotics.com"}
        )
        assert response.status_code == 200

        body = response.json()
        assert body["profile"]["first_name"]["value"] == "Jane"
        assert body["profile"]["company"]["value"] == "Acmerobotics"
        assert body["providers_succeeded"] == ["inference"]

    def test_signature_block_improves_the_result(self, client: TestClient) -> None:
        response = client.post(
            "/v1/enrich",
            json={"email": "jane.doe@acmerobotics.com", "signature_block": SIGNATURE},
        )
        body = response.json()
        assert body["profile"]["title"]["value"] == "Chief Technology Officer"
        assert body["profile"]["title"]["confidence"] == "high"

    def test_rejects_malformed_address(self, client: TestClient) -> None:
        response = client.post("/v1/enrich", json={"email": "not-an-email"})
        assert response.status_code == 422

    def test_requires_an_email_field(self, client: TestClient) -> None:
        assert client.post("/v1/enrich", json={}).status_code == 422

    def test_every_attribute_carries_provenance(self, client: TestClient) -> None:
        body = client.post("/v1/enrich", json={"email": "jane.doe@acme.com"}).json()
        for value in body["profile"].values():
            if value is not None:
                assert "confidence" in value
                assert "source" in value


class TestBatchEndpoint:
    def test_enriches_a_batch(self, client: TestClient) -> None:
        response = client.post(
            "/v1/enrich/batch",
            json={"emails": ["jane.doe@acme.com", "ivan.petrov@corp.io"]},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2

    def test_rejects_an_empty_batch(self, client: TestClient) -> None:
        assert client.post("/v1/enrich/batch", json={"emails": []}).status_code == 422

    def test_enforces_the_batch_ceiling(self, client: TestClient) -> None:
        emails = [f"user{i}@acme.com" for i in range(101)]
        response = client.post("/v1/enrich/batch", json={"emails": emails})
        assert response.status_code == 422


class TestClassifyEndpoint:
    @pytest.mark.parametrize(
        ("email", "expected"),
        [
            ("jane@gmail.com", "personal"),
            ("jane@acme.com", "corporate"),
            ("sales@acme.com", "role"),
            ("noreply@acme.com", "no_reply"),
        ],
    )
    def test_classifies(self, client: TestClient, email: str, expected: str) -> None:
        response = client.post("/v1/classify", json={"email": email})
        assert response.status_code == 200
        assert response.json()["kind"] == expected

    def test_rejects_malformed_address(self, client: TestClient) -> None:
        assert client.post("/v1/classify", json={"email": "nope"}).status_code == 422


class TestCorrelationId:
    def test_response_carries_a_correlation_id(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.headers.get("X-Correlation-ID")

    def test_inbound_correlation_id_is_echoed(self, client: TestClient) -> None:
        response = client.get("/health", headers={"X-Correlation-ID": "trace-abc"})
        assert response.headers["X-Correlation-ID"] == "trace-abc"


class TestMCPServer:
    def test_tool_manifest_is_well_formed(self) -> None:
        names = {tool["name"] for tool in TOOL_DEFINITIONS}
        assert names == {"enrich_email", "enrich_emails_batch", "classify_email"}

        for tool in TOOL_DEFINITIONS:
            assert tool["description"]
            assert tool["inputSchema"]["type"] == "object"
            assert tool["inputSchema"]["required"]

    async def test_enrich_email_tool(self) -> None:
        server = EnrichmentMCPServer()
        result = await server.call_tool(
            "enrich_email", {"email": "jane.doe@acmerobotics.com"}
        )
        assert result["isError"] is False
        assert "Jane" in result["content"][0]["text"]
        await server.close()

    async def test_classify_email_tool(self) -> None:
        server = EnrichmentMCPServer()
        result = await server.call_tool("classify_email", {"email": "sales@acme.com"})
        assert result["isError"] is False
        assert "role" in result["content"][0]["text"]
        await server.close()

    async def test_batch_tool(self) -> None:
        server = EnrichmentMCPServer()
        result = await server.call_tool(
            "enrich_emails_batch", {"emails": ["a.smith@acme.com", "bad"]}
        )
        assert result["isError"] is False
        await server.close()

    async def test_unknown_tool_returns_an_error_payload(self) -> None:
        server = EnrichmentMCPServer()
        result = await server.call_tool("nope", {})
        assert result["isError"] is True
        assert "unknown tool" in result["content"][0]["text"]
        await server.close()

    async def test_invalid_email_returns_an_error_payload(self) -> None:
        """Errors come back as content, not exceptions, so clients see them."""
        server = EnrichmentMCPServer()
        result = await server.call_tool("enrich_email", {"email": "not-an-email"})
        assert result["isError"] is True
        await server.close()
