"""MCP server exposing enrichment as tools.

Speaks the Model Context Protocol over stdio, so any MCP client (Claude
Desktop, Claude Code, or a custom agent runtime) can call these tools without
bespoke integration code.

This module is a thin adapter. All logic lives in
:class:`~..service.EnrichmentService`, so the tool surface stays declarative
and the same behaviour is reachable over REST.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import get_settings
from ..errors import InvalidEmailError
from ..logging import configure_logging, get_logger
from ..models import EnrichmentRequest
from ..parsing import parse_email
from ..providers.base import available_providers
from ..service import EnrichmentService

logger = get_logger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "enrich_email",
        "description": (
            "Find information about a person from their email address. "
            "Returns name, company, title, and contact details where they can "
            "be determined, each with a confidence level and the source that "
            "produced it. Supplying the sender's signature block substantially "
            "improves results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address to enrich.",
                },
                "signature_block": {
                    "type": "string",
                    "description": (
                        "Optional raw signature text from the person's email. "
                        "Yields high-confidence name, title, and phone."
                    ),
                },
                "skip_cache": {
                    "type": "boolean",
                    "description": "Bypass the cache and force a fresh lookup.",
                    "default": False,
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "enrich_emails_batch",
        "description": (
            "Enrich many email addresses at once. Runs concurrently under the "
            "configured rate limit. Individual failures are isolated and "
            "reported rather than failing the batch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "emails": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email addresses to enrich (max 100).",
                    "maxItems": 100,
                },
                "skip_cache": {"type": "boolean", "default": False},
            },
            "required": ["emails"],
        },
    },
    {
        "name": "classify_email",
        "description": (
            "Classify an address without enriching it. Returns whether it is "
            "personal, corporate, a role account (sales@, info@), or an "
            "unattended sender (noreply@). Fast and free: use it to filter a "
            "list before spending enrichment calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The address to classify."}
            },
            "required": ["email"],
        },
    },
]


class EnrichmentMCPServer:
    """Dispatches MCP tool calls to the enrichment service."""

    def __init__(self, service: EnrichmentService | None = None) -> None:
        self._service = service or EnrichmentService()

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool manifest advertised to clients."""
        return TOOL_DEFINITIONS

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call.

        Errors are returned as structured payloads with ``isError`` set rather
        than raised, so a client sees an actionable message instead of a
        transport-level failure.
        """
        try:
            match name:
                case "enrich_email":
                    return await self._enrich_email(arguments)
                case "enrich_emails_batch":
                    return await self._enrich_batch(arguments)
                case "classify_email":
                    return await self._classify(arguments)
                case _:
                    return _error(f"unknown tool: {name}")
        except InvalidEmailError as exc:
            return _error(f"invalid email: {exc}")
        except Exception as exc:
            logger.exception("tool call failed", extra={"tool": name})
            return _error(f"{type(exc).__name__}: {exc}")

    # -- handlers --------------------------------------------------------

    async def _enrich_email(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = EnrichmentRequest(
            email=arguments["email"],
            signature_block=arguments.get("signature_block"),
            skip_cache=arguments.get("skip_cache", False),
        )
        result = await self._service.enrich(request)
        return _ok(result.model_dump(mode="json"))

    async def _enrich_batch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._service.enrich_batch(
            arguments["emails"], skip_cache=arguments.get("skip_cache", False)
        )
        return _ok(result.model_dump(mode="json"))

    async def _classify(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = parse_email(arguments["email"])
        except ValueError as exc:
            raise InvalidEmailError(str(exc)) from exc
        return _ok(parsed.model_dump(mode="json"))

    async def close(self) -> None:
        await self._service.close()


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a successful result in MCP content format."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
        "isError": False,
    }


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


async def serve_stdio() -> None:
    """Run the server over stdio using the official MCP SDK.

    The SDK is an optional dependency: the REST transport and the service
    itself work without it, so the import failure is reported with an
    actionable message rather than crashing at module import.
    """
    try:
        import mcp.server.stdio
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The MCP SDK is not installed. Install it with: pip install 'mcp>=1.2'"
        ) from exc

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    handler = EnrichmentMCPServer()
    server: Server = Server("email-enrichment")

    # The MCP SDK's decorators are untyped, so strict mode cannot see through
    # them. The handler bodies below are still fully checked.
    @server.list_tools()  # type: ignore[untyped-decorator]
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in TOOL_DEFINITIONS
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        response = await handler.call_tool(name, arguments)
        return [
            TextContent(type="text", text=block["text"]) for block in response["content"]
        ]

    logger.info(
        "mcp server starting",
        extra={"providers": available_providers(), "tools": len(TOOL_DEFINITIONS)},
    )

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    """Console-script entry point."""
    import asyncio

    asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
