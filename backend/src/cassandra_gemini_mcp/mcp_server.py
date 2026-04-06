"""FastMCP server for Cassandra Gemini MCP — grounded search and Q&A via Gemini Interactions API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from cassandra_mcp_auth import AclMiddleware, DiscoveryTransform
from cassandra_gemini_mcp.auth import McpKeyAuthProvider, build_auth
from cassandra_gemini_mcp.clients.gemini import GeminiClient
from cassandra_gemini_mcp.config import Settings

logger = logging.getLogger(__name__)

SERVICE_ID = "gemini-mcp"


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server with auth and all tools."""

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. "
            "Get your API key from https://aistudio.google.com/app/apikey"
        )

    # Auth
    auth_provider = None
    mcp_key_provider = None
    if settings.auth_url and settings.auth_secret:
        if (
            settings.workos_client_id
            and settings.workos_authkit_domain
            and settings.base_url
        ):
            auth_provider, mcp_key_provider = build_auth(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
                base_url=settings.base_url,
                workos_client_id=settings.workos_client_id,
                workos_authkit_domain=settings.workos_authkit_domain,
            )
        else:
            mcp_key_provider = McpKeyAuthProvider(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
            )
            auth_provider = mcp_key_provider

    gemini_client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)

    # Set fallback so tools work even without lifespan (gateway embedding)
    from cassandra_gemini_mcp.tools._helpers import set_fallback_client
    set_fallback_client(gemini_client)

    @asynccontextmanager
    async def lifespan(server):
        yield {
            "gemini_client": gemini_client,
        }
        await gemini_client.close()
        if mcp_key_provider is not None:
            mcp_key_provider.close()

    acl_mw = AclMiddleware(service_id=SERVICE_ID, acl_path=settings.auth_yaml_path)

    mcp_kwargs: dict = {
        "name": "Cassandra Gemini",
        "instructions": (
            "# Cassandra Gemini\n\n"
            "Grounded web search and Q&A powered by Gemini Interactions API with "
            "automatic Google Search grounding.\n\n"
            "## Tools\n\n"
            "- **search** — Quick web search with structured results (titles, URLs, snippets)\n"
            "- **ask** — Deep reasoning with grounded answers, supports stateful follow-ups "
            "via interaction_id\n\n"
            "All responses include source citations. Use `interaction_id` from any response "
            "to continue the conversation with full context."
        ),
        "lifespan": lifespan,
        "middleware": [acl_mw] if acl_mw._enabled else [],  # noqa: SLF001
    }
    if settings.code_mode:
        mcp_kwargs["transforms"] = [DiscoveryTransform(service_id=SERVICE_ID)]
    if auth_provider:
        mcp_kwargs["auth"] = auth_provider

    mcp = FastMCP(**mcp_kwargs)

    # Health check
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request):  # noqa: ANN001, ARG001
        from starlette.responses import JSONResponse  # noqa: PLC0415

        return JSONResponse({"ok": True, "service": "cassandra-gemini-mcp"})

    # Register tools
    from cassandra_gemini_mcp.tools import register_all  # noqa: PLC0415

    register_all(mcp, settings)

    return mcp
