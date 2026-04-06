"""Shared helpers for tool modules."""

from __future__ import annotations

from fastmcp.server.context import Context

from cassandra_gemini_mcp.clients.gemini import GeminiClient

_fallback_client: GeminiClient | None = None


def set_fallback_client(client: GeminiClient) -> None:
    global _fallback_client  # noqa: PLW0603
    _fallback_client = client


def resolve_gemini_client(ctx: Context) -> GeminiClient:
    """Get the Gemini client from lifespan context or fallback."""
    try:
        return ctx.request_context.lifespan_context["gemini_client"]
    except (AttributeError, KeyError, TypeError):
        pass
    if _fallback_client is not None:
        return _fallback_client
    raise RuntimeError("No GeminiClient available — check lifespan or fallback")
