"""Gemini research tools — search and ask with grounded responses."""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP
from fastmcp.server.context import Context
from mcp.types import ToolAnnotations

from cassandra_gemini_mcp.clients.gemini import format_response
from cassandra_gemini_mcp.config import Settings
from cassandra_gemini_mcp.tools._helpers import resolve_gemini_client


def register(mcp: FastMCP, settings: Settings) -> None:
    _ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

    @mcp.tool(annotations=_ro)
    async def search(
        query: str,
        ctx: Context,
        max_results: int = 10,
    ) -> str:
        """Quick web search via Gemini with Google Search grounding.

        Returns structured results with titles, URLs, and snippets. Uses minimal
        thinking for fast turnaround.

        Args:
            query: Search query.
            max_results: Maximum number of results to return (default: 10).
        """
        client = resolve_gemini_client(ctx)
        system_instruction = (
            f"Search for the query and return results in this exact format:\n\n"
            f"---\n"
            f"TITLE: [page title]\n"
            f"URL: [full url]\n"
            f"SNIPPET: [2-3 sentence excerpt]\n"
            f"---\n\n"
            f"Return up to {max_results} results. No additional commentary or analysis."
        )
        result = await client.create_interaction(
            input_content=query,
            thinking_level="minimal",
            system_instruction=system_instruction,
            max_tokens=4096,
        )
        return format_response(result)

    @mcp.tool(annotations=_ro)
    async def ask(
        query: str,
        ctx: Context,
        interaction_id: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> str:
        """Get grounded answers from Gemini with deep reasoning.

        Model automatically searches the web when needed for current information.
        Pass interaction_id from a previous response to continue that conversation.

        Args:
            query: Your question.
            interaction_id: Pass from a previous response to continue that conversation.
            max_tokens: Maximum response length (default: 8192).
        """
        client = resolve_gemini_client(ctx)
        result = await client.create_interaction(
            input_content=query,
            thinking_level="high",
            previous_interaction_id=interaction_id,
            max_tokens=max_tokens,
            system_instruction="Be concise and factual. Cite sources when using web information.",
        )
        return format_response(result)
