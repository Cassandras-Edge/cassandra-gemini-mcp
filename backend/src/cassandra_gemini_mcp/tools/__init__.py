from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from cassandra_gemini_mcp.config import Settings


def register_all(mcp: FastMCP, settings: Settings) -> None:
    from .research import register as reg_research

    reg_research(mcp, settings)
