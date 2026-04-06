# cassandra-gemini-mcp

Gemini Interactions API MCP server — grounded web search and Q&A with stateful conversations.

## Architecture

- **Interactions API** — stateful conversations with `interaction_id` for follow-ups
- **Auto-grounding** — Google Search + URL Context tools, model decides when to use them
- **Thinking levels** — minimal (search), high (ask) for reasoning depth control

## Tools

| Tool | Thinking | Purpose |
|------|----------|---------|
| `search` | minimal | Quick web search, structured results |
| `ask` | high | Deep reasoning with grounded answers |

## Config

| Env Var | Required | Default |
|---------|----------|---------|
| `GEMINI_API_KEY` | yes | — |
| `GEMINI_MODEL` | no | `gemini-3.1-flash-lite-preview` |
| `MCP_PORT` | no | `3003` |

## Dev

```bash
cd backend
uv sync
GEMINI_API_KEY=... uv run cassandra-gemini-mcp
```
