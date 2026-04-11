# cassandra-gemini-mcp

MCP server for Google's [Gemini Interactions API](https://ai.google.dev/) — stateful, grounded web search and Q&A. Auto-grounds via Google Search + URL Context tools; the model decides when to use them.

## Tools

| Tool | Thinking Level | Purpose |
|------|---------------|---------|
| `search` | minimal | Quick web search with structured results |
| `ask` | high | Deep reasoning with grounded answers and citations |

Conversations are stateful — every response returns an `interaction_id` that can be passed to follow-up calls for continuation.

## Architecture

```
MCP client → gemini.cassandrasedge.com (CF Tunnel)
  → FastMCP sidecar (port 3003)
    → McpKeyAuthProvider → /keys/validate (auth service)
    → Gemini Interactions API
```

Auth uses the shared `cassandra-mcp-auth` sidecar pattern: `Bearer mcp_...` tokens validated against the auth service, per-tool ACL enforced locally from baked-in `acl.yaml`.

## Config

| Env Var | Required | Default |
|---------|----------|---------|
| `GEMINI_API_KEY` | Yes | — |
| `GEMINI_MODEL` | No | `gemini-3.1-flash-lite-preview` |
| `AUTH_URL` | Yes | — |
| `AUTH_SECRET` | Yes | — |
| `MCP_PORT` | No | `3003` |

## Dev

```bash
cd backend
uv sync
GEMINI_API_KEY=... uv run cassandra-gemini-mcp
```

## Deploy

Auto-deploys on push to main via Woodpecker CI → BuildKit → local registry → ArgoCD (`cassandra-k8s/apps/gemini-mcp/`).

Part of the [Cassandra](https://github.com/Cassandras-Edge) stack.
