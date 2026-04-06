"""Settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    # Gemini
    gemini_api_key: str
    gemini_model: str

    # Auth (ACL service)
    auth_url: str
    auth_secret: str
    auth_yaml_path: str

    # WorkOS OAuth
    workos_client_id: str
    workos_authkit_domain: str
    base_url: str

    # Server
    host: str
    mcp_port: int
    code_mode: bool


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        auth_url=os.environ.get("AUTH_URL", ""),
        auth_secret=os.environ.get("AUTH_SECRET", ""),
        auth_yaml_path=os.environ.get("AUTH_YAML_PATH", "/app/acl.yaml"),
        workos_client_id=os.environ.get("WORKOS_CLIENT_ID", ""),
        workos_authkit_domain=os.environ.get("WORKOS_AUTHKIT_DOMAIN", ""),
        base_url=os.environ.get("BASE_URL", ""),
        host=os.environ.get("HOST", "0.0.0.0"),
        mcp_port=int(os.environ.get("MCP_PORT", "3003")),
        code_mode=os.environ.get("CODE_MODE", "true").lower() in ("true", "1", "yes"),
    )
