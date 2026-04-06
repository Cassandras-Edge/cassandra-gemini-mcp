"""Gemini Interactions API client.

Wraps the stateful Interactions API with Google Search grounding and URL context.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"


class GeminiClient:
    """HTTP client for the Gemini Interactions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._http = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def create_interaction(
        self,
        input_content: str | list,
        thinking_level: Literal["minimal", "low", "medium", "high"] = "medium",
        previous_interaction_id: Optional[str] = None,
        max_tokens: int = 8192,
        system_instruction: Optional[str] = None,
    ) -> dict:
        """Create an interaction. Returns parsed response with text, sources, interaction_id."""
        payload: dict = {
            "model": self._model,
            "input": input_content,
            "store": True,
            "generation_config": {
                "thinking_level": thinking_level,
                "max_output_tokens": max_tokens,
            },
            "tools": [
                {"type": "google_search"},
                {"type": "url_context"},
            ],
        }

        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id

        if system_instruction:
            payload["system_instruction"] = system_instruction

        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            response = await self._http.post(
                INTERACTIONS_ENDPOINT, json=payload, headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return _parse_interaction_response(data)
        except httpx.HTTPStatusError as e:
            return {
                "error": f"API error: {e.response.status_code} - {e.response.text}",
                "interaction_id": None,
                "status": "failed",
            }
        except Exception as e:
            return {
                "error": f"Request failed: {str(e)}",
                "interaction_id": None,
                "status": "failed",
            }


def _parse_interaction_response(data: dict) -> dict:
    """Parse the interaction response into a structured format."""
    result: dict = {
        "interaction_id": data.get("id"),
        "status": data.get("status"),
        "text": "",
        "sources": [],
        "usage": data.get("usage", {}),
    }

    for output in data.get("outputs", []):
        output_type = output.get("type")

        if output_type == "text":
            result["text"] += output.get("text", "")
            for ann in output.get("annotations", []):
                source = ann.get("source")
                if source and source not in result["sources"]:
                    result["sources"].append(source)

        elif output_type == "google_search_result":
            for item in output.get("result", []):
                source = {"url": item.get("url"), "title": item.get("title")}
                if source["url"] and source not in result["sources"]:
                    result["sources"].append(source)

        elif output_type == "url_context_result":
            for item in output.get("result", []):
                if item.get("status") == "success":
                    source = {"url": item.get("url"), "title": "URL Context"}
                    if source["url"] and source not in result["sources"]:
                        result["sources"].append(source)

    return result


def _resolve_redirect_url(url: str) -> str:
    """Resolve Google's grounding redirect URLs to actual source URLs."""
    if not url or "vertexaisearch.cloud.google.com/grounding-api-redirect" not in url:
        return url
    try:
        with httpx.Client(timeout=5.0, follow_redirects=False) as client:
            response = client.head(url)
            if response.status_code in (301, 302, 303, 307, 308):
                return response.headers.get("location", url)
    except Exception:
        pass
    return url


def resolve_all_urls(sources: list) -> list:
    """Resolve all redirect URLs in parallel."""
    urls_to_resolve = []
    for source in sources:
        url = source.get("url", "") if isinstance(source, dict) else source
        if url and "vertexaisearch.cloud.google.com/grounding-api-redirect" in url:
            urls_to_resolve.append(url)

    if not urls_to_resolve:
        return sources

    url_map: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_resolve_redirect_url, u): u for u in urls_to_resolve}
        for future in as_completed(futures):
            original_url = futures[future]
            try:
                url_map[original_url] = future.result()
            except Exception:
                url_map[original_url] = original_url

    resolved = []
    for source in sources:
        if isinstance(source, dict):
            url = source.get("url", "")
            resolved.append({"title": source.get("title", "Untitled"), "url": url_map.get(url, url)})
        else:
            resolved.append(url_map.get(source, source))
    return resolved


def resolve_text_urls(text: str) -> str:
    """Resolve redirect URLs embedded in the response text."""
    redirect_pattern = (
        r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^\s\)\]\"\'<>]+"
    )
    urls = list(set(re.findall(redirect_pattern, text)))
    if not urls:
        return text

    url_map: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_resolve_redirect_url, u): u for u in urls}
        for future in as_completed(futures):
            original = futures[future]
            try:
                url_map[original] = future.result()
            except Exception:
                url_map[original] = original

    for original, resolved in url_map.items():
        text = text.replace(original, resolved)
    return text


def format_response(result: dict) -> str:
    """Format the parsed result into a readable string."""
    if "error" in result:
        return f"Error: {result['error']}"

    output = [resolve_text_urls(result.get("text", ""))]

    sources = result.get("sources", [])
    if sources:
        resolved_sources = resolve_all_urls(sources)
        output.append("\n\nSources:")
        for i, source in enumerate(resolved_sources, 1):
            if isinstance(source, dict):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                output.append(f"{i}. [{title}]({url})")
            else:
                output.append(f"{i}. {source}")

    output.append("\n---")
    output.append(f"To follow up, use interaction_id: {result.get('interaction_id', 'N/A')}")

    return "\n".join(output)
