"""Tools for working with Desert/FTM AI assistant conversations.

Layer 1: fetch recent conversation turns for the current user so the agent
can "recall" what was said earlier in the thread. This is a thin wrapper
around Laravel's /api/ai-assistant/history endpoint.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.desert.api_client_log import (
    log_desert_get_http_error,
    log_desert_get_ok,
    log_desert_get_request_failed,
    log_desert_get_start,
    log_desert_tool_config_error,
)
from app.tools.desert.resolve import resolve_desert_base_and_token
from config.settings import Settings


class _RecallArgs(BaseModel):
    """Arguments for the recall tool.

    The user question itself stays in the main agent input; this tool just
    controls how many turns to fetch and whether to target a specific
    conversation id.
    """

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of recent turns to retrieve (1-100).",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional stable conversation id. If omitted, the most recent "
            "conversation for this user is used."
        ),
    )


def build_recall_conversation_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """Create a tool that fetches recent conversation turns from Desert.

    This does *not* do vector search yet; it simply returns the last N turns
    so the agent can answer questions like "what did I tell you earlier" or
    "summarize our last few messages" from an authoritative server copy.
    """

    async def _run(limit: int = 20, conversation_id: str | None = None) -> str:
        base, token = resolve_desert_base_and_token(
            settings, request_base=request_base, request_token=request_token
        )
        if not token:
            log_desert_tool_config_error(
                "desert_ai_recall", base, "missing Desert API bearer token"
            )
            return (
                "error: cannot recall conversation history because no Desert "
                "API token is configured."
            )
        if not base:
            log_desert_tool_config_error(
                "desert_ai_recall", base, "missing Desert API base URL"
            )
            return (
                "error: cannot recall conversation history because no Desert "
                "API base URL was provided."
            )

        params: list[tuple[str, str]] = [("limit", str(limit))]
        if conversation_id:
            params.append(("conversation_id", conversation_id))

        path = "/ai-assistant/history"
        url = f"{base}{path}"
        log_desert_get_start("desert_ai_recall", base, path)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            body: Any = r.json()
        except httpx.HTTPStatusError as e:
            snippet = (e.response.text or "")[:1000]
            log_desert_get_http_error(
                "desert_ai_recall", base, path, e.response.status_code, snippet
            )
            print(
                f"[desert.api] ai_recall http_error status={e.response.status_code}",
                file=sys.stderr,
                flush=True,
            )
            return (
                f"error: Desert API returned HTTP {e.response.status_code} for "
                f"GET {path}. Response snippet: {snippet!r}"
            )
        except httpx.HTTPError as e:
            log_desert_get_request_failed("desert_ai_recall", base, path, str(e))
            print(
                f"[desert.api] ai_recall request_failed error={e!s}",
                file=sys.stderr,
                flush=True,
            )
            return f"error calling Desert API for AI history: {e!s}"

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            return "error: Desert AI history endpoint returned an unexpected payload."

        conv_id = data.get("conversation_id")
        turns = data.get("turns") or []
        log_desert_get_ok(
            "desert_ai_recall", base, path, 200, ["conversation_id", "turns"]
        )

        if not turns:
            return "No prior AI assistant messages were found for this user."

        lines: list[str] = []
        lines.append(
            "Recent AI assistant conversation turns (oldest to newest):"
        )
        if conv_id:
            lines.append(f"conversation_id: {conv_id}")
        for t in turns:
            role = (t.get("role") or "").lower()
            content = t.get("content") or ""
            if not content:
                continue
            if role not in {"user", "assistant"}:
                role = "assistant" if role == "tool" else role or "assistant"
            prefix = "USER" if role == "user" else "ASSISTANT"
            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

    return StructuredTool.from_function(
        name="desert_ai_recall_conversation",
        description=(
            "Recall recent turns from the user's AI assistant conversation via "
            "Desert /api/ai-assistant/history. Use this when the user asks you "
            "to summarize what you've talked about recently, or refers to "
            "earlier parts of the conversation and you need an authoritative "
            "server-side history."
        ),
        args_schema=_RecallArgs,
        coroutine=_run,
    )
