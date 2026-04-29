"""HTTP client for the Desert AI long-term memory store.

The actual storage lives in the tenant DB on the Desert (Laravel) side, behind
three endpoints:

    GET   /api/ai-memories/relevant
    POST  /api/ai-memories
    PATCH /api/ai-memories/{id}/touch

This module exposes thin async helpers that wrap those calls so the rest of
the assistant code (chat handler + extractor chain) doesn't have to think
about HTTP.

Why HTTP and not direct DB access:

- The ai-assistant runs as a separate ECS service with no DB credentials.
- Multi-tenancy is enforced by the existing AiAssistantController flow:
  Laravel mints a per-request bearer token tied to a specific user, and the
  Python service uses that same token for every API call. The tenant DB
  connection is automatically scoped by stancl/tenancy on the Laravel side.
- Reusing the existing auth path means we don't introduce a second secret
  surface for the AI memory subsystem.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import httpx

from app.tools.desert.api_client_log import (
    log_desert_get_http_error,
    log_desert_get_ok,
    log_desert_get_request_failed,
    log_desert_get_start,
)


@dataclass
class Memory:
    """A single AI memory row, as returned by GET /api/ai-memories/relevant."""

    id: int
    kind: str
    visibility: str
    key: str | None
    content: str
    metadata: dict[str, Any] | None
    user_id: int | None
    is_pinned: bool
    relevance_score: float
    use_count: int
    last_used_at: str | None
    source_conversation_id: str | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            id=int(data.get("id") or 0),
            kind=str(data.get("kind") or ""),
            visibility=str(data.get("visibility") or "tenant_all"),
            key=data.get("key"),
            content=str(data.get("content") or ""),
            metadata=data.get("metadata")
            if isinstance(data.get("metadata"), dict)
            else None,
            user_id=(int(data["user_id"]) if data.get("user_id") is not None else None),
            is_pinned=bool(data.get("is_pinned") or False),
            relevance_score=float(data.get("relevance_score") or 0.0),
            use_count=int(data.get("use_count") or 0),
            last_used_at=data.get("last_used_at"),
            source_conversation_id=data.get("source_conversation_id"),
        )


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }


async def load_relevant_memories(
    base: str,
    token: str,
    *,
    user_id: int,
    user_role: str,
    kinds: list[str] | None = None,
    limit: int = 20,
) -> list[Memory]:
    """Fetch memories the given user is allowed to see, ranked by recency.

    Returns an empty list on any error (we never want memory failures to
    block the chat). Errors are logged for diagnostics.
    """
    tool_name = "ai_memory.load_relevant"
    if not base or not token:
        return []

    params: list[tuple[str, str]] = [
        ("user_id", str(user_id)),
        ("user_role", user_role),
        ("limit", str(limit)),
    ]
    for k in kinds or []:
        params.append(("kinds[]", k))

    path = "/ai-memories/relevant"
    log_desert_get_start(tool_name, base, path)
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(
                f"{base}{path}", params=params, headers=_headers(token)
            )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:1000]
        log_desert_get_http_error(tool_name, base, path, e.response.status_code, body)
        print(
            f"[ai_memory] load_relevant http_error status={e.response.status_code}",
            file=sys.stderr,
            flush=True,
        )
        return []
    except httpx.HTTPError as e:
        log_desert_get_request_failed(tool_name, base, path, str(e))
        print(
            f"[ai_memory] load_relevant request_failed error={e!s}",
            file=sys.stderr,
            flush=True,
        )
        return []

    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []

    log_desert_get_ok(tool_name, base, path, 200, [f"count={len(rows)}"])
    return [Memory.from_api(row) for row in rows if isinstance(row, dict)]


async def save_memory(
    base: str,
    token: str,
    *,
    kind: str,
    content: str,
    user_id: int | None = None,
    visibility: str = "tenant_all",
    key: str | None = None,
    metadata: dict[str, Any] | None = None,
    source_conversation_id: str | None = None,
    is_pinned: bool = False,
) -> int | None:
    """Upsert a memory. Returns the new (or updated) row id, or None on failure."""
    tool_name = "ai_memory.save"
    if not base or not token:
        return None

    payload: dict[str, Any] = {
        "kind": kind,
        "content": content,
        "visibility": visibility,
        "is_pinned": is_pinned,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    if key is not None:
        payload["key"] = key
    if metadata is not None:
        payload["metadata"] = metadata
    if source_conversation_id is not None:
        payload["source_conversation_id"] = source_conversation_id

    path = "/ai-memories"
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.post(
                f"{base}{path}", json=payload, headers=_headers(token)
            )
        r.raise_for_status()
        body = r.json()
    except httpx.HTTPStatusError as e:
        body_text = (e.response.text or "")[:1000]
        print(
            f"[ai_memory] save http_error status={e.response.status_code} body={body_text!r}",
            file=sys.stderr,
            flush=True,
        )
        return None
    except httpx.HTTPError as e:
        print(f"[ai_memory] save request_failed error={e!s}", file=sys.stderr, flush=True)
        return None

    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, dict) and data.get("id") is not None:
        try:
            return int(data["id"])
        except (TypeError, ValueError):
            return None
    return None


async def touch_memory(base: str, token: str, *, memory_id: int) -> bool:
    """Bump last_used_at + use_count on a memory. Returns True on success."""
    if not base or not token or memory_id <= 0:
        return False

    path = f"/ai-memories/{memory_id}/touch"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.patch(f"{base}{path}", headers=_headers(token))
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(
            f"[ai_memory] touch failed memory_id={memory_id} error={e!s}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def format_memories_for_prompt(memories: list[Memory]) -> str:
    """Render memories into a compact block to inject into the system prompt.

    The format intentionally exposes the memory id so the assistant *could*
    reference one in its reasoning, though we don't currently parse that
    back out.
    """
    if not memories:
        return ""

    lines: list[str] = []
    for m in memories:
        prefix = "📌 " if m.is_pinned else "• "
        kind_label = f"[{m.kind}]"
        scope_bits = []
        if m.user_id is not None:
            scope_bits.append(f"user_id={m.user_id}")
        scope_bits.append(f"visibility={m.visibility}")
        scope = " (" + ", ".join(scope_bits) + ")"
        lines.append(f"{prefix}{kind_label}{scope} {m.content}")

    return "\n".join(lines)
