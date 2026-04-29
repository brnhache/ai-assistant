"""Custom-forms tools for the Desert AI assistant.

Two tools are exposed:

- desert_list_custom_forms — what custom forms (Form definitions) exist for
  this tenant. Each result includes submissions_count.
- desert_list_form_submissions — submissions of a specific form (by name,
  slug, or numeric id). The Laravel route binding accepts all three.

Domain note (so this stays correct after future me reads it):

  Custom Forms (Form / FormSubmission models) are SEPARATE documents from
  Field Tickets. Submissions optionally link to a workorder via
  form_submissions.workorder_id, and a single field ticket usually has
  many form submissions (e.g. one ticket might have 5 pumpjack-inspection
  submissions linked).

  Do NOT confuse these with "workorder modules" (WorkorderModuleConfig /
  WorkorderModuleData) which are configurable JSON sections embedded
  inside the workorder itself.

  See ~/.openclaw/workspace/memory/desert-domain.md for the full mental
  model.
"""

from __future__ import annotations

import sys

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
from app.tools.desert.shape import shape_paginated
from config.settings import Settings


class _ListFormsArgs(BaseModel):
    note: str = Field(
        default="",
        description="Optional context from the user question (ignored for the HTTP call).",
    )


class _ListFormSubmissionsArgs(BaseModel):
    form: str = Field(
        ...,
        description=(
            "Form identifier: numeric id, slug, or exact name. The server accepts "
            "all three; prefer slug or name (e.g. 'pumpjack-inspection' or 'hazard_assessment') if known."
        ),
    )
    per_page: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Submissions per page (1-200, default 50).",
    )
    page: int = Field(default=1, ge=1, description="1-indexed page number.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_or_error(
    settings: Settings,
    request_base: str | None,
    request_token: str | None,
    tool_name: str,
) -> tuple[str | None, str | None, str | None]:
    """Return (base, token, error_or_none). On error, log + return a user-facing string."""
    base, token = resolve_desert_base_and_token(
        settings, request_base=request_base, request_token=request_token
    )
    print(
        f"[desert.api] {tool_name} resolve base={base or '(empty)'} token_len={len(token or '')}",
        file=sys.stderr,
        flush=True,
    )
    if not token:
        log_desert_tool_config_error(
            tool_name,
            base,
            "missing Desert API bearer token (no desert_api_token and DESERT_SERVICE_TOKEN unset)",
        )
        return None, None, (
            "error: no Desert API token "
            "(set DESERT_SERVICE_TOKEN or pass desert_api_token from Laravel)"
        )
    if not base:
        log_desert_tool_config_error(
            tool_name,
            base,
            "missing Desert API base URL",
        )
        return None, None, (
            "error: no Desert API base URL. Laravel must send desert_api_base_url "
            "(e.g. https://your-tenant.app/api) for the correct tenant."
        )
    return base, token, None


async def _http_get_json(
    base: str, path: str, token: str, tool_name: str
) -> object:
    """Issue an authenticated GET and return parsed JSON, or a string starting with 'error:'."""
    url = f"{base}{path}"
    print(f"[desert.api] {tool_name} GET {url}", file=sys.stderr, flush=True)
    log_desert_get_start(tool_name, base, path)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:3000]
        log_desert_get_http_error(tool_name, base, path, e.response.status_code, body)
        print(
            f"[desert.api] {tool_name} http_error status={e.response.status_code}",
            file=sys.stderr,
            flush=True,
        )
        return (
            f"error: Desert API returned HTTP {e.response.status_code} for GET {path}. "
            f"Details: {e!s}. Response (truncated): {body!r}"
        )
    except httpx.HTTPError as e:
        log_desert_get_request_failed(tool_name, base, path, str(e))
        print(
            f"[desert.api] {tool_name} request_failed error={str(e) or '(empty)'}",
            file=sys.stderr,
            flush=True,
        )
        return f"error calling Desert API: {e!s}"

    keys = list(data.keys()) if isinstance(data, dict) else ["<list>"]
    log_desert_get_ok(tool_name, base, path, r.status_code, keys)
    print(
        f"[desert.api] {tool_name} ok status={r.status_code} keys={','.join(keys) if keys else '(none)'}",
        file=sys.stderr,
        flush=True,
    )
    return data


# ---------------------------------------------------------------------------
# Name/slug resolution helpers (for LLM-friendly matching)
# ---------------------------------------------------------------------------


def _normalize_form_phrase(text: str) -> tuple[str, list[str]]:
    """Normalize a human phrase or form name/slug into a token list.

    Lowercases, normalizes pump jack → pumpjack, replaces dashes/underscores
    with spaces, strips punctuation, and splits on whitespace.
    """
    import re

    t = (text or "").lower()
    # Normalize common domain-specific variants first.
    t = t.replace("pump jack", "pumpjack")
    # Replace underscores/dashes with spaces to align slugs and names.
    t = re.sub(r"[_-]+", " ", t)
    # Drop non-alphanumeric chars except space.
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # Collapse whitespace and split.
    tokens = [tok for tok in t.split() if tok]
    return t.strip(), tokens


def _score_form_match(query_tokens: list[str], form_tokens: list[str]) -> float:
    """Return a heuristic similarity score between 0 and 1.

    - Base score from token overlap.
    - Bonuses for core domain tokens (pumpjack, hazard, inspection, installation, flha, worksite).
    - Small penalty when query asks for installation but form only mentions inspection (and vice versa).
    """
    if not query_tokens or not form_tokens:
        return 0.0

    qset = set(query_tokens)
    fset = set(form_tokens)
    shared = qset & fset
    base = len(shared) / max(len(qset), 1)

    score = base

    core_tokens = {"pumpjack", "hazard", "inspection", "install", "installation", "flha", "worksite"}
    core_shared = core_tokens & shared
    if core_shared:
        score += 0.15 * len(core_shared)

    # Distinguish installation vs inspection questions a bit.
    q_has_install = any(tok.startswith("install") for tok in qset)
    q_has_inspect = any(tok.startswith("inspect") for tok in qset)
    f_has_install = any(tok.startswith("install") for tok in fset)
    f_has_inspect = any(tok.startswith("inspect") for tok in fset)

    if q_has_install and f_has_inspect and not f_has_install:
        score -= 0.15
    if q_has_inspect and f_has_install and not f_has_inspect:
        score -= 0.15

    # Clamp to [0,1].
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score


class _ResolveFormArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language phrase the user used to refer to a custom form, "
            "such as 'pumpjack installations', 'hazard assessments', or "
            "'those FLHA forms'."
        ),
    )
    max_candidates: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Maximum number of candidate forms to return, sorted by relevance "
            "score (highest first)."
        ),
    )

# ---------------------------------------------------------------------------
# Public tool builders
# ---------------------------------------------------------------------------


def build_list_custom_forms_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """List Form definitions (custom forms) for the current tenant.

    Calls GET /api/forms which returns a flat list (no pagination) wrapped
    as {status, data: [...]}. Each form includes a `submissions_count`
    populated via withCount('submissions').
    """

    async def _run(note: str = "") -> str:
        tool_name = "desert_list_custom_forms"
        base, token, err = _resolve_or_error(
            settings, request_base, request_token, tool_name
        )
        if err is not None:
            return err

        path = "/forms"
        data = await _http_get_json(base, path, token, tool_name)
        if isinstance(data, str):  # error string from _http_get_json
            return data

        return shape_paginated(data, items_key="forms")

    return StructuredTool.from_function(
        name="desert_list_custom_forms",
        description=(
            "List custom-form DEFINITIONS for this tenant via Desert GET /forms. "
            "Returns JSON with: total (count of distinct form definitions), "
            "items_key=`forms` (each entry has id, name, slug, submissions_count, "
            "is_active, expose_to_clients, config). Use this to answer 'what "
            "custom forms exist?' or 'how many submissions does the X form "
            "have?' (read submissions_count). "
            "IMPORTANT: custom forms are DIFFERENT from workorder modules. "
            "Workorder modules (e.g. pumpjack_information) are JSON sections "
            "embedded INSIDE a field ticket; custom forms (e.g. "
            "pumpjack-inspection) are separate documents that may be linked to "
            "a field ticket via workorder_id. Don't conflate them."
        ),
        args_schema=_ListFormsArgs,
        coroutine=_run,
    )


def build_list_form_submissions_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """List submissions of a specific custom form.

    Calls GET /api/forms/{form}/submissions — `{form}` accepts numeric id,
    slug, or exact name (RouteServiceProvider binds it). Response is
    paginated; meta lives in the `meta` envelope, which shape_paginated
    handles.
    """

    async def _run(form: str, per_page: int = 50, page: int = 1) -> str:
        tool_name = "desert_list_form_submissions"
        base, token, err = _resolve_or_error(
            settings, request_base, request_token, tool_name
        )
        if err is not None:
            return err

        # Don't trust the LLM to URL-encode. Even a literal form_name with
        # underscores or hyphens passes through fine, but spaces/&/? would
        # break. Use httpx.URL for safety.
        from urllib.parse import quote

        form_path = quote(str(form), safe="")
        path = f"/forms/{form_path}/submissions?per_page={per_page}&page={page}"
        data = await _http_get_json(base, path, token, tool_name)
        if isinstance(data, str):
            return data

        return shape_paginated(data, items_key="submissions")

    return StructuredTool.from_function(
        name="desert_list_form_submissions",
        description=(
            "List SUBMISSIONS of a specific custom form via Desert "
            "GET /forms/{form}/submissions. Args: form (id, slug, or name; "
            "e.g. 'pumpjack-inspection' or 'hazard_assessment'), per_page (default 50, max 200), "
            "page (default 1). Returns JSON with: total (authoritative count "
            "of submissions for this form, possibly across multiple pages), "
            "showing (rows in THIS page), page, per_page, last_page, "
            "items_key=`submissions` (each has id, form_id, workorder_id, "
            "data (the form fields), created_by_name, created_at). "
            "Use this to answer 'how many pumpjack inspections are there?' "
            "(quote `total`) or 'list the recent hazard assessments'. "
            "If total > showing, fetch additional pages with `page=2` etc. "
            "before claiming you've seen them all."
        ),
        args_schema=_ListFormSubmissionsArgs,
        coroutine=_run,
    )


def build_resolve_form_name_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """Resolve a natural-language form phrase to likely custom-form candidates.

    This helper is designed to bridge between how humans talk about forms
    ("pumpjack installations", "hazard assessments", "those FLHA forms") and
    the actual `name`/`slug` fields on Form records. It always looks at the
    real forms list first (GET /forms) and returns a scored candidate list.

    The LLM should use this to decide which form id/slug to pass to
    `desert_list_form_submissions` instead of guessing a slug directly.
    """

    async def _run(query: str, max_candidates: int = 5) -> str:
        tool_name = "desert_resolve_form_name"
        base, token, err = _resolve_or_error(
            settings, request_base, request_token, tool_name
        )
        if err is not None:
            return err

        path = "/forms"
        data = await _http_get_json(base, path, token, tool_name)
        if isinstance(data, str):
            return data

        forms = []
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            forms = data["data"]
        elif isinstance(data, list):
            forms = data

        norm_query, q_tokens = _normalize_form_phrase(query)
        scored = []
        for f in forms:
            name = f.get("name") or ""
            slug = f.get("slug") or ""
            norm_name, name_tokens = _normalize_form_phrase(name)
            norm_slug, slug_tokens = _normalize_form_phrase(slug)
            tokens = list({*name_tokens, *slug_tokens})
            score = _score_form_match(q_tokens, tokens)
            scored.append(
                {
                    "id": f.get("id"),
                    "name": name,
                    "slug": slug,
                    "submissions_count": f.get("submissions_count"),
                    "is_active": f.get("is_active"),
                    "expose_to_clients": f.get("expose_to_clients"),
                    "score": round(score, 4),
                }
            )

        # Sort by score desc and keep top N with score > 0.
        scored.sort(key=lambda x: x["score"], reverse=True)
        candidates = [c for c in scored if c["score"] > 0][:max_candidates]

        import json as _json

        return _json.dumps(
            {
                "query": query,
                "normalized_query": norm_query,
                "candidates": candidates,
            },
            indent=2,
            default=str,
        )

    return StructuredTool.from_function(
        name="desert_resolve_form_name",
        description=(
            "Given a natural-language phrase like 'pumpjack installations' or "
            "'hazard assessments', look at the tenant's actual custom forms "
            "list (GET /forms) and return likely matches sorted by relevance. "
            "Use this to decide which form id/slug to pass to "
            "desert_list_form_submissions. If no candidates are returned, "
            "tell the user you don't see any forms that look like that phrase "
            "and show them what forms DO exist instead of assuming 0."
        ),
        args_schema=_ResolveFormArgs,
        coroutine=_run,
    )
