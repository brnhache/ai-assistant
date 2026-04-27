import json

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


class _WorkordersArgs(BaseModel):
    note: str = Field(
        default="",
        description="Optional context from the user question (ignored for the HTTP call).",
    )


def build_list_workorders_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """request_* comes from Laravel each chat (tenant /api base + user token). Do not rely on env host."""

    async def _run(note: str = "") -> str:
        base, token = resolve_desert_base_and_token(
            settings, request_base=request_base, request_token=request_token
        )
        if not token:
            log_desert_tool_config_error(
                "desert_list_workorders",
                base,
                "missing Desert API bearer token (no desert_api_token and DESERT_SERVICE_TOKEN unset)",
            )
            return (
                "error: no Desert API token "
                "(set DESERT_SERVICE_TOKEN or pass desert_api_token from Laravel)"
            )
        if not base:
            log_desert_tool_config_error(
                "desert_list_workorders",
                base,
                "missing Desert API base URL (no desert_api_base_url and DESERT_API_BASE_URL empty)",
            )
            return (
                "error: no Desert API base URL. Laravel must send desert_api_base_url "
                "(e.g. https://your-tenant.app/api) for the correct tenant."
            )
        # Smaller page = faster response for the LLM; tenant can raise per_page up to 500 in Laravel
        path = "/workorders?per_page=100&page=1"
        url = f"{base}{path}"
        log_desert_get_start("desert_list_workorders", base, path)
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
            log_desert_get_http_error(
                "desert_list_workorders", base, path, e.response.status_code, body
            )
            return (
                f"error: Desert API returned HTTP {e.response.status_code} for GET /workorders. "
                f"Details: {e!s}. Response (truncated): {body!r}"
            )
        except httpx.HTTPError as e:
            log_desert_get_request_failed("desert_list_workorders", base, path, str(e))
            return f"error calling Desert API: {e!s}"
        keys = list(data.keys()) if isinstance(data, dict) else ["<list>"]
        log_desert_get_ok("desert_list_workorders", base, path, r.status_code, keys)
        text = json.dumps(data, indent=2, default=str)
        if len(text) > 24_000:
            return text[:24_000] + "\n…(truncated)"
        return text

    return StructuredTool.from_function(
        name="desert_list_workorders",
        description=(
            "List field tickets (workorders) for the tenant via Desert GET /workorders."
        ),
        args_schema=_WorkordersArgs,
        coroutine=_run,
    )
