import json

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.desert.api_client_log import (
    log_desert_get_http_error,
    log_desert_get_ok,
    log_desert_get_request_failed,
    log_desert_get_start,
)
from app.tools.desert.resolve import resolve_desert_base_and_token
from config.settings import Settings


class _EquipmentArgs(BaseModel):
    """Narrow tool args so the model does not over-parameterize."""

    note: str = Field(
        default="",
        description="Optional context from the user question (ignored for the HTTP call).",
    )


def build_list_equipment_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    async def _run(note: str = "") -> str:
        base, token = resolve_desert_base_and_token(
            settings, request_base=request_base, request_token=request_token
        )
        if not token:
            return (
                "error: no Desert API token "
                "(set DESERT_SERVICE_TOKEN or pass desert_api_token from Laravel)"
            )
        if not base:
            return (
                "error: no Desert API base URL. Laravel must send desert_api_base_url "
                "for the correct tenant."
            )
        path = "/fleet/equipment"
        url = f"{base}{path}"
        log_desert_get_start("desert_list_equipment", base, path)
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
                "desert_list_equipment", base, path, e.response.status_code, body
            )
            return (
                f"error: Desert API returned HTTP {e.response.status_code} for GET /fleet/equipment. "
                f"Details: {e!s}. Response (truncated): {body!r}"
            )
        except httpx.HTTPError as e:
            log_desert_get_request_failed("desert_list_equipment", base, path, str(e))
            return f"error calling Desert API: {e!s}"
        keys = list(data.keys()) if isinstance(data, dict) else ["<list>"]
        log_desert_get_ok("desert_list_equipment", base, path, r.status_code, keys)
        text = json.dumps(data, indent=2, default=str)
        if len(text) > 24_000:
            return text[:24_000] + "\n…(truncated)"
        return text

    return StructuredTool.from_function(
        name="desert_list_equipment",
        description="List all equipment for the current tenant via Desert GET /fleet/equipment.",
        args_schema=_EquipmentArgs,
        coroutine=_run,
    )
