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
        import sys

        base, token = resolve_desert_base_and_token(
            settings, request_base=request_base, request_token=request_token
        )
        print(
            "[desert.api] list_equipment resolve base=%s token_len=%s"
            % (base or "(empty)", len(token or "")),
            file=sys.stderr,
            flush=True,
        )
        if not token:
            log_desert_tool_config_error(
                "desert_list_equipment",
                base,
                "missing Desert API bearer token (no desert_api_token and DESERT_SERVICE_TOKEN unset)",
            )
            print(
                "[desert.api] list_equipment config_error: missing token (no desert_api_token and DESERT_SERVICE_TOKEN unset)",
                file=sys.stderr,
                flush=True,
            )
            return (
                "error: no Desert API token "
                "(set DESERT_SERVICE_TOKEN or pass desert_api_token from Laravel)"
            )
        if not base:
            log_desert_tool_config_error(
                "desert_list_equipment",
                base,
                "missing Desert API base URL (no desert_api_base_url and DESERT_API_BASE_URL empty)",
            )
            print(
                "[desert.api] list_equipment config_error: missing base URL (no desert_api_base_url and DESERT_API_BASE_URL empty)",
                file=sys.stderr,
                flush=True,
            )
            return (
                "error: no Desert API base URL. Laravel must send desert_api_base_url "
                "for the correct tenant."
            )
        path = "/fleet/equipment"
        url = f"{base}{path}"
        print("[desert.api] list_equipment GET %s" % url, file=sys.stderr, flush=True)
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
            print(
                "[desert.api] list_equipment http_error status=%s" % e.response.status_code,
                file=sys.stderr,
                flush=True,
            )
            return (
                f"error: Desert API returned HTTP {e.response.status_code} for GET /fleet/equipment. "
                f"Details: {e!s}. Response (truncated): {body!r}"
            )
        except httpx.HTTPError as e:
            log_desert_get_request_failed("desert_list_equipment", base, path, str(e))
            print(
                "[desert.api] list_equipment request_failed error=%s" % (str(e) or "(empty)"),
                file=sys.stderr,
                flush=True,
            )
            return f"error calling Desert API: {e!s}"
        keys = list(data.keys()) if isinstance(data, dict) else ["<list>"]
        log_desert_get_ok("desert_list_equipment", base, path, r.status_code, keys)
        print(
            "[desert.api] list_equipment ok status=%s keys=%s"
            % (r.status_code, ",".join(keys) if keys else "(none)"),
            file=sys.stderr,
            flush=True,
        )
        # Structured shape so the LLM quotes `total` instead of counting items.
        return shape_paginated(data, items_key="equipment")

    return StructuredTool.from_function(
        name="desert_list_equipment",
        description=(
            "List all equipment for the current tenant via Desert GET /fleet/equipment. "
            "Returns JSON with: total (authoritative count from server), showing, "
            "page, per_page, equipment (array). Use `total` for count questions \u2014 "
            "never count `equipment` by hand and never invent rows."
        ),
        args_schema=_EquipmentArgs,
        coroutine=_run,
    )
