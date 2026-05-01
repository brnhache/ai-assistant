"""Tools for AI-generated report exports.

These tools let the Desert AI assistant hand a tabular structure (columns + rows)
back to Laravel so it can generate a downloadable CSV/PDF on the tenant's disk.

The assistant is responsible for:
- calling other Desert tools to fetch the underlying data from the DB,
- analysing / grouping / ranking it,
- deciding which columns and rows belong in the final report.

This module only wraps the POST /api/ai-reports/export endpoint.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.desert.api_client_log import (
    log_desert_get_request_failed,
    log_desert_tool_config_error,
)
from app.tools.desert.resolve import resolve_desert_base_and_token
from config.settings import Settings


class _ExportAiReportArgs(BaseModel):
    title: str = Field(
        default="AI Report",
        description="Short title for the report (used in the filename and inside the PDF).",
    )
    format: str = Field(
        default="csv",
        description="Desired output format: csv, xlsx (treated as Excel-friendly CSV), or pdf.",
    )
    columns: List[Dict[str, str]] = Field(
        ...,
        description=(
            "Ordered list of column definitions. Each entry must have 'key' (field name) and "
            "'header' (human label). The keys must match the keys used in each row object."
        ),
    )
    rows: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Tabular rows for the report. Each row is a dict keyed by the column 'key' values. "
            "Values should be scalars or short strings; arrays will be JSON-encoded."
        ),
    )


def _resolve_or_error(
    settings: Settings,
    request_base: str | None,
    request_token: str | None,
    tool_name: str,
) -> tuple[str | None, str | None, str | None]:
    """Return (base, token, error_or_none)."""

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


async def _http_post_json(base: str, path: str, token: str, tool_name: str, payload: Dict[str, Any]) -> Any:
    """Issue an authenticated POST and return parsed JSON, or an error string."""

    url = f"{base}{path}"
    print(f"[desert.api] {tool_name} POST {url}", file=sys.stderr, flush=True)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:2000]
        log_desert_get_request_failed(tool_name, base, path, f"http_status={e.response.status_code} body={body!r}")
        print(
            f"[desert.api] {tool_name} http_error status={e.response.status_code}",
            file=sys.stderr,
            flush=True,
        )
        return (
            f"error: Desert API returned HTTP {e.response.status_code} for POST {path}. "
            f"Response (truncated): {body!r}"
        )
    except httpx.HTTPError as e:
        log_desert_get_request_failed(tool_name, base, path, str(e))
        print(
            f"[desert.api] {tool_name} request_failed error={e!s}",
            file=sys.stderr,
            flush=True,
        )
        return f"error calling Desert API: {e!s}"


# ---------------------------------------------------------------------------
# Public tool builder
# ---------------------------------------------------------------------------


def build_export_ai_report_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """Create an AI-generated report file from tabular data.

    The assistant should call other Desert tools first to fetch and analyse
    data, then assemble a columns[] + rows[] payload and call this tool when
    the user wants a downloadable CSV/XLSX/PDF.
    """

    tool_name = "desert_export_ai_report"

    async def _run(args: _ExportAiReportArgs) -> Any:
        base, token, err = _resolve_or_error(
            settings,
            request_base=request_base,
            request_token=request_token,
            tool_name=tool_name,
        )
        if err is not None:
            return err
        assert base and token

        payload: Dict[str, Any] = {
            "title": args.title,
            "format": args.format,
            "columns": args.columns,
            "rows": args.rows,
        }
        data = await _http_post_json(base, "/ai-reports/export", token, tool_name, payload)
        return data

    return StructuredTool.from_function(
        name=tool_name,
        description=(
            "Generate a downloadable report file (CSV/XLSX/PDF) from analysed data. "
            "Use this ONLY after you have already fetched and processed the relevant "
            "data using other Desert tools. The columns and rows you pass here will "
            "appear exactly in the exported file."
        ),
        func=_run,
        args_schema=_ExportAiReportArgs,
        coroutine=_run,
    )
