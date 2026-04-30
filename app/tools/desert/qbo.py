"""Tools for interacting with QuickBooks Online via Desert's API.

These are **read-only** helpers that rely on the existing Laravel
QuickBooks integration (WorkorderController + Quickbooks trait). The
Python service never talks to QBO directly; it always goes through the
same tenant-scoped API the SPA uses.
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


class _QboConnectionArgs(BaseModel):
    note: str = Field(
        default="",
        description=(
            "Optional context from the user question (ignored for the HTTP call)."
        ),
    )


class _QboCustomersArgs(BaseModel):
    note: str = Field(
        default="",
        description=(
            "Optional context from the user question (ignored for the HTTP call)."
        ),
    )


async def _get_json(
    *,
    settings: Settings,
    request_base: str | None,
    request_token: str | None,
    path: str,
    tool_name: str,
) -> dict[str, Any] | str:
    """Shared helper for simple GET endpoints. Returns dict on success or
    a human-readable error string on failure.
    """

    base, token = resolve_desert_base_and_token(
        settings, request_base=request_base, request_token=request_token
    )
    if not token:
        log_desert_tool_config_error(
            tool_name,
            base,
            "missing Desert API bearer token (no desert_api_token and DESERT_SERVICE_TOKEN unset)",
        )
        return (
            "error: cannot reach QuickBooks because no Desert API token is "
            "configured for this request."
        )
    if not base:
        log_desert_tool_config_error(
            tool_name,
            base,
            "missing Desert API base URL (no desert_api_base_url and DESERT_API_BASE_URL empty)",
        )
        return (
            "error: cannot reach QuickBooks because no Desert API base URL "
            "was provided (missing desert_api_base_url)."
        )

    url = f"{base}{path}"
    log_desert_get_start(tool_name, base, path)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        r.raise_for_status()
        body: Any = r.json()
    except httpx.HTTPStatusError as e:
        snippet = (e.response.text or "")[:1000]
        log_desert_get_http_error(tool_name, base, path, e.response.status_code, snippet)
        print(
            f"[desert.api] {tool_name} http_error status={e.response.status_code}",
            file=sys.stderr,
            flush=True,
        )
        return (
            f"error: Desert API returned HTTP {e.response.status_code} for GET {path}. "
            f"Response snippet: {snippet!r}"
        )
    except httpx.HTTPError as e:
        log_desert_get_request_failed(tool_name, base, path, str(e))
        print(
            f"[desert.api] {tool_name} request_failed error={e!s}",
            file=sys.stderr,
            flush=True,
        )
        return f"error calling Desert API: {e!s}"

    data = body if isinstance(body, dict) else {}
    log_desert_get_ok(tool_name, base, path, 200, list(data.keys()))
    return data


def build_qbo_connection_status_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """Check QuickBooks Online connection status for the current user.

    Wraps GET /api/workorder/qbo-connection-status. Useful for answering
    questions like "am I connected to QuickBooks?" or "can you see my
    QuickBooks data?".
    """

    async def _run(note: str = "") -> str:  # noqa: ARG001
        data = await _get_json(
            settings=settings,
            request_base=request_base,
            request_token=request_token,
            path="/workorder/qbo-connection-status",
            tool_name="desert_qbo_connection_status",
        )
        if isinstance(data, str):
            return data

        # Shape the status into a compact, human-readable summary.
        connected = bool(data.get("connected"))
        env = data.get("environment") or data.get("env") or "unknown"
        company = data.get("company_name") or data.get("realm_id") or "unknown"
        last_sync = data.get("last_sync_at") or data.get("last_sync")

        parts = []
        parts.append(f"connected: {'yes' if connected else 'no'}")
        parts.append(f"environment: {env}")
        parts.append(f"company: {company}")
        if last_sync:
            parts.append(f"last_sync_at: {last_sync}")

        return "QuickBooks connection status (current user): " + "; ".join(parts)

    return StructuredTool.from_function(
        name="desert_qbo_connection_status",
        description=(
            "Check whether QuickBooks Online is connected for this user and "
            "which company/environment is linked. Use this before attempting "
            "to answer questions that depend on live QuickBooks data."
        ),
        args_schema=_QboConnectionArgs,
        coroutine=_run,
    )


def build_qbo_list_customers_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """List QuickBooks customers via Desert.

    Wraps GET /api/workorder/get-qbo-customers. Returns a concise table of
    customer names + ids so the agent can answer questions like "which
    customers are synced to QuickBooks?" or resolve a name to an id before
    more specific QBO queries.
    """

    async def _run(note: str = "") -> str:  # noqa: ARG001
        data = await _get_json(
            settings=settings,
            request_base=request_base,
            request_token=request_token,
            path="/workorder/get-qbo-customers",
            tool_name="desert_qbo_list_customers",
        )
        if isinstance(data, str):
            return data

        customers = data.get("customers")
        if not isinstance(customers, list) or not customers:
            message = data.get("message") or "No QuickBooks customers found."
            return message

        lines: list[str] = []
        lines.append("QuickBooks customers (from Desert):")
        for c in customers:
            # The exact shape depends on the QuickBooks SDK; be defensive. In
            # our Laravel trait (Quickbooks::quickbooksGetCustomers) we return
            # a simplified array with keys: name, id, description. Fall back
            # through the richer SDK field names as well just in case.
            name = (
                c.get("name")
                or c.get("DisplayName")
                or c.get("CompanyName")
                or c.get("FullyQualifiedName")
                or c.get("Name")
            )
            desc = c.get("description")
            cid = c.get("Id") or c.get("id")
            email = None
            primary_email = c.get("PrimaryEmailAddr") or {}
            if isinstance(primary_email, dict):
                email = primary_email.get("Address") or primary_email.get("address")
            line_bits = []
            if cid is not None:
                line_bits.append(f"id={cid}")
            if name:
                line_bits.append(f"name={name}")
            if desc:
                line_bits.append(f"description={desc}")
            if email:
                line_bits.append(f"email={email}")
            if line_bits:
                lines.append(" - " + ", ".join(line_bits))

        return "\n".join(lines)

    return StructuredTool.from_function(
        name="desert_qbo_list_customers",
        description=(
            "List customers from QuickBooks Online via Desert's existing "
            "integration. Useful when reconciling field tickets to QBO or "
            "when the user asks which customers are synced."
        ),
        args_schema=_QboCustomersArgs,
        coroutine=_run,
    )
