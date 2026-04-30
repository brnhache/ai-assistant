"""High-level reconciliation helpers between FTM and QuickBooks.

Read-only helpers that:
- Resolve a QuickBooks customer by name/description via Desert.
- Fetch that customer's QBO invoices.
- Fetch field tickets (workorders) from Desert and filter by client name.
- Compare the two to show matches vs unmatched items.

This is intentionally conservative and best-effort: if we can't resolve a
customer cleanly or the APIs fail, we return an explanatory message
instead of guessing.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

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


class _ReconcileCustomerArgs(BaseModel):
    customer_phrase: str = Field(
        description=(
            "Customer name or phrase, e.g. 'Whitecap' or 'Baytex Energy'. "
            "Used to resolve the QuickBooks customer and filter field tickets."
        )
    )
    note: str = Field(
        default="",
        description="Optional extra context from the user (ignored for HTTP calls).",
    )


async def _get_json(
    *,
    base: str,
    token: str,
    path: str,
    tool_name: str,
) -> Dict[str, Any] | str:
    """Internal helper (slightly simplified variant of desert.qbo._get_json)."""

    log_desert_get_start(tool_name, base, path)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            r = await client.get(f"{base}{path}", headers=headers)
        r.raise_for_status()
        body: Any = r.json()
    except httpx.HTTPStatusError as e:
        snippet = (e.response.text or "")[:800]
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


def _normalize(s: str | None) -> str:
    return (s or "").strip().lower()


def _resolve_qbo_customer(
    phrase: str,
    customers: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Pick the best matching QBO customer for a phrase.

    Returns (match, close_candidates). match is None if ambiguous or none.
    """

    p = _normalize(phrase)
    if not p or not customers:
        return None, []

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for c in customers:
        name = c.get("name") or c.get("DisplayName") or c.get("CompanyName") or ""
        desc = c.get("description") or ""
        blob = f"{name} {desc}".lower()
        score = 0
        if p in blob:
            score += 10
        # crude token overlap
        for token in p.split():
            if token and token in blob:
                score += 1
        if score > 0:
            scored.append((score, c))

    if not scored:
        return None, []

    scored.sort(key=lambda t: t[0], reverse=True)
    top_score = scored[0][0]
    top = [c for s, c in scored if s == top_score]

    if len(top) == 1:
        return top[0], []

    # ambiguous: return the top few so the LLM can ask the user.
    candidates = [c for _, c in scored[:5]]
    return None, candidates


def build_qbo_reconcile_customer_tool(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> StructuredTool:
    """Reconcile a named customer between FTM field tickets and QBO invoices.

    Read-only: uses Desert APIs to fetch QBO customers + invoices and FTM
    field tickets, then compares them. Designed for questions like:
    "What else can you see in QBO about Whitecap?" or
    "For Baytex, which tickets have invoices and which don't?".
    """

    async def _run(customer_phrase: str, note: str = "") -> str:  # noqa: ARG001
        base, token = resolve_desert_base_and_token(
            settings, request_base=request_base, request_token=request_token
        )
        if not token:
            log_desert_tool_config_error(
                "desert_qbo_reconcile_customer",
                base,
                "missing Desert API bearer token",
            )
            return (
                "error: cannot reconcile because no Desert API token is "
                "configured for this request."
            )
        if not base:
            log_desert_tool_config_error(
                "desert_qbo_reconcile_customer",
                base,
                "missing Desert API base URL",
            )
            return (
                "error: cannot reconcile because no Desert API base URL was "
                "provided (missing desert_api_base_url)."
            )

        # 1) Check QBO connection.
        status = await _get_json(
            base=base,
            token=token,
            path="/workorder/qbo-connection-status",
            tool_name="desert_qbo_reconcile_customer",
        )
        if isinstance(status, str):
            return status
        if not bool(status.get("connected")):
            env = status.get("environment") or status.get("env") or "unknown"
            return (
                "QuickBooks is not connected for this user (environment: "
                f"{env}). I can't see any QBO data to reconcile."
            )

        # 2) Load QBO customers and resolve the phrase.
        customers_payload = await _get_json(
            base=base,
            token=token,
            path="/workorder/get-qbo-customers",
            tool_name="desert_qbo_reconcile_customer_customers",
        )
        if isinstance(customers_payload, str):
            return customers_payload
        customers = customers_payload.get("customers") or []
        if not isinstance(customers, list) or not customers:
            msg = customers_payload.get("message") or "No QuickBooks customers were returned."
            return msg

        match, candidates = _resolve_qbo_customer(customer_phrase, customers)
        if match is None and candidates:
            lines = [
                "I found multiple QuickBooks customers that could match "
                f"{customer_phrase!r}. Please choose one by name or id:",
            ]
            for c in candidates:
                cid = c.get("id") or c.get("Id")
                name = c.get("name") or c.get("DisplayName") or c.get("CompanyName")
                desc = c.get("description") or ""
                bits = []
                if cid is not None:
                    bits.append(f"id={cid}")
                if name:
                    bits.append(f"name={name}")
                if desc:
                    bits.append(f"description={desc}")
                if bits:
                    lines.append(" - " + ", ".join(bits))
            return "\n".join(lines)
        if match is None:
            return (
                f"I couldn't find any QuickBooks customer whose name "
                f"matches {customer_phrase!r}."
            )

        qbo_cust_id = match.get("id") or match.get("Id")
        qbo_cust_name = (
            match.get("name")
            or match.get("DisplayName")
            or match.get("CompanyName")
            or "(unknown)"
        )
        if not qbo_cust_id:
            return (
                "I found a potential QuickBooks customer match, but it had no "
                "Id field. I can't safely fetch invoices without that."
            )

        # 3) Fetch QBO invoices for that customer.
        invoices_payload = await _get_json(
            base=base,
            token=token,
            path=f"/workorder/qbo-invoices?customer_id={qbo_cust_id}",
            tool_name="desert_qbo_reconcile_customer_invoices",
        )
        if isinstance(invoices_payload, str):
            return invoices_payload
        invoices = invoices_payload.get("invoices") or []
        if not isinstance(invoices, list):
            invoices = []

        # Index invoices by doc_number for matching to field tickets.
        inv_by_doc: Dict[str, Dict[str, Any]] = {}
        for inv in invoices:
            doc = inv.get("doc_number") or inv.get("doc")
            if isinstance(doc, str) and doc:
                inv_by_doc[doc] = inv

        # 4) Fetch a page of field tickets and filter by client name containing the phrase.
        workorders_payload = await _get_json(
            base=base,
            token=token,
            path="/workorders?per_page=100&page=1",
            tool_name="desert_qbo_reconcile_customer_workorders",
        )
        if isinstance(workorders_payload, str):
            return workorders_payload
        items = workorders_payload.get("items") or []
        if not isinstance(items, list):
            items = []

        tickets_for_customer: List[Dict[str, Any]] = []
        norm_phrase = _normalize(customer_phrase)
        for wo in items:
            # Try multiple fields for client name depending on API shape.
            client_name = None
            client = wo.get("client") or {}
            org = client.get("organization") or {}
            if isinstance(org, dict) and org.get("name"):
                client_name = org.get("name")
            elif isinstance(client, dict) and client.get("name"):
                client_name = client.get("name")
            elif wo.get("client_name"):
                client_name = wo.get("client_name")

            if client_name and norm_phrase in _normalize(client_name):
                tickets_for_customer.append(wo)

        # 5) Compare.
        matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        tickets_without_invoice: List[Dict[str, Any]] = []
        invoices_without_ticket: Dict[str, Dict[str, Any]] = dict(inv_by_doc)

        for wo in tickets_for_customer:
            inv_num = wo.get("invoice_number") or wo.get("invoiceNumber")
            if isinstance(inv_num, str) and inv_num in inv_by_doc:
                matched.append((wo, inv_by_doc[inv_num]))
                invoices_without_ticket.pop(inv_num, None)
            else:
                tickets_without_invoice.append(wo)

        lines: List[str] = []
        lines.append(
            f"Reconciliation for customer {qbo_cust_name!r} (QuickBooks Id {qbo_cust_id}):"
        )
        lines.append("")
        lines.append(f"QBO invoices found: {len(invoices)}")
        lines.append(f"FTM field tickets matched to this customer (by client name): {len(tickets_for_customer)}")
        lines.append("")

        if matched:
            lines.append("Tickets with matching QBO invoices (by invoice number):")
            for wo, inv in matched:
                wid = wo.get("id")
                wnum = wo.get("workorder_number") or wo.get("number")
                inv_num = inv.get("doc_number") or inv.get("doc")
                total = inv.get("total")
                bits = []
                if wid is not None:
                    bits.append(f"ticket_id={wid}")
                if wnum:
                    bits.append(f"ticket_number={wnum}")
                if inv_num:
                    bits.append(f"invoice={inv_num}")
                if total is not None:
                    bits.append(f"total={total}")
                lines.append(" - " + ", ".join(bits))
            lines.append("")
        else:
            lines.append("No tickets with matching invoice numbers were found on this page.")
            lines.append("")

        if tickets_without_invoice:
            lines.append("Tickets for this customer with NO invoice number recorded:")
            for wo in tickets_without_invoice:
                wid = wo.get("id")
                wnum = wo.get("workorder_number") or wo.get("number")
                bits = []
                if wid is not None:
                    bits.append(f"ticket_id={wid}")
                if wnum:
                    bits.append(f"ticket_number={wnum}")
                lines.append(" - " + ", ".join(bits))
            lines.append("")

        if invoices_without_ticket:
            lines.append("QBO invoices for this customer with NO matching ticket on this page:")
            for inv in invoices_without_ticket.values():
                inv_num = inv.get("doc_number") or inv.get("doc")
                date = inv.get("txn_date") or inv.get("date")
                total = inv.get("total")
                bits = []
                if inv_num:
                    bits.append(f"invoice={inv_num}")
                if date:
                    bits.append(f"date={date}")
                if total is not None:
                    bits.append(f"total={total}")
                lines.append(" - " + ", ".join(bits))
        else:
            lines.append(
                "All QBO invoices for this customer on this query appear to have matching tickets on this page."
            )

        lines.append("")
        lines.append(
            "Note: this comparison only uses the first page of field tickets (per_page=100) "
            "and invoices returned by the current Desert/QBO endpoints. It is a best-effort "
            "summary, not an exhaustive reconciliation."
        )

        return "\n".join(lines)

    return StructuredTool.from_function(
        name="desert_qbo_reconcile_customer",
        description=(
            "Compare QuickBooks invoices and FTM field tickets for a named "
            "customer. Read-only helper for reconciliation questions like "
            "'what else do you see in QBO about Whitecap?'"
        ),
        args_schema=_ReconcileCustomerArgs,
        coroutine=_run,
    )
