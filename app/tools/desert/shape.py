"""Shape Desert/Laravel responses into a count-stable structure for LLM tools.

Laravel paginators return {data, total, per_page, current_page, last_page, ...}.
The LLM struggles to count `data` items by hand and tends to confabulate counts
or invent rows when its guess and the visible items disagree (we got bitten by
gpt-4o-mini fabricating FT09999 / FT10000 to retrofit a wrong count).

This helper:
- Always exposes an authoritative `total` field at the top of the JSON object.
- Renames `data` -> `items` so the LLM doesn't try to count `data` itself.
- Preserves pagination meta (`page`, `per_page`, `last_page`).
- Falls back gracefully for raw-list and unknown response shapes.
- Truncates items if the response is too big, BUT keeps `total` accurate and
  flags the truncation explicitly so the LLM can be honest about it.
"""

from __future__ import annotations

import json

# Maximum size of the JSON we hand back to the LLM. Big enough for ~100 ticket
# rows with full nested client/employee data; small enough to keep latency
# reasonable. If a tenant has thousands of items we'll truncate and tell the
# LLM we did.
_MAX_BYTES = 96_000


def shape_paginated(data: object, *, items_key: str = "items") -> str:
    """Return a JSON string with stable count-fields for the LLM.

    Args:
        data: parsed JSON from the Desert API call.
        items_key: name to use for the rows array in the output (default "items").
                   We deliberately rename `data` -> `items` to discourage the
                   LLM from counting `data` by eye.

    Returns:
        A pretty-printed JSON string. May be truncated if the full response
        exceeds _MAX_BYTES; truncation is reflected in the returned object via
        `items_truncated: true` and `items_shown_in_response: N`. The `total`
        field always reflects the SERVER total, never the truncated count.
    """
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
        # Some endpoints (e.g. /api/workorders) put pagination at the top level.
        # Others (e.g. /api/forms/{form}/submissions) wrap it in `meta`. Try
        # top-level first, then meta. Treat what's not paginated as a flat list.
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

        def _pick(key: str):
            v = data.get(key)
            if v is None:
                v = meta.get(key)
            return v

        total_raw = _pick("total")
        total_known = isinstance(total_raw, int)
        out: dict = {
            "total": total_raw if total_known else len(items),
            "total_known": total_known,
            "showing": len(items),
            "page": _pick("current_page"),
            "per_page": _pick("per_page"),
            "last_page": _pick("last_page"),
            items_key: items,
        }
    elif isinstance(data, list):
        out = {
            "total": len(data),
            "total_known": True,
            "showing": len(data),
            "page": 1,
            "per_page": len(data),
            "last_page": 1,
            items_key: data,
        }
    else:
        # Unknown shape — hand it back as-is rather than mangling.
        return json.dumps({"raw": data}, indent=2, default=str)[:_MAX_BYTES]

    text = json.dumps(out, indent=2, default=str)
    if len(text) <= _MAX_BYTES:
        return text

    # Truncate items but keep meta intact so total is preserved.
    truncated: list = []
    base_text = json.dumps({**out, items_key: []}, indent=2, default=str)
    running = len(base_text)
    for it in out[items_key]:
        piece = json.dumps(it, indent=2, default=str)
        if running + len(piece) > _MAX_BYTES - 1_000:
            break
        truncated.append(it)
        running += len(piece) + 4
    out[items_key] = truncated
    out["items_truncated"] = True
    out["items_shown_in_response"] = len(truncated)
    return json.dumps(out, indent=2, default=str)
