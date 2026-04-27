"""Structured logs for HTTP calls to the Desert Laravel API from tools (CloudWatch)."""

import logging
from collections.abc import Sequence

log = logging.getLogger("desert.api")


def _safe_base(base: str | None) -> str:
    if not base:
        return "(empty)"
    b = str(base).strip()[:200]
    return b


def log_desert_get_start(tool: str, base: str | None, path: str) -> None:
    log.info(
        "desert_tool request tool=%s base_url=%s path=%s",
        tool,
        _safe_base(base),
        path,
    )


def log_desert_get_ok(tool: str, base: str | None, path: str, status: int, data_keys: Sequence[str]) -> None:
    log.info(
        "desert_tool ok tool=%s base_url=%s path=%s http_status=%s top_level_keys=%s",
        tool,
        _safe_base(base),
        path,
        status,
        list(data_keys)[:20] if data_keys is not None else None,
    )


def log_desert_get_http_error(
    tool: str,
    base: str | None,
    path: str,
    status: int,
    body_snippet: str,
) -> None:
    snippet = (body_snippet or "")[:500]
    log.warning(
        "desert_tool http_error tool=%s base_url=%s path=%s http_status=%s body_snippet=%r",
        tool,
        _safe_base(base),
        path,
        status,
        snippet,
    )


def log_desert_get_request_failed(tool: str, base: str | None, path: str, err: str) -> None:
    log.warning(
        "desert_tool request_failed tool=%s base_url=%s path=%s error=%s",
        tool,
        _safe_base(base),
        path,
        err,
    )


def log_desert_tool_config_error(tool: str, base: str | None, reason: str) -> None:
    """Log non-HTTP configuration / context errors (e.g. missing token or base URL).

    These do not trigger an HTTP call, so they won't be captured by the normal
    request_* helpers above. Using this ensures CloudWatch shows why a tool
    immediately returned an "error:" string to the LLM.
    """
    log.warning(
        "desert_tool config_error tool=%s base_url=%s reason=%s",
        tool,
        _safe_base(base),
        reason,
    )
