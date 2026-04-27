"""Per-request API URL and token (must not rely on env default host for multi-tenant)."""

from config.settings import Settings

from app.tools.desert.context import get_desert_api_base, get_desert_bearer_token


def resolve_desert_base_and_token(
    settings: Settings,
    *,
    request_base: str | None,
    request_token: str | None,
) -> tuple[str, str]:
    """
    Return (api_base, bearer) for this chat turn.

    Laravel must send the tenant's API base (e.g. https://tenant.app/api) on each chat. The
    service env DESERT_API_BASE_URL is only a dev fallback; using the central host here would
    return data for no tenant and break tool calls.
    """
    default_base = settings.desert_api_base_url.rstrip("/")
    token = (request_token or "").strip()
    if not token:
        token = get_desert_bearer_token(settings.desert_service_token)
    base = (request_base or "").strip().rstrip("/")
    if not base:
        base = get_desert_api_base(default_base)
    return base, token
