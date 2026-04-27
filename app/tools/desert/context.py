"""Per-request Desert API URL + bearer token (from Laravel) for correct tenant host."""

from contextvars import ContextVar, Token

_desert_api_token: ContextVar[str | None] = ContextVar("desert_api_token", default=None)
_desert_api_base: ContextVar[str | None] = ContextVar("desert_api_base", default=None)


def set_desert_api_context(
    *,
    base_url: str | None,
    token: str | None,
) -> tuple[Token | None, Token | None]:
    t_tok: Token | None = None
    t_base: Token | None = None
    if token not in (None, ""):
        t_tok = _desert_api_token.set(token)
    if base_url not in (None, ""):
        t_base = _desert_api_base.set(base_url.rstrip("/"))
    return (t_base, t_tok)


def reset_desert_api_context(tokens: tuple[Token | None, Token | None]) -> None:
    t_base, t_tok = tokens
    if t_tok is not None:
        _desert_api_token.reset(t_tok)
    if t_base is not None:
        _desert_api_base.reset(t_base)


def get_desert_api_base(fallback: str) -> str:
    b = _desert_api_base.get()
    if b:
        return b

    return fallback.rstrip("/")


def get_desert_bearer_token(fallback: str) -> str:
    t = _desert_api_token.get()
    if t:
        return t

    return fallback
