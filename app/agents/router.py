"""Select LLM model id by capability / chain (stub — extend per docs/PROJECT.md)."""

from config.settings import Settings


def resolve_chat_model(settings: Settings) -> str:
    """Fast path default; later: map request capabilities to fast vs strong tier."""
    return settings.openai_model or settings.openai_model_fast
