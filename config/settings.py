from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration. See `.env.example`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inbound_service_token: str = Field(
        ...,
        description="Bearer token Laravel sends when calling this service",
        alias="INBOUND_SERVICE_TOKEN",
    )

    desert_api_base_url: str = Field(..., alias="DESERT_API_BASE_URL")
    desert_service_token: str = Field(default="", alias="DESERT_SERVICE_TOKEN")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    # Kept as the documented OpenAI fallback chat model when Anthropic is unavailable.
    openai_model: str = Field(default="gpt-5.1", alias="OPENAI_MODEL")
    openai_model_fast: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_FAST")
    # Cheap model used for the AI long-term memory extractor (post-conversation).
    openai_memory_extractor_model: str = Field(default="gpt-5-mini", alias="OPENAI_MEMORY_EXTRACTOR_MODEL")
    openai_model_strong: str = Field(default="gpt-4o", alias="OPENAI_MODEL_STRONG")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Primary chat model. Opus chosen for faithfulness on tool-calling agents — see
    # docs note in app/agents/main_agent.py.
    anthropic_model: str = Field(default="claude-opus-4-7", alias="ANTHROPIC_MODEL")

    # When true (default), use Anthropic as primary and fall back to OpenAI on failure.
    # When false, use OpenAI directly (for testing or if you want to disable Anthropic).
    use_anthropic_primary: bool = Field(default=True, alias="USE_ANTHROPIC_PRIMARY")

    database_url: str = Field(default="", alias="DATABASE_URL")
    redis_url: str = Field(default="", alias="REDIS_URL")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
