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
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_model_fast: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_FAST")
    openai_model_strong: str = Field(default="gpt-4o", alias="OPENAI_MODEL_STRONG")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest", alias="ANTHROPIC_MODEL")

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
