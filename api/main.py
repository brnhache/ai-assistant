from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import briefing, chat, field_ticket_preflight, health, reconciliation
from config.settings import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Warm settings cache (loads .env)
    get_settings()
    yield


def _configure_logging() -> None:
    """Ensure our loggers (desert.chat, desert.api, etc.) go to stdout.

    We don't touch uvicorn's own loggers; instead we attach a handler to the
    "desert" logger subtree so `desert.chat` / `desert.api` reliably show up
    in ECS/CloudWatch regardless of how uvicorn configures root.
    """
    base_logger = logging.getLogger("desert")
    if any(isinstance(h, logging.StreamHandler) for h in base_logger.handlers):
        # Already configured (tests or prior init).
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    handler.setFormatter(formatter)
    base_logger.addHandler(handler)
    # Respect LOG_LEVEL from settings (default info)
    try:
        level_name = get_settings().log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
    except Exception:
        level = logging.INFO
    base_logger.setLevel(level)
    # Children (desert.chat, desert.api, etc.) will propagate to this logger.
    base_logger.propagate = False


def create_app() -> FastAPI:
    _configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="Desert AI Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(briefing.router, prefix="/api/v1")
    app.include_router(field_ticket_preflight.router, prefix="/api/v1")
    app.include_router(reconciliation.router, prefix="/api/v1")
    return app


app = create_app()
