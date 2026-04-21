from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import briefing, chat, field_ticket_preflight, health, reconciliation
from config.settings import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Warm settings cache (loads .env)
    get_settings()
    yield


def create_app() -> FastAPI:
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
