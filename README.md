# Desert AI Assistant

Python **FastAPI** + **LangChain** / **LangGraph** service that integrates with **Field Ticket Master** (Desert Laravel app).

## Spec

Canonical architecture, API contract, and rollout rules: **[`../docs/PROJECT.md`](../docs/PROJECT.md)** (in the `desert-platform` workspace).

## Quick start

```bash
cd ai-assistant
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env: INBOUND_SERVICE_TOKEN, DESERT_* , OPENAI_API_KEY
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Chat (requires Bearer token + body per PROJECT.md): `POST http://localhost:8000/api/v1/chat`

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Repo layout

This directory is intended as its **own git repository** (sibling to `desert/`). The parent `desert-platform/` folder holds shared docs and Cursor rules.

## Tests

```bash
pytest
```
