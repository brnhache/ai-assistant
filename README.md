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

Build a production image from the `ai-assistant` repo root:

```bash
docker build -f docker/Dockerfile -t desert-ai-assistant:latest .
```

## Hosting (production)

Desert runs on **Laravel Vapor**; this service must run **elsewhere** (never inside the Vapor bundle). The plan in `docs/PROJECT.md` is **AWS ECS on Fargate** behind an **HTTPS** load balancer. Other PaaS options work the same way if they give you a stable URL and TLS.

### 1. Runtime requirements

- **Image**: the Dockerfile above (Python 3.12, FastAPI on port **8000**).
- **Health check**: `GET /health` → `{"status":"ok"}` (use for ALB/target group health).
- **Secrets** (store in **AWS Secrets Manager** or SSM Parameter Store, inject as env — do not commit):
  - `INBOUND_SERVICE_TOKEN` — shared secret; Laravel’s `AI_ASSISTANT_SERVICE_TOKEN` must **match exactly**.
  - `OPENAI_API_KEY` (and/or other LLM keys per your router).
  - `DESERT_API_BASE_URL` — **fallback only** (e.g. local dev). In production, Laravel sends **`desert_api_base_url`** on every chat (the tenant’s `https://{subdomain}/api`); the agent must use that, not the central marketing host, or tool calls hit the wrong database.
  - Optional: `DESERT_SERVICE_TOKEN` only if you ever call Desert **without** Laravel forwarding per-user tokens (Laravel normally sends `desert_api_token` + `desert_api_base_url` on each chat).
- **Outbound**: the task must reach **OpenAI** (or your LLM vendor), **each tenant’s Desert API** (`https://{tenant-host}/api/...`), and optionally **LangSmith**.

**Logs (debugging tool → Laravel calls):** In **CloudWatch** for the ECS service, look for `desert.chat` (inbound chat: `tenant_id`, `user_id`, `desert_api_base_url`, `token_len` — no secrets) and `desert.api` (per tool: `desert_tool` request/ok/http_error with `base_url` + `path`, and a short `body_snippet` on non-2xx). On **Laravel (Vapor)** application logs, each proxy logs `AI assistant chat proxy` with the same `desert_api_base_url` and `token_source` (`request_bearer` vs `minted_pat`).

### 2. AWS ECS Fargate (recommended if you already use AWS)

1. **ECR**: create a repository, push `desert-ai-assistant:latest` from CI (e.g. GitHub Actions `aws ecr get-login-password` + `docker push`).
2. **VPC**: use the same VPC as RDS/other services if you later add Postgres/Redis in-VPC; for chat-only, public subnets + NAT for outbound are enough.
3. **ECS cluster** → **Fargate task definition**:
   - Container port **8000**, protocol HTTP.
   - CPU/memory: start **1 vCPU / 2 GB**; raise if you see OOM or slow cold starts.
   - Env vars from secrets (see list above).
4. **Application Load Balancer** (HTTPS):
   - Listener **443** with ACM certificate on a hostname you control, e.g. `assistant.yourdomain.com`.
   - Target group → ECS service, health check `GET /health`, **generous** interval (LLM load can slow readiness if misconfigured).
5. **Service**: desired count ≥ 1 per environment; enable **deployment circuit breaker**; optional **auto scaling** on CPU or request count.
6. **Security groups**:
   - ALB: allow **443** from the internet (or from CloudFront only if you front it).
   - ECS tasks: allow inbound **only from the ALB security group** on port 8000; allow **outbound HTTPS** to the world (LLM APIs) and to Desert tenant hosts.

### 3. Wire Laravel (Vapor)

In the **Vapor** (or `.env`) environment for each Desert stage:

```env
AI_ASSISTANT_GATEWAY_ENABLED=true
AI_ASSISTANT_BASE_URL=https://assistant.yourdomain.com
AI_ASSISTANT_SERVICE_TOKEN=<same as INBOUND_SERVICE_TOKEN>
```

Redeploy Desert. Tenant admins still turn the feature on under **Settings → AI Assistant**.

### 4. Staging first

- Deploy the assistant to a **staging** URL and set the same variables on **Desert staging** only.
- Confirm: `curl -sS https://assistant.../health` and one authenticated chat from the app.

### 5. Simpler alternatives (good for staging or early prod)

- **Fly.io**, **Railway**, **Render**, **Google Cloud Run**: container on port 8000, set the same env vars, attach HTTPS URL, put that URL in `AI_ASSISTANT_BASE_URL`.

### 6. Operational notes

- **Timeouts**: ALB idle timeout and Laravel `AiAssistantClient` HTTP timeout should be **≥ 120s** for long LLM turns (already 120s in Laravel).
- **Cold starts**: Fargate min capacity 1 avoids first-request latency if traffic is low.
- **Logs**: ship container stdout to **CloudWatch Logs**; add request IDs in FastAPI later if needed.

## Repo layout

This directory is intended as its **own git repository** (sibling to `desert/`). The parent `desert-platform/` folder holds shared docs and Cursor rules.

## Tests

```bash
pytest
```
