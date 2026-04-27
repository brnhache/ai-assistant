FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir --upgrade pip \
    && adduser --disabled-password --gecos "" app

COPY pyproject.toml ./
COPY api ./api
COPY app ./app
COPY config ./config
COPY scripts ./scripts

RUN pip install --no-cache-dir . \
    && chown -R app:app /app

USER app

EXPOSE 8000
# Single worker per container; scale with more ECS tasks. LLM calls are long — avoid many workers per task.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
