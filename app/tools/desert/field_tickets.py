import json

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import Settings


class _WorkordersArgs(BaseModel):
    note: str = Field(
        default="",
        description="Optional context from the user question (ignored for the HTTP call).",
    )


def build_list_workorders_tool(settings: Settings) -> StructuredTool:
    base = settings.desert_api_base_url.rstrip("/")

    async def _run(note: str = "") -> str:
        if not settings.desert_service_token:
            return "error: DESERT_SERVICE_TOKEN is not configured"
        url = f"{base}/workorders"
        headers = {
            "Authorization": f"Bearer {settings.desert_service_token}",
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            return f"error calling Desert API: {e!s}"
        text = json.dumps(data, indent=2, default=str)
        if len(text) > 24_000:
            return text[:24_000] + "\n…(truncated)"
        return text

    return StructuredTool.from_function(
        name="desert_list_workorders",
        description=(
            "List field tickets (workorders) for the tenant via Desert GET /workorders."
        ),
        args_schema=_WorkordersArgs,
        coroutine=_run,
    )
