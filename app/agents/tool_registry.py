"""Central registry for Desert/FTM tools + metadata.

Right now this is thin: it builds the small set of existing tools and tags
each with a risk tier. As we add write/external tools, this becomes the
single place where we track which ones require confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain_core.tools import BaseTool

from app.tools.desert.conversation import build_recall_conversation_tool
from app.tools.desert.custom_forms import (
    build_list_custom_forms_tool,
    build_list_form_submissions_tool,
    build_resolve_form_name_tool,
)
from app.tools.desert.equipment import build_list_equipment_tool
from app.tools.desert.field_tickets import build_list_workorders_tool
from app.tools.desert.qbo import (
    build_qbo_connection_status_tool,
    build_qbo_list_customers_tool,
)
from config.settings import Settings


@dataclass(frozen=True)
class ToolMeta:
    name: str
    risk: str  # "read" | "write_low" | "write_medium" | "write_high" | "external"


@dataclass
class BuiltTools:
    tools: List[BaseTool]
    metadata: Dict[str, ToolMeta]


def build_desert_tools(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
) -> BuiltTools:
    """Build all Desert/FTM tools + their metadata for the main chat agent."""

    tools: List[BaseTool] = []
    meta: Dict[str, ToolMeta] = {}

    def _add(tool: BaseTool, risk: str) -> None:
        tools.append(tool)
        meta[tool.name] = ToolMeta(name=tool.name, risk=risk)

    _add(
        build_list_equipment_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )
    _add(
        build_list_workorders_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )
    _add(
        build_recall_conversation_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )
    _add(
        build_list_custom_forms_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )
    _add(
        build_list_form_submissions_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )
    _add(
        build_resolve_form_name_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="read",
    )

    # QuickBooks (read-only, via Desert API; still "external" from FTM's POV).
    _add(
        build_qbo_connection_status_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="external",
    )
    _add(
        build_qbo_list_customers_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        risk="external",
    )

    return BuiltTools(tools=tools, metadata=meta)
