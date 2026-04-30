"""Skeleton for tool confirmation / risk gating.

Today this is a no-op wrapper that simply returns the tools unchanged, but it
is the hook point for future work where we want to intercept high-risk tool
calls and require explicit confirmation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from langchain_core.tools import BaseTool

from app.agents.tool_registry import ToolMeta


def apply_confirmation_policy(
    tools: Iterable[BaseTool],
    metadata: Dict[str, ToolMeta],
    *,
    surface: str = "chat",  # later: "voice" | "chat" | etc.
) -> List[BaseTool]:
    """Apply confirmation / gating policy to a set of tools.

    For now this is intentionally a no-op; all existing tools are read-only.
    When we introduce write/external tools, this function becomes the single
    place we wrap or replace those tools with confirmation-aware versions.
    """

    # Future sketch (not implemented yet):
    # - Wrap high-risk tools to emit a structured "proposed_action" object
    #   instead of executing immediately.
    # - The frontend (chat/voice) surfaces that to the user for approval.
    # - On approval, a follow-up call executes the actual tool.

    return list(tools)
