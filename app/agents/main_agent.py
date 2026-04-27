from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.agents.router import resolve_chat_model
from app.models.requests import ChatHistoryMessage
from app.tools.desert.equipment import build_list_equipment_tool
from app.tools.desert.field_tickets import build_list_workorders_tool
from config.settings import Settings

_MAX_TURNS = 40


def _load_main_system_prompt() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "prompts" / "main_system.txt"
    return path.read_text(encoding="utf-8")


def _build_agent_graph(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
):
    if not settings.openai_api_key.strip():
        return None
    tools = [
        build_list_equipment_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        build_list_workorders_tool(
            settings, request_base=request_base, request_token=request_token
        ),
    ]
    model = resolve_chat_model(settings)
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model,
        temperature=0,
    )
    system = _load_main_system_prompt()
    return create_agent(llm, tools, system_prompt=system)


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts).strip()
    return ""


def _turns_to_lc_messages(turns: list[Any] | list[ChatHistoryMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for turn in turns or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
        else:
            role = getattr(turn, "role", None)
            content = (getattr(turn, "content", None) or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


async def invoke_chat_agent(
    settings: Settings,
    message: str,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
    message_history: list[ChatHistoryMessage] | list[dict[str, str]] | None = None,
) -> str:
    graph = _build_agent_graph(
        settings, request_base=request_base, request_token=request_token
    )
    if graph is None:
        raise RuntimeError("OPENAI_API_KEY is not set; chat agent is unavailable")
    prior: list[BaseMessage] = _turns_to_lc_messages(list(message_history or []))
    if len(prior) > _MAX_TURNS:
        prior = prior[-_MAX_TURNS :]
    transcript = prior + [HumanMessage(content=message)]
    result = await graph.ainvoke({"messages": transcript})
    messages = result.get("messages", [])
    text = _last_ai_text(messages)
    return text or str(result)
