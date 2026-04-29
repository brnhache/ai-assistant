import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - dep missing in local dev
    ChatAnthropic = None  # type: ignore[assignment]

from app.agents.router import resolve_chat_model
from app.models.requests import ChatHistoryMessage
from app.tools.desert.custom_forms import (
    build_list_custom_forms_tool,
    build_list_form_submissions_tool,
    build_resolve_form_name_tool,
)
from app.tools.desert.equipment import build_list_equipment_tool
from app.tools.desert.field_tickets import build_list_workorders_tool
from config.settings import Settings

_MAX_TURNS = 40


def _load_main_system_prompt() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "prompts" / "main_system.txt"
    return path.read_text(encoding="utf-8")


def _build_chat_llm(settings: Settings):
    """Build the chat LLM with primary + fallback.

    Why Anthropic primary:
        Tool-calling agents that hand the model tool output and ask it to
        summarise / count / quote are sensitive to confabulation. Claude
        Opus is markedly better than the GPT-4o family at saying "I see N
        items" without inventing extras. We had a real incident where
        gpt-4o-mini returned 30 tickets when the tool output had 28, then
        fabricated FT09999 + FT10000 with fake clients to retrofit the
        wrong count. Opus avoids that class of failure.

    Why OpenAI fallback:
        If Anthropic has an outage we still want the assistant to work.
        gpt-5.1 (current frontier OpenAI) is roughly comparable for tool
        faithfulness and noticeably better than 4o-mini.
    """
    has_anthropic = (
        settings.use_anthropic_primary
        and bool(settings.anthropic_api_key.strip())
        and ChatAnthropic is not None
    )
    has_openai = bool(settings.openai_api_key.strip())

    openai_llm = None
    if has_openai:
        openai_llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=resolve_chat_model(settings),
            temperature=0,
        )

    if has_anthropic:
        anthropic_llm = ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=0,
            max_tokens=4096,
        )
        if openai_llm is not None:
            print(
                f"[desert.chat] llm primary=anthropic/{settings.anthropic_model} "
                f"fallback=openai/{resolve_chat_model(settings)}",
                file=sys.stderr,
                flush=True,
            )
            return anthropic_llm.with_fallbacks([openai_llm])
        print(
            f"[desert.chat] llm primary=anthropic/{settings.anthropic_model} (no fallback configured)",
            file=sys.stderr,
            flush=True,
        )
        return anthropic_llm

    if openai_llm is not None:
        print(
            f"[desert.chat] llm primary=openai/{resolve_chat_model(settings)} (anthropic disabled or unavailable)",
            file=sys.stderr,
            flush=True,
        )
        return openai_llm

    return None


def _build_agent_graph(
    settings: Settings,
    *,
    request_base: str | None = None,
    request_token: str | None = None,
):
    llm = _build_chat_llm(settings)
    if llm is None:
        return None
    tools = [
        build_list_equipment_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        build_list_workorders_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        build_list_custom_forms_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        build_list_form_submissions_tool(
            settings, request_base=request_base, request_token=request_token
        ),
        build_resolve_form_name_tool(
            settings, request_base=request_base, request_token=request_token
        ),
    ]
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
        raise RuntimeError(
            "No chat LLM configured. Set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY."
        )
    prior: list[BaseMessage] = _turns_to_lc_messages(list(message_history or []))
    if len(prior) > _MAX_TURNS:
        prior = prior[-_MAX_TURNS :]
    transcript = prior + [HumanMessage(content=message)]
    result = await graph.ainvoke({"messages": transcript})
    messages = result.get("messages", [])
    text = _last_ai_text(messages)
    return text or str(result)
