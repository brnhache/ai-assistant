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
from app.agents.tool_registry import build_desert_tools
from app.agents.confirmation import apply_confirmation_policy
from app.memory.extractor import schedule_extractor
from app.memory.store import (
    Memory,
    format_memories_for_prompt,
    load_relevant_memories,
    touch_memory,
)
from app.models.requests import ChatHistoryMessage
from app.tools.desert.custom_forms import (
    build_list_custom_forms_tool,
    build_list_form_submissions_tool,
    build_resolve_form_name_tool,
)
from app.tools.desert.equipment import build_list_equipment_tool
from app.tools.desert.field_tickets import build_list_workorders_tool
from app.tools.desert.conversation import build_recall_conversation_tool
from config.settings import Settings

_MAX_TURNS = 40
_MEMORY_KINDS = ["mapping", "preference", "fact"]


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
    system_prompt_override: str | None = None,
):
    llm = _build_chat_llm(settings)
    if llm is None:
        return None

    built = build_desert_tools(
        settings, request_base=request_base, request_token=request_token
    )
    tools = apply_confirmation_policy(built.tools, built.metadata, surface="chat")

    system = system_prompt_override or _load_main_system_prompt()
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
    user_id: int | None = None,
    user_role: str | None = None,
    conversation_id: str | None = None,
) -> str:
    # 1. Load relevant long-term memories for this user (if we know who they are).
    memories: list[Memory] = []
    if request_base and request_token and user_id is not None and user_role:
        try:
            memories = await load_relevant_memories(
                request_base,
                request_token,
                user_id=user_id,
                user_role=user_role,
                kinds=_MEMORY_KINDS,
                limit=20,
            )
        except Exception as e:  # pragma: no cover - never block chat on memory
            print(
                f"[desert.chat] memory_load_failed user_id={user_id} error={e!s}",
                file=sys.stderr,
                flush=True,
            )
            memories = []

    base_system = _load_main_system_prompt()
    if memories:
        memory_block = format_memories_for_prompt(memories)
        system_prompt = (
            f"{base_system}\n\n"
            "# Long-term memory (things you've learned about this tenant/user)\n\n"
            "Treat these as durable facts about the current account. Prefer them over guesses, "
            "but if a tool call returns data that contradicts a memory, trust the live data and "
            "surface the discrepancy in plain language.\n\n"
            f"{memory_block}"
        )
    else:
        system_prompt = base_system

    graph = _build_agent_graph(
        settings,
        request_base=request_base,
        request_token=request_token,
        system_prompt_override=system_prompt,
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

    # 2. Touch memories that were available this turn so frequently-used ones
    #    rank higher next time. Cheap; we don't await individually.
    if memories and request_base and request_token:
        for m in memories:
            try:
                await touch_memory(request_base, request_token, memory_id=m.id)
            except Exception:  # pragma: no cover
                pass

    # 3. Fire-and-forget memory extraction over this conversation.
    if request_base and request_token and user_id is not None:
        full_transcript = transcript + [AIMessage(content=text or "")]
        schedule_extractor(
            settings,
            base=request_base,
            token=request_token,
            user_id=user_id,
            transcript=full_transcript,
            existing_memories=memories,
            source_conversation_id=conversation_id,
        )

    return text or str(result)
