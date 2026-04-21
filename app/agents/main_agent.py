from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.agents.router import resolve_chat_model
from app.tools.desert.equipment import build_list_equipment_tool
from app.tools.desert.field_tickets import build_list_workorders_tool
from config.settings import Settings


def _load_main_system_prompt() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "prompts" / "main_system.txt"
    return path.read_text(encoding="utf-8")


def _build_agent_graph(settings: Settings):
    if not settings.openai_api_key.strip():
        return None
    tools = [
        build_list_equipment_tool(settings),
        build_list_workorders_tool(settings),
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


async def invoke_chat_agent(settings: Settings, message: str) -> str:
    graph = _build_agent_graph(settings)
    if graph is None:
        raise RuntimeError("OPENAI_API_KEY is not set; chat agent is unavailable")
    result = await graph.ainvoke({"messages": [HumanMessage(content=message)]})
    messages = result.get("messages", [])
    text = _last_ai_text(messages)
    return text or str(result)
