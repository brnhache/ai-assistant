"""Post-conversation memory extractor.

After every chat turn we run a tiny, cheap LLM chain that looks at the
conversation and decides what (if anything) is worth saving as long-term
memory for the FTM AI assistant.

Why a separate chain (instead of asking the main agent to also write
memories): keeping extraction out of the user-facing turn means

- the user never waits on it,
- we don't bloat the main agent's prompt with "also remember things"
  instructions that compete with the actual task,
- we can run it on a much cheaper model (gpt-5-mini) than the
  Anthropic Opus agent without affecting answer quality.

The extractor outputs structured JSON via Pydantic so we can validate it
before writing to the DB, and silently drop anything malformed.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.memory.store import Memory, save_memory
from config.settings import Settings


class _NewMemory(BaseModel):
    """A single proposed memory the extractor wants to save."""

    kind: str = Field(
        description=(
            "One of: 'mapping' (phrase -> slug/entity), 'preference' "
            "(user preference about answer style or workflow), 'fact' "
            "(durable fact about the tenant or its operations), or "
            "'summary' (condensed summary of this conversation). Use the "
            "most specific kind that fits."
        )
    )
    content: str = Field(
        description=(
            "Short natural-language sentence describing the memory. "
            "Must read like something the assistant could be told at the "
            "start of a future conversation, e.g. 'When the user says "
            "\\\"pumpjack installations\\\" they mean the form with slug "
            "pumpjack-installation-form.'"
        )
    )
    key: str | None = Field(
        default=None,
        description=(
            "Optional stable lookup key for upsert (e.g. "
            "'form_phrase:pumpjack installations'). Use the same key for "
            "the same kind of memory across conversations so we update in "
            "place instead of duplicating. Leave null for free-form facts."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional structured payload, e.g. {'phrase': '...', "
            "'form_slug': '...'}. Useful for mappings."
        ),
    )
    visibility: str = Field(
        default="tenant_all",
        description=(
            "One of: 'tenant_all', 'manager_plus', 'admin_only', 'user_only'. "
            "Default to 'tenant_all' unless the memory is clearly about a "
            "single user's personal preference (use 'user_only')."
        ),
    )


class _ExtractorOutput(BaseModel):
    memories: list[_NewMemory] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are the long-term-memory extractor for an oilfield-services ERP "
    "AI assistant (Field Ticket Master). After every conversation, you "
    "look at the messages and the assistant's existing known memories, "
    "and decide if anything new is worth remembering for future "
    "conversations in this same tenant.\n"
    "\n"
    "Save a memory ONLY when it is durable and useful next time:\n"
    "- phrase mappings the user implicitly or explicitly taught the "
    "assistant (e.g. their words for a custom form);\n"
    "- preferences about how the user wants answers;\n"
    "- durable facts about how this tenant operates that the assistant "
    "should know going forward.\n"
    "\n"
    "Do NOT save:\n"
    "- transient question/answer pairs ('how many tickets today?');\n"
    "- anything sensitive about specific employees' activity;\n"
    "- anything that duplicates or contradicts an existing memory listed "
    "in the input.\n"
    "\n"
    "When in doubt, save nothing. It's better to have zero memories than "
    "noisy or wrong ones."
)


def _format_existing(memories: list[Memory]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(
        f"- [{m.kind}] key={m.key!r} content={m.content!r}" for m in memories
    )


def _format_transcript(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "USER"
        elif isinstance(msg, AIMessage):
            role = "ASSISTANT"
        else:
            continue
        content = msg.content
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict)
            )
        lines.append(f"{role}: {content}".strip())
    return "\n\n".join(lines)


async def extract_and_save_memories(
    settings: Settings,
    *,
    base: str,
    token: str,
    user_id: int | None,
    transcript: list[BaseMessage],
    existing_memories: list[Memory],
    source_conversation_id: str | None = None,
) -> int:
    """Run the extractor on a finished conversation and persist new memories.

    Returns the number of memories actually saved.

    Failures are swallowed (logged to stderr) — memory extraction must
    never propagate an exception back to the chat path.
    """
    if not base or not token:
        return 0
    if not settings.openai_api_key:
        # The extractor is OpenAI-only (cheap). If no key, just no-op.
        return 0
    if not transcript:
        return 0

    # Cap transcript size — last N messages is more than enough to extract from.
    recent = transcript[-12:]
    transcript_text = _format_transcript(recent)
    if not transcript_text.strip():
        return 0

    existing_text = _format_existing(existing_memories)

    user_prompt = (
        f"EXISTING MEMORIES (do not duplicate):\n{existing_text}\n\n"
        f"CONVERSATION:\n{transcript_text}\n\n"
        "Return zero or more memories worth saving. If unsure, return none."
    )

    try:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_memory_extractor_model,
            temperature=0,
        )
        structured = llm.with_structured_output(_ExtractorOutput)
        result = await structured.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:  # pragma: no cover - network failures
        print(
            f"[ai_memory.extractor] llm_failed error={e!s}",
            file=sys.stderr,
            flush=True,
        )
        return 0

    memories = getattr(result, "memories", []) or []
    if not memories:
        return 0

    saved = 0
    for new in memories:
        try:
            mem = (
                new
                if isinstance(new, _NewMemory)
                else _NewMemory.model_validate(new)
            )
        except ValidationError:
            continue

        memory_id = await save_memory(
            base,
            token,
            kind=mem.kind,
            content=mem.content,
            user_id=user_id if mem.visibility == "user_only" else None,
            visibility=mem.visibility,
            key=mem.key,
            metadata=mem.metadata,
            source_conversation_id=source_conversation_id,
        )
        if memory_id is not None:
            saved += 1

    print(
        f"[ai_memory.extractor] saved={saved} of {len(memories)} proposed",
        file=sys.stderr,
        flush=True,
    )
    return saved


def schedule_extractor(
    settings: Settings,
    *,
    base: str,
    token: str,
    user_id: int | None,
    transcript: list[BaseMessage],
    existing_memories: list[Memory],
    source_conversation_id: str | None = None,
) -> None:
    """Fire-and-forget wrapper for use from inside a request handler."""
    try:
        asyncio.create_task(
            extract_and_save_memories(
                settings,
                base=base,
                token=token,
                user_id=user_id,
                transcript=transcript,
                existing_memories=existing_memories,
                source_conversation_id=source_conversation_id,
            )
        )
    except RuntimeError:
        # No running loop — happens in some test contexts. Skip silently.
        pass
