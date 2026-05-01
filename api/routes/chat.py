import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import verify_inbound_service_token
from app.agents.main_agent import invoke_chat_agent
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse
from app.tools.desert.context import reset_desert_api_context, set_desert_api_context
from config.settings import Settings, get_settings

router = APIRouter()
log = logging.getLogger("desert.chat")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _token: str = Depends(verify_inbound_service_token),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    if "chat" not in body.capabilities:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Capability "chat" not included in capabilities for this request',
        )
    if not (settings.anthropic_api_key.strip() or settings.openai_api_key.strip()):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM is not configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY)",
        )
    import sys

    tok = body.desert_api_token or ""
    # Print to stderr for ECS/CloudWatch even if logging config is odd.
    print(
        "[desert.chat] chat_inbound tenant_id=%s user_id=%s base=%s token_len=%s"
        % (
            body.tenant_id,
            body.user_id,
            (body.desert_api_base_url or "")[:200] or None,
            len(tok) if tok else 0,
        ),
        file=sys.stderr,
        flush=True,
    )
    log.info(
        "chat_inbound tenant_id=%s user_id=%s desert_api_base_url=%s token_len=%s",
        body.tenant_id,
        body.user_id,
        (body.desert_api_base_url or "")[:200] or None,
        len(tok) if tok else 0,
    )
    ctx = set_desert_api_context(
        base_url=body.desert_api_base_url,
        token=body.desert_api_token,
    )
    user_role_raw = body.context.get("user_role") if isinstance(body.context, dict) else None
    user_role = str(user_role_raw).lower() if user_role_raw else "user"
    if user_role not in {"admin", "manager", "user"}:
        user_role = "user"
    llm_mode_raw = body.context.get("llm_mode") if isinstance(body.context, dict) else None
    llm_mode = str(llm_mode_raw) if llm_mode_raw else None
    conv = body.conversation_id or str(uuid.uuid4())
    try:
        reply = await invoke_chat_agent(
            settings,
            body.message,
            request_base=body.desert_api_base_url,
            request_token=body.desert_api_token,
            message_history=body.message_history,
            user_id=body.user_id,
            user_role=user_role,
            conversation_id=conv,
            llm_mode=llm_mode,
        )
        print(
            "[desert.chat] chat_reply ok tenant_id=%s user_id=%s"
            % (body.tenant_id, body.user_id),
            file=sys.stderr,
            flush=True,
        )
    except RuntimeError as e:
        print(
            "[desert.chat] chat_runtime_error tenant_id=%s user_id=%s detail=%s"
            % (body.tenant_id, body.user_id, str(e)),
            file=sys.stderr,
            flush=True,
        )
        log.error(
            "chat_runtime_error tenant_id=%s user_id=%s detail=%s",
            body.tenant_id,
            body.user_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except Exception:
        print(
            "[desert.chat] chat_agent_failed tenant_id=%s user_id=%s"
            % (body.tenant_id, body.user_id),
            file=sys.stderr,
            flush=True,
        )
        log.exception(
            "chat_agent_failed tenant_id=%s user_id=%s",
            body.tenant_id,
            body.user_id,
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed",
        ) from None
    finally:
        reset_desert_api_context(ctx)
    return ChatResponse(reply=reply, conversation_id=conv)
