import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import verify_inbound_service_token
from app.agents.main_agent import invoke_chat_agent
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse
from config.settings import Settings, get_settings

router = APIRouter()


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
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM is not configured (set OPENAI_API_KEY)",
        )
    try:
        reply = await invoke_chat_agent(settings, body.message)
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    conv = body.conversation_id or str(uuid.uuid4())
    return ChatResponse(reply=reply, conversation_id=conv)
