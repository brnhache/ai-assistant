from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import verify_inbound_service_token
from app.models.requests import BriefingRequest
from app.models.responses import BriefingResponse

router = APIRouter()


@router.post("/briefing", response_model=BriefingResponse)
async def briefing(
    body: BriefingRequest,
    _token: str = Depends(verify_inbound_service_token),
) -> BriefingResponse:
    if "briefing" not in body.capabilities:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Capability "briefing" not included in capabilities for this request',
        )
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Briefing chain not implemented yet",
    )
