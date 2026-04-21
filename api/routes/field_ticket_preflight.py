from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import verify_inbound_service_token
from app.models.requests import FieldTicketPreflightRequest
from app.models.responses import FieldTicketPreflightResponse

router = APIRouter()


@router.post("/field-tickets/preflight", response_model=FieldTicketPreflightResponse)
async def field_ticket_preflight(
    body: FieldTicketPreflightRequest,
    _token: str = Depends(verify_inbound_service_token),
) -> FieldTicketPreflightResponse:
    required = {"field_ticket_preflight", "fuel_surcharge_suggest"}
    if not set(body.capabilities) & required:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=(
                "At least one of field_ticket_preflight or "
                "fuel_surcharge_suggest must be listed"
            ),
        )
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Field ticket preflight chain not implemented yet",
    )
