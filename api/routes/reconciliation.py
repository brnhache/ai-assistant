from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import verify_inbound_service_token
from app.models.requests import ReconcileRequest
from app.models.responses import ReconcileResponse

router = APIRouter()


@router.post("/reconcile", response_model=ReconcileResponse)
async def reconcile(
    body: ReconcileRequest,
    _token: str = Depends(verify_inbound_service_token),
) -> ReconcileResponse:
    if "reconcile" not in body.capabilities:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Capability "reconcile" not included in capabilities for this request',
        )
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Reconciliation chain not implemented yet",
    )
