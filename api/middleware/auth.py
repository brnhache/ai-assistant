from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


async def verify_inbound_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    if credentials.credentials != settings.inbound_service_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )
    return credentials.credentials
