"""Pydantic mirrors of Desert domain objects — expand as tools need structured parsing."""

from pydantic import BaseModel


class EquipmentStub(BaseModel):
    id: int
    name: str | None = None
