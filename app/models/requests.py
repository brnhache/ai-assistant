from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    capabilities: list[str] = Field(default_factory=list)
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def capabilities_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("capabilities must be non-empty")
        return v


class BriefingRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    capabilities: list[str] = Field(default_factory=list)
    type: str = Field(default="morning")
    preferences: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def capabilities_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("capabilities must be non-empty")
        return v


class FieldTicketPreflightRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    workorder_id: int = Field(..., ge=1)
    capabilities: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def capabilities_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("capabilities must be non-empty")
        return v


class ReconcileRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    capabilities: list[str] = Field(default_factory=list)
    type: str = Field(default="field_tickets_vs_invoices")
    date_range: dict[str, str] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def capabilities_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("capabilities must be non-empty")
        return v
