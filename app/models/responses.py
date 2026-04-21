from typing import Any

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    type: str
    id: int | None = None
    url: str | None = None


class SuggestedAction(BaseModel):
    label: str
    url: str | None = None
    action: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[SourceRef] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    conversation_id: str | None = None


class BriefingSection(BaseModel):
    title: str
    content: str
    priority: str = "medium"
    links: list[str] = Field(default_factory=list)


class BriefingResponse(BaseModel):
    sections: list[BriefingSection] = Field(default_factory=list)
    generated_at: str | None = None


class PreflightFinding(BaseModel):
    code: str
    severity: str
    message: str
    field_hints: list[str] = Field(default_factory=list)


class FuelSurchargeBlock(BaseModel):
    eligible: bool = False
    suggested_misc_rate: float | None = None
    suggested_amount: float | None = None
    currency: str | None = None
    policy_version: str | None = None
    index: dict[str, Any] | None = None
    rationale: str | None = None
    disclaimer: str | None = None


class FieldTicketPreflightResponse(BaseModel):
    workorder_id: int
    findings: list[PreflightFinding] = Field(default_factory=list)
    fuel_surcharge: FuelSurchargeBlock | None = None
    model_trace_id: str | None = None


class ReconcileResponse(BaseModel):
    summary: str = ""
    discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
