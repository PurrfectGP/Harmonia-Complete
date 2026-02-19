from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, Any

class ProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    version: int
    sins: dict
    quality_score: float
    quality_tier: str
    response_styles: Optional[dict] = None
    flags: list[str] = []
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}

class SinScore(BaseModel):
    score: float
    confidence: float
    evidence: str
    evidence_start: int = -1
    evidence_end: int = -1
    variance: Optional[float] = None

class TraitBreakdown(BaseModel):
    sin: str
    aggregated_score: float
    confidence: float
    variance: float
    weight: float
    questions_contributing: int
