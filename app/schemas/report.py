from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Any

class CustomerSummary(BaseModel):
    display_score: int
    badges: list[str]
    synopsis: dict  # {headline: str, body: str}
    compatibility_breakdown: dict  # {physical, personality, chemistry}
    shared_traits: list[str]
    conversation_starters: list[str]

class EvidenceEntry(BaseModel):
    sin: str
    score: float
    confidence: float
    evidence_snippet: str
    snippet_location: dict  # {start: int, end: int}
    interpretation: Optional[str] = None

class QuestionEvidenceMap(BaseModel):
    response_text: str
    sin_recognitions: list[EvidenceEntry]

class UserEvidenceMap(BaseModel):
    user_id: UUID
    evidence_map: dict[str, QuestionEvidenceMap]

class CalibrationExampleResponse(BaseModel):
    id: UUID
    question_number: int
    response_text: str
    sin: str
    gemini_raw_score: float
    gemini_raw_confidence: float
    gemini_raw_evidence: str
    validated_score: Optional[float] = None
    validated_by: Optional[str] = None
    review_status: str
    review_notes: Optional[str] = None

    model_config = {"from_attributes": True}

class CalibrationReviewRequest(BaseModel):
    action: str  # approve/correct/reject
    validated_score: Optional[float] = None
    notes: Optional[str] = None
    reviewer: str

class CalibrationBulkReviewItem(BaseModel):
    example_id: UUID
    action: str
    validated_score: Optional[float] = None
    notes: Optional[str] = None

class CalibrationBulkReviewRequest(BaseModel):
    reviews: list[CalibrationBulkReviewItem]
    reviewer: str

class CalibrationStatsResponse(BaseModel):
    total: int
    pending: int
    approved: int
    corrected: int
    rejected: int
    coverage_map: dict
    avg_correction_magnitude: Optional[float] = None
