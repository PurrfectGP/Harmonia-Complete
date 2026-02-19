from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, Any

class MatchCalculateResponse(BaseModel):
    match_id: UUID
    user_a_id: UUID
    user_b_id: UUID
    wtm_score: float
    s_vis_a_to_b: float
    s_vis_b_to_a: float
    s_psych: float
    s_bio: Optional[float] = None
    customer_summary: dict

class MatchListItem(BaseModel):
    match_id: UUID
    other_user_id: UUID
    other_user_name: str
    wtm_score: float
    created_at: datetime

class SwipeCreate(BaseModel):
    swiper_id: UUID
    target_id: UUID
    direction: str = "right"  # left/right/superlike

class SwipeResponse(BaseModel):
    status: str
    is_mutual_match: bool
    match_id: Optional[UUID] = None

class VisualCalibrateRequest(BaseModel):
    user_id: UUID
    ratings: list[dict]  # [{image_id: str, rating: int}]

class VisualScoreRequest(BaseModel):
    user_id: UUID
    target_user_id: UUID

class HLAUploadResponse(BaseModel):
    status: str
    snp_count: int
    imputation_confidence: float
