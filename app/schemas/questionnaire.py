from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from typing import Optional

class QuestionResponse(BaseModel):
    user_id: UUID
    question_number: int = Field(ge=1, le=6)
    response_text: str

    @field_validator("response_text")
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count < 25:
            raise ValueError(f"Response too short: {word_count} words (minimum 25)")
        if word_count > 150:
            raise ValueError(f"Response too long: {word_count} words (maximum 150)")
        return v

class QuestionResponseBatch(BaseModel):
    user_id: UUID
    responses: list[QuestionResponse]

    # Note: each QuestionResponse in batch doesn't need user_id repeated
    # but for simplicity we keep it flat

class QuestionSubmitResponse(BaseModel):
    status: str
    question_number: int
    word_count: int

class BatchSubmitResponse(BaseModel):
    status: str
    profile_id: Optional[UUID] = None
    quality_tier: Optional[str] = None
    quality_score: Optional[float] = None

class SingleResponseSubmit(BaseModel):
    user_id: UUID
    question_number: int = Field(ge=1, le=6)
    response_text: str

    @field_validator("response_text")
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count < 25:
            raise ValueError(f"Response too short: {word_count} words (minimum 25)")
        if word_count > 150:
            raise ValueError(f"Response too long: {word_count} words (maximum 150)")
        return v

class BatchResponseSubmit(BaseModel):
    user_id: UUID
    responses: list[dict]  # [{question_number: int, response_text: str}]
