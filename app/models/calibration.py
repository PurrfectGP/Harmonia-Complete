"""
Harmonia V3 — CalibrationExample model (ground-truth calibration data).
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class CalibrationExample(Base):
    __tablename__ = "calibration_examples"
    __table_args__ = (
        Index(
            "ix_calibration_question_sin_status",
            "question_number",
            "sin",
            "review_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="1-6"
    )
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    sin: Mapped[str] = mapped_column(String, nullable=False)
    gemini_raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    gemini_raw_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    gemini_raw_evidence: Mapped[str] = mapped_column(Text, nullable=False)
    validated_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    review_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        server_default="pending",
        comment="pending / approved / corrected / rejected",
    )
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("personality_profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    source_profile: Mapped["PersonalityProfile"] = relationship(
        "PersonalityProfile", back_populates="calibration_examples"
    )

    def __repr__(self) -> str:
        return (
            f"<CalibrationExample q={self.question_number} sin={self.sin!r} "
            f"status={self.review_status!r}>"
        )
