"""
Harmonia V3 — ParsingEvidence model (per-question, per-sin scoring evidence).
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ParsingEvidence(Base):
    __tablename__ = "parsing_evidence"
    __table_args__ = (
        Index(
            "ix_parsing_evidence_user_question_sin",
            "user_id",
            "question_number",
            "sin",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    question_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="1-6"
    )
    sin: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    snippet_start_index: Mapped[int] = mapped_column(Integer, nullable=False)
    snippet_end_index: Mapped[int] = mapped_column(Integer, nullable=False)
    interpretation: Mapped[str | None] = mapped_column(Text, nullable=True)
    observer_persona: Mapped[str | None] = mapped_column(String, nullable=True)
    gemini_model_used: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="parsing_evidence")

    def __repr__(self) -> str:
        return (
            f"<ParsingEvidence user={self.user_id} "
            f"q={self.question_number} sin={self.sin!r} score={self.score}>"
        )
