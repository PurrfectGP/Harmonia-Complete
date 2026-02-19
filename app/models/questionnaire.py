"""
Harmonia V3 — Questionnaire models (responses + reference questions).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class QuestionnaireResponse(Base):
    __tablename__ = "questionnaire_responses"
    __table_args__ = (
        UniqueConstraint("user_id", "question_number", name="uq_user_question"),
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
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="questionnaire_responses")

    def __repr__(self) -> str:
        return (
            f"<QuestionnaireResponse user={self.user_id} "
            f"q={self.question_number} words={self.word_count}>"
        )


class Question(Base):
    """Reference table holding the six canonical Harmonia questions."""

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    question_number: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)

    def __repr__(self) -> str:
        return f"<Question #{self.question_number} category={self.category!r}>"
