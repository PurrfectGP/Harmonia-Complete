"""
Harmonia V3 — User model.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    photos: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Array of photo URLs"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    questionnaire_responses: Mapped[list["QuestionnaireResponse"]] = relationship(
        "QuestionnaireResponse", back_populates="user", cascade="all, delete-orphan"
    )
    personality_profile: Mapped["PersonalityProfile"] = relationship(
        "PersonalityProfile", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    visual_preference: Mapped["VisualPreference"] = relationship(
        "VisualPreference", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    visual_ratings: Mapped[list["VisualRating"]] = relationship(
        "VisualRating", back_populates="user", cascade="all, delete-orphan"
    )
    hla_data: Mapped["HLAData"] = relationship(
        "HLAData", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    parsing_evidence: Mapped[list["ParsingEvidence"]] = relationship(
        "ParsingEvidence", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email!r} id={self.id}>"
