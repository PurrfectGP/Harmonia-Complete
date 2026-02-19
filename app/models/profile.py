"""
Harmonia V3 — PersonalityProfile model (aggregated sin scores + quality).
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PersonalityProfile(Base):
    __tablename__ = "personality_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    sins: Mapped[dict] = mapped_column(
        JSONB, nullable=False, comment="7 aggregated sin scores"
    )
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_tier: Mapped[str] = mapped_column(
        String, nullable=False, comment="high / moderate / low / rejected"
    )
    response_styles: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    flags: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Array of flag strings"
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True, comment="Extra metadata (column name: metadata)"
    )
    source: Mapped[str] = mapped_column(
        String, nullable=False, comment="real_user / claude_agent"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="personality_profile")
    calibration_examples: Mapped[list["CalibrationExample"]] = relationship(
        "CalibrationExample", back_populates="source_profile"
    )

    def __repr__(self) -> str:
        return (
            f"<PersonalityProfile user={self.user_id} "
            f"tier={self.quality_tier!r} v={self.version}>"
        )
