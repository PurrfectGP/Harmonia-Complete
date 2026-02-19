"""
Harmonia V3 — Visual preference and rating models.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class VisualPreference(Base):
    __tablename__ = "visual_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    support_set_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    mandatory_traits: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    preferred_traits: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    aversion_traits: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    adapted_weights_key: Mapped[str | None] = mapped_column(
        String, nullable=True, comment="Redis key for adapted weights"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="visual_preference")

    def __repr__(self) -> str:
        return f"<VisualPreference user={self.user_id}>"


class VisualRating(Base):
    __tablename__ = "visual_ratings"
    __table_args__ = (
        Index("ix_visual_ratings_user_rating", "user_id", "rating"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[str] = mapped_column(String, nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="1-5"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="visual_ratings")

    def __repr__(self) -> str:
        return f"<VisualRating user={self.user_id} image={self.image_id!r} rating={self.rating}>"
